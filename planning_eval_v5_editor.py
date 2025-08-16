# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import yaml
import os
import numpy as np
import lpips
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
import json
import shutil

from diffusers.models import AutoencoderKL

### evo evaluation library ###
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync, metrics
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation

from diffusion import create_diffusion
from datasets_v2 import TrajectoryEvalDataset
from isolated_nwm_infer_v5 import model_forward_wrapper
from misc import calculate_delta_yaw, get_action_torch, save_planning_pred, log_viz_single, transform, unnormalize_data
from isolated_nwm_eval import save_metric_to_disk
import distributed as dist
from models_tpz_v5 import CDiT_models


with open("config/data_config.yaml", "r") as f:
    data_config = yaml.safe_load(f)

with open("config/data_hyperparams_plan.yaml", "r") as f:
    data_hyperparams = yaml.safe_load(f)

ACTION_STATS_TORCH = {}
for key in data_config['action_stats']:
    ACTION_STATS_TORCH[key] = torch.tensor(data_config['action_stats'][key])

# ===========================
# NEW: Editor 输出工具
# ===========================
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _tensor_to_uint8_rgb(x):
    """
    x: (C,H,W) tensor in [-1,1] or [0,1] -> uint8 HWC
    """
    x = x.detach().to(torch.float32).cpu()
    if x.min() < 0:
        x = (x + 1) / 2
    x = x.clamp(0, 1)
    x = (x * 255.0).round().to(torch.uint8)
    return x.permute(1, 2, 0).numpy()

def _save_png(tchw, path):
    Image.fromarray(_tensor_to_uint8_rgb(tchw)).save(path)

def _save_clean_grid(imgs, nrow=None, save_path="grid.png", figsize=(12, 12)):
    with torch.no_grad():
        if imgs.min() < 0:
            imgs = (imgs + 1) / 2
        imgs = imgs.clamp(0, 1)
        if nrow is None:
            N = imgs.size(0)
            nrow = max(1, int(N ** 0.5))
        grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
        np_grid = grid.to(torch.float32).permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(np_grid)
    ax.axis("off")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ----- 你原有的绘图函数（保留） -----
def plot_images_with_losses(preds, losses, save_path="predictions_with_losses.png"):
    preds = (preds + 1) / 2
    ncol = int(preds.size(0)**0.5)
    nrow = preds.size(0) // ncol
    if ncol * nrow < preds.size(0):
        nrow += 1
    grid_img = vutils.make_grid(preds, nrow=ncol, padding=2)
    np_grid = grid_img.to(torch.float32).permute(1, 2, 0).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(50, 50))
    ax.imshow(np_grid)
    ax.axis("off")

    img_height, img_width = np_grid.shape[0] // nrow, np_grid.shape[1] // ncol
    for idx, loss in enumerate(losses):
        row = idx // ncol
        col = idx % ncol
        x = col * img_width
        y = row * img_height
        text = "GT Goal" if idx == 0 else f"ID: {idx - 1}  Loss: {loss:.2f}"
        ax.text(x + img_width / 2, y + 15, text, color="white", 
                ha="center", va="top", fontsize=50, backgroundcolor="black")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_batch_final(init_imgs, pred_imgs, goal_imgs, idxs, losses, save_path="final_plan.png"):
    imgs_for_plotting = torch.cat([init_imgs, pred_imgs, goal_imgs])
    imgs_for_plotting = (imgs_for_plotting + 1) / 2
    ncol = init_imgs.shape[0]
    grid_img = vutils.make_grid(imgs_for_plotting, nrow=ncol, padding=2)
    np_grid = grid_img.to(torch.float32).permute(1, 2, 0).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(ncol * 10, 30))
    ax.imshow(np_grid)
    ax.axis("off")

    img_height, img_width = np_grid.shape[0] // 3, np_grid.shape[1] // ncol
    for i in range(ncol):
        x = i * img_width
        y_pred = img_height
        ax.text(x + img_width / 2, y_pred + 15, f"ID: {int(idxs[i].item())} Loss: {losses[i]:.2f}", 
                color="white", ha="center", va="top", fontsize=40, backgroundcolor="black")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
def plot_batch_trajectories(init_imgs, pred_imgs_seq, goal_imgs, idxs, save_path="trajectory_grid.png"):
    B, T, C, H, W = pred_imgs_seq.shape
    init_imgs = (init_imgs + 1) / 2
    goal_imgs = (goal_imgs + 1) / 2
    pred_imgs_seq = (pred_imgs_seq + 1) / 2

    tiles = []
    for b in range(B):
        traj = [init_imgs[b]] + [pred_imgs_seq[b, t] for t in range(T)] + [goal_imgs[b]]
        tiles.append(torch.stack(traj))
    tiles = torch.cat(tiles, dim=0)

    grid_img = vutils.make_grid(tiles, nrow=T+2, padding=2)
    np_grid = grid_img.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(max(12, (T+2) * 4), B * 4))
    ax.imshow(np_grid)
    ax.axis("off")

    H, W = init_imgs.shape[-2:]
    img_height, img_width = H + 2, W + 2
    for b in range(B):
        for t in range(T + 2):
            x = t * img_width
            y = b * img_height
            label = "Init" if t == 0 else ("Goal" if t == T + 1 else f"Step {t}")
            ax.text(x + img_width / 2, y + 20, f"ID:{int(idxs[b].item())} {label}",
                    color="white", ha="center", va="top", fontsize=16, backgroundcolor="black")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def get_dataset_eval(config, dataset_name, predefined_index=False):
    data_cfg = config["eval_datasets"][dataset_name]
    predefined_index = f"data_splits/{dataset_name}/test/navigation_eval.pkl" if predefined_index else None
    return TrajectoryEvalDataset(
        data_folder=data_cfg["data_folder"],
        data_split_folder=data_cfg["test"],
        dataset_name=dataset_name,
        image_size=config["image_size"],
        min_dist_cat=config["trajectory_eval_distance"]["min_dist_cat"],
        max_dist_cat=config["trajectory_eval_distance"]["max_dist_cat"],
        len_traj_pred=config["trajectory_eval_len_traj_pred"],
        traj_stride=config["traj_stride"], 
        context_size=config["trajectory_eval_context_size"],
        normalize=config["normalize"],
        transform=transform,
        predefined_index=predefined_index,
        traj_names="rollout_traj_names.txt"
    )

class WM_Planning_Evaluator:
    def __init__(self, args):
        super().__init__()  
        self.args = args
        self.exp = args.exp
        _, _, device, _ = dist.init_distributed()
        self.device = torch.device(device)
        
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        
        self.exp_eval = self.exp
        self.get_eval_name()

        with open("config/eval_config.yaml", "r") as f:
            default_config = yaml.safe_load(f)
        self.config = default_config

        with open(self.exp_eval, "r") as f:
            user_config = yaml.safe_load(f)
        self.config.update(user_config)

        latent_size = self.config['image_size'] // 8
        self.latent_size = self.config['image_size'] // 8
        self.num_cond = self.config['eval_context_size']
        
        if self.args.save_preds:
            exp_name = os.path.basename(self.args.exp).split('.')[0]
            self.args.save_output_dir = os.path.join(args.output_dir, exp_name)
            os.makedirs(self.args.save_output_dir, exist_ok=True)
                
        self.dataset_names = self.args.datasets.split(',')
        self.datasets = {}
        for dataset_name in self.dataset_names:
            dataset_val = get_dataset_eval(self.config, dataset_name, predefined_index=False)
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Eval dataset not divisible by #processes; entries will be duplicated slightly.')

            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

            curr_data_loader = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=False
            )
            self.datasets[dataset_name] = curr_data_loader
        
        print("loading")
        model = CDiT_models[self.config['model']](
            context_size=self.num_cond,
            input_size=latent_size,
        )

        ckp = torch.load(f'{self.config["results_dir"]}/{self.config["run_name"]}/checkpoints/{args.ckp}.pth.tar', map_location='cpu', weights_only=False)
        model.load_state_dict(ckp["ema"], strict=True)
        model.eval()
        model.to(self.device)
        self.model = torch.compile(model)
        self.diffusion = create_diffusion(str(250))
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device], find_unused_parameters=False)
        self.model_without_ddp = self.model.module
         
        # ---- LPIPS 初始化（放在同一 device）并打补丁把 scaling_layer 常量迁走 ----
        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)  # 用于逐帧 loss
        try:
            sl = self.loss_fn.scaling_layer  # 一些版本里存在
            sl.shift = sl.shift.to(self.device)
            sl.scale = sl.scale.to(self.device)
        except Exception as e:
            print("[LPIPS hotfix] scaling_layer device patch failed:", e)
        # -----------------------------------------------------------------------

        self.mode = 'cem'
        self.num_samples = self.args.num_samples
        self.topk = self.args.topk
        self.opt_steps = self.args.opt_steps
        self.num_repeat_eval = self.args.num_repeat_eval
        self.action_dim = 4

    def init_mu_sigma(self, obs_0, traj_len):
        n_evals = obs_0.shape[0]
        mu = torch.zeros(n_evals, self.action_dim) 
        mu[:, ] = torch.tensor(data_hyperparams[self.args.datasets]['mu'])
        sigma = torch.ones([n_evals, self.action_dim])
        sigma[:, ] = torch.tensor(data_hyperparams[self.args.datasets]['var_scale']) 
        return mu, sigma
        
    def generate_actions(self, dataset_save_output_dir, dataset_name, idxs, obs_image, goal_image, gt_actions, len_traj_pred, aug_image):
        idx_int = int(idxs.flatten()[0].item())
        image_plot_dir = _ensure_dir(os.path.join(dataset_save_output_dir, 'plots'))
        videos_plot_dir = _ensure_dir(os.path.join(dataset_save_output_dir, 'videos'))

        # editor/run_{idx_string}/
        editor_run_dir = _ensure_dir(os.path.join(videos_plot_dir, "editor", f"run_{idx_int:03d}"))

        n_evals = obs_image.shape[0]
        mu, sigma = self.init_mu_sigma(obs_image, len_traj_pred)
        mu, sigma = mu.to(self.device), sigma.to(self.device)

        for i in range(self.opt_steps):
            losses = []
            for traj in range(n_evals):
                traj_id = int(idxs.flatten()[traj].item())
                sample = (torch.randn(self.num_samples, self.action_dim).to(self.device) * sigma[traj] + mu[traj])
                single_delta = sample[:, :3]
                deltas = single_delta.unsqueeze(1).repeat(1, len_traj_pred, 1)
                unnorm_deltas = unnormalize_data(deltas, ACTION_STATS_TORCH)
                delta_yaw = calculate_delta_yaw(unnorm_deltas)
                deltas = torch.cat((deltas, delta_yaw.to(deltas.device)), dim=-1)
                deltas[:, -1, -1] += sample[:, -1] * np.pi

                cur_obs_image = obs_image[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1) 
                cur_goal_image = goal_image[traj].unsqueeze(0).repeat(self.args.num_samples, 1, 1, 1, 1).squeeze(1)
                
                if self.num_repeat_eval * self.num_samples > 120:
                    cur_losses = []
                    for _ in range(self.num_repeat_eval):
                        preds = self.autoregressive_rollout(cur_obs_image, deltas, self.args.rollout_stride, aug_image=aug_image)
                        preds = preds[:, -1]
                        loss = self.loss_fn(preds.to(self.device), cur_goal_image.to(self.device)).flatten(0)
                        cur_losses.append(loss)
                    loss = torch.stack(cur_losses).mean(dim=0)
                else:
                    expanded_deltas = deltas.repeat(self.num_repeat_eval, 1, 1) 
                    expanded_obs_image = cur_obs_image.repeat(self.num_repeat_eval, 1, 1, 1, 1) 
                    expanded_goal_image = cur_goal_image.repeat(self.num_repeat_eval, 1, 1, 1) 

                    preds = self.autoregressive_rollout(expanded_obs_image, expanded_deltas, self.args.rollout_stride, aug_image=aug_image)
                    preds = preds[:, -1]

                    loss = self.loss_fn(preds.to(self.device), expanded_goal_image.to(self.device)).flatten(0)
                    loss = loss.view(self.num_repeat_eval, -1).mean(dim=0)
                    preds = preds[:self.args.num_samples]

                sorted_idx = torch.argsort(loss)
                topk_idx = sorted_idx[:self.topk]
                topk_action = deltas[topk_idx][:, -1]
                losses.append(loss[topk_idx[0]].item())   
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

                if self.args.plot:
                    self.visualize_trajectories(
                        dataset_name, gt_actions, image_plot_dir, i, traj, traj_id,
                        deltas, cur_obs_image, cur_goal_image, preds, loss, topk_idx
                    )                    
        
        # Final rollout 
        deltas = mu[:, :3]
        deltas = deltas.unsqueeze(1).repeat(1, len_traj_pred, 1)

        unnorm_deltas = unnormalize_data(deltas, ACTION_STATS_TORCH)
        delta_yaw = calculate_delta_yaw(unnorm_deltas)
        deltas = torch.cat((deltas, delta_yaw.to(deltas.device)), dim=-1)
        deltas[:, -1, -1] += mu[:, -1] * np.pi

        preds = self.autoregressive_rollout(obs_image, deltas, self.args.rollout_stride, aug_image=aug_image)
        preds_completed = preds          # (B,T,C,H,W)
        preds_last = preds[:, -1]        # (B,C,H,W)

        final_loss = self.loss_fn(preds_last.to(self.device), goal_image.squeeze(1).to(self.device)).flatten(0)

        if self.args.save_preds:
            save_planning_pred(dataset_save_output_dir, n_evals, idxs, obs_image, goal_image, preds_last, deltas, final_loss, gt_actions, preds_completed)
        
        # ---------- NEW: 每样本逐帧原图 + 每帧 loss 写入 JSON（不依赖 --plot） ----------
        # 逐帧 loss 定义：LPIPS(frame, goal)；对 init、每个 step 与 goal 都计算（goal 自己设 0.0）
        B = obs_image.shape[0]
        T = preds_completed.shape[1]
        init_imgs = obs_image[:, -1]               # (B,C,H,W)
        goal_imgs = goal_image.squeeze(1)          # (B,C,H,W)

        for b in range(B):
            sid = int(idxs[b].item())
            frames_dir = _ensure_dir(os.path.join(editor_run_dir, "frames"))

            # 1) 存图：init, step_*, goal
            _save_png(init_imgs[b], os.path.join(frames_dir, "init.png"))
            for t in range(T):
                _save_png(preds_completed[b, t], os.path.join(frames_dir, f"step_{t:03d}.png"))
            _save_png(goal_imgs[b], os.path.join(frames_dir, "goal.png"))

            # 2) 逐帧 LPIPS loss
            init_loss = float(self.loss_fn(init_imgs[b:b+1].to(self.device),
                                           goal_imgs[b:b+1].to(self.device)).item())
            step_losses = []
            for t in range(T):
                step_losses.append(float(self.loss_fn(preds_completed[b, t:t+1].to(self.device),
                                                      goal_imgs[b:b+1].to(self.device)).item()))
            goal_loss = 0.0

            # 3) 写 metadata.json（结构对齐你的示例）
            # - num_frames = T + 1（INIT + T步）；sequence = [0..T-1]
            # - frames[0] 为 INIT，其后 frames[1..] 动作为 0..T-1
            H, W = init_imgs.shape[-2:]
            meta = {
                "run_index": sid,                      # 用样本 id 作为 run_index（更直观）
                "video_mp4": "",                       # evaluator 无 mp4，这里留空；如需可后处理生成
                "fps": int(self.args.rollout_stride) if self.args.rollout_stride > 0 else 1,
                "resolution": {"width": int(W), "height": int(H)},
                "num_frames": T + 1,
                "sequence": list(range(T)),
                "commands": {},                        # 评估为连续动作，无离散 commands
                "frames": []
            }
            # INIT
            meta["frames"].append({
                "frame": 0,
                "png": "frames/init.png",
                "action": "INIT",
                "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "theta_rad": 0.0},     # 若有真实位姿可在此填入
                "delta": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dtheta": 0.0},
                "loss": init_loss
            })
            # STEPS
            for t in range(T):
                meta["frames"].append({
                    "frame": t + 1,
                    "png": f"frames/step_{t:03d}.png",
                    "action": t,   # 数字动作 0..T-1，与示例一致
                    "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "theta_rad": 0.0},  # 可填真实
                    "delta": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dtheta": 0.0},  # 可填真实
                    "loss": step_losses[t]
                })
            # （可选）也可把 goal 作为最后一帧写入
            # meta["frames"].append({
            #     "frame": T + 1,
            #     "png": "frames/goal.png",
            #     "action": "GOAL",
            #     "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "theta_rad": 0.0},
            #     "delta": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dtheta": 0.0},
            #     "loss": 0.0
            # })

            _write_json(os.path.join(editor_run_dir, "metadata.json"), meta)
        # -------------------------------------------------------------------------

        # 仍保留原有画图（--plot 才会生成）
        if self.args.plot:
            idx_string = "_".join(map(str, idxs.flatten().int().tolist())) 
            img_name = os.path.join(image_plot_dir, f'FINAL_{idx_string}.png')
            traj_name = os.path.join(image_plot_dir, f"TRAJ_{idx_string}.png")
            plot_batch_final(obs_image[:, -1].to(self.device), preds_last, goal_image.squeeze(1).to(self.device), idxs, final_loss.tolist(), save_path=img_name)
            plot_batch_trajectories(obs_image[:, -1].to(self.device), preds_completed, goal_image.squeeze(1).to(self.device), idxs, save_path=traj_name)

        pred_actions = get_action_torch(deltas[:, :, :3], ACTION_STATS_TORCH)
        pred_yaw = deltas[:, :, -1].sum(1)
        return pred_actions, pred_yaw

    def visualize_trajectories(self, dataset_name, gt_actions, image_plot_dir, i, traj, traj_id, deltas, cur_obs_image, cur_goal_image, preds, loss, topk_idx):
        img_for_plotting = torch.cat([cur_goal_image[0:1].to(self.device), preds])
        loss_for_plotting = torch.cat((torch.tensor([0]).to(self.device), loss))
        img_name = os.path.join(image_plot_dir, f'idx{traj_id}_iter{i}.png')
        plot_images_with_losses(img_for_plotting, loss_for_plotting, save_path=img_name)
        plot_name = os.path.join(image_plot_dir, f'idx{traj_id}_iter{i}_trajs.png')
        num_plot = self.args.num_samples
        log_viz_single(
            dataset_name, 
            cur_obs_image[0], 
            cur_goal_image[0], 
            preds[:num_plot], 
            deltas[:num_plot], 
            loss[:num_plot], 
            topk_idx[0:1], 
            gt_actions[traj], 
            ACTION_STATS_TORCH, 
            plan_iter=i, 
            output_dir=plot_name
        )
    
    def autoregressive_rollout(self, obs_image, deltas, rollout_stride, aug_image):
        deltas = deltas.unflatten(1, (-1, rollout_stride)).sum(2)
        preds = []
        curr_obs = obs_image.clone().to(self.device)
        for i in range(deltas.shape[1]):
            curr_delta = deltas[:, i:i+1]
            all_models = self.model, self.diffusion, self.vae
            x_pred_pixels = model_forward_wrapper(all_models, curr_obs, curr_delta, self.args.rollout_stride, self.latent_size, num_cond=self.num_cond, device=self.device, x_supervised=aug_image)
            x_pred_pixels = x_pred_pixels.unsqueeze(1)
            curr_obs = torch.cat((curr_obs, x_pred_pixels), dim=1)
            curr_obs = curr_obs[:, 1:]
            preds.append(x_pred_pixels)
        preds = torch.cat(preds, 1)
        return preds
    
    def get_eval_name(self):
        self.eval_name = f'CEM_N{self.args.num_samples}_K{self.args.topk}_RS{self.args.rollout_stride}_rep{self.args.num_repeat_eval}_OPT{self.args.opt_steps}'
        
    def actions_to_traj(self, actions):
        positions_xyz = torch.zeros((actions.shape[0], 3))
        positions_xyz[:, :3] = actions
        orientations_quat_wxyz = torch.zeros((actions.shape[0], 4))
        orientations_quat_wxyz[:, -1] = 1
        timestamps = torch.arange(actions.shape[0], dtype=torch.float64)
        traj = PoseTrajectory3D(positions_xyz=positions_xyz, orientations_quat_wxyz=orientations_quat_wxyz, timestamps=timestamps)
        return traj
    
    @torch.no_grad
    def evaluate(self):
        for dataset_name in self.dataset_names:
            metric_logger = dist.MetricLogger(delimiter="  ")
            header = 'Test:'
            eval_save_output_dir = None
            
            if self.args.save_preds:
                dataset_save_output_dir = _ensure_dir(os.path.join(self.args.save_output_dir, dataset_name))
                eval_save_output_dir = _ensure_dir(os.path.join(dataset_save_output_dir, self.eval_name))
            
            curr_data_loader = self.datasets[dataset_name]
            for (idxs, obs_image, goal_image, gt_actions, goal_pos, aug_image) in metric_logger.log_every(curr_data_loader, 1, header):
                obs_image = obs_image[:, -self.num_cond:]
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    pred_actions, pred_yaw = self.generate_actions(eval_save_output_dir, dataset_name, idxs, obs_image, goal_image, gt_actions, self.config["trajectory_eval_len_traj_pred"], aug_image=aug_image)
                for i in range(len(obs_image)):
                    pred_traj_i = self.actions_to_traj(pred_actions[i, :, :3])
                    gt_traj_i = self.actions_to_traj(gt_actions[i, :, :3])
                    
                    ate, rpe_trans, _ = self.eval_metrics(gt_traj_i, pred_traj_i)

                    pred_final_pos = pred_actions[i, -1, :3].to('cpu')
                    pred_final_yaw = pred_yaw[i].to('cpu')
                    goal_final_pos = goal_pos[i, 0, :3]
                    goal_final_yaw = goal_pos[i, 0, -1]
                    pos_diff_norm = torch.norm(pred_final_pos - goal_final_pos)
                    yaw_diff = pred_final_yaw - goal_final_yaw
                    yaw_diff_norm = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).abs()
                    
                    metric_logger.meters[f'{dataset_name}_ate'].update(ate, n=1)
                    metric_logger.meters[f'{dataset_name}_rpe_trans'].update(rpe_trans, n=1)
                    metric_logger.meters[f'{dataset_name}_pos_diff_norm'].update(pos_diff_norm, n=1)   
                    metric_logger.meters[f'{dataset_name}_yaw_diff_norm'].update(yaw_diff_norm, n=1)   
            output_fn = os.path.join(self.args.save_output_dir, f'{dataset_name}_{self.eval_name}.json')
            save_metric_to_disk(metric_logger, output_fn)

        metric_logger.synchronize_between_processes()
            
    def eval_metrics(self, traj_ref, traj_pred):
        traj_ref, traj_pred = sync.associate_trajectories(traj_ref, traj_pred)
        
        result = main_ape.ape(traj_ref, traj_pred, est_name='traj',
            pose_relation=PoseRelation.translation_part, align=False, correct_scale=False)
        ate = result.stats['rmse']

        result = main_rpe.rpe(traj_ref, traj_pred, est_name='traj',
            pose_relation=PoseRelation.rotation_angle_deg, align=False, correct_scale=False,
            delta=1.0, delta_unit=metrics.Unit.frames, rel_delta_tol=0.1)
        rpe_rot = result.stats['rmse']

        result = main_rpe.rpe(traj_ref, traj_pred, est_name='traj',
            pose_relation=PoseRelation.translation_part, align=False, correct_scale=False,
            delta=1.0, delta_unit=metrics.Unit.frames, rel_delta_tol=0.1)
        rpe_trans = result.stats['rmse']

        return ate, rpe_trans, rpe_rot
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Default Args
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument("--ckp", type=str, default='0100000', help="checkpoint id")

    parser.add_argument("--datasets", type=str, default=None, help="dataset name(s), comma-separated")
    parser.add_argument("--output_dir", type=str, default=None, help="output dir to save model predictions")
    parser.add_argument('--save_preds', action='store_true', default=False, help='whether to save prediction tensors')
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    
    # Planning Specific Args
    parser.add_argument("--num_samples", type=int, default=10, help="num CEM samples")
    parser.add_argument("--rollout_stride", type=int, default=1, help="rollout stride")
    parser.add_argument("--topk", type=int, default=5, help="top k for CEM")
    parser.add_argument("--opt_steps", type=int, default=15, help="CEM iterations")
    parser.add_argument("--num_repeat_eval", type=int, default=1, help="repeats per sample for stability")
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()
    
    evaluator = WM_Planning_Evaluator(args)
    _ = int(os.environ.get("LOCAL_RANK", 0))
    _ = torch.cuda.current_device()
    evaluator.evaluate()
