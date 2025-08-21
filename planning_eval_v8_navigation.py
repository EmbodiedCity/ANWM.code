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

from diffusers.models import AutoencoderKL
from noise_trajectory_generation import trajectory_generation_random as trajectory_generation

### evo evaluation library ###
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync, metrics
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation

from diffusion import create_diffusion
from datasets_v3 import TrajectoryEvalDataset
from isolated_nwm_infer_v8 import model_forward_wrapper
from misc import calculate_delta_yaw, get_action_torch, save_planning_pred, log_viz_single, transform, unnormalize_data
from isolated_nwm_eval import save_metric_to_disk
import distributed as dist
from models_tpz_v8 import CDiT_models


with open("config/data_config.yaml", "r") as f:
    data_config = yaml.safe_load(f)

with open("config/data_hyperparams_plan.yaml", "r") as f:
    data_hyperparams = yaml.safe_load(f)

ACTION_STATS_TORCH = {}
for key in data_config['action_stats']:
    ACTION_STATS_TORCH[key] = torch.tensor(data_config['action_stats'][key])

# ========= Helper utils for editor outputs =========
def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _tensor_to_uint8_rgb(x: torch.Tensor):
    """
    x: (C,H,W) tensor in [-1,1] or [0,1] -> numpy uint8 HWC
    """
    x = x.detach().to(torch.float32).cpu()
    if x.min() < 0:
        x = (x + 1) / 2
    x = x.clamp(0, 1)
    x = (x * 255.0).round().to(torch.uint8)
    return x.permute(1, 2, 0).numpy()

def _save_png(tchw: torch.Tensor, path: str):
    Image.fromarray(_tensor_to_uint8_rgb(tchw)).save(path)

def _write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ======= (原) 绘图函数 =======
def plot_images_with_losses(preds, losses, save_path="predictions_with_losses.png"):
    # Denormalize images from [-1, 1] to [0, 1]
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

    # Overlay the losses on each image
    for idx, loss in enumerate(losses):
        row = idx // ncol
        col = idx % ncol
        x = col * img_width
        y = row * img_height
        if idx == 0:
            text = f"GT Goal"
        else:
            text = f"ID: {idx - 1}  Loss: {loss:.2f}"
        ax.text(x + img_width / 2, y + 15, text, color="white", 
                ha="center", va="top", fontsize=50, backgroundcolor="black")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_batch_final(init_imgs, pred_imgs, goal_imgs, idxs, losses, save_path="final_plan.png"):
    # images are (B, c, h, w)
    imgs_for_plotting = torch.cat([init_imgs, pred_imgs, goal_imgs])
    imgs_for_plotting = (imgs_for_plotting + 1) / 2
    ncol = init_imgs.shape[0]
    grid_img = vutils.make_grid(imgs_for_plotting, nrow=ncol, padding=2)
    np_grid = grid_img.to(torch.float32).permute(1, 2, 0).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(ncol * 10, 30))  # Adjust size as needed
    ax.imshow(np_grid)
    ax.axis("off")

    img_height, img_width = np_grid.shape[0] // 3, np_grid.shape[1] // ncol

    # Overlay the IDs and losses on each image pair in the grid
    for i in range(ncol):
        x = i * img_width
        y_pred = img_height
        ax.text(x + img_width / 2, y_pred + 15, f"ID: {int(idxs[i].item())} Loss: {losses[i]:.2f}", 
                color="white", ha="center", va="top", fontsize=40, backgroundcolor="black")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
def plot_batch_trajectories(init_imgs, pred_imgs_seq, goal_imgs, idxs, save_path="trajectory_grid.png"):
    """
    Visualize a batch of image trajectories:
    - init_imgs: (B, C, H, W)
    - pred_imgs_seq: (B, T, C, H, W)  --> multiple steps
    - goal_imgs: (B, C, H, W)
    - idxs: (B,) tensor of IDs
    """
    B, T, C, H, W = pred_imgs_seq.shape
    device = init_imgs.device
    imgs_for_plotting = []

    # Denormalize from [-1, 1] to [0, 1]
    init_imgs = (init_imgs + 1) / 2
    goal_imgs = (goal_imgs + 1) / 2
    pred_imgs_seq = (pred_imgs_seq + 1) / 2

    for b in range(B):
        traj = [init_imgs[b]]  # Start with initial image
        traj += [pred_imgs_seq[b, t] for t in range(T)]  # Add each predicted step
        traj += [goal_imgs[b]]  # End with goal image
        imgs_for_plotting.append(torch.stack(traj))

    # Stack all batch trajectories vertically: shape → (B*(T+2), C, H, W)
    imgs_for_plotting = torch.cat(imgs_for_plotting, dim=0)

    # Create image grid: (B rows, T+2 cols)
    grid_img = vutils.make_grid(imgs_for_plotting, nrow=T+2, padding=2)
    np_grid = grid_img.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(max(12, (T+2) * 4), B * 4))
    ax.imshow(np_grid)
    ax.axis("off")

    img_height, img_width = H + 2, W + 2  # account for padding
    for b in range(B):
        for t in range(T + 2):  # init + T preds + goal
            x = t * img_width
            y = b * img_height
            if t == 0:
                label = f"ID:{int(idxs[b].item())} Init"
            elif t == T + 1:
                label = "Goal"
            else:
                label = f"Step {t}"
            ax.text(x + img_width / 2, y + 20, label, color="white",
                    ha="center", va="top", fontsize=16, backgroundcolor="black")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def get_dataset_eval(config, dataset_name, predefined_index=True):
    data_config = config["eval_datasets"][dataset_name]
    if predefined_index:
        predefined_index = f"data_splits/{dataset_name}/test/navigation_eval.pkl"
    else:
        predefined_index = None

    dataset = TrajectoryEvalDataset(
                data_folder=data_config["data_folder"],
                data_split_folder=data_config["test"],
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
    
    return dataset

class WM_Planning_Evaluator:
    def __init__(self, args):
        super().__init__()  
        self.args = args
        self.exp = args.exp
        _, _, device, _ = dist.init_distributed()
        self.device = torch.device(device)
        
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        
        # Setting up Config
        # self.exp_eval = f'{self.exp}_nomad_eval' # local paths etc.
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
        
        # logging directory
        if self.args.save_preds:
            exp_name = os.path.basename(self.args.exp).split('.')[0]
            self.args.save_output_dir = os.path.join(args.output_dir, exp_name)
            os.makedirs(self.args.save_output_dir, exist_ok=True)
                
        # Loading Datasets
        self.dataset_names = self.args.datasets.split(',')
        self.datasets = {}
        for dataset_name in self.dataset_names:
            dataset_val = get_dataset_eval(self.config, dataset_name, predefined_index=True)
            
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
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
        
        # Loading Model
        print("loading")
        model = CDiT_models[self.config['model']](
            context_size=self.num_cond+1,
            input_size=latent_size,
        )

        ckp = torch.load(f'{self.config["results_dir"]}/{self.config["run_name"]}/checkpoints/{args.ckp}.pth.tar', map_location='cpu', weights_only=False)
        model.load_state_dict(ckp["ema"], strict=True)
        model.eval()
        model.to(self.device)
        self.model = torch.compile(model)
        self.diffusion = create_diffusion(str(250))
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device], find_unused_parameters=False)
        self.model_without_ddp = self.model.module
         
        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.mode = 'cem' # assume CEM for planning
        self.num_samples = self.args.num_samples
        self.topk = self.args.topk
        self.opt_steps = self.args.opt_steps
        self.num_repeat_eval = self.args.num_repeat_eval
        self.action_dim = 4 # hardcoded (delta_x, delta_y, delta_z, delta_yaw)
        
    def generate_actions(self, dataset_save_output_dir, dataset_name, idxs, obs_image, goal_image, gt_actions, len_traj_pred, aug_image, camera_mats):
        idx_string = "_".join(map(str, idxs.flatten().int().tolist())) 
        image_plot_dir = os.path.join(dataset_save_output_dir, 'plots')
        os.makedirs(image_plot_dir, exist_ok=True)
        videos_plot_dir = os.path.join(dataset_save_output_dir, 'videos')
        os.makedirs(videos_plot_dir, exist_ok=True)
        
        n_evals = obs_image.shape[0]
        candidate_number = self.num_samples 
        all_deltas = []
        all_losses = []
        all_preds = []

        for traj in range(n_evals):
            traj_id = int(idxs.flatten()[traj].item())

            # === 构造 GT 轨迹（含起点），用于你的随机轨迹生成器 ===
            gt_deltas_xyz = gt_actions[traj, :, :3].to('cpu').numpy()  # [T, 3]
            gt_xyz = np.concatenate([np.zeros((1, 3), dtype=np.float32),
                                     np.cumsum(gt_deltas_xyz, axis=0).astype(np.float32)], axis=0)  # [T+1, 3]
            T = gt_xyz.shape[0]
            gt_yaw = np.zeros((T,), dtype=np.float32)  # 相似度只用 xyz，yaw 给 0 即可
            GT_traj = [(float(x), float(y), float(z), float(yaw)) for (x, y, z), yaw in zip(gt_xyz, gt_yaw)]

            # 生成候选 trajectory poses
            candidate_trajectories = trajectory_generation(GT_traj, candidate_number=candidate_number)
            candidate_trajectories = np.array(candidate_trajectories, dtype=np.float32)  # [N, T, 4] 其中 T = len_traj_pred + 1

            # 转换为 delta（轨迹点之间的变化）
            candidate_deltas = candidate_trajectories[:, 1:, :] - candidate_trajectories[:, :-1, :]  # [N, steps, 4]
            deltas = torch.tensor(candidate_deltas, dtype=torch.float32, device=self.device)

            cur_obs_image = obs_image[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1) 
            cur_goal_image = goal_image[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1).squeeze(1)
            cur_aug_image = aug_image[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1)   # (num_samples, 1, C, H, W)
            cur_cam = camera_mats[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1)          # (num_samples, num_cond+1, 4, 4)            
            # WM is stochastic, so we can repeat the evaluation of each trajectory and average to reduce variance
            if self.num_repeat_eval * self.num_samples > 120:
                cur_losses = []
                for r in range(self.num_repeat_eval):
                    preds = self.autoregressive_rollout(cur_obs_image, deltas, self.args.rollout_stride, aug_image=cur_aug_image, camera_mats=cur_cam)
                    preds = preds[:, -1] # take the last predicted image
                    loss = self.loss_fn(preds.to(self.device), cur_goal_image.to(self.device)).flatten(0)
                    cur_losses.append(loss)

                loss = torch.stack(cur_losses).mean(dim=0)
            else:
                expanded_deltas = deltas.repeat(self.num_repeat_eval, 1, 1) 
                expanded_obs_image = cur_obs_image.repeat(self.num_repeat_eval, 1, 1, 1, 1) 
                expanded_goal_image = cur_goal_image.repeat(self.num_repeat_eval, 1, 1, 1) 
                expanded_aug = cur_aug_image.repeat(self.num_repeat_eval, 1, 1, 1, 1)
                expanded_cams = cur_cam.repeat(self.num_repeat_eval, 1, 1, 1)  

                preds = self.autoregressive_rollout(expanded_obs_image, expanded_deltas, self.args.rollout_stride, aug_image=expanded_aug, camera_mats=expanded_cams)
                preds = preds[:, -1]

                loss = self.loss_fn(preds.to(self.device), expanded_goal_image.to(self.device)).flatten(0)
                loss = loss.view(self.num_repeat_eval, -1)
                loss = loss.mean(dim=0)

                preds = preds[:self.num_samples]

            sorted_idx = torch.argsort(loss)
            topk_idx = sorted_idx[:self.topk]
            best_idx = sorted_idx[0]
            best_delta = deltas[best_idx]  # [steps, 4]

            all_deltas.append(best_delta.unsqueeze(0))
            all_losses.append(loss[best_idx].item())
            all_preds.append(preds[best_idx].unsqueeze(0))

            if self.args.plot:
                self.visualize_trajectories(dataset_name, gt_actions, image_plot_dir, 1, traj, traj_id, deltas, cur_obs_image, cur_goal_image, preds, loss, topk_idx)                    
    
        # Final rollout for selected deltas
        final_deltas = torch.cat(all_deltas, dim=0)  # [n_evals, steps, 4]
        preds = self.autoregressive_rollout(obs_image, final_deltas, self.args.rollout_stride, aug_image=aug_image, camera_mats=camera_mats)
        preds_completed = preds
        preds = preds[:, -1] # take the last predicted image

        loss = self.loss_fn(preds.to(self.device), goal_image.squeeze(1).to(self.device)).flatten(0)

        if self.args.save_preds:
            # 注意：此处应保存 final_deltas（不是局部循环里的 deltas）
            save_planning_pred(dataset_save_output_dir, n_evals, idxs, obs_image, goal_image, preds, final_deltas, loss, gt_actions, preds_completed)

        # ============== NEW: 保存逐帧 PNG + metadata.json（Editor 友好） ==============
        # 逐帧 loss：LPIPS(frame, goal)；对 init、每个 step 与 goal 都计算（goal 自己设 0.0）
        if self.args.save_preds:
            B = obs_image.shape[0]
            T = preds_completed.shape[1]
            init_imgs = obs_image[:, -1]               # (B,C,H,W)
            goal_imgs = goal_image.squeeze(1)          # (B,C,H,W)

            editor_dir = _ensure_dir(os.path.join(dataset_save_output_dir, "editor"))
            for b in range(B):
                sid = int(idxs[b].item())
                run_dir = _ensure_dir(os.path.join(editor_dir, f"run_{sid:03d}"))
                frames_dir = _ensure_dir(os.path.join(run_dir, "frames"))

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
                # 3) 写 metadata.json
                H, W = init_imgs.shape[-2:]
                meta = {
                    "run_index": sid,
                    "video_mp4": "",
                    "fps": int(self.args.rollout_stride) if self.args.rollout_stride > 0 else 1,
                    "resolution": {"width": int(W), "height": int(H)},
                    "num_frames": T + 1,            # INIT + T steps
                    "sequence": list(range(T)),     # 0..T-1
                    "commands": {},
                    "frames": []
                }
                # INIT
                meta["frames"].append({
                    "frame": 0,
                    "png": "frames/init.png",
                    "action": "INIT",
                    "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "theta_rad": 0.0},
                    "delta": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dtheta": 0.0},
                    "loss": init_loss
                })
                # STEPS
                for t in range(T):
                    meta["frames"].append({
                        "frame": t + 1,
                        "png": f"frames/step_{t:03d}.png",
                        "action": t,
                        "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "theta_rad": 0.0},
                        "delta": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dtheta": 0.0},
                        "loss": step_losses[t]
                    })
                _write_json(os.path.join(run_dir, "metadata.json"), meta)
        # ==========================================================================

        if self.args.plot:
            img_name = os.path.join(image_plot_dir, f'FINAL_{idx_string}.png')
            traj_name = os.path.join(image_plot_dir, f"TRAJ_{idx_string}.png")
            plot_batch_final(obs_image[:, -1].to(self.device), preds, goal_image.squeeze(1).to(self.device), idxs, all_losses, save_path=img_name)
            plot_batch_trajectories(obs_image[:, -1].to(self.device), preds_completed, goal_image.squeeze(1).to(self.device), idxs, save_path=traj_name)

        pred_actions = get_action_torch(final_deltas[:, :, :3], ACTION_STATS_TORCH)
        pred_yaw = final_deltas[:, :, -1].sum(1)
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
    
    def autoregressive_rollout(self, obs_image, deltas, rollout_stride, aug_image, camera_mats):
        deltas = deltas.unflatten(1, (-1, rollout_stride)).sum(2)
        preds = []
        curr_obs = obs_image.clone().to(self.device)
        
        for i in range(deltas.shape[1]):
            curr_delta = deltas[:, i:i+1]
            all_models = self.model, self.diffusion, self.vae
            x_pred_pixels = model_forward_wrapper(all_models, curr_obs, curr_delta, self.args.rollout_stride, self.latent_size, num_cond=self.num_cond, device=self.device, x_supervised=aug_image, camera_mats=camera_mats)
            x_pred_pixels = x_pred_pixels.unsqueeze(1)
            
            curr_obs = torch.cat((curr_obs, x_pred_pixels), dim=1) # append current prediction
            curr_obs = curr_obs[:, 1:] # remove first observation
            preds.append(x_pred_pixels)
        
        preds = torch.cat(preds, 1)
        return preds
    
    def get_eval_name(self):
        # Get evaluation name for logging. Should overwrite for specific experiments
        self.eval_name = f'CEM_N{self.args.num_samples}_K{self.args.topk}_RS{self.args.rollout_stride}_rep{self.args.num_repeat_eval}_OPT{self.args.opt_steps}'
        
    def actions_to_traj(self, actions):
        positions_xyz = torch.zeros((actions.shape[0], 3))
        positions_xyz[:, :3] = actions
        orientations_quat_wxyz = torch.zeros((actions.shape[0], 4)) # Define identity quaternion
        orientations_quat_wxyz[:, -1] = 1 # Define identity quaternion
        timestamps = torch.arange(actions.shape[0], dtype=torch.float64)
        traj = PoseTrajectory3D(positions_xyz=positions_xyz, orientations_quat_wxyz=orientations_quat_wxyz, timestamps=timestamps)
        return traj
    
    @torch.no_grad()
    def evaluate(self):
        
        for dataset_name in self.dataset_names:
            metric_logger = dist.MetricLogger(delimiter="  ")
            header = 'Test:'
            eval_save_output_dir = None
            
            if self.args.save_preds:
                dataset_save_output_dir = os.path.join(self.args.save_output_dir, dataset_name)
                os.makedirs(dataset_save_output_dir, exist_ok=True)
                eval_save_output_dir = os.path.join(dataset_save_output_dir, self.eval_name)
                os.makedirs(eval_save_output_dir, exist_ok=True)
            
            curr_data_loader = self.datasets[dataset_name]
            for (idxs, obs_image, goal_image, gt_actions, goal_pos, aug_image, camera_ctx, camera_goal) in metric_logger.log_every(curr_data_loader, 1, header):
                obs_image = obs_image[:, -self.num_cond:]
                camera_ctx  = camera_ctx[:, -self.num_cond:]
                camera_mats = torch.cat([camera_ctx, camera_goal], dim=1)  # (B, num_cond+1, 4, 4)
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    pred_actions, pred_yaw = self.generate_actions(eval_save_output_dir, dataset_name, idxs, obs_image, goal_image, gt_actions, self.config["trajectory_eval_len_traj_pred"], aug_image=aug_image, camera_mats=camera_mats)
                for i in range(len(obs_image)):
                    pred_traj_i = self.actions_to_traj(pred_actions[i, :, :3])
                    gt_traj_i = self.actions_to_traj(gt_actions[i, :, :3])
                    
                    ate, rpe_trans, _ = self.eval_metrics(gt_traj_i, pred_traj_i)

                    pred_final_pos = pred_actions[i, -1, :3].to('cpu') # (3,)
                    pred_final_yaw = pred_yaw[i].to('cpu') # 
                    goal_final_pos = goal_pos[i, 0, :3] # (3,)
                    goal_final_yaw = goal_pos[i, 0, -1] # (B,)
                    pos_diff_norm = torch.norm(pred_final_pos - goal_final_pos)
                    yaw_diff = pred_final_yaw - goal_final_yaw  # 
                    yaw_diff_norm = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).abs()
                    
                    metric_logger.meters['{}_ate'.format(dataset_name)].update(ate, n=1)
                    metric_logger.meters['{}_rpe_trans'.format(dataset_name)].update(rpe_trans, n=1)
                    metric_logger.meters['{}_pos_diff_norm'.format(dataset_name)].update(pos_diff_norm, n=1)   
                    metric_logger.meters['{}_yaw_diff_norm'.format(dataset_name)].update(yaw_diff_norm, n=1)   
            output_fn = os.path.join(self.args.save_output_dir, f'{dataset_name}_{self.eval_name}.json')
            save_metric_to_disk(metric_logger, output_fn)

        # gather the stats from all processes
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
    parser.add_argument("--ckp", type=str, default='0100000', help="experiment name")

    parser.add_argument("--datasets", type=str, default=None, help="dataset name")
    parser.add_argument("--output_dir", type=str, default=None, help="output dir to save model predictions")
    parser.add_argument('--save_preds', action='store_true', default=False, help='whether to save prediction tensors or not')
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    
    # Planning Specific Args
    parser.add_argument("--num_samples", type=int, default=10, help="num nomad samples to predict")
    parser.add_argument("--rollout_stride", type=int, default=1, help="rollout stride")
    parser.add_argument("--topk", type=int, default=5, help="top k samples to take mean and var for CEM")
    parser.add_argument("--opt_steps", type=int, default=15, help="num iterations for CEM")
    parser.add_argument("--num_repeat_eval", type=int, default=1, help="number of evals for one action")
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()
    
    evaluator = WM_Planning_Evaluator(args)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpu_id = torch.cuda.current_device()  # Or args.gpu if explicitly set
    evaluator.evaluate()
