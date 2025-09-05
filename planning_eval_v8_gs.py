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

from diffusers.models import AutoencoderKL

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
        self.mode = 'gs' # 3dgs
        self.num_samples = self.args.num_samples
        self.topk = self.args.topk
        self.opt_steps = self.args.opt_steps
        self.num_repeat_eval = self.args.num_repeat_eval
        self.action_dim = 4 # hardcoded (delta_x, delta_y, delta_z, delta_yaw)

    def init_mu_sigma(self, obs_0, traj_len):
        n_evals = obs_0.shape[0]
        mu = torch.zeros(n_evals, self.action_dim) 
        mu[:, ] = torch.tensor(data_hyperparams[self.args.datasets]['mu'])
        sigma = torch.ones([n_evals, self.action_dim])
        sigma[:, ] = torch.tensor(data_hyperparams[self.args.datasets]['var_scale']) 
        return mu, sigma

    # ======================= 可微优化（替代 CEM） =======================
    def _optimize_single_traj(self, obs_img_1, goal_img_1, T, aug_img_1, cam_1, steps, lr=0.05, smooth_w=1e-3,
                              # >>> NEW <<< 这些参数仅用于可视化；不影响原有优化/返回
                              dataset_name=None, gt_actions=None, image_plot_dir=None, traj_idx=None, traj_id=None,
                              viz_every=1):
        """
        对 batch 中单条样本做反向优化（保持外部接口不变）。
          obs_img_1:   (1, num_cond, C, H, W)
          goal_img_1:  (1, C, H, W)
          aug_img_1:   (1, 1, C, H, W)
          cam_1:       (1, num_cond+1, 4, 4)
        返回：
          deltas_1:    (1, T, 4)
          preds_seq_1: (1, T, C, H, W)
          final_lpips: (1,) tensor
        """
        device = self.device
        # 以数据超参的均值作为动作初值（与原风格一致）
        mu0, _ = self.init_mu_sigma(obs_img_1, T)   # (1,4)
        mu0 = mu0.to(device)
        norm_xyz = mu0[:, :3].unsqueeze(1).repeat(1, T, 1).clone().detach().requires_grad_(True)  # (1,T,3)
        last_yaw_bias = torch.zeros(1, device=device).requires_grad_(True)  # 最后一帧 yaw 偏置

        # 冻结模型权重
        for p in self.model.parameters(): 
            p.requires_grad_(False)

        opt = torch.optim.Adam([norm_xyz, last_yaw_bias], lr=lr)

        # 在外层 no_grad 环境下，显式开启梯度
        with torch.enable_grad():
            for it in range(max(20, steps * 4)):
                opt.zero_grad(set_to_none=True)
                # 构建 deltas：前三维仍是归一化；yaw 由反归一化速度几何计算，可微
                unnorm_xyz = unnormalize_data(norm_xyz, ACTION_STATS_TORCH)        # (1,T,3)
                d_yaw = calculate_delta_yaw(unnorm_xyz)                            # (1,T,1)
                deltas_1 = torch.cat([norm_xyz, d_yaw.to(norm_xyz.device)], dim=-1)  # (1,T,4)
                deltas_1[:, -1, -1] += last_yaw_bias * np.pi

                preds_seq = self.autoregressive_rollout(
                    obs_img_1, deltas_1, self.args.rollout_stride,
                    aug_image=aug_img_1, camera_mats=cam_1
                )                                                                   # (1,T,C,H,W)
                pred_last = preds_seq[:, -1]                                        # (1,C,H,W)

                # >>> NEW <<< GS 每次迭代可视化：与 CEM 输出保持一致，复用 visualize_trajectories
                if self.args.plot and (dataset_name is not None) and (image_plot_dir is not None):
                    last_it = max(20, steps * 4) - 1
                    if (it % viz_every == 0) or (it == last_it):
                        preds_for_viz = preds_seq[:, -1].detach()                                # (1,C,H,W)
                        loss_img = self.loss_fn(preds_for_viz, goal_img_1).flatten(0).detach()   # (1,)
                        topk_idx = torch.tensor([0], device=preds_for_viz.device)                # 单候选，高亮 0
                        self.visualize_trajectories(
                            dataset_name=dataset_name,
                            gt_actions=gt_actions,
                            image_plot_dir=image_plot_dir,
                            i=it,                                  # 当前迭代步
                            traj=0 if traj_idx is None else traj_idx,
                            traj_id=-1 if traj_id is None else traj_id,
                            deltas=deltas_1.detach(),              # (1,T,4)
                            cur_obs_image=obs_img_1.detach(),      # (1,num_cond,C,H,W)
                            cur_goal_image=goal_img_1.detach(),    # (1,C,H,W)
                            preds=preds_for_viz,                   # (1,C,H,W)
                            loss=loss_img,                         # (1,)
                            topk_idx=topk_idx                      # (1,)
                        )
                # >>> NEW <<<

                # 最后一帧 LPIPS（原口径）
                img_loss = self.loss_fn(pred_last, goal_img_1).flatten(0).mean()

                # === 新增 1：中间帧 LPIPS ===
                # 将中间每步也向目标图靠近（稳定过渡），默认权重 0.2
                if preds_seq.shape[1] > 1:
                    mid = preds_seq[:, :-1].contiguous().view(-1, *preds_seq.shape[2:])  # ((T-1),C,H,W)
                    tgt = goal_img_1.expand(preds_seq.shape[1]-1, -1, -1, -1).contiguous()
                    mid_img_loss = self.loss_fn(mid, tgt).mean()
                else:
                    mid_img_loss = 0.0

                # 一阶平滑：动作 L2 + 时间差分
                l2 = norm_xyz.pow(2).mean()
                diff = (norm_xyz[:, 1:] - norm_xyz[:, :-1]).pow(2).mean()
                smooth_loss = l2 + diff

                # === 新增 2：二阶差分（jerk）正则 ===
                # 抑制加速度突变，默认权重 1e-3
                if norm_xyz.shape[1] > 2:
                    jerk = (norm_xyz[:, 2:] - 2*norm_xyz[:, 1:-1] + norm_xyz[:, :-2]).pow(2).mean()
                else:
                    jerk = 0.0

                mid_w  = 0.2   # 中间帧 LPIPS 权重
                jerk_w = 1e-3  # 二阶平滑权重
                loss = img_loss + smooth_w * smooth_loss + mid_w * mid_img_loss + jerk_w * jerk

                loss.backward()
                torch.nn.utils.clip_grad_norm_([norm_xyz, last_yaw_bias], 1.0)
                opt.step()

        # 最终输出
        with torch.no_grad():
            unnorm_xyz = unnormalize_data(norm_xyz, ACTION_STATS_TORCH)
            d_yaw = calculate_delta_yaw(unnorm_xyz)
            deltas_1 = torch.cat([norm_xyz, d_yaw.to(norm_xyz.device)], dim=-1)
            deltas_1[:, -1, -1] += last_yaw_bias * np.pi
            preds_seq = self.autoregressive_rollout(
                obs_img_1, deltas_1, self.args.rollout_stride,
                aug_image=aug_img_1, camera_mats=cam_1
            )
            final_lpips = self.loss_fn(preds_seq[:, -1], goal_img_1).flatten(0)

        return deltas_1.detach(), preds_seq.detach(), final_lpips.detach()
    # ==================================================================

    def generate_actions(self, dataset_save_output_dir, dataset_name, idxs, obs_image, goal_image, gt_actions, len_traj_pred, aug_image, camera_mats):
        # —— 完全替代 CEM：逐样本用世界模型反向优化 —— #
        idx_string = "_".join(map(str, idxs.flatten().int().tolist())) 
        image_plot_dir = os.path.join(dataset_save_output_dir, 'plots')
        os.makedirs(image_plot_dir, exist_ok=True)
        videos_plot_dir = os.path.join(dataset_save_output_dir, 'videos')
        os.makedirs(videos_plot_dir, exist_ok=True)
        
        B = obs_image.shape[0]
        T = len_traj_pred

        all_deltas = []
        all_preds_seq = []
        all_final_losses = []

        # 将目标图 squeeze 成 (B,C,H,W) 以复用原 LPIPS 口径
        goal_img_B = goal_image.squeeze(1).to(self.device)

        for b in range(B):
            # 单条样本打包成 batch=1，复用现有 rollout
            obs_1 = obs_image[b:b+1]
            aug_1 = aug_image[b:b+1]
            cam_1 = camera_mats[b:b+1]
            goal_1 = goal_img_B[b:b+1]

            # —— 评估阶段外层有 @torch.no_grad，需要在内部显式开梯度 —— #
            with torch.enable_grad():
                deltas_1, preds_seq_1, final_lpips = self._optimize_single_traj(
                    obs_img_1=obs_1, goal_img_1=goal_1, T=T,
                    aug_img_1=aug_1, cam_1=cam_1,
                    steps=self.opt_steps, lr=0.05, smooth_w=1e-3,
                    # >>> NEW <<< 传入可视化上下文
                    dataset_name=dataset_name,
                    gt_actions=gt_actions,
                    image_plot_dir=image_plot_dir,
                    traj_idx=b,
                    traj_id=int(idxs[b].item()),
                    viz_every=100
                    # >>> NEW <<<
                )
            all_deltas.append(deltas_1)
            all_preds_seq.append(preds_seq_1)
            all_final_losses.append(final_lpips)

        deltas = torch.cat(all_deltas, dim=0)                 # (B,T,4)
        preds_completed = torch.cat(all_preds_seq, dim=0)     # (B,T,C,H,W)
        preds = preds_completed[:, -1]                        # (B,C,H,W)
        loss = torch.cat(all_final_losses, dim=0)             # (B,)

        # 保持原有保存/绘图/返回接口不变
        if self.args.save_preds:
            save_planning_pred(dataset_save_output_dir, B, idxs, obs_image, goal_image, preds, deltas, loss, gt_actions, preds_completed)
        
        if self.args.plot:
            img_name = os.path.join(image_plot_dir, f'FINAL_{idx_string}.png')
            traj_name = os.path.join(image_plot_dir, f"TRAJ_{idx_string}.png")
            plot_batch_final(obs_image[:, -1].to(self.device), preds, goal_img_B, idxs, loss.detach().cpu().tolist(), save_path=img_name)
            plot_batch_trajectories(obs_image[:, -1].to(self.device), preds_completed, goal_img_B, idxs, save_path=traj_name)

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
        self.eval_name = f'3DGS_N{self.args.num_samples}_K{self.args.topk}_RS{self.args.rollout_stride}_rep{self.args.num_repeat_eval}_OPT{self.args.opt_steps}'
        
    def actions_to_traj(self, actions):
        positions_xyz = torch.zeros((actions.shape[0], 3))
        positions_xyz[:, :3] = actions
        orientations_quat_wxyz = torch.zeros((actions.shape[0], 4)) # Define identity quaternion
        orientations_quat_wxyz[:, -1] = 1 # Define identity quaternion
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
                dataset_save_output_dir = os.path.join(self.args.save_output_dir, dataset_name)
                os.makedirs(dataset_save_output_dir, exist_ok=True)
                eval_save_output_dir = os.path.join(dataset_save_output_dir, self.eval_name)
                os.makedirs(eval_save_output_dir, exist_ok=True)
            
            curr_data_loader = self.datasets[dataset_name]
            for (idxs, obs_image, goal_image, gt_actions, goal_pos, aug_image, camera_ctx, camera_goal) in metric_logger.log_every(curr_data_loader, 1, header):
                obs_image = obs_image[:, -self.num_cond:]
                camera_ctx  = camera_ctx[:, -self.num_cond:]
                camera_mats = torch.cat([camera_ctx, camera_goal], dim=1)  # (B, num_cond+1, 4, 4)
                # 原代码里这里用了 autocast；我们不在 autocast 下做优化
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    pass

                # —— 评估函数整体 no_grad，这里调用优化前显式开启梯度 —— #
                with torch.enable_grad():
                    pred_actions, pred_yaw = self.generate_actions(
                        eval_save_output_dir, dataset_name, idxs, obs_image, goal_image, gt_actions,
                        self.config["trajectory_eval_len_traj_pred"], aug_image=aug_image, camera_mats=camera_mats
                    )

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
    
    # Planning Specific Args（保留原参数，便于对比；内部已用反向优化，不再采样）
    parser.add_argument("--num_samples", type=int, default=10, help="num nomad samples to predict")
    parser.add_argument("--rollout_stride", type=int, default=1, help="rollout stride")
    parser.add_argument("--topk", type=int, default=5, help="top k samples to take mean and var for CEM")
    parser.add_argument("--opt_steps", type=int, default=15, help="num iterations for CEM（此处作为反向优化步数基数）")
    parser.add_argument("--num_repeat_eval", type=int, default=1, help="number of evals for one action")
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()
    
    evaluator = WM_Planning_Evaluator(args)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpu_id = torch.cuda.current_device()  # Or args.gpu if explicitly set
    evaluator.evaluate()
