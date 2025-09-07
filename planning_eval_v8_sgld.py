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
from misc import (
    calculate_delta_yaw, get_action_torch, save_planning_pred, log_viz_single,
    transform, unnormalize_data
)
from isolated_nwm_eval import save_metric_to_disk
import distributed as dist
from models_tpz_v8 import CDiT_models


# ---------------------- 全局配置加载 ----------------------
with open("config/data_config.yaml", "r") as f:
    data_config = yaml.safe_load(f)

with open("config/data_hyperparams_plan.yaml", "r") as f:
    data_hyperparams = yaml.safe_load(f)

ACTION_STATS_TORCH = {}
for key in data_config['action_stats']:
    ACTION_STATS_TORCH[key] = torch.tensor(data_config['action_stats'][key])


# ---------------------- 可视化工具 ----------------------
def plot_images_with_losses(preds, losses, save_path="predictions_with_losses.png"):
    # Denormalize images from [-1, 1] to [0, 1]
    preds = (preds + 1) / 2
    ncol = int(preds.size(0) ** 0.5)
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


def plot_traj3d(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor = None,
                save_path: str = "traj3d.png", title: str = None):
    """
    3D 轨迹绘制（绝对坐标），方便直观看到 Z 维变化。
    pred_xyz: (T, 3)；gt_xyz: (T, 3) 可选
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pred = pred_xyz.detach().cpu().numpy()
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], linewidth=2, label="Pred")
    if gt_xyz is not None:
        gt = gt_xyz.detach().cpu().numpy()
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], linestyle='--', linewidth=1.5, label="GT")

    ax.scatter(pred[0, 0], pred[0, 1], pred[0, 2], s=50, marker='o', label="Start")
    ax.scatter(pred[-1, 0], pred[-1, 1], pred[-1, 2], s=50, marker='^', label="End")

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    if title: ax.set_title(title)
    ax.legend(); ax.grid(True)
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# ---------------------- 数据集 ----------------------
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


# ================================================================
#                          核心评估器
# ================================================================
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
                print('Warning: distributed eval with non-divisible dataset; duplicates will be added.')

            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
            curr_data_loader = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True, drop_last=False
            )
            self.datasets[dataset_name] = curr_data_loader

        # Loading Model
        print("loading")
        model = CDiT_models[self.config['model']](
            context_size=self.num_cond + 1,
            input_size=latent_size,
        )
        ckp = torch.load(
            f'{self.config["results_dir"]}/{self.config["run_name"]}/checkpoints/{args.ckp}.pth.tar',
            map_location='cpu', weights_only=False
        )
        model.load_state_dict(ckp["ema"], strict=True)
        model.eval()
        model.to(self.device)
        self.model = torch.compile(model)
        self.diffusion = create_diffusion(str(250))
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[self.device], find_unused_parameters=False
        )
        self.model_without_ddp = self.model.module

        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.mode = 'sgld'  # sgld
        self.num_samples = self.args.num_samples
        self.topk = self.args.topk
        self.opt_steps = self.args.opt_steps
        self.num_repeat_eval = self.args.num_repeat_eval
        self.action_dim = 4  # (delta_x, delta_y, delta_z, delta_yaw)

    # ---------------------- 初始化动作均值/方差 ----------------------
    def init_mu_sigma(self, obs_0, traj_len):
        n_evals = obs_0.shape[0]
        mu = torch.zeros(n_evals, self.action_dim)
        mu[:, ] = torch.tensor(data_hyperparams[self.args.datasets]['mu'])
        sigma = torch.ones([n_evals, self.action_dim])
        sigma[:, ] = torch.tensor(data_hyperparams[self.args.datasets]['var_scale'])
        return mu, sigma

    # ================================================================
    # 可微优化（点估计；之前我们替代 CEM 的版本仍保留）
    # ================================================================
    def _optimize_single_traj(
        self, obs_img_1, goal_img_1, T, aug_img_1, cam_1, steps, lr=0.05, smooth_w=1e-3,
        # 仅用于可视化
        dataset_name=None, gt_actions=None, image_plot_dir=None, traj_idx=None, traj_id=None,
        viz_every=1
    ):
        """
        对 batch 中单条样本做反向优化（保持外部接口不变）。
        【注意：优化变量在“归一化动作空间” norm_xyz ∈ R^(T×3)，yaw 不直接优化，而是由 unnormalize 后的 (dx,dy,dz) 几何推导】
        """
        device = self.device
        mu0, _ = self.init_mu_sigma(obs_img_1, T)  # (1,4)
        mu0 = mu0.to(device)
        # --------- 采样/优化空间差异说明 ----------
        # 之前：CEM 在“归一化动作空间”（前三维 dx,dy,dz）做高斯采样；我们替代为梯度下降也在相同空间优化。
        # 现在：SGLD 采样仍然在“归一化动作空间”上进行（与 CEM/优化一致），而不是图像空间或像素空间。
        # yaw 依旧由 (dx,dy,dz) 通过 calculate_delta_yaw() 推导，不作为独立采样/优化变量。
        # ----------------------------------------
        norm_xyz = mu0[:, :3].unsqueeze(1).repeat(1, T, 1).clone().detach().requires_grad_(True)
        last_yaw_bias = torch.zeros(1, device=device).requires_grad_(True)

        for p in self.model.parameters():
            p.requires_grad_(False)

        opt = torch.optim.Adam([norm_xyz, last_yaw_bias], lr=lr)

        with torch.enable_grad():
            for it in range(max(20, steps * 4)):
                opt.zero_grad(set_to_none=True)
                unnorm_xyz = unnormalize_data(norm_xyz, ACTION_STATS_TORCH)        # (1,T,3)
                d_yaw = calculate_delta_yaw(unnorm_xyz)                            # (1,T,1)
                deltas_1 = torch.cat([norm_xyz, d_yaw.to(norm_xyz.device)], dim=-1)  # (1,T,4)
                deltas_1[:, -1, -1] += last_yaw_bias * np.pi

                preds_seq = self.autoregressive_rollout(
                    obs_img_1, deltas_1, self.args.rollout_stride,
                    aug_image=aug_img_1, camera_mats=cam_1
                )
                pred_last = preds_seq[:, -1]

                # （可选）迭代可视化
                if self.args.plot and (dataset_name is not None) and (image_plot_dir is not None):
                    last_it = max(20, steps * 4) - 1
                    if (it % viz_every == 0) or (it == last_it):
                        with torch.no_grad():
                            viz_num = int(self.args.num_samples)
                            # 围绕当前 norm_xyz 做轻微扰动可视化
                            var_sc = data_hyperparams[self.args.datasets]['var_scale']
                            base_scale = float(var_sc[0]) if isinstance(var_sc, (list, tuple)) else float(var_sc)
                            viz_sigma = 0.5 * base_scale if base_scale > 0 else 0.5

                            norm_xyz_center = norm_xyz.detach()
                            norm_xyz_viz = norm_xyz_center.expand(viz_num, -1, -1).clone()
                            norm_xyz_viz = norm_xyz_viz + viz_sigma * torch.randn_like(norm_xyz_viz)

                            unnorm_viz = unnormalize_data(norm_xyz_viz, ACTION_STATS_TORCH)
                            d_yaw_viz = calculate_delta_yaw(unnorm_viz)
                            deltas_viz = torch.cat([norm_xyz_viz, d_yaw_viz.to(norm_xyz_viz.device)], dim=-1)
                            deltas_viz[:, -1, -1] += last_yaw_bias.detach() * np.pi

                            cur_obs = obs_img_1.repeat(viz_num, 1, 1, 1, 1)
                            cur_aug = aug_img_1.repeat(viz_num, 1, 1, 1, 1)
                            cur_cam = cam_1.repeat(viz_num, 1, 1, 1)

                            preds_seq_viz = self.autoregressive_rollout(
                                cur_obs, deltas_viz, self.args.rollout_stride,
                                aug_image=cur_aug, camera_mats=cur_cam
                            )
                            preds_last_viz = preds_seq_viz[:, -1]

                            goal_rep = goal_img_1.expand(viz_num, -1, -1, -1)
                            loss_viz = self.loss_fn(preds_last_viz, goal_rep).flatten(0)
                            sorted_idx = torch.argsort(loss_viz)
                            topk_k = int(self.topk) if self.topk is not None else min(5, viz_num)
                            topk_idx = sorted_idx[:max(1, min(topk_k, viz_num))]

                        self.visualize_trajectories(
                            dataset_name=dataset_name,
                            gt_actions=gt_actions,
                            image_plot_dir=image_plot_dir,
                            i=it,
                            traj=0 if traj_idx is None else traj_idx,
                            traj_id=-1 if traj_id is None else traj_id,
                            deltas=deltas_viz.detach(),
                            cur_obs_image=cur_obs.detach(),
                            cur_goal_image=goal_img_1.detach(),
                            preds=preds_last_viz.detach(),
                            loss=loss_viz.detach(),
                            topk_idx=topk_idx.detach()
                        )

                # 能量（loss）= 末帧 LPIPS + 中间帧 LPIPS + 平滑 + jerk
                img_loss = self.loss_fn(pred_last, goal_img_1).flatten(0).mean()
                if preds_seq.shape[1] > 1:
                    mid = preds_seq[:, :-1].contiguous().view(-1, *preds_seq.shape[2:])
                    tgt = goal_img_1.expand(preds_seq.shape[1]-1, -1, -1, -1).contiguous()
                    mid_img_loss = self.loss_fn(mid, tgt).mean()
                else:
                    mid_img_loss = 0.0

                l2 = norm_xyz.pow(2).mean()
                diff = (norm_xyz[:, 1:] - norm_xyz[:, :-1]).pow(2).mean()
                if norm_xyz.shape[1] > 2:
                    jerk = (norm_xyz[:, 2:] - 2*norm_xyz[:, 1:-1] + norm_xyz[:, :-2]).pow(2).mean()
                else:
                    jerk = 0.0

                mid_w = 0.2
                jerk_w = 1e-3
                loss = img_loss + smooth_w * (l2 + diff) + mid_w * mid_img_loss + jerk_w * jerk

                loss.backward()
                torch.nn.utils.clip_grad_norm_([norm_xyz, last_yaw_bias], 1.0)
                opt.step()

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

    # ================================================================
    # 概率场采样：SGLD 在“归一化动作空间 (dx,dy,dz)”上采样
    # ================================================================
    def _energy(self, obs_img, goal_img, deltas_norm, aug_img, cam_mats,
                mid_w=0.2, smooth_w=1e-3, jerk_w=1e-3, prior_w=1.0):
        """
        deltas_norm: (N, T, 4)，但只使用前三维(归一化 dx,dy,dz)，yaw 由几何推导。返回:
          E: (N,)；preds_seq: (N, T, C, H, W)
        """
        with torch.enable_grad():
            norm_xyz = deltas_norm[..., :3]                           # 采样/优化空间：归一化动作 (dx, dy, dz)
            unnorm_xyz = unnormalize_data(norm_xyz, ACTION_STATS_TORCH)
            d_yaw = calculate_delta_yaw(unnorm_xyz)
            deltas = torch.cat([norm_xyz, d_yaw.to(norm_xyz.device)], dim=-1)

            preds_seq = self.autoregressive_rollout(
                obs_img, deltas, self.args.rollout_stride, aug_image=aug_img, camera_mats=cam_mats
            )
            pred_last = preds_seq[:, -1]

            goal_rep = goal_img.expand_as(pred_last)
            img_loss = self.loss_fn(pred_last, goal_rep).flatten(1).mean(dim=1)

            if preds_seq.shape[1] > 1:
                mid = preds_seq[:, :-1].contiguous().view(-1, *preds_seq.shape[2:])
                tgt = goal_img.expand(preds_seq.shape[0]*(preds_seq.shape[1]-1), -1, -1, -1).contiguous()
                mid_img_loss = self.loss_fn(mid, tgt).flatten(1).mean(dim=1)
            else:
                mid_img_loss = torch.zeros_like(img_loss)

            l2 = norm_xyz.pow(2).mean(dim=(1, 2))
            diff = (norm_xyz[:, 1:] - norm_xyz[:, :-1]).pow(2).mean(dim=(1, 2))
            if norm_xyz.shape[1] > 2:
                jerk = (norm_xyz[:, 2:] - 2*norm_xyz[:, 1:-1] + norm_xyz[:, :-2]).pow(2).mean(dim=(1, 2))
            else:
                jerk = torch.zeros_like(img_loss)

            # 高斯先验：N(mu0, s^2 I)（在归一化域，用 var_scale 控制宽度）
            N, T, _ = norm_xyz.shape
            mu0, _ = self.init_mu_sigma(obs_img[:1, -1], T)  # (1,4)
            mu0 = mu0.to(norm_xyz.device)[..., :3].view(1, 1, 3).expand(N, T, 3)
            var_scale = torch.tensor(
                data_hyperparams[self.args.datasets]['var_scale'],
                device=norm_xyz.device, dtype=norm_xyz.dtype
            )
            if var_scale.ndim == 0:
                var_scale = var_scale.repeat(3)
            sigma2 = (self.args.prior_scale * var_scale[:3])**2
            prior = ((norm_xyz - mu0)**2 / sigma2.view(1, 1, 3)).mean(dim=(1, 2))

            E = img_loss + mid_w * mid_img_loss + smooth_w * (l2 + diff) + jerk_w * jerk + prior_w * prior
            return E, preds_seq

    def _sample_langevin(self, obs_img_1, goal_img_1, T, aug_img_1, cam_1,
                         N=64, steps=200, eta=5e-3, beta=50.0):
        """
        在“归一化动作空间 (dx,dy,dz)”上做 SGLD（而不是像素/图像空间）；yaw 由几何推导，不直接采样。
        返回:
          deltas_N:  (N, T, 4)   —— 归一化 xyz + yaw
          preds_seq_N: (N, T, C, H, W)
          energy_N: (N,)
          weights:   (N,)  ~ softmax(-beta * (E - minE)) 作为重要性权重
        """
        device = self.device

        # 初始化：以 mu0 为中心的高斯
        mu0, _ = self.init_mu_sigma(obs_img_1[:, -1], T)    # (1,4)
        mu0 = mu0.to(device)[..., :3].view(1, 1, 3)
        var_scale = torch.tensor(
            data_hyperparams[self.args.datasets]['var_scale'],
            device=device, dtype=torch.float32
        )
        if var_scale.ndim == 0:
            var_scale = var_scale.repeat(3)
        init_std = (self.args.prior_scale * var_scale[:3]).view(1, 1, 3)

        norm_xyz = mu0.expand(N, T, 3) + init_std.expand(N, T, 3) * torch.randn(N, T, 3, device=device)
        norm_xyz.requires_grad_(True)

        # 纯粹为了 zero_grad 使用的“空”优化器
        opt_like = torch.optim.SGD([norm_xyz], lr=1.0)

        for k in range(steps):
            opt_like.zero_grad(set_to_none=True)
            deltas_norm = torch.cat([norm_xyz, torch.zeros(N, T, 1, device=device)], dim=-1)
            E, _ = self._energy(
                obs_img_1.repeat(N, 1, 1, 1, 1),
                goal_img_1.repeat(N, 1, 1, 1),
                deltas_norm,
                aug_img_1.repeat(N, 1, 1, 1, 1),
                cam_1.repeat(N, 1, 1, 1)
            )
            (beta * E.mean()).backward()

            with torch.no_grad():
                grad = norm_xyz.grad
                # SGLD：x <- x - 0.5*eta*grad + sqrt(eta)*Noise
                norm_xyz += -0.5 * eta * grad + (eta ** 0.5) * torch.randn_like(norm_xyz)
                norm_xyz.clamp_(-5.0, 5.0)
            norm_xyz.requires_grad_(True)

        with torch.no_grad():
            deltas_norm = torch.cat([norm_xyz, torch.zeros(N, T, 1, device=device)], dim=-1)
            E, preds_seq = self._energy(
                obs_img_1.repeat(N, 1, 1, 1, 1),
                goal_img_1.repeat(N, 1, 1, 1),
                deltas_norm,
                aug_img_1.repeat(N, 1, 1, 1, 1),
                cam_1.repeat(N, 1, 1, 1)
            )
            unnorm_xyz = unnormalize_data(norm_xyz, ACTION_STATS_TORCH)
            d_yaw = calculate_delta_yaw(unnorm_xyz)
            deltas_N = torch.cat([norm_xyz, d_yaw.to(device)], dim=-1)

            # 权重：避免数值下溢
            E_shift = E - E.min()
            w = torch.softmax(-beta * E_shift, dim=0)

        return deltas_N, preds_seq, E, w

    # ---------------------- 统一动作生成接口 ----------------------
    def generate_actions(self, dataset_save_output_dir, dataset_name, idxs, obs_image, goal_image, gt_actions, len_traj_pred, aug_image, camera_mats):
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

        # squeeze 成 (B,C,H,W)
        goal_img_B = goal_image.squeeze(1).to(self.device)

        pred_actions_list = []
        pred_yaw_list = []

        for b in range(B):
            obs_1 = obs_image[b:b + 1]
            aug_1 = aug_image[b:b + 1]
            cam_1 = camera_mats[b:b + 1]
            goal_1 = goal_img_B[b:b + 1]

            if self.args.sampler == "langevin":
                # ---------- 概率采样：在“归一化动作空间”上用 SGLD ----------
                deltas_N, preds_seq_N, E_N, w_N = self._sample_langevin(
                    obs_1, goal_1, T, aug_1, cam_1,
                    N=self.args.samples, steps=self.args.sgld_steps,
                    eta=self.args.sgld_lr, beta=self.args.beta
                )
                # 用重要性权重做样本集的加权平均，得到一个“点估计轨迹”（你也可以返回全体样本）
                deltas_agg = torch.einsum('n,ntd->td', w_N, deltas_N[..., :4]).unsqueeze(0)  # (1,T,4)

                # 下游接口仍然使用 get_action_torch（内部含反归一化/尺度映射）
                pred_actions_b = get_action_torch(deltas_agg[..., :3], ACTION_STATS_TORCH)  # (1,T,3) —— 注意：这里仍是“动作域参数”，评估时会 cumsum 得到绝对轨迹
                pred_yaw_b = deltas_agg[..., -1].sum(1)                                     # (1,)

                # （可选）保存样本末帧可视化
                if self.args.save_preds:
                    with torch.no_grad():
                        preds_last = preds_seq_N[:, -1]  # (N,C,H,W)
                        loss_viz = self.loss_fn(preds_last, goal_1.repeat(self.args.samples, 1, 1, 1)).flatten(0)
                        topk_idx = torch.argsort(loss_viz)[:min(self.topk, self.args.samples)]
                        self.visualize_trajectories(
                            dataset_name, gt_actions, image_plot_dir,
                            i=self.args.sgld_steps, traj=b, traj_id=int(idxs[b].item()),
                            deltas=deltas_N, cur_obs_image=obs_1.repeat(self.args.samples, 1, 1, 1, 1),
                            cur_goal_image=goal_1, preds=preds_last, loss=loss_viz, topk_idx=topk_idx
                        )

                    # 导出 3D 样本分布（绝对坐标）便于后处理
                    samples_dir = os.path.join(dataset_save_output_dir, "samples")
                    os.makedirs(samples_dir, exist_ok=True)
                    with torch.no_grad():
                        # 将 (N,T,3 delta-物理域) -> (N,T,3 绝对坐标)
                        # 注意：get_action_torch 把归一化动作映射回物理位移（delta），我们需要 cumsum 得到绝对轨迹
                        deltas_xyz_phys = get_action_torch(deltas_N[..., :3], ACTION_STATS_TORCH)  # (N,T,3) delta(物理域)
                        xyz_abs_N = torch.cumsum(deltas_xyz_phys, dim=1)                           # (N,T,3)
                        torch.save(
                            {
                                "xyz_abs": xyz_abs_N.detach().cpu(),
                                "weights": w_N.detach().cpu(),
                                "energy": E_N.detach().cpu()
                            },
                            os.path.join(samples_dir, f"samples_idx{int(idxs[b].item())}.pt")
                        )

                pred_actions_list.append(pred_actions_b)
                pred_yaw_list.append(pred_yaw_b)

                # 用于与旧接口保持一致的保存/绘图（点估计轨迹）
                all_deltas.append(deltas_agg.detach())
                all_preds_seq.append(preds_seq_N[0:1].detach())  # 随便取一个样本的序列用于旧图；也可重新用 deltas_agg rollout
                with torch.no_grad():
                    # 使用 deltas_agg 重新 rollout 得到一致的 preds/preds_completed
                    preds_seq_agg = self.autoregressive_rollout(
                        obs_1, deltas_agg, self.args.rollout_stride, aug_image=aug_1, camera_mats=cam_1
                    )
                    final_lpips = self.loss_fn(preds_seq_agg[:, -1], goal_1).flatten(0)
                    all_preds_seq[-1] = preds_seq_agg.detach()
                    all_final_losses.append(final_lpips)

            else:
                # ---------- 点估计（梯度优化） ----------
                with torch.enable_grad():
                    deltas_1, preds_seq_1, final_lpips = self._optimize_single_traj(
                        obs_img_1=obs_1, goal_img_1=goal_1, T=T,
                        aug_img_1=aug_1, cam_1=cam_1,
                        steps=self.opt_steps, lr=0.05, smooth_w=1e-3,
                        dataset_name=dataset_name, gt_actions=gt_actions, image_plot_dir=image_plot_dir,
                        traj_idx=b, traj_id=int(idxs[b].item()), viz_every=100
                    )

                pred_actions_b = get_action_torch(deltas_1[..., :3], ACTION_STATS_TORCH)
                pred_yaw_b = deltas_1[..., -1].sum(1)

                pred_actions_list.append(pred_actions_b)
                pred_yaw_list.append(pred_yaw_b)
                all_deltas.append(deltas_1)
                all_preds_seq.append(preds_seq_1)
                all_final_losses.append(final_lpips)

        deltas = torch.cat(all_deltas, dim=0)                 # (B,T,4)
        preds_completed = torch.cat(all_preds_seq, dim=0)     # (B,T,C,H,W)
        preds = preds_completed[:, -1]                        # (B,C,H,W)
        loss = torch.cat(all_final_losses, dim=0) if len(all_final_losses) > 0 else torch.zeros(B)

        # 保存/绘图（点估计）
        if self.args.save_preds:
            save_planning_pred(dataset_save_output_dir, B, idxs, obs_image, goal_image, preds, deltas, loss, gt_actions, preds_completed)
        if self.args.plot:
            img_name = os.path.join(image_plot_dir, f'FINAL_{idx_string}.png')
            traj_name = os.path.join(image_plot_dir, f"TRAJ_{idx_string}.png")
            plot_batch_final(obs_image[:, -1].to(self.device), preds, goal_img_B, idxs, loss.detach().cpu().tolist(), save_path=img_name)
            plot_batch_trajectories(obs_image[:, -1].to(self.device), preds_completed, goal_img_B, idxs, save_path=traj_name)

        pred_actions = torch.cat(pred_actions_list, dim=0)  # (B,T,3) —— 动作域(物理 delta)
        pred_yaw = torch.cat(pred_yaw_list, dim=0)          # (B,)
        return pred_actions, pred_yaw

    # ---------------------- 图像级可视化（沿用旧接口） ----------------------
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

    # ---------------------- 世界模型自回归 rollout ----------------------
    def autoregressive_rollout(self, obs_image, deltas, rollout_stride, aug_image, camera_mats):
        # stride 合并
        deltas = deltas.unflatten(1, (-1, rollout_stride)).sum(2)
        preds = []
        curr_obs = obs_image.clone().to(self.device)

        for i in range(deltas.shape[1]):
            curr_delta = deltas[:, i:i + 1]
            all_models = self.model, self.diffusion, self.vae
            x_pred_pixels = model_forward_wrapper(
                all_models, curr_obs, curr_delta, self.args.rollout_stride,
                self.latent_size, num_cond=self.num_cond, device=self.device,
                x_supervised=aug_image, camera_mats=camera_mats
            )
            x_pred_pixels = x_pred_pixels.unsqueeze(1)
            curr_obs = torch.cat((curr_obs, x_pred_pixels), dim=1)  # append current prediction
            curr_obs = curr_obs[:, 1:]  # remove first observation
            preds.append(x_pred_pixels)

        preds = torch.cat(preds, 1)
        return preds

    def get_eval_name(self):
        # 名称仍保留原参数，便于对比
        self.eval_name = f'SGLD_N{self.args.num_samples}_K{self.args.topk}_RS{self.args.rollout_stride}_rep{self.args.num_repeat_eval}_OPT{self.args.opt_steps}'

    def actions_to_traj(self, actions, is_delta: bool = True):
        """
        将动作序列转 PoseTrajectory3D。
        actions: (T, 3) —— 通常是“物理域位移 (dx,dy,dz)”，需要 cumsum 得到绝对坐标。
        is_delta=True 表示需要积分；如果上游已经是绝对坐标，置 False。
        """
        actions = actions.to(torch.float64)
        if is_delta:
            positions_xyz = torch.cumsum(actions, dim=0)  # (T,3) 绝对
        else:
            positions_xyz = actions
        orientations_quat_wxyz = torch.zeros((positions_xyz.shape[0], 4), dtype=torch.float64)
        orientations_quat_wxyz[:, -1] = 1.0
        timestamps = torch.arange(positions_xyz.shape[0], dtype=torch.float64)
        traj = PoseTrajectory3D(
            positions_xyz=positions_xyz,
            orientations_quat_wxyz=orientations_quat_wxyz,
            timestamps=timestamps
        )
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
                camera_ctx = camera_ctx[:, -self.num_cond:]
                camera_mats = torch.cat([camera_ctx, camera_goal], dim=1)  # (B, num_cond+1, 4, 4)

                # 注意：采样（SGLD）或优化前需要 enable_grad（内部已处理）
                pred_actions, pred_yaw = self.generate_actions(
                    eval_save_output_dir, dataset_name, idxs, obs_image, goal_image, gt_actions,
                    self.config["trajectory_eval_len_traj_pred"], aug_image=aug_image, camera_mats=camera_mats
                )

                # 评估：将“动作域 delta(物理)”-> “绝对 3D 轨迹”，并画 3D
                for i in range(len(obs_image)):
                    pred_traj_i = self.actions_to_traj(pred_actions[i, :, :3], is_delta=True)
                    gt_traj_i = self.actions_to_traj(gt_actions[i, :, :3], is_delta=True)

                    ate, rpe_trans, _ = self.eval_metrics(gt_traj_i, pred_traj_i)

                    pred_final_pos = pred_actions[i, -1, :3].to('cpu')  # (3,)
                    pred_final_yaw = pred_yaw[i].to('cpu')
                    goal_final_pos = goal_pos[i, 0, :3]  # (3,)
                    goal_final_yaw = goal_pos[i, 0, -1]
                    pos_diff_norm = torch.norm(pred_final_pos - goal_final_pos)
                    yaw_diff = pred_final_yaw - goal_final_yaw
                    yaw_diff_norm = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).abs()

                    metric_logger.meters['{}_ate'.format(dataset_name)].update(ate, n=1)
                    metric_logger.meters['{}_rpe_trans'.format(dataset_name)].update(rpe_trans, n=1)
                    metric_logger.meters['{}_pos_diff_norm'.format(dataset_name)].update(pos_diff_norm, n=1)
                    metric_logger.meters['{}_yaw_diff_norm'.format(dataset_name)].update(yaw_diff_norm, n=1)

                    if self.args.save_preds and self.args.plot3d:
                        out_dir = os.path.join(self.args.save_output_dir, dataset_name, self.eval_name, "traj3d")
                        os.makedirs(out_dir, exist_ok=True)
                        # 取绝对坐标用于绘图
                        pred_xyz = torch.cumsum(pred_actions[i, :, :3], dim=0)
                        gt_xyz = torch.cumsum(gt_actions[i, :, :3], dim=0)
                        fn = os.path.join(out_dir, f"traj3d_idx{int(idxs[i].item())}.png")
                        plot_traj3d(pred_xyz, gt_xyz, save_path=fn, title=f"{dataset_name} idx={int(idxs[i].item())}")

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


# ---------------------- 入口 ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Default Args
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument("--ckp", type=str, default='0100000', help="checkpoint id (e.g., 0100000)")

    parser.add_argument("--datasets", type=str, default=None, help="dataset name(s), comma-separated")
    parser.add_argument("--output_dir", type=str, default=None, help="dir to save model predictions")
    parser.add_argument('--save_preds', action='store_true', default=False, help='whether to save prediction tensors')
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")

    # Planning Specific Args（原字段保留便于对比）
    parser.add_argument("--num_samples", type=int, default=10, help="visualization sample count")
    parser.add_argument("--rollout_stride", type=int, default=1, help="rollout stride")
    parser.add_argument("--topk", type=int, default=5, help="top-k for visualization highlight")
    parser.add_argument("--opt_steps", type=int, default=15, help="baseline gradient steps for opt-mode")
    parser.add_argument("--num_repeat_eval", type=int, default=1, help="number of evals for one action")
    parser.add_argument('--plot', action='store_true', default=False)

    # ---------- 新增：概率场采样配置 ----------
    parser.add_argument("--sampler", type=str, default="opt", choices=["opt", "langevin"],
                        help="opt: gradient descent point estimate; langevin: SGLD sampling in normalized action space")
    parser.add_argument("--samples", type=int, default=64, help="number of trajectory samples to draw (langevin)")
    parser.add_argument("--sgld_steps", type=int, default=200, help="SGLD steps")
    parser.add_argument("--sgld_lr", type=float, default=5e-3, help="SGLD step size (eta)")
    parser.add_argument("--beta", type=float, default=50.0, help="energy temperature for posterior weighting")
    parser.add_argument("--prior_scale", type=float, default=1.0, help="N(mu0, prior_scale*var_scale) prior width in normalized domain")
    parser.add_argument('--plot3d', action='store_true', default=False, help='save 3D trajectory plots')

    args = parser.parse_args()

    evaluator = WM_Planning_Evaluator(args)
    # local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 如需调试本地 rank 可启用
    # gpu_id = torch.cuda.current_device()
    evaluator.evaluate()
