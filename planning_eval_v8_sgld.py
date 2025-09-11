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
import torch.distributed as tdist  # 用于优雅销毁进程组

# ===================== 常量 =====================
SGLD_LOG_EVERY   = 10   # SGLD 迭代打印频率
SGLD_LOG_KEEP    = 8    # SGLD 打印时最多展示样本数
ROLLOUT_LOG_KEEP = 8    # 每步 rollout 打印样本数

# ★ 物理域单步位移上限（米）：依据“单步<~5m”的先验，避免轨迹暴涨/跑偏
STEP_CAP_METERS  = 5.0

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
    imgs_for_plotting = []

    init_imgs = (init_imgs + 1) / 2
    goal_imgs = (goal_imgs + 1) / 2
    pred_imgs_seq = (pred_imgs_seq + 1) / 2

    for b in range(B):
        traj = [init_imgs[b]]
        traj += [pred_imgs_seq[b, t] for t in range(T)]
        traj += [goal_imgs[b]]
        imgs_for_plotting.append(torch.stack(traj))

    imgs_for_plotting = torch.cat(imgs_for_plotting, dim=0)
    grid_img = vutils.make_grid(imgs_for_plotting, nrow=T+2, padding=2)
    np_grid = grid_img.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(max(12, (T+2) * 4), B * 4))
    ax.imshow(np_grid)
    ax.axis("off")

    img_height, img_width = H + 2, W + 2
    for b in range(B):
        for t in range(T + 2):
            x = t * img_width
            y = b * img_height
            label = "Goal" if t == T + 1 else ("Init" if t == 0 else f"Step {t}")
            if t == 0:
                label = f"ID:{int(idxs[b].item())} Init"
            ax.text(x + img_width / 2, y + 20, label, color="white",
                    ha="center", va="top", fontsize=16, backgroundcolor="black")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_traj3d(pred_xyz: torch.Tensor,
                gt_xyz: torch.Tensor = None,
                yaw_seq: torch.Tensor = None,
                save_path: str = "traj3d.png",
                title: str = None):
    """3D 轨迹绘制（绝对坐标）"""
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pred = pred_xyz.detach().cpu().numpy()
    x, y, z = pred[:, 0], pred[:, 1], pred[:, 2]

    if gt_xyz is not None:
        gt_np = gt_xyz.detach().cpu().numpy()
        gx, gy, gz = gt_np[:, 0], gt_np[:, 1], gt_np[:, 2]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    try:
        ax.set_proj_type('persp')
    except Exception:
        pass

    ax.plot(x, y, z, linewidth=2, label="Pred")
    if gt_xyz is not None:
        ax.plot(gx, gy, gz, linestyle='--', linewidth=1.5, label="GT")

    ax.scatter(x[0], y[0], z[0], s=50, marker='o', label="Start")
    ax.scatter(x[-1], y[-1], z[-1], s=50, marker='^', label="End")

    if yaw_seq is not None:
        yaw_np = yaw_seq.detach().cpu().numpy()
        step = max(1, len(yaw_np) // 20)
        u = np.cos(yaw_np[::step])
        v = np.sin(yaw_np[::step])
        w = np.zeros_like(u)
        ax.quiver(x[::step], y[::step], z[::step], u, v, w,
                  length=max(np.ptp(x), np.ptp(y)) * 0.05,
                  normalize=True)

    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        max_range = np.array([np.ptp(x), np.ptp(y), np.ptp(z)]).max()
        if max_range == 0:
            max_range = 1.0
        mx, my, mz = (x.min() + x.max())/2, (y.min() + y.max())/2, (z.min() + z.max())/2
        ax.set_xlim(mx - max_range/2, mx + max_range/2)
        ax.set_ylim(my - max_range/2, my + max_range/2)
        ax.set_zlim(mz - max_range/2, mz + max_range/2)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    if title: ax.set_title(title)
    ax.legend(); ax.grid(True)
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

# ==== 概率场云（Point Cloud）可视化，叠加 GT 参考 ====
def plot_probability_cloud(xyz_abs: torch.Tensor,
                           weights: torch.Tensor,
                           save_prefix: str,
                           title: str = "",
                           gt_xyz_abs: torch.Tensor = None,
                           max_points: int = 200_000):
    """
    xyz_abs: (N, T, 3)  -> 采样得到的绝对轨迹点（已 cumsum）
    weights: (N,)       -> 每条轨迹的后验权重（softmax 后）
    gt_xyz_abs: (T, 3)  -> GT 绝对轨迹，用折线叠加作为参考
    save_prefix: 保存文件路径前缀；会生成 *_3d.png 和 *_xy.png
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    xyz = xyz_abs.detach().cpu().reshape(-1, 3)          # (N*T,3)
    N, T, _ = xyz_abs.shape
    w = weights.detach().cpu()                           # (N,)
    w_rep = w.unsqueeze(1).expand(N, T).reshape(-1)      # (N*T,)

    # 归一化权重到[0,1]
    if w_rep.numel() > 0:
        w_min, w_max = float(w_rep.min()), float(w_rep.max())
        w_norm = (w_rep - w_min) / (w_max - w_min + 1e-12) if (w_max > w_min) else torch.ones_like(w_rep)
    else:
        w_norm = torch.ones_like(w_rep)

    # 下采样
    M = xyz.shape[0]
    if M > max_points:
        idx = torch.randperm(M)[:max_points]
        xyz = xyz[idx]
        w_norm = w_norm[idx]

    sizes = 5.0 + 45.0 * w_norm.numpy()   # 点大小随权重
    alphas = 0.10 + 0.50 * w_norm.numpy() # 透明度随权重

    # --- 3D 云 ---
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=sizes, alpha=alphas)
    if gt_xyz_abs is not None:
        g = gt_xyz_abs.detach().cpu().numpy()
        ax.plot(g[:, 0], g[:, 1], g[:, 2], linestyle='--', linewidth=2.0, label='GT')
        ax.scatter(g[0, 0], g[0, 1], g[0, 2], s=40, c='k', marker='o')  # GT start
        ax.scatter(g[-1, 0], g[-1, 1], g[-1, 2], s=40, c='k', marker='^')  # GT end
        ax.legend()

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    if title: ax.set_title(title + " (3D cloud)")
    try: ax.set_box_aspect([1,1,1])
    except Exception: pass
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    plt.savefig(f"{save_prefix}_3d.png", dpi=200)
    plt.close(fig)

    # --- 俯视 (X-Y) 云：方形坐标范围 + 外边距，避免直线时过窄 ---
    fig2, ax2 = plt.subplots(figsize=(8, 8))  # 方形画布
    ax2.scatter(xyz[:, 0], xyz[:, 1], s=sizes, alpha=alphas)
    if gt_xyz_abs is not None:
        g = gt_xyz_abs.detach().cpu().numpy()
        ax2.plot(g[:, 0], g[:, 1], linestyle='--', linewidth=2.0, label='GT')
        ax2.scatter(g[0, 0], g[0, 1], s=30, c='k', marker='o')
        ax2.scatter(g[-1, 0], g[-1, 1], s=30, c='k', marker='^')
        ax2.legend()

    ax2.set_xlabel("X"); ax2.set_ylabel("Y")
    ax2.set_aspect("equal", adjustable="box")

    # 计算云点+GT的总体范围，构造方形边界并加 padding
    all_x = xyz[:, 0].numpy()
    all_y = xyz[:, 1].numpy()
    if gt_xyz_abs is not None:
        all_x = np.concatenate([all_x, g[:, 0]])
        all_y = np.concatenate([all_y, g[:, 1]])

    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    rng_x, rng_y = x_max - x_min, y_max - y_min
    max_rng = max(rng_x, rng_y)

    # 最小显示跨度，直线或近零范围时保证可视化；再加 15% 边距
    MIN_SPAN = 1.0  # 米
    PAD = 0.15
    span = max(max_rng, MIN_SPAN)
    half = 0.5 * span * (1.0 + PAD)
    ax2.set_xlim(cx - half, cx + half)
    ax2.set_ylim(cy - half, cy + half)

    if title: ax2.set_title(title + " (top-down)")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_xy.png", dpi=200)
    plt.close(fig2)

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
        self.mode = 'sgld'
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
    # 可微优化（点估计；替代 CEM）
    # ================================================================
    def _optimize_single_traj(
        self, obs_img_1, goal_img_1, T, aug_img_1, cam_1, steps, lr=0.05, smooth_w=1e-3,
        dataset_name=None, gt_actions=None, image_plot_dir=None, traj_idx=None, traj_id=None,
        viz_every=1
    ):
        device = self.device
        mu0, _ = self.init_mu_sigma(obs_img_1, T)
        mu0 = mu0.to(device)

        norm_xyz = mu0[:, :3].unsqueeze(1).repeat(1, T, 1).clone().detach().requires_grad_(True)
        last_yaw_bias = torch.zeros(1, device=device).requires_grad_(True)

        for p in self.model.parameters():
            p.requires_grad_((False))

        opt = torch.optim.Adam([norm_xyz, last_yaw_bias], lr=lr)

        with torch.enable_grad():
            for it in range(max(20, steps * 4)):
                opt.zero_grad(set_to_none=True)
                unnorm_xyz = unnormalize_data(norm_xyz, ACTION_STATS_TORCH)
                d_yaw = calculate_delta_yaw(unnorm_xyz)
                deltas_1 = torch.cat([norm_xyz, d_yaw.to(norm_xyz.device)], dim=-1)
                deltas_1[:, -1, -1] += last_yaw_bias * np.pi

                preds_seq = self.autoregressive_rollout(
                    obs_img_1, deltas_1, self.args.rollout_stride,
                    aug_image=aug_img_1, camera_mats=cam_1
                )
                pred_last = preds_seq[:, -1]

                if self.args.plot and (dataset_name is not None) and (image_plot_dir is not None):
                    last_it = max(20, steps * 4) - 1
                    if (it % viz_every == 0) or (it == last_it):
                        with torch.no_grad():
                            viz_num = int(self.args.num_samples)
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
    # 仅正则/先验能量（对 norm_xyz 可导）
    # ================================================================
    def _regularizer_energy(
        self,
        norm_xyz: torch.Tensor,   # (N,T,3)
        obs_img: torch.Tensor,    # (N, num_cond+1, C, H, W)
        T: int,
        var_scale: torch.Tensor,
        prior_scale: float,
        smooth_w: float = 1e-3,
        jerk_w: float = 1e-3,
    ):
        N = norm_xyz.shape[0]

        mu0_all, _ = self.init_mu_sigma(obs_img[:, -1], T)          # (N,4)
        mu0_xyz = mu0_all.to(norm_xyz.device)[..., :3]               # (N,3)
        mu0_xyz = mu0_xyz.unsqueeze(1).expand(N, T, 3)               # (N,T,3)

        l2   = norm_xyz.pow(2).mean(dim=(1, 2))
        diff = (norm_xyz[:, 1:] - norm_xyz[:, :-1]).pow(2).mean(dim=(1, 2)) if T > 1 else torch.zeros_like(l2)
        jerk = (norm_xyz[:, 2:] - 2*norm_xyz[:, 1:-1] + norm_xyz[:, :-2]).pow(2).mean(dim=(1, 2)) if T > 2 else torch.zeros_like(l2)

        if var_scale.ndim == 0:
            var_scale = var_scale.repeat(3)
        sigma2 = (prior_scale * var_scale[:3])**2
        prior = ((norm_xyz - mu0_xyz)**2 / sigma2.view(1, 1, 3)).mean(dim=(1, 2))

        E_reg = smooth_w * (l2 + diff) + jerk_w * jerk + prior
        return E_reg

    # ================================================================
    # 计算能量（用于 SGLD） —— 图像项 no_grad；正则可导
    # ================================================================
    def _energy(
        self, obs_img, goal_img, deltas_norm, aug_img, cam_mats,
        mid_w=0.2, smooth_w=1e-3, jerk_w=1e-3, prior_w=1.0
    ):
        norm_xyz = deltas_norm[..., :3]                               # (N,T,3)

        # ---------- 图像打分（不构图） ----------
        with torch.no_grad():
            unnorm_xyz = unnormalize_data(norm_xyz, ACTION_STATS_TORCH)
            d_yaw = calculate_delta_yaw(unnorm_xyz)
            deltas = torch.cat([norm_xyz, d_yaw.to(norm_xyz.device)], dim=-1)

            preds_seq = self.autoregressive_rollout(
                obs_img, deltas, self.args.rollout_stride,
                aug_image=aug_img, camera_mats=cam_mats
            )  # (N, T, C, H, W)

            pred_last = preds_seq[:, -1]  # (N, C, H, W)
            img_loss = self.loss_fn(pred_last, goal_img).flatten(1).mean(dim=1)  # (N,)

            if preds_seq.shape[1] > 1:
                N_, T_, C, H, W = preds_seq.shape
                mid = preds_seq[:, :-1].contiguous().view(N_*(T_-1), C, H, W)
                tgt = goal_img.unsqueeze(1).expand(N_, T_-1, C, H, W).contiguous().view(N_*(T_-1), C, H, W)
                mid_lpips_each = self.loss_fn(mid, tgt).flatten(1).mean(dim=1)
                mid_img_loss = mid_lpips_each.view(N_, T_-1).mean(dim=1)
            else:
                mid_img_loss = torch.zeros_like(img_loss)

            img_term = (img_loss + mid_w * mid_img_loss).detach()    # (N,)

        # ---------- 正则/先验能量（可导） ----------
        var_scale = torch.tensor(
            data_hyperparams[self.args.datasets]['var_scale'],
            device=norm_xyz.device, dtype=norm_xyz.dtype
        )
        E_reg = self._regularizer_energy(
            norm_xyz, obs_img, norm_xyz.shape[1],
            var_scale=var_scale, prior_scale=self.args.prior_scale,
            smooth_w=smooth_w, jerk_w=jerk_w
        )

        E = img_term + prior_w * E_reg
        return E, preds_seq

    # ---------------------- 每步 rollout 后打印当前累计末端坐标 ----------------------
    def _print_rollout_step(self, deltas_collapsed, step_idx):
        try:
            rank = dist.get_rank()
        except Exception:
            rank = 0
        if rank != 0:
            return

        with torch.no_grad():
            B, T, _ = deltas_collapsed.shape
            n_show = min(B, ROLLOUT_LOG_KEEP)
            deltas_phys = get_action_torch(deltas_collapsed[:n_show, :step_idx+1, :3], ACTION_STATS_TORCH)
            xyz_abs = torch.cumsum(deltas_phys, dim=1)

            for b in range(n_show):
                last = xyz_abs[b, -1]
                x, y, z = float(last[0]), float(last[1]), float(last[2])
                print(f"[ROLL][step {step_idx+1}/{T}] sample={b} x={x:.6f} y={y:.6f} z={z:.6f}")

    # ================================================================
    # 概率场采样：SGLD
    # ================================================================
    def _sample_langevin(self, obs_img_1, goal_img_1, T, aug_img_1, cam_1,
                         N=64, steps=200, eta=5e-3, beta=50.0):
        device = self.device

        mu0, _ = self.init_mu_sigma(obs_img_1[:, -1], T)
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

        try:
            print_rank = (dist.get_rank() == 0)
        except Exception:
            print_rank = True

        if print_rank:
            with torch.no_grad():
                mu_vec  = mu0[0, 0].detach().cpu()
                std_vec = init_std[0, 0].detach().cpu()
                print(f"[INIT][prior_norm] N={N} T={T} mu={mu_vec.tolist()} std={std_vec.tolist()}", flush=True)

                nz = norm_xyz.detach().cpu().reshape(-1, 3)
                m_norm = nz.mean(0); s_norm = nz.std(0, unbiased=False)
                print(f"[INIT][empirical_norm] mean={m_norm.tolist()} std={s_norm.tolist()}", flush=True)

                deltas_phys = get_action_torch(norm_xyz.detach(), ACTION_STATS_TORCH).detach().cpu().reshape(-1, 3)
                m_phy = deltas_phys.mean(0); s_phy = deltas_phys.std(0, unbiased=False)
                zmin = float(deltas_phys[:, 2].min()); zmax = float(deltas_phys[:, 2].max())
                print(f"[INIT][empirical_phys] mean={m_phy.tolist()} std={s_phy.tolist()} z[min,max]=({zmin:.6f},{zmax:.6f})", flush=True)

        def _print_snapshot(step, note=""):
            if not print_rank:
                return
            with torch.no_grad():
                n_save = int(min(SGLD_LOG_KEEP, norm_xyz.shape[0]))
                deltas_xyz_phys = get_action_torch(norm_xyz[:n_save].detach(), ACTION_STATS_TORCH)  # (n_save,T,3)
                xyz_abs = torch.cumsum(deltas_xyz_phys, dim=1)
                z_vals = xyz_abs[..., 2].detach().cpu()
                z_min = float(z_vals.min()); z_max = float(z_vals.max())
                z_rng = float(z_max - z_min)
                z_rng_mean = float((z_vals.max(dim=1).values - z_vals.min(dim=1).values).mean())

                norm_small = norm_xyz[:n_save].detach().cpu().reshape(-1, 3)
                std_x = norm_small[:, 0].std().item()
                std_y = norm_small[:, 1].std().item()
                std_z = norm_small[:, 2].std().item()

                print(f"[SGLD][{note}] step={step:04d}/{steps} n={n_save} "
                      f"z_min={z_min:.6f} z_max={z_max:.6f} z_range={z_rng:.6f} "
                      f"mean_range_per_traj={z_rng_mean:.6f} | "
                      f"norm_std(x,y,z)=({std_x:.4f},{std_y:.4f},{std_z:.4f})")

        # ================== SGLD 主循环（带物理步长钳制） ==================
        for k in range(steps):
            norm_xyz = norm_xyz.detach().requires_grad_(True)

            # 能量（图像项 no_grad；正则可导）
            deltas_norm = torch.cat([norm_xyz, torch.zeros(N, T, 1, device=device)], dim=-1)
            E, _ = self._energy(
                obs_img_1.repeat(N, 1, 1, 1, 1),
                goal_img_1.repeat(N, 1, 1, 1),
                deltas_norm,
                aug_img_1.repeat(N, 1, 1, 1, 1),
                cam_1.repeat(N, 1, 1, 1)
            )

            loss_all = beta * E.mean()
            grad = None
            if loss_all.requires_grad:
                grad = torch.autograd.grad(loss_all, norm_xyz, retain_graph=False, allow_unused=True)[0]

            if (grad is None) or (not torch.isfinite(grad).all()):
                # 回退到纯正则梯度
                var_scale_local = torch.tensor(
                    data_hyperparams[self.args.datasets]['var_scale'],
                    device=norm_xyz.device, dtype=norm_xyz.dtype
                )
                E_reg = self._regularizer_energy(
                    norm_xyz, obs_img_1, T,
                    var_scale=var_scale_local,
                    prior_scale=self.args.prior_scale,
                    smooth_w=1e-3, jerk_w=1e-3
                )
                loss_reg = beta * E_reg.mean()
                grad = torch.autograd.grad(loss_reg, norm_xyz, retain_graph=False)[0]

            with torch.no_grad():
                # Langevin 更新（归一化域）
                norm_xyz.add_(-0.5 * eta * grad)
                norm_xyz.add_((eta ** 0.5) * torch.randn_like(norm_xyz))

                # 物理域单步位移上限（~5m）：先计算物理模长，再按比例缩小归一化向量
                deltas_phys = get_action_torch(norm_xyz, ACTION_STATS_TORCH)  # (N,T,3) 物理域单步
                mag_phys = torch.linalg.norm(deltas_phys, dim=-1, keepdim=True)  # (N,T,1)
                scale_phys = torch.clamp(STEP_CAP_METERS / (mag_phys + 1e-8), max=1.0)
                norm_xyz.mul_(scale_phys)

                # 额外硬边界（极端安全网）
                norm_xyz.clamp_(-5.0, 5.0)

            if (k % max(1, SGLD_LOG_EVERY) == 0):
                _print_snapshot(k, note="iter")
        # ================== 主循环结束 ==================

        _print_snapshot(steps, note="final")

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

            E_shift = E - E.min()
            w = torch.softmax(-beta * E_shift, dim=0)

        return deltas_N, preds_seq, E, w

    # ---------------------- 世界模型自回归 rollout ----------------------
    def autoregressive_rollout(self, obs_image, deltas, rollout_stride, aug_image, camera_mats):
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
            curr_obs = torch.cat((curr_obs, x_pred_pixels), dim=1)
            curr_obs = curr_obs[:, 1:]
            preds.append(x_pred_pixels)

            self._print_rollout_step(deltas, i)

        preds = torch.cat(preds, 1)
        return preds

    def get_eval_name(self):
        self.eval_name = f'SGLD_N{self.args.num_samples}_K{self.args.topk}_RS{self.args.rollout_stride}_rep{self.args.num_repeat_eval}_OPT{self.args.opt_steps}'

    def actions_to_traj(self, actions, is_delta: bool = True):
        actions = actions.detach().to(dtype=torch.float64, device="cpu")

        if is_delta:
            positions_xyz = torch.cumsum(actions, dim=0)
        else:
            positions_xyz = actions

        T_len = positions_xyz.shape[0]
        orientations_quat_wxyz = torch.zeros((T_len, 4), dtype=torch.float64)
        orientations_quat_wxyz[:, 0] = 1.0  # w=1, x=y=z=0

        timestamps = torch.arange(T_len, dtype=torch.float64)

        traj = PoseTrajectory3D(
            positions_xyz=positions_xyz.numpy(),
            orientations_quat_wxyz=orientations_quat_wxyz.numpy(),
            timestamps=timestamps.numpy()
        )
        return traj

    # ---------------------- 统一动作生成接口 ----------------------
    def generate_actions(self, dataset_save_output_dir, dataset_name, idxs, obs_image, goal_image, gt_actions, len_traj_pred, aug_image, camera_mats):
        idx_string = "_".join(map(str, idxs.flatten().int().tolist()))
        image_plot_dir = os.path.join(dataset_save_output_dir, 'plots') if dataset_save_output_dir else None
        if image_plot_dir:
            os.makedirs(image_plot_dir, exist_ok=True)
        cloud_dir = os.path.join(dataset_save_output_dir, 'cloud') if dataset_save_output_dir else None
        if cloud_dir:
            os.makedirs(cloud_dir, exist_ok=True)

        B = obs_image.shape[0]
        T = len_traj_pred

        all_deltas = []
        all_preds_seq = []
        all_final_losses = []

        goal_img_B = goal_image.squeeze(1).to(self.device)

        pred_actions_list = []
        pred_yaw_list = []

        for b in range(B):
            obs_1 = obs_image[b:b + 1]
            aug_1 = aug_image[b:b + 1]
            cam_1 = camera_mats[b:b + 1]
            goal_1 = goal_img_B[b:b + 1]

            if self.args.sampler == "langevin":
                with torch.enable_grad():
                    deltas_N, preds_seq_N, E_N, w_N = self._sample_langevin(
                        obs_1, goal_1, T, aug_1, cam_1,
                        N=self.args.samples, steps=self.args.sgld_steps,
                        eta=self.args.sgld_lr, beta=self.args.beta
                    )
                deltas_agg = torch.einsum('n,ntd->td', w_N, deltas_N[..., :4]).unsqueeze(0)  # (1,T,4)

                pred_actions_b = get_action_torch(deltas_agg[..., :3], ACTION_STATS_TORCH)  # (1,T,3)
                pred_yaw_b = deltas_agg[..., -1].sum(1)                                     # (1,)

                # 概率场点云（叠加 GT）
                try:
                    with torch.no_grad():
                        deltas_xyz_phys = get_action_torch(deltas_N[..., :3], ACTION_STATS_TORCH)  # (N,T,3)
                        xyz_abs_all = torch.cumsum(deltas_xyz_phys, dim=1)                         # (N,T,3)
                        gt_xyz_abs = torch.cumsum(gt_actions[b, :, :3], dim=0)                    # (T,3)

                        save_prefix = os.path.join(cloud_dir, f"idx{int(idxs[b].item())}_cloud") if cloud_dir else f"idx{int(idxs[b].item())}_cloud"
                        plot_probability_cloud(
                            xyz_abs_all, w_N,
                            save_prefix=save_prefix,
                            title=f"{dataset_name} idx={int(idxs[b].item())} (all)",
                            gt_xyz_abs=gt_xyz_abs
                        )
                        topk_k = int(self.topk) if self.topk is not None else min(5, self.args.samples)
                        topk_idx = torch.argsort(w_N, descending=True)[:max(1, min(topk_k, self.args.samples))]
                        plot_probability_cloud(
                            xyz_abs_all[topk_idx], w_N[topk_idx],
                            save_prefix=save_prefix + "_topk",
                            title=f"{dataset_name} idx={int(idxs[b].item())} (topk)",
                            gt_xyz_abs=gt_xyz_abs
                        )
                except Exception as e:
                    print(f"[WARN] plot cloud failed for idx {int(idxs[b].item())}: {e}")

                if self.args.save_preds and image_plot_dir is not None:
                    with torch.no_grad():
                        preds_last = preds_seq_N[:, -1]
                        loss_viz = self.loss_fn(preds_last, goal_1.repeat(self.args.samples, 1, 1, 1)).flatten(0)
                        topk_idx_vis = torch.argsort(loss_viz)[:min(self.topk, self.args.samples)]
                        self.visualize_trajectories(
                            dataset_name, gt_actions, image_plot_dir,
                            i=self.args.sgld_steps, traj=b, traj_id=int(idxs[b].item()),
                            deltas=deltas_N, cur_obs_image=obs_1.repeat(self.args.samples, 1, 1, 1, 1),
                            cur_goal_image=goal_1, preds=preds_last, loss=loss_viz, topk_idx=topk_idx_vis
                        )

                    samples_dir = os.path.join(dataset_save_output_dir, "samples") if dataset_save_output_dir else None
                    if samples_dir:
                        os.makedirs(samples_dir, exist_ok=True)
                        with torch.no_grad():
                            deltas_xyz_phys_all = get_action_torch(deltas_N[..., :3], ACTION_STATS_TORCH)
                            xyz_abs_N = torch.cumsum(deltas_xyz_phys_all, dim=1)
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

                all_deltas.append(deltas_agg.detach())
                all_preds_seq.append(preds_seq_N[0:1].detach())
                with torch.no_grad():
                    preds_seq_agg = self.autoregressive_rollout(
                        obs_1, deltas_agg, self.args.rollout_stride, aug_image=aug_1, camera_mats=cam_1
                    )
                    final_lpips = self.loss_fn(preds_seq_agg[:, -1], goal_1).flatten(0)
                    all_preds_seq[-1] = preds_seq_agg.detach()
                    all_final_losses.append(final_lpips)

            else:
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

        if self.args.save_preds and dataset_save_output_dir is not None:
            save_planning_pred(dataset_save_output_dir, B, idxs, obs_image, goal_image, preds, deltas, loss, gt_actions, preds_completed)
        if self.args.plot and image_plot_dir is not None:
            img_name = os.path.join(image_plot_dir, f'FINAL_{idx_string}.png')
            traj_name = os.path.join(image_plot_dir, f"TRAJ_{idx_string}.png")
            plot_batch_final(obs_image[:, -1].to(self.device), preds, goal_img_B, idxs, loss.detach().cpu().tolist(), save_path=img_name)
            plot_batch_trajectories(obs_image[:, -1].to(self.device), preds_completed, goal_img_B, idxs, save_path=traj_name)

        pred_actions = torch.cat(pred_actions_list, dim=0)  # (B,T,3)
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

    # ------------ 评估 ------------
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

                # 不关梯度，确保 SGLD/优化可回传
                pred_actions, pred_yaw = self.generate_actions(
                    eval_save_output_dir, dataset_name, idxs, obs_image, goal_image, gt_actions,
                    self.config["trajectory_eval_len_traj_pred"], aug_image=aug_image, camera_mats=camera_mats
                )

                # === 统一设备，避免 cuda/cpu 混用 ===
                goal_pos = goal_pos.to(pred_actions.device)

                # 指标计算与绘图可在 no_grad 中进行
                with torch.no_grad():
                    for i in range(len(obs_image)):
                        pred_traj_i = self.actions_to_traj(pred_actions[i, :, :3], is_delta=True)
                        gt_traj_i = self.actions_to_traj(gt_actions[i, :, :3], is_delta=True)

                        ate, rpe_trans, _ = self.eval_metrics(gt_traj_i, pred_traj_i)

                        pred_final_pos = pred_actions[i, -1, :3]
                        pred_final_yaw = pred_yaw[i]
                        goal_final_pos = goal_pos[i, 0, :3]       # 已转到 pred 的 device
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
                            pred_xyz = torch.cumsum(pred_actions[i, :, :3], dim=0)
                            gt_xyz   = torch.cumsum(gt_actions[i,  :, :3], dim=0)
                            d_yaw_seq = calculate_delta_yaw(pred_actions[i:i+1, :, :3])  # (1,T,1)
                            yaw_seq   = torch.cumsum(d_yaw_seq[0, :, 0], dim=0)          # (T,)
                            fn = os.path.join(out_dir, f"traj3d_idx{int(idxs[i].item())}.png")
                            plot_traj3d(pred_xyz, gt_xyz, yaw_seq=yaw_seq, save_path=fn,
                                        title=f"{dataset_name} idx={int(idxs[i].item())}")

            output_fn = os.path.join(self.args.save_output_dir, f'{dataset_name}_{self.eval_name}.json') if self.args.save_preds else None
            if output_fn:
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

    # Planning Specific Args
    parser.add_argument("--num_samples", type=int, default=10, help="visualization sample count")
    parser.add_argument("--rollout_stride", type=int, default=1, help="rollout stride")
    parser.add_argument("--topk", type=int, default=5, help="top-k for visualization highlight")
    parser.add_argument("--opt_steps", type=int, default=15, help="baseline gradient steps for opt-mode")
    parser.add_argument("--num_repeat_eval", type=int, default=1, help="number of evals for one action")
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--plot3d', action='store_true', default=False, help='save 3D trajectory plots')

    # 概率场采样配置
    parser.add_argument("--sampler", type=str, default="opt", choices=["opt", "langevin"],
                        help="opt: gradient descent point estimate; langevin: SGLD sampling in normalized action space")
    parser.add_argument("--samples", type=int, default=64, help="number of trajectory samples to draw (langevin)")
    parser.add_argument("--sgld_steps", type=int, default=200, help="SGLD steps")
    parser.add_argument("--sgld_lr", type=float, default=5e-3, help="SGLD step size (eta)")
    parser.add_argument("--beta", type=float, default=50.0, help="energy temperature for posterior weighting")
    parser.add_argument("--prior_scale", type=float, default=1.0, help="N(mu0, prior_scale*var_scale) prior width in normalized domain")

    args = parser.parse_args()

    evaluator = WM_Planning_Evaluator(args)

    # 优雅销毁分布式进程组，避免 NCCL 警告/阻塞
    try:
        evaluator.evaluate()
    finally:
        if tdist.is_available() and tdist.is_initialized():
            tdist.destroy_process_group()
