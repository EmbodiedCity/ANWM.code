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
import pickle
import numpy as np
import lpips
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
import json

# ======== [PANEL BEGIN] extra imports for paper panel ========
from glob import glob
from PIL import ImageDraw, ImageFont
# ======== [PANEL END]   extra imports for paper panel ========
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

# NEW: 仅用于设置进程组超时
import datetime  # NEW
import torch.distributed as tdist  # NEW


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

# ======== [PANEL BEGIN] file-level helpers for montage ========
def _pil_open_safe(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[panel][skip] cannot open {path}: {e}")
        return None

def _stack_images_grid(img_paths, cols=4, unit_h=360, pad=12, bg=(255,255,255), caption=True):
    """
    将若干图片按网格拼接：
      - cols: 每行列数
      - unit_h: 单图等高缩放后的高度
      - pad: 内边距
      - caption: 是否在图下写文件名
    """
    imgs, names = [], []
    for p in img_paths:
        im = _pil_open_safe(p)
        if im is None:
            continue
        w, h = im.size
        if h <= 0:
            continue
        new_w = max(1, int(round(w * (unit_h / float(h)))))
        im = im.resize((new_w, unit_h), Image.BICUBIC)
        imgs.append(im)
        names.append(os.path.basename(p))
    if not imgs:
        raise RuntimeError("No images to montage.")

    cap_h = 40 if caption else 0
    rows = (len(imgs) + cols - 1) // cols

    # 每列最大宽度，保证列内水平居中
    col_w_max = [0] * cols
    for i, im in enumerate(imgs):
        c = i % cols
        col_w_max[c] = max(col_w_max[c], im.size[0])

    grid_w = sum(col_w_max) + pad * (cols + 1)
    grid_h = rows * (unit_h + cap_h) + pad * (rows + 1)
    canvas = Image.new("RGB", (grid_w, grid_h), bg)

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    x_offsets = [pad]
    for c in range(1, cols):
        x_offsets.append(x_offsets[-1] + col_w_max[c-1] + pad)

    for idx, im in enumerate(imgs):
        r, c = divmod(idx, cols)
        x0 = x_offsets[c] + (col_w_max[c] - im.size[0]) // 2
        y0 = pad + r * (unit_h + cap_h + pad)
        canvas.paste(im, (x0, y0))
        if cap_h and font is not None:
            cap = names[idx]
            if len(cap) > 60:
                cap = cap[:57] + "..."
            # ======== [A FIX BEGIN] Pillow>=10 兼容文本测量 ========
            try:
                bbox = draw.textbbox((0, 0), cap, font=font)  # (l,t,r,b)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                try:
                    tw, th = draw.textsize(cap, font=font)  # 旧版 Pillow
                except Exception:
                    tw, th = 0, 0
                    cap = None
            if cap:
                draw.text((x0 + (im.size[0]-tw)//2, y0 + unit_h + (cap_h-th)//2),
                          cap, fill=(0,0,0), font=font)
            # ======== [A FIX END]   Pillow>=10 兼容文本测量 ========
    return canvas
# ======== [PANEL END]   file-level helpers for montage ========

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
        predefined_index = f"data_splits/{dataset_name}/test/navigation_eval_16_long.pkl"
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
                # traj_names="rollout_traj_names.txt"
                traj_names="traj_names.txt"
            )

    return dataset

# NEW: 在本脚本内初始化默认进程组，并把 collective 超时拉长
def _init_pg_timeout(hours: int = 4) -> int:
    """
    初始化默认进程组，并设置更长的 collective 超时。
    返回本进程 local_rank（可作为 GPU id）。
    """
    if not tdist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        tdist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(hours=hours),
        )
        return local_rank
    return int(os.environ.get("LOCAL_RANK", 0))


class WM_Planning_Evaluator:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.exp = args.exp

        # NEW: 直接在本脚本中初始化分布式进程组，设置 4 小时超时
        _gpu_id = _init_pg_timeout(hours=4)  # NEW
        self.device = torch.device(f"cuda:{_gpu_id}")  # NEW
        device = self.device  # NEW: 兼容下面 to(device) 的用法

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

        # 总是设置 save_output_dir（用于保存指标文件），即使 save_preds 为 False
        if self.args.output_dir is not None:
            exp_name = os.path.basename(self.args.exp).split('.')[0]
            self.args.save_output_dir = os.path.join(args.output_dir, exp_name)
            os.makedirs(self.args.save_output_dir, exist_ok=True)  # 所有 rank 都创建目录（os.makedirs 是安全的）
        else:
            # 如果没有指定 output_dir，使用当前目录
            self.args.save_output_dir = "."

        # Loading Datasets
        self.dataset_names = self.args.datasets.split(',')
        self.datasets = {}
        # Load pre-sampled candidate trajectories from pkl files
        for dataset_name in self.dataset_names:
            candidate_traj_path = f"data_splits/{dataset_name}/test/{dataset_name}_{self.args.num_samples}_trajectories_long.pkl"
            with open(candidate_traj_path, "rb") as f:
                self.sampled_trajectories = pickle.load(f)
            print(f"Loaded {len(self.sampled_trajectories)} pre-sampled candidate trajectories from {candidate_traj_path}")
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
            # curr_data_loader = [next(iter(curr_data_loader))]
            self.datasets[dataset_name] = curr_data_loader

        # Loading Model
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
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)

        # NEW: 用整数 gpu id 绑定 DDP 设备，更稳妥
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[_gpu_id], output_device=_gpu_id, find_unused_parameters=False
        )  # NEW
        self.model_without_ddp = self.model.module

        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.mode = 'rule' # assume RULE for planning
        self.num_samples = self.args.num_samples
        self.topk = self.args.topk
        self.opt_steps = self.args.opt_steps
        self.num_repeat_eval = self.args.num_repeat_eval
        self.action_dim = 4 # hardcoded (delta_x, delta_y, delta_z, delta_yaw)

    # ====== NEW: 通用 Editor 保存（可用于最终或候选轨迹） ======
    def _save_editor_run(self, base_dir: str, sid: int,
                         init_img: torch.Tensor,     # (C,H,W) in [-1,1]
                         step_seq: torch.Tensor,     # (T,C,H,W) in [-1,1]
                         goal_img: torch.Tensor,     # (C,H,W) in [-1,1]
                         deltas: torch.Tensor = None,# (1,T,4) or (T,4)
                         tag: str = "final",         # "final" / "candidate"
                         cand_rank: int = None,
                         cand_id: int = None,
                         extra_meta: dict = None):
        """
        将一条轨迹（init + T steps + goal）保存为逐帧 PNG + metadata.json。
        若 tag != 'final'，则落到 run_xxx/candidates/cand_yyy 目录，并保存 deltas.npy。

        仅两点增强：
        1) 若提供 deltas，自动聚合成与 step_seq 帧数 T 对齐（np.array_split 分组后求和）；
        2) 用聚合后的 deltas 累计得到逐帧 pose（避免 metadata 中全 0）。
        """
        editor_dir = _ensure_dir(os.path.join(base_dir, "editor"))
        run_dir = _ensure_dir(os.path.join(editor_dir, f"run_{sid:03d}"))
        if tag != "final":
            run_dir = _ensure_dir(os.path.join(run_dir, "candidates", f"cand_{cand_rank:03d}"))

        frames_dir = _ensure_dir(os.path.join(run_dir, "frames"))

        # 存图
        _save_png(init_img, os.path.join(frames_dir, "init.png"))
        T = step_seq.shape[0]
        for t in range(T):
            _save_png(step_seq[t], os.path.join(frames_dir, f"step_{t:03d}.png"))
        _save_png(goal_img, os.path.join(frames_dir, "goal.png"))

        # 逐帧 LPIPS
        init_loss = float(self.loss_fn(init_img[None].to(self.device),
                                       goal_img[None].to(self.device)).item())
        step_losses = []
        for t in range(T):
            step_losses.append(float(self.loss_fn(step_seq[t:t+1].to(self.device),
                                                  goal_img[None].to(self.device)).item()))

        # ====== 小幅增强：将 raw deltas 聚合到 T，并累计 pose ======
        deltas_eff = None  # (T,4) or None
        pose_xyz = None
        pose_yaw = None
        if deltas is not None:
            if isinstance(deltas, torch.Tensor):
                d = deltas.detach().cpu().numpy()
            else:
                d = np.asarray(deltas)
            if d.ndim == 3 and d.shape[0] == 1:
                d = d[0]  # -> (T_raw,4)
            assert d.ndim == 2 and d.shape[1] == 4, f"Expected (T_raw,4), got {d.shape}"

            T_raw = d.shape[0]
            if T_raw == T:
                deltas_eff = d.astype(np.float32, copy=False)
            else:
                idxs = np.arange(T_raw)
                chunks = np.array_split(idxs, T)
                summed = []
                for ch in chunks:
                    if len(ch) == 0:
                        summed.append(np.zeros((4,), dtype=np.float32))
                    else:
                        summed.append(d[ch].sum(0).astype(np.float32))
                deltas_eff = np.stack(summed, axis=0)  # (T,4)

            # 累计 pose
            pose_xyz = deltas_eff[:, :3].cumsum(axis=0)   # (T,3)
            pose_yaw = deltas_eff[:, 3].cumsum(axis=0)    # (T,)

            # 仅 candidate 目录落一个 deltas.npy（与你之前的约定一致）
            if tag != "final":
                np.save(os.path.join(run_dir, "deltas.npy"), deltas_eff)

        # metadata
        H, W = init_img.shape[-2:]
        meta = {
            "run_index": int(sid),
            "type": tag,
            "candidate_rank": None if cand_rank is None else int(cand_rank),
            "candidate_id": None if cand_id is None else int(cand_id),
            "video_mp4": "",
            "fps": int(self.args.rollout_stride) if self.args.rollout_stride > 0 else 1,
            "resolution": {"width": int(W), "height": int(H)},
            "num_frames": T + 1,
            "sequence": list(range(T)),
            "commands": {},
            "frames": []
        }
        if extra_meta:
            meta.update(extra_meta)

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
            if deltas_eff is None:
                dx = dy = dz = dtheta = 0.0
                px = py = pz = pyaw = 0.0
            else:
                dx, dy, dz, dtheta = [float(v) for v in deltas_eff[t]]
                px, py, pz = [float(v) for v in pose_xyz[t]]
                pyaw = float(pose_yaw[t])

            meta["frames"].append({
                "frame": t + 1,
                "png": f"frames/step_{t:03d}.png",
                "action": int(t),
                "pose": {"x": px, "y": py, "z": pz, "theta_rad": pyaw},
                "delta": {"dx": dx, "dy": dy, "dz": dz, "dtheta": dtheta},
                "loss": step_losses[t]
            })

        _write_json(os.path.join(run_dir, "metadata.json"), meta)
        return run_dir, meta
    # ====== END NEW ======

    # ======== [SAMPLE PANEL] per-sample figure ========
    def save_single_sample_panel(self,
                                 obs_img_tchw: torch.Tensor,        # (C,H,W) in [-1,1]
                                 goal_img_tchw: torch.Tensor,       # (C,H,W) in [-1,1]
                                 cand_pred_tchw_list,               # [ (C,H,W) x K ] in [-1,1]
                                 cand_deltas_world_list,            # [ (T,4) x K ] in meters/rad
                                 cand_losses_list,                  # [float] length=K
                                 out_path: str,
                                 goal_xy=None,                      # (x,y) or (x,y,z)
                                 labels=None):                      # ["P1","P2","P3"]
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        def _to_np01(tchw):
            x = tchw.detach().to(torch.float32).cpu()
            if x.min() < 0: x = (x + 1) / 2
            x = x.clamp(0, 1)
            return x.permute(1, 2, 0).numpy()

        K = len(cand_pred_tchw_list)
        if labels is None:
            labels = [f"P{i+1}" for i in range(K)]

        # 准备图像
        obs_np  = _to_np01(obs_img_tchw)
        goal_np = _to_np01(goal_img_tchw)
        preds_np = [_to_np01(p) for p in cand_pred_tchw_list]

        # 布局：Obs | Goal | Traj(3D) | Pred1 | Pred2 | Pred3
        ncols = 3 + K
        fig_w = 4 * ncols
        fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 4), constrained_layout=True)

        # 0) Observation
        axes[0].imshow(obs_np); axes[0].set_title("Observation"); axes[0].axis("off")
        # 1) Goal
        axes[1].imshow(goal_np); axes[1].set_title("Goal"); axes[1].axis("off")

        # 2) Trajectory plot —— 3D 版本
        ax_traj = fig.add_subplot(1, ncols, 3, projection="3d")
        color_cycle = ["#F4A259", "#E4572E", "#2FBF71", "#4C78A8", "#B279A2"]

        for i, dw in enumerate(cand_deltas_world_list):
            dw = np.asarray(dw)  # (T, 4)
            xyz = dw[:, :3].cumsum(axis=0)  # 起点(0,0,0) 累加
            xs, ys, zs = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            ax_traj.plot3D(xs, ys, zs, color=color_cycle[i % len(color_cycle)], linewidth=3)
            if xyz.shape[0] > 0:
                x0, y0, z0 = xyz[0, 0], xyz[0, 1], xyz[0, 2]
            else:
                x0 = y0 = z0 = 0.0
            ax_traj.text(x0, y0, z0, labels[i],
                         fontsize=9, color=color_cycle[i % len(color_cycle)], weight="bold")

        # goal_xy 兼容 2D/3D：2D 时 z=0
        if goal_xy is not None:
            gx, gy = float(goal_xy[0]), float(goal_xy[1])
            gz = float(goal_xy[2]) if (isinstance(goal_xy, (list, tuple, np.ndarray)) and len(goal_xy) >= 3) else 0.0
            ax_traj.scatter(gx, gy, gz, c="#2066E0", s=40, depthshade=True)
            ax_traj.text(gx, gy, gz, "Goal", color="#2066E0", fontsize=9)

        ax_traj.set_title("Trajectories (3D)")
        ax_traj.set_xlabel("X (m)")
        ax_traj.set_ylabel("Y (m)")
        ax_traj.set_zlabel("Z (m)")
        ax_traj.view_init(elev=22, azim=-60)
        ax_traj.grid(True, alpha=0.2)

        # 3..) Predictions with Loss
        best_j = int(np.argmin(np.asarray(cand_losses_list)))
        for j in range(K):
            ax = axes[3+j]
            ax.imshow(preds_np[j]); ax.axis("off")
            ax.set_title(f"Prediction {j+1}")
            txt = f"Loss: {cand_losses_list[j]:.3f}"
            if j == best_j:
                rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                         fill=False, linewidth=6, edgecolor="#2FBF71")
                ax.add_patch(rect)
            ax.text(0.5, 0.02, txt, transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=11, color="black",
                    bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.15"))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    # ======== [END SAMPLE PANEL] ========

    def generate_actions(self, dataset_save_output_dir, dataset_name, idxs, obs_image, goal_image, gt_actions, len_traj_pred, aug_image):
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

        # [NEW] 为每个样本收集一致性与 APE 统计（Top-K 范围内）
        agree_flags_all = []
        selected_apes_all = []
        oracle_min_apes_all = []
        # [SR NEW] 收集 noisy-GT 选择率（每样本一个 0/1）
        sr_flags_all = []
        # [END NEW]

        for traj in range(n_evals):
            traj_id = int(idxs.flatten()[traj].item())

            # === 构造 GT 轨迹（含起点）
            gt_deltas_xyz = gt_actions[traj, :, :3].to('cpu').numpy()  # [T, 3]
            gt_xyz = np.concatenate([np.zeros((1, 3), dtype=np.float32),
                                     np.cumsum(gt_deltas_xyz, axis=0).astype(np.float32)], axis=0)  # [T+1, 3]
            T = gt_xyz.shape[0]
            gt_yaw = np.zeros((T,), dtype=np.float32)
            GT_traj = [(float(x), float(y), float(z), float(yaw)) for (x, y, z), yaw in zip(gt_xyz, gt_yaw)]

            # 从pkl文件加载候选轨迹
            candidate_trajectories = self.sampled_trajectories[traj_id]  # [N, T+1, 4]
            candidate_trajectories = np.array(candidate_trajectories, dtype=np.float32)

            # 转换为 delta（轨迹点之间的变化）
            candidate_deltas = candidate_trajectories[:, 1:, :] - candidate_trajectories[:, :-1, :]  # [N, steps, 4]

            ##### [CHANGE BLOCK - 与 CEM 保持一致：米→waypoint→[-1,1]；yaw=弧度不归一化]
            deltas_world = torch.tensor(candidate_deltas, dtype=torch.float32, device=self.device)  # (N, T, 4)
            spacing = float(data_config[dataset_name]['metric_waypoint_spacing'])
            d_xyz_waypt = deltas_world[..., :3] / spacing
            mins = ACTION_STATS_TORCH['min'].to(self.device)[:3]
            maxs = ACTION_STATS_TORCH['max'].to(self.device)[:3]
            d_xyz_norm = ((d_xyz_waypt - mins) / (maxs - mins)) * 2.0 - 1.0
            delta_yaw = calculate_delta_yaw(d_xyz_waypt)
            deltas_model = torch.cat([d_xyz_norm, delta_yaw], dim=-1)
            ##### [END CHANGE BLOCK]

            cur_obs_image = obs_image[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1)
            cur_goal_image = goal_image[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1).squeeze(1)
            cur_aug_image = aug_image[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1)

            if self.num_repeat_eval * self.num_samples > 12:
                cur_losses = []
                for r in range(self.num_repeat_eval):
                    preds = self.autoregressive_rollout(cur_obs_image, deltas_model, self.args.rollout_stride, aug_image=cur_aug_image)
                    preds = preds[:, -1]
                    loss = self.loss_fn(preds.to(self.device), cur_goal_image.to(self.device)).flatten(0)
                    cur_losses.append(loss)
                loss = torch.stack(cur_losses).mean(dim=0)
            else:
                expanded_deltas = deltas_model.repeat(self.num_repeat_eval, 1, 1)
                expanded_obs_image = cur_obs_image.repeat(self.num_repeat_eval, 1, 1, 1, 1)
                expanded_goal_image = cur_goal_image.repeat(self.num_repeat_eval, 1, 1, 1)
                expanded_aug = cur_aug_image.repeat(self.num_repeat_eval, 1, 1, 1, 1)

                preds = self.autoregressive_rollout(expanded_obs_image, expanded_deltas, self.args.rollout_stride, aug_image=expanded_aug)
                preds = preds[:, -1]

                loss = self.loss_fn(preds.to(self.device), expanded_goal_image.to(self.device)).flatten(0)
                loss = loss.view(self.num_repeat_eval, -1)
                loss = loss.mean(dim=0)

                preds = preds[:self.num_samples]

            sorted_idx = torch.argsort(loss)
            topk_idx = sorted_idx[:self.topk]
            best_idx = sorted_idx[0]

            # ---------- [SR NEW] 噪声GT 是否被选中 ----------
            noisy_idx  = 0  # 第 0 个是 random(GT+noise)
            loss_noisy = float(loss[noisy_idx].item())
            best_loss  = float(loss[best_idx].item())
            sr_flag    = 1.0 if loss_noisy <= best_loss else 0.0
            sr_flags_all.append(sr_flag)
            # ---------------------------------------------

            # ======== [SAMPLE PANEL CALL] 每样本一张图，轨迹窗为 3D ========
            try:
                panel_ids = topk_idx[:3].tolist()  # Top-3
                cand_pred_list = [preds[i].detach().cpu() for i in panel_ids]
                cand_dw_list   = [deltas_world[i].detach().cpu().numpy() for i in panel_ids]
                cand_loss_list = [float(loss[i].item()) for i in panel_ids]
                goal_xyz = gt_xyz[-1, :3]  # <<< 改为 3D 终点
                self.save_single_sample_panel(
                    obs_img_tchw = obs_image[traj, -1],
                    goal_img_tchw = goal_image[traj].squeeze(0),
                    cand_pred_tchw_list = cand_pred_list,
                    cand_deltas_world_list = cand_dw_list,
                    cand_losses_list = cand_loss_list,
                    out_path = os.path.join(image_plot_dir, f"sample_panel_idx{traj_id}.png"),
                    goal_xy = goal_xyz,  # <<< 传入 3D
                    labels = [f"P{i+1}" for i in range(len(panel_ids))]
                )
            except Exception as e:
                print(f"[sample_panel warn] idx{traj_id}: {e}")
            # ======== [END SAMPLE PANEL CALL] ========

            # ====== [NEW] Top-K 内部 APE 一致性评估 ======
            gt_traj_this = self.actions_to_traj(gt_actions[traj, :, :3])
            cids = topk_idx.tolist()

            ape_partial = []
            for cid in cids:
                cand_actions = get_action_torch(
                    deltas_model[cid:cid+1, :, :3], ACTION_STATS_TORCH
                )[0]  # (T,3) cpu
                cand_traj = self.actions_to_traj(cand_actions)
                cand_ate, _, _ = self.eval_metrics(gt_traj_this, cand_traj)
                ape_partial.append((int(cid), float(cand_ate)))

            ape_tensor = torch.full((self.num_samples,), float("inf"), device=loss.device)
            for cid, ate_val in ape_partial:
                ape_tensor[cid] = ate_val

            local_best = topk_idx[torch.argmin(ape_tensor[topk_idx])]
            min_ape_idx = int(local_best.item())

            agree_minloss_minape = int(int(best_idx.item()) == min_ape_idx)
            selected_ape = float(ape_tensor[int(best_idx.item())].item())
            oracle_min_ape = float(ape_tensor[min_ape_idx].item())

            agree_flags_all.append(agree_minloss_minape)
            selected_apes_all.append(selected_ape)
            oracle_min_apes_all.append(oracle_min_ape)
            # ====== [END NEW] ======

            # ====== [DETAIL RECORD] 收集所有候选轨迹的详细指标（不改变原有逻辑） ======
            if self.args.save_preds:
                # 计算所有候选轨迹的APE（用于详细记录）
                all_candidate_metrics = []
                for cid in range(self.num_samples):
                    cand_actions = get_action_torch(
                        deltas_model[cid:cid+1, :, :3], ACTION_STATS_TORCH
                    )[0]  # (T,3) cpu
                    cand_traj = self.actions_to_traj(cand_actions)
                    cand_ate, cand_rpe_trans, cand_rpe_rot = self.eval_metrics(gt_traj_this, cand_traj)
                    all_candidate_metrics.append({
                        "candidate_id": int(cid),
                        "loss": float(loss[cid].item()),
                        "ape": float(cand_ate),
                        "rpe_trans": float(cand_rpe_trans),
                        "rpe_rot": float(cand_rpe_rot),
                        "is_topk": int(cid in topk_idx.tolist()),
                        "is_selected": int(cid == best_idx.item()),
                        "is_noisy_gt": int(cid == noisy_idx)
                    })
                
                # 计算selected在APE排序中的位置
                ape_sorted = sorted([(cid, all_candidate_metrics[cid]["ape"]) for cid in range(self.num_samples)], 
                                  key=lambda x: x[1])
                ape_rank_of_selected = -1
                for rank, (cid, _) in enumerate(ape_sorted):
                    if cid == best_idx.item():
                        ape_rank_of_selected = rank
                        break
                
                # 构建详细记录
                sample_detail = {
                    "traj_id": int(traj_id),
                    "model_selection": {
                        "selected_candidate_id": int(best_idx.item()),
                        "selected_loss": float(loss[best_idx.item()].item()),
                        "selected_ape": float(selected_ape),
                        "selected_rank": int(torch.where(sorted_idx == best_idx)[0].item())
                    },
                    "oracle_selection": {
                        "oracle_candidate_id": int(min_ape_idx),
                        "oracle_ape": float(oracle_min_ape),
                        "oracle_loss": float(loss[min_ape_idx].item()),
                        "oracle_rank": int(torch.where(sorted_idx == min_ape_idx)[0].item()) if min_ape_idx in sorted_idx else -1
                    },
                    "noisy_gt": {
                        "candidate_id": int(noisy_idx),
                        "loss": float(loss_noisy),
                        "ape": float(all_candidate_metrics[noisy_idx]["ape"]),
                        "is_selected": int(sr_flag),
                        "rank": int(torch.where(sorted_idx == noisy_idx)[0].item()) if noisy_idx in sorted_idx else -1
                    },
                    "consistency": {
                        "agree_minloss_minape": int(agree_minloss_minape),
                        "loss_rank_of_oracle": int(torch.where(sorted_idx == min_ape_idx)[0].item()) if min_ape_idx in sorted_idx else -1,
                        "ape_rank_of_selected": ape_rank_of_selected
                    },
                    "all_candidates": all_candidate_metrics,
                    "topk_ids": topk_idx.tolist(),
                    "gt_trajectory": {
                        "num_steps": int(T),
                        "final_position": [float(gt_xyz[-1, 0]), float(gt_xyz[-1, 1]), float(gt_xyz[-1, 2])],
                        "total_distance": float(np.linalg.norm(gt_xyz[-1, :3]))
                    },
                    "config": {
                        "num_samples": int(self.num_samples),
                        "topk": int(self.topk),
                        "num_repeat_eval": int(self.num_repeat_eval),
                        "rollout_stride": int(self.args.rollout_stride)
                    }
                }
                
                # 保存到JSON文件
                detail_dir = os.path.join(dataset_save_output_dir, "case_details")
                _ensure_dir(detail_dir)
                detail_file = os.path.join(detail_dir, f"case_{traj_id:04d}.json")
                _write_json(detail_file, sample_detail)
            # ====== [END DETAIL RECORD] ======

            # 保留两份：模型用（归一化平移+弧度yaw）；可读保存用（世界系 米/弧度）
            best_delta_model = deltas_model[best_idx]
            best_delta_world = deltas_world[best_idx]

            all_deltas.append(best_delta_model.unsqueeze(0))
            all_losses.append(loss[best_idx].item())
            all_preds.append(preds[best_idx].unsqueeze(0))

            # ====== 保存候选轨迹（Top-K）逐帧 PNG + metadata.json + deltas.npy ======
            if self.args.save_preds:
                topk_loop_ids = topk_idx

                editor_base_dir = dataset_save_output_dir
                sid = traj_id

                single_obs  = obs_image[traj:traj+1]
                single_aug  = aug_image[traj:traj+1]
                single_goal = goal_image[traj].squeeze(0)

                cand_summaries = []
                for rank, cid in enumerate(topk_loop_ids.tolist()):
                    cand_delta_model = deltas_model[cid:cid+1]
                    cand_delta_world = deltas_world[cid:cid+1]

                    cand_seq = self.autoregressive_rollout(
                        single_obs, cand_delta_model, self.args.rollout_stride,
                        aug_image=single_aug
                    )[0]

                    cand_final_lpips = float(loss[cid].item())

                    cand_ape_val = float(ape_tensor[int(cid)].item())
                    _, meta = self._save_editor_run(
                        base_dir=editor_base_dir,
                        sid=sid,
                        init_img=single_obs[0, -1],
                        step_seq=cand_seq,
                        goal_img=single_goal,
                        deltas=cand_delta_world,
                        tag="candidate",
                        cand_rank=rank,
                        cand_id=int(cid),
                        extra_meta={"final_lpips": cand_final_lpips,
                                   "cand_ape": cand_ape_val}
                    )

                    cand_summaries.append({
                        "rank": rank,
                        "cand_id": int(cid),
                        "final_lpips": cand_final_lpips,
                        "cand_ape": cand_ape_val
                    })

                run_cands_dir = os.path.join(editor_base_dir, "editor", f"run_{sid:03d}", "candidates")
                _ensure_dir(run_cands_dir)
                _write_json(os.path.join(run_cands_dir, "summary.json"),
                            {"num_candidates_saved": len(cand_summaries),
                             "topk": int(len(cand_summaries)),
                             "items": cand_summaries})
            # ====== END ======

            if self.args.plot:
                self.visualize_trajectories(dataset_name, gt_actions, image_plot_dir, 1, traj, traj_id, deltas_model, cur_obs_image, cur_goal_image, preds, loss, topk_idx)

        # Final rollout for selected deltas
        final_deltas = torch.cat(all_deltas, dim=0)
        preds = self.autoregressive_rollout(obs_image, final_deltas, self.args.rollout_stride, aug_image=aug_image)
        preds_completed = preds
        preds = preds[:, -1]

        loss = self.loss_fn(preds.to(self.device), goal_image.squeeze(1).to(self.device)).flatten(0)

        if self.args.save_preds:
            save_planning_pred(dataset_save_output_dir, n_evals, idxs, obs_image, goal_image, preds, final_deltas, loss, gt_actions, preds_completed)

        # 逐样本 Editor JSON/PNG
        if self.args.save_preds:
            B = obs_image.shape[0]
            init_imgs = obs_image[:, -1]
            goal_imgs = goal_image.squeeze(1)
            for b in range(B):
                sid = int(idxs[b].item())
                self._save_editor_run(
                    base_dir=dataset_save_output_dir,
                    sid=sid,
                    init_img=init_imgs[b],
                    step_seq=preds_completed[b],
                    goal_img=goal_imgs[b],
                    deltas=final_deltas[b:b+1],
                    tag="final",
                    cand_rank=None,
                    cand_id=None,
                    extra_meta={
                        "final_lpips": float(loss[b].item()),
                        "selected_ape": float(selected_apes_all[b]),
                        "oracle_min_ape": float(oracle_min_apes_all[b]),
                        "agree_minloss_minape": int(agree_flags_all[b])
                    }
                )

        if self.args.plot:
            img_name = os.path.join(image_plot_dir, f'FINAL_{idx_string}.png')
            traj_name = os.path.join(image_plot_dir, f"TRAJ_{idx_string}.png")
            plot_batch_final(obs_image[:, -1].to(self.device), preds, goal_image.squeeze(1).to(self.device), idxs, all_losses, save_path=img_name)
            plot_batch_trajectories(obs_image[:, -1].to(self.device), preds_completed, goal_image.squeeze(1).to(self.device), idxs, save_path=traj_name)

        pred_actions = get_action_torch(final_deltas[:, :, :3], ACTION_STATS_TORCH)
        pred_yaw = final_deltas[:, :, -1].sum(1)

        agree_tensor = torch.tensor(agree_flags_all, dtype=torch.float32)
        selected_apes_tensor = torch.tensor(selected_apes_all, dtype=torch.float32)
        oracle_min_apes_tensor = torch.tensor(oracle_min_apes_all, dtype=torch.float32)
        sr_tensor = torch.tensor(sr_flags_all, dtype=torch.float32)  # [SR NEW] 每样本 0/1

        return pred_actions, pred_yaw, agree_tensor, selected_apes_tensor, oracle_min_apes_tensor, sr_tensor

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
        self.eval_name = f'RULE_N{self.args.num_samples}_K{self.args.topk}_RS{self.args.rollout_stride}_rep{self.args.num_repeat_eval}_OPT{self.args.opt_steps}'

    def actions_to_traj(self, actions):
        positions_xyz = torch.zeros((actions.shape[0], 3))
        positions_xyz[:, :3] = actions
        orientations_quat_wxyz = torch.zeros((actions.shape[0], 4))
        orientations_quat_wxyz[:, -1] = 1
        timestamps = torch.arange(actions.shape[0], dtype=torch.float64)
        traj = PoseTrajectory3D(positions_xyz=positions_xyz, orientations_quat_wxyz=orientations_quat_wxyz, timestamps=timestamps)
        return traj

    # ======== [PANEL BEGIN] class method: make a paper panel without CLI ========
    def make_paper_panel_for_dataset(self,
                                     dataset_name: str,
                                     out_name: str = "paper_panel.png",
                                     cols: int = 4,
                                     unit_h: int = 360,
                                     pad: int = 12,
                                     patterns=("idx*_iter1.png","idx*_iter1_trajs.png","FINAL_*.png","TRAJ_*.png"),
                                     only_indices=None,
                                     caption: bool = True):
        """
        在 <output_dir>/<dataset>/<eval_name>/plots/ 收集指定模式的图片，拼成一张大图。
        - 自动保存至 <output_dir>/<dataset>/<eval_name>/{out_name}
        - only_indices: 例如 [5,11,17,...] 只拼带有 "idx{n}" 的样本（FINAL_*/TRAJ_* 始终保留）
        """
        base_dir  = os.path.join(self.args.save_output_dir, dataset_name, self.eval_name)
        plots_dir = os.path.join(base_dir, "plots")
        if not os.path.isdir(plots_dir):
            raise FileNotFoundError(f"[panel] plots dir not found: {plots_dir}")

        cand = []
        for pat in patterns:
            cand.extend(sorted(glob(os.path.join(plots_dir, pat))))

        if only_indices:
            idx_strs = {f"idx{int(i)}" for i in only_indices}
            def keep(p):
                bn = os.path.basename(p)
                if bn.startswith("FINAL_") or bn.startswith("TRAJ_"):
                    return True
                return any(s in bn for s in idx_strs)
            cand = [p for p in cand if keep(p)]

        if not cand:
            raise RuntimeError(f"[panel] no images matched in {plots_dir} with patterns={patterns} and indices={only_indices}")

        panel = _stack_images_grid(cand, cols=cols, unit_h=unit_h, pad=pad, caption=caption)
        out_path = os.path.join(base_dir, out_name)
        panel.save(out_path)
        print(f"[panel saved] {out_path}  (images={len(cand)}, cols={cols}, unit_h={unit_h})")
    # ======== [PANEL END]   class method: make a paper panel without CLI ========

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
            for (idxs, obs_image, goal_image, gt_actions, goal_pos, aug_image) in metric_logger.log_every(curr_data_loader, 1, header):
                obs_image = obs_image[:, -self.num_cond:]
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    # [SR NEW] 多接收一个 sr_tensor
                    pred_actions, pred_yaw, agree_flags, selected_apes, oracle_min_apes, sr_tensor = \
                        self.generate_actions(eval_save_output_dir, dataset_name, idxs, obs_image, goal_image, gt_actions, self.config["trajectory_eval_len_traj_pred"], aug_image=aug_image)
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

                    metric_logger.meters[f'{dataset_name}_agree_minloss_minape'].update(float(agree_flags[i].item()), n=1)
                    metric_logger.meters[f'{dataset_name}_selected_ape'].update(float(selected_apes[i].item()), n=1)
                    metric_logger.meters[f'{dataset_name}_oracle_min_ape'].update(float(oracle_min_apes[i].item()), n=1)
                    metric_logger.meters[f'{dataset_name}_sr'].update(float(sr_tensor[i].item()), n=1)  # [SR NEW]

                # ======== [B BEGIN] 每个 batch 结束后增量导出一张 live 面板 ========
                try:
                    if self.args.save_preds and self.args.plot:
                        self.make_paper_panel_for_dataset(
                            dataset_name=dataset_name,
                            out_name="paper_panel_live.png",
                            cols=4,
                            unit_h=360,
                            pad=12,
                            patterns=("idx*_iter1.png","idx*_iter1_trajs.png","FINAL_*.png","TRAJ_*.png"),
                            only_indices=None,
                            caption=True
                        )
                except Exception as e:
                    print(f"[panel][live warn] {e}")
                # ======== [B END]   每个 batch 结束后增量导出一张 live 面板 ========

            # 所有 rank 都调用 save_metric_to_disk（内部会处理同步，只在 rank0 写文件）
            output_fn = os.path.join(self.args.save_output_dir, f'{dataset_name}_{self.eval_name}.json')
            save_metric_to_disk(metric_logger, output_fn)

            # ======== [PANEL BEGIN] auto make paper panel ========
            try:
                if self.args.save_preds:
                    self.make_paper_panel_for_dataset(
                        dataset_name=dataset_name,
                        out_name="paper_panel.png",
                        cols=4,
                        unit_h=360,
                        pad=12,
                        patterns=("idx*_iter1.png","idx*_iter1_trajs.png","FINAL_*.png","TRAJ_*.png"),
                        only_indices=None,
                        caption=True
                    )
            except Exception as e:
                print(f"[panel][warn] failed to build panel for {dataset_name}: {e}")
            # ======== [PANEL END]   auto make paper panel ========

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
    parser.add_argument("--topk", type=int, default=5, help="top k samples to take mean and var for RULE")
    parser.add_argument("--opt_steps", type=int, default=15, help="num iterations for RULE")
    parser.add_argument("--num_repeat_eval", type=int, default=1, help="number of evals for one action")
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()

    evaluator = WM_Planning_Evaluator(args)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpu_id = torch.cuda.current_device()  # Or args.gpu if explicitly set
    evaluator.evaluate()
