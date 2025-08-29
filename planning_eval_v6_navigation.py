# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import json
import argparse

import torch
import yaml
import numpy as np
import lpips
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image

from diffusers.models import AutoencoderKL

### evo evaluation library ###
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync, metrics
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation

from diffusion import create_diffusion
from datasets_v3 import TrajectoryEvalDataset
from isolated_nwm_infer_v6 import model_forward_wrapper
from misc import (
    calculate_delta_yaw,
    get_action_torch,
    save_planning_pred,
    log_viz_single,
    transform,
    unnormalize_data,
)
from isolated_nwm_eval import save_metric_to_disk
import distributed as dist
from models_tpz_v6_2 import CDiT_models

# =================== Configs ===================
with open("config/data_config.yaml", "r") as f:
    data_config = yaml.safe_load(f)

with open("config/data_hyperparams_plan.yaml", "r") as f:
    data_hyperparams = yaml.safe_load(f)

ACTION_STATS_TORCH = {k: torch.tensor(v) for k, v in data_config["action_stats"].items()}

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

# ======= 绘图函数 =======
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
        ax.text(
            x + img_width / 2,
            y + 15,
            text,
            color="white",
            ha="center",
            va="top",
            fontsize=50,
            backgroundcolor="black",
        )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_batch_final(init_imgs, pred_imgs, goal_imgs, idxs, losses, save_path="final_plan.png"):
    # images are (B, c, h, w)
    imgs_for_plotting = torch.cat([init_imgs, pred_imgs, goal_imgs])
    imgs_for_plotting = (imgs_for_plotting + 1) / 2
    ncol = init_imgs.shape[0]
    grid_img = vutils.make_grid(imgs_for_plotting, nrow=ncol, padding=2)
    np_grid = grid_img.to(torch.float32).permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(ncol * 10, 30))
    ax.imshow(np_grid)
    ax.axis("off")

    img_height, img_width = np_grid.shape[0] // 3, np_grid.shape[1] // ncol

    # Overlay the IDs and losses on each image pair in the grid
    for i in range(ncol):
        x = i * img_width
        y_pred = img_height
        ax.text(
            x + img_width / 2,
            y_pred + 15,
            f"ID: {int(idxs[i].item())} Loss: {losses[i]:.2f}",
            color="white",
            ha="center",
            va="top",
            fontsize=40,
            backgroundcolor="black",
        )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_batch_trajectories(init_imgs, pred_imgs_seq, goal_imgs, idxs, save_path="trajectory_grid.png"):
    """
    Visualize a batch of image trajectories:
    - init_imgs: (B, C, H, W)
    - pred_imgs_seq: (B, T, C, H, W)
    - goal_imgs: (B, C, H, W)
    - idxs: (B,) tensor of IDs
    """
    B, T, C, H, W = pred_imgs_seq.shape

    # Denormalize from [-1, 1] to [0, 1]
    init_imgs = (init_imgs + 1) / 2
    goal_imgs = (goal_imgs + 1) / 2
    pred_imgs_seq = (pred_imgs_seq + 1) / 2

    imgs_for_plotting = []
    for b in range(B):
        traj = [init_imgs[b]]
        traj += [pred_imgs_seq[b, t] for t in range(T)]
        traj += [goal_imgs[b]]
        imgs_for_plotting.append(torch.stack(traj))

    imgs_for_plotting = torch.cat(imgs_for_plotting, dim=0)  # (B*(T+2), C, H, W)

    grid_img = vutils.make_grid(imgs_for_plotting, nrow=T + 2, padding=2)
    np_grid = grid_img.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(max(12, (T + 2) * 4), B * 4))
    ax.imshow(np_grid)
    ax.axis("off")

    img_height, img_width = H + 2, W + 2
    for b in range(B):
        for t in range(T + 2):
            x = t * img_width
            y = b * img_height
            if t == 0:
                label = f"ID:{int(idxs[b].item())} Init"
            elif t == T + 1:
                label = "Goal"
            else:
                label = f"Step {t}"
            ax.text(
                x + img_width / 2,
                y + 20,
                label,
                color="white",
                ha="center",
                va="top",
                fontsize=16,
                backgroundcolor="black",
            )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# =================== Dataset loader ===================
def get_dataset_eval(config, dataset_name, predefined_index=True):
    data_cfg = config["eval_datasets"][dataset_name]
    if predefined_index:
        predefined_index = f"data_splits/{dataset_name}/test/navigation_eval.pkl"
    else:
        predefined_index = None

    dataset = TrajectoryEvalDataset(
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
        traj_names="rollout_traj_names.txt",
    )
    return dataset

# =================== Evaluator ===================
class WM_Planning_Evaluator:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.exp = args.exp
        _, _, device, _ = dist.init_distributed()
        self.device = torch.device(device)
        self.device_index = torch.cuda.current_device() if torch.cuda.is_available() else None

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()

        # Config
        self.exp_eval = self.exp
        self.get_eval_name()

        with open("config/eval_config.yaml", "r") as f:
            default_config = yaml.safe_load(f)
        self.config = default_config

        with open(self.exp_eval, "r") as f:
            user_config = yaml.safe_load(f)
        self.config.update(user_config)

        latent_size = self.config["image_size"] // 8
        self.latent_size = self.config["image_size"] // 8
        self.num_cond = self.config["eval_context_size"]

        # logging directory
        if self.args.save_preds:
            exp_name = os.path.basename(self.args.exp).split(".")[0]
            self.args.save_output_dir = os.path.join(args.output_dir, exp_name)
            os.makedirs(self.args.save_output_dir, exist_ok=True)

        # Datasets
        self.dataset_names = self.args.datasets.split(",")
        self.datasets = {}
        for dataset_name in self.dataset_names:
            dataset_val = get_dataset_eval(self.config, dataset_name, predefined_index=True)

            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: distributed eval with a dataset not divisible by num_tasks; duplicates will be added."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )

            curr_data_loader = torch.utils.data.DataLoader(
                dataset_val,
                sampler=sampler_val,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            self.datasets[dataset_name] = curr_data_loader

        # Model
        print("loading")
        model = CDiT_models[self.config["model"]](
            context_size=self.num_cond + 1,
            input_size=latent_size,
        )

        ckp = torch.load(
            f'{self.config["results_dir"]}/{self.config["run_name"]}/checkpoints/{args.ckp}.pth.tar',
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(ckp["ema"], strict=True)
        model.eval()
        model.to(self.device)
        self.model = torch.compile(model)
        self.diffusion = create_diffusion(str(250))
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)
        if torch.cuda.is_available():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.device_index], find_unused_parameters=False
            )
        else:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=False)
        self.model_without_ddp = self.model.module

        self.loss_fn = lpips.LPIPS(net="alex").to(self.device)
        self.num_samples = self.args.num_samples
        self.topk = self.args.topk
        self.num_repeat_eval = self.args.num_repeat_eval
        self.action_dim = 4  # (dx, dy, dz, dyaw)

    # ====== Editor 保存 ======
    def _save_editor_run(
        self,
        base_dir: str,
        sid: int,
        init_img: torch.Tensor,  # (C,H,W) in [-1,1]
        step_seq: torch.Tensor,  # (T,C,H,W) in [-1,1]
        goal_img: torch.Tensor,  # (C,H,W) in [-1,1]
        deltas: torch.Tensor = None,  # (1,T,4) or (T,4)
        tag: str = "final",  # "final" / "candidate"
        cand_rank: int = None,
        cand_id: int = None,
        extra_meta: dict = None,
    ):
        editor_dir = _ensure_dir(os.path.join(base_dir, "editor"))
        run_dir = _ensure_dir(os.path.join(editor_dir, f"run_{sid:03d}"))
        if tag != "final":
            run_dir = _ensure_dir(os.path.join(run_dir, "candidates", f"cand_{cand_rank:03d}"))

        frames_dir = _ensure_dir(os.path.join(run_dir, "frames"))

        _save_png(init_img, os.path.join(frames_dir, "init.png"))
        T = step_seq.shape[0]
        for t in range(T):
            _save_png(step_seq[t], os.path.join(frames_dir, f"step_{t:03d}.png"))
        _save_png(goal_img, os.path.join(frames_dir, "goal.png"))

        init_loss = float(self.loss_fn(init_img[None].to(self.device), goal_img[None].to(self.device)).item())
        step_losses = []
        for t in range(T):
            step_losses.append(
                float(self.loss_fn(step_seq[t : t + 1].to(self.device), goal_img[None].to(self.device)).item())
            )

        deltas_eff = None
        pose_xyz = None
        pose_yaw = None
        if deltas is not None:
            d = deltas.detach().cpu().numpy() if isinstance(deltas, torch.Tensor) else np.asarray(deltas)
            if d.ndim == 3 and d.shape[0] == 1:
                d = d[0]
            assert d.ndim == 2 and d.shape[1] == 4
            T_raw = d.shape[0]
            if T_raw == T:
                deltas_eff = d.astype(np.float32, copy=False)
            else:
                idxs = np.arange(T_raw)
                chunks = np.array_split(idxs, T)
                summed = [d[ch].sum(0).astype(np.float32) if len(ch) > 0 else np.zeros((4,), dtype=np.float32) for ch in chunks]
                deltas_eff = np.stack(summed, axis=0)
            pose_xyz = deltas_eff[:, :3].cumsum(axis=0)
            pose_yaw = deltas_eff[:, 3].cumsum(axis=0)
            if tag != "final":
                np.save(os.path.join(run_dir, "deltas.npy"), deltas_eff)

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
            "frames": [],
        }
        if extra_meta:
            meta.update(extra_meta)

        meta["frames"].append(
            {
                "frame": 0,
                "png": "frames/init.png",
                "action": "INIT",
                "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "theta_rad": 0.0},
                "delta": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dtheta": 0.0},
                "loss": init_loss,
            }
        )
        for t in range(T):
            if deltas_eff is None:
                dx = dy = dz = dtheta = 0.0
                px = py = pz = pyaw = 0.0
            else:
                dx, dy, dz, dtheta = [float(v) for v in deltas_eff[t]]
                px, py, pz = [float(v) for v in pose_xyz[t]]
                pyaw = float(pose_yaw[t])

            meta["frames"].append(
                {
                    "frame": t + 1,
                    "png": f"frames/step_{t:03d}.png",
                    "action": int(t),
                    "pose": {"x": px, "y": py, "z": pz, "theta_rad": pyaw},
                    "delta": {"dx": dx, "dy": dy, "dz": dz, "dtheta": dtheta},
                    "loss": step_losses[t],
                }
            )

        _write_json(os.path.join(run_dir, "metadata.json"), meta)
        return run_dir, meta

    def generate_actions(
        self,
        dataset_save_output_dir,
        dataset_name,
        idxs,
        obs_image,
        goal_image,
        gt_actions,
        len_traj_pred,
        aug_image,
    ):
        idx_string = "_".join(map(str, idxs.flatten().int().tolist()))
        image_plot_dir = os.path.join(dataset_save_output_dir, "plots")
        _ensure_dir(image_plot_dir)
        videos_plot_dir = os.path.join(dataset_save_output_dir, "videos")
        _ensure_dir(videos_plot_dir)

        n_evals = obs_image.shape[0]
        candidate_number = self.num_samples
        all_deltas, all_losses, all_preds = [], [], []

        for traj in range(n_evals):
            traj_id = int(idxs.flatten()[traj].item())

            # === 构造 GT 轨迹（含起点） ===
            gt_deltas_xyz = gt_actions[traj, :, :3].to("cpu").numpy()  # [T, 3]
            gt_xyz = np.concatenate(
                [
                    np.zeros((1, 3), dtype=np.float32),
                    np.cumsum(gt_deltas_xyz, axis=0).astype(np.float32),
                ],
                axis=0,
            )  # [T+1, 3]
            T = gt_xyz.shape[0]
            gt_yaw = np.zeros((T,), dtype=np.float32)
            GT_traj = [
                (float(x), float(y), float(z), float(yaw)) for (x, y, z), yaw in zip(gt_xyz, gt_yaw)
            ]

            # 生成候选 trajectory poses
            from noise_trajectory_generation import trajectory_generation_random as trajectory_generation

            candidate_trajectories = trajectory_generation(GT_traj, candidate_number=candidate_number)
            candidate_trajectories = np.array(candidate_trajectories, dtype=np.float32)  # [N, T, 4]

            # 转换为 delta（相邻差分）
            candidate_deltas = candidate_trajectories[:, 1:, :] - candidate_trajectories[:, :-1, :]  # [N, steps, 4]
            deltas = torch.tensor(candidate_deltas, dtype=torch.float32, device=self.device)

            cur_obs_image = obs_image[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1)
            cur_goal_image = (
                goal_image[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1).squeeze(1)
            )
            cur_aug_image = aug_image[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1)

            # variance reduction by repeat evaluation
            if self.num_repeat_eval * self.num_samples > 120:
                cur_losses = []
                for _ in range(self.num_repeat_eval):
                    preds = self.autoregressive_rollout(
                        cur_obs_image, deltas, self.args.rollout_stride, aug_image=cur_aug_image
                    )
                    preds_last = preds[:, -1]
                    loss = self.loss_fn(preds_last.to(self.device), cur_goal_image.to(self.device)).flatten(0)
                    cur_losses.append(loss)
                loss = torch.stack(cur_losses).mean(dim=0)
            else:
                expanded_deltas = deltas.repeat(self.num_repeat_eval, 1, 1)
                expanded_obs_image = cur_obs_image.repeat(self.num_repeat_eval, 1, 1, 1, 1)
                expanded_goal_image = cur_goal_image.repeat(self.num_repeat_eval, 1, 1, 1)
                expanded_aug = cur_aug_image.repeat(self.num_repeat_eval, 1, 1, 1, 1)

                preds = self.autoregressive_rollout(
                    expanded_obs_image, expanded_deltas, self.args.rollout_stride, aug_image=expanded_aug
                )
                preds_last = preds[:, -1]

                loss = self.loss_fn(preds_last.to(self.device), expanded_goal_image.to(self.device)).flatten(0)
                loss = loss.view(self.num_repeat_eval, -1)
                loss = loss.mean(dim=0)

                preds_last = preds_last[: self.num_samples]

            sorted_idx = torch.argsort(loss)
            topk_idx = sorted_idx[: self.topk]
            best_idx = sorted_idx[0]
            best_delta = deltas[best_idx]  # [steps, 4]

            all_deltas.append(best_delta.unsqueeze(0))
            all_losses.append(loss[best_idx].item())
            all_preds.append(preds_last[best_idx].unsqueeze(0))

            # ====== 保存候选（Top-K）逐帧 PNG + metadata.json + deltas.npy ======
            if self.args.save_preds:
                editor_base_dir = dataset_save_output_dir
                sid = traj_id
                single_obs = obs_image[traj : traj + 1]
                single_aug = aug_image[traj : traj + 1]
                single_goal = goal_image[traj].squeeze(0)

                cand_summaries = []
                for rank, cid in enumerate(topk_idx.tolist()):
                    cand_delta = deltas[cid : cid + 1]  # (1, steps, 4)
                    cand_seq = self.autoregressive_rollout(
                        single_obs, cand_delta, self.args.rollout_stride, aug_image=single_aug
                    )[0]

                    cand_final_lpips = float(loss[cid].item())

                    _, meta = self._save_editor_run(
                        base_dir=editor_base_dir,
                        sid=sid,
                        init_img=single_obs[0, -1],
                        step_seq=cand_seq,
                        goal_img=single_goal,
                        deltas=cand_delta,
                        tag="candidate",
                        cand_rank=rank,
                        cand_id=int(cid),
                        extra_meta={"final_lpips": cand_final_lpips},
                    )
                    cand_summaries.append({"rank": rank, "cand_id": int(cid), "final_lpips": cand_final_lpips})

                run_cands_dir = os.path.join(editor_base_dir, "editor", f"run_{sid:03d}", "candidates")
                _ensure_dir(run_cands_dir)
                _write_json(
                    os.path.join(run_cands_dir, "summary.json"),
                    {"num_candidates_saved": len(cand_summaries), "topk": int(len(cand_summaries)), "items": cand_summaries},
                )

            if self.args.plot:
                self.visualize_trajectories(
                    dataset_name,
                    gt_actions,
                    image_plot_dir,
                    0,
                    traj,
                    traj_id,
                    deltas,
                    cur_obs_image,
                    cur_goal_image,
                    preds_last,
                    loss,
                    topk_idx,
                )

        # Final rollout for selected best deltas per sample
        final_deltas = torch.cat(all_deltas, dim=0)  # [n_evals, steps, 4]
        preds = self.autoregressive_rollout(obs_image, final_deltas, self.args.rollout_stride, aug_image=aug_image)
        preds_completed = preds
        preds_last = preds[:, -1]

        loss = self.loss_fn(preds_last.to(self.device), goal_image.squeeze(1).to(self.device)).flatten(0)

        if self.args.save_preds:
            save_planning_pred(
                dataset_save_output_dir,
                n_evals,
                idxs,
                obs_image,
                goal_image,
                preds_last,
                final_deltas,
                loss,
                gt_actions,
                preds_completed,
            )

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
                    deltas=final_deltas[b : b + 1],
                    tag="final",
                    cand_rank=None,
                    cand_id=None,
                    extra_meta={"final_lpips": float(loss[b].item())},
                )

        if self.args.plot:
            img_name = os.path.join(image_plot_dir, f"FINAL_{idx_string}.png")
            traj_name = os.path.join(image_plot_dir, f"TRAJ_{idx_string}.png")
            plot_batch_final(
                obs_image[:, -1].to(self.device),
                preds_last,
                goal_image.squeeze(1).to(self.device),
                idxs,
                loss.tolist(),
                save_path=img_name,
            )
            plot_batch_trajectories(
                obs_image[:, -1].to(self.device), preds_completed, goal_image.squeeze(1).to(self.device), idxs, save_path=traj_name
            )

        pred_actions = get_action_torch(final_deltas[:, :, :3], ACTION_STATS_TORCH)
        pred_yaw = final_deltas[:, :, -1].sum(1)
        return pred_actions, pred_yaw

    def visualize_trajectories(
        self,
        dataset_name,
        gt_actions,
        image_plot_dir,
        i,
        traj,
        traj_id,
        deltas,
        cur_obs_image,
        cur_goal_image,
        preds,
        loss,
        topk_idx,
    ):
        img_for_plotting = torch.cat([cur_goal_image[0:1].to(self.device), preds])
        loss_for_plotting = torch.cat((torch.tensor([0], device=self.device), loss))
        img_name = os.path.join(image_plot_dir, f"idx{traj_id}_iter{i}.png")
        plot_images_with_losses(img_for_plotting, loss_for_plotting, save_path=img_name)
        plot_name = os.path.join(image_plot_dir, f"idx{traj_id}_iter{i}_trajs.png")
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
            output_dir=plot_name,
        )

    def autoregressive_rollout(self, obs_image, deltas, rollout_stride, aug_image):
        deltas = deltas.unflatten(1, (-1, rollout_stride)).sum(2)
        preds = []
        curr_obs = obs_image.clone().to(self.device)

        for i in range(deltas.shape[1]):
            curr_delta = deltas[:, i : i + 1]
            all_models = self.model, self.diffusion, self.vae
            x_pred_pixels = model_forward_wrapper(
                all_models,
                curr_obs,
                curr_delta,
                self.args.rollout_stride,
                self.latent_size,
                num_cond=self.num_cond,
                device=self.device,
                x_supervised=aug_image,
            )
            x_pred_pixels = x_pred_pixels.unsqueeze(1)

            curr_obs = torch.cat((curr_obs, x_pred_pixels), dim=1)
            curr_obs = curr_obs[:, 1:]
            preds.append(x_pred_pixels)

        preds = torch.cat(preds, 1)
        return preds

    def get_eval_name(self):
        self.eval_name = f"CAND_N{self.args.num_samples}_K{self.args.topk}_RS{self.args.rollout_stride}_rep{self.args.num_repeat_eval}"

    def actions_to_traj(self, actions):
        positions_xyz = torch.zeros((actions.shape[0], 3))
        positions_xyz[:, :3] = actions
        orientations_quat_wxyz = torch.zeros((actions.shape[0], 4))
        orientations_quat_wxyz[:, -1] = 1
        timestamps = torch.arange(actions.shape[0], dtype=torch.float64)
        traj = PoseTrajectory3D(
            positions_xyz=positions_xyz,
            orientations_quat_wxyz=orientations_quat_wxyz,
            timestamps=timestamps,
        )
        return traj

    @torch.no_grad()
    def evaluate(self):
        for dataset_name in self.dataset_names:
            metric_logger = dist.MetricLogger(delimiter="  ")
            header = "Test:"

            if self.args.save_preds:
                dataset_save_output_dir = os.path.join(self.args.save_output_dir, dataset_name)
                os.makedirs(dataset_save_output_dir, exist_ok=True)
                eval_save_output_dir = os.path.join(dataset_save_output_dir, self.eval_name)
                os.makedirs(eval_save_output_dir, exist_ok=True)
            else:
                eval_save_output_dir = None

            curr_data_loader = self.datasets[dataset_name]
            for (
                idxs,
                obs_image,
                goal_image,
                gt_actions,
                goal_pos,
                aug_image,
                camera_ctx,
                camera_goal,
            ) in metric_logger.log_every(curr_data_loader, 1, header):
                obs_image = obs_image[:, -self.num_cond :]
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    pred_actions, pred_yaw = self.generate_actions(
                        eval_save_output_dir,
                        dataset_name,
                        idxs,
                        obs_image,
                        goal_image,
                        gt_actions,
                        self.config["trajectory_eval_len_traj_pred"],
                        aug_image=aug_image,
                    )
                for i in range(len(obs_image)):
                    pred_traj_i = self.actions_to_traj(pred_actions[i, :, :3])
                    gt_traj_i = self.actions_to_traj(gt_actions[i, :, :3])

                    ate, rpe_trans, _ = self.eval_metrics(gt_traj_i, pred_traj_i)

                    pred_final_pos = pred_actions[i, -1, :3].to("cpu")
                    pred_final_yaw = pred_yaw[i].to("cpu")
                    goal_final_pos = goal_pos[i, 0, :3]
                    goal_final_yaw = goal_pos[i, 0, -1]
                    pos_diff_norm = torch.norm(pred_final_pos - goal_final_pos)
                    yaw_diff = pred_final_yaw - goal_final_yaw
                    yaw_diff_norm = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).abs()

                    metric_logger.meters[f"{dataset_name}_ate"].update(ate, n=1)
                    metric_logger.meters[f"{dataset_name}_rpe_trans"].update(rpe_trans, n=1)
                    metric_logger.meters[f"{dataset_name}_pos_diff_norm"].update(pos_diff_norm, n=1)
                    metric_logger.meters[f"{dataset_name}_yaw_diff_norm"].update(yaw_diff_norm, n=1)

            output_fn = os.path.join(self.args.save_output_dir, f"{dataset_name}_{self.eval_name}.json")
            save_metric_to_disk(metric_logger, output_fn)

        metric_logger.synchronize_between_processes()

    def eval_metrics(self, traj_ref, traj_pred):
        traj_ref, traj_pred = sync.associate_trajectories(traj_ref, traj_pred)

        result = main_ape.ape(
            traj_ref,
            traj_pred,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=False,
            correct_scale=False,
        )
        ate = result.stats["rmse"]

        result = main_rpe.rpe(
            traj_ref,
            traj_pred,
            est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            align=False,
            correct_scale=False,
            delta=1.0,
            delta_unit=metrics.Unit.frames,
            rel_delta_tol=0.1,
        )
        rpe_rot = result.stats["rmse"]

        result = main_rpe.rpe(
            traj_ref,
            traj_pred,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=False,
            correct_scale=False,
            delta=1.0,
            delta_unit=metrics.Unit.frames,
            rel_delta_tol=0.1,
        )
        rpe_trans = result.stats["rmse"]

        return ate, rpe_trans, rpe_rot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Default Args
    parser.add_argument("--exp", type=str, default=None, help="experiment name (path to eval_config yaml)")
    parser.add_argument("--ckp", type=str, default="0100000", help="checkpoint step, e.g., 0100000")

    parser.add_argument("--datasets", type=str, default=None, help="dataset names, comma separated")
    parser.add_argument("--output_dir", type=str, default=None, help="output dir to save model predictions")
    parser.add_argument("--save_preds", action="store_true", default=False, help="whether to save prediction tensors")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")

    # Planning Specific Args
    parser.add_argument("--num_samples", type=int, default=10, help="num candidate trajectories")
    parser.add_argument("--rollout_stride", type=int, default=1, help="rollout stride")
    parser.add_argument("--topk", type=int, default=5, help="top k candidates to save")
    parser.add_argument("--num_repeat_eval", type=int, default=1, help="number of eval repeats per candidate")
    parser.add_argument("--plot", action="store_true", default=False)
    
    parser.add_argument("--opt_steps", type=int, default=0,
                help="(ignored) kept for CLI compatibility; no CEM in this run")

    args = parser.parse_args()

    evaluator = WM_Planning_Evaluator(args)
    _ = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        _ = torch.cuda.current_device()
    evaluator.evaluate()
