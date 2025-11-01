# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------
# Inherited from dataset v5 and v6, for visualization
import cv2
import numpy as np
import torch
import os, json
from PIL import Image
from typing import Tuple
import yaml
import pickle
import tqdm
from torch.utils.data import Dataset
from misc import angle_difference, get_data_path, get_delta_np, normalize_data, to_local_coords
from project_functions import (
    reproject_depth_to_other_pose_2seq,
    project_to_2d_image_2seq,
    reproject_depth_to_other_pose_seq2seq,
    project_to_2d_image_seq2seq,
    resize_image_half,
)

# ======== depth saving helper ========
def save_depth_pair(depth_map, out_png16, out_vis, vmax=None):
    if depth_map is None:
        return
    dm = np.asarray(depth_map)

    # 16-bit 原始
    if dm.dtype in (np.float32, np.float64):
        mm = dm.copy()
        mm[~np.isfinite(mm)] = 0.0
        dmax = float(np.max(mm)) if np.isfinite(mm).any() else 0.0
        mm = np.clip(mm, 0.0, dmax) * 1000.0
        png16 = np.clip(np.round(mm), 0, 65535).astype(np.uint16)
    elif dm.dtype == np.uint16:
        png16 = dm
    else:
        dm_f = dm.astype(np.float32)
        dm_f[~np.isfinite(dm_f)] = 0.0
        dmax = float(np.max(dm_f)) if np.isfinite(dm_f).any() else 1.0
        scale = 65535.0 / max(dmax, 1e-6)
        png16 = np.clip(np.round(dm_f * scale), 0, 65535).astype(np.uint16)
    Image.fromarray(png16).save(out_png16)

    # 伪彩
    vis = dm.astype(np.float32)
    vis[~np.isfinite(vis)] = 0.0
    if vmax is None:
        flat = vis[np.isfinite(vis)]
        vmax = np.percentile(flat, 95) if flat.size > 0 else (float(vis.max()) if np.isfinite(vis).any() else 1.0)
        vmax = max(vmax, 1e-6)
    vis_u8 = np.clip((vis / vmax) * 255.0, 0, 255).astype(np.uint8)
    vis_color_bgr = cv2.applyColorMap(vis_u8, cv2.COLORMAP_JET)
    vis_color_rgb = vis_color_bgr[..., ::-1]
    Image.fromarray(vis_color_rgb).save(out_vis)
# ===================================================

def _to_uint8(img_np):
    """把任意 [0,1] 或 [0,255] 或 float/uint8 的 RGB 图统一转成 uint8 RGB。"""
    arr = np.asarray(img_np)
    if arr.dtype != np.uint8:
        mx = float(arr.max()) if np.isfinite(arr).any() else 0.0
        if mx <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def _mse_rgb(a_u8, b_u8):
    """输入 uint8 RGB，返回均方误差（0~1 范围）。"""
    a = a_u8.astype(np.float32) / 255.0
    b = b_u8.astype(np.float32) / 255.0
    return float(np.mean((a - b) ** 2))

class BaseDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        self.goals_per_obs = goals_per_obs

        traj_names_file = os.path.join(data_split_folder, traj_names)
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.distance_categories = list(range(min_dist_cat, max_dist_cat + 1))
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.len_traj_pred = len_traj_pred
        self.traj_stride = traj_stride

        self.context_size = context_size
        self.normalize = normalize

        # load data/data_config.yaml
        with open("config/data_config.yaml", "r") as f:
            all_data_config = yaml.safe_load(f)

        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        self.data_config = all_data_config[self.dataset_name]
        self.transform = transform

        # 最小改动：联合重投影使用的历史帧数（默认等于 context_size，不影响原行为）
        self.num_cond_pro = context_size

        self._load_index(predefined_index)
        self.ACTION_STATS = {}
        for key in all_data_config['action_stats']:
            self.ACTION_STATS[key] = np.expand_dims(all_data_config['action_stats'][key], axis=0)

    def _load_index(self, predefined_index) -> None:
        if predefined_index:
            print(f"****** Using a predefined evaluation index... {predefined_index}******")
            with open(predefined_index, "rb") as f:
                self.index_to_data = pickle.load(f)
                return
        else:
            print("****** Evaluating from NON PREDEFINED index... ******")
            index_to_data_path = os.path.join(
                self.data_split_folder,
                f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_n{self.context_size}_len_traj_pred_{self.len_traj_pred}.pkl",
            )
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _build_index(self, use_tqdm: bool = False):
        samples_index = []
        goals_index = []
        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])
            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size - 1
            end_time = traj_len - self.len_traj_pred
            for curr_time in range(begin_time, end_time, self.traj_stride):
                max_goal_distance = min(self.max_dist_cat, traj_len - curr_time - 1)
                min_goal_distance = max(self.min_dist_cat, -curr_time)
                samples_index.append((traj_name, curr_time, min_goal_distance, max_goal_distance))
        return samples_index, goals_index
  
    def _get_trajectory(self, trajectory_name):
        with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
            traj_data = pickle.load(f)
        for k, v in traj_data.items():
            traj_data[k] = v.astype('float')
        return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def _compute_projected_image_o(self, traj_data, curr_time, goal_time, rgb_img):
        pose_src = traj_data["pose"][curr_time]
        pose_dst = traj_data["pose"][goal_time]
        depth_map = traj_data["depth"][curr_time]
        K = traj_data["K"]
        projected_images = self.generate_augmented_image_o(
            K=K, depth_map=depth_map, rgb_img=rgb_img, pose_src=pose_src, pose_dst=pose_dst
        )
        return projected_images
    
    def generate_augmented_image_o(self, K, depth_map, rgb_img, pose_src, pose_dst) -> np.ndarray:
        image_size = depth_map.shape  # (H, W)
        if rgb_img.shape[:2] != image_size:
            rgb_img = resize_image_half(rgb_img)
        points_3d, colors = reproject_depth_to_other_pose_2seq(K, depth_map, rgb_img, pose_src, pose_dst) 
        images = project_to_2d_image_2seq(K, points_3d, colors, image_size) # (H, W, 3, goal_time)
        return images

    # ============ seq2seq ============
    def _compute_projected_images(self, traj_data, context_times, rgb_seq, goal_times_np):
        K = traj_data["K"]
        depth_seq = traj_data["depth"][context_times]     # (T, H, W)
        poses_src_seq = traj_data["pose"][context_times]  # (T, 4, 4)
        H, W = depth_seq.shape[-2:]
        poses_dst_seq = traj_data["pose"][goal_times_np]  # (B, 4, 4)

        points_3d_all, colors_all = reproject_depth_to_other_pose_seq2seq(
            K=K,
            depth_maps=depth_seq,
            rgb_imgs=rgb_seq,          # (T,H,W,3)
            poses_src=poses_src_seq,
            poses_dst=poses_dst_seq
        )
        images = project_to_2d_image_seq2seq(
            K=K,
            points_3d=points_3d_all,
            colors=colors_all,
            image_size=(H, W)
        )  # (B, H, W, 3)
        return images
    # ============================================================

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1
        yaw = traj_data["yaw"][start_index:end_index]
        positions = traj_data["point"][start_index:end_index]
        goal_pos = traj_data["point"][goal_time]
        goal_yaw = traj_data["yaw"][goal_time]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            raise ValueError("is used?")

        waypoints_pos = to_local_coords(positions, positions[0], yaw[0])
        waypoints_yaw = angle_difference(yaw[0], yaw)
        actions = np.concatenate([waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)
        actions = actions[1:]
        
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])
        goal_yaw = angle_difference(yaw[0], goal_yaw)
        
        if self.normalize:
            actions[:, :3] /= self.data_config["metric_waypoint_spacing"]
            goal_pos[:, :3] /= self.data_config["metric_waypoint_spacing"]
        
        goal_pos = np.concatenate([goal_pos, goal_yaw.reshape(-1, 1)], axis=-1)

        return actions, goal_pos

class TrainingDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str = 'traj_names.txt',
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
            len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1, size=(self.goals_per_obs))
            goal_time = (curr_time + goal_offset).astype('int')  # (B,)
            rel_time = (goal_offset).astype('float')/(128.)

            # 历史帧时间序列
            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            true_context = [(f_curr, t) for t in context_times]
            goal_context = [(f_curr, t) for t in goal_time]
            context = true_context + goal_context

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])

            # 历史帧 RGB（用于单帧可视化）
            rgb_imgs = [cv2.imread(get_data_path(self.data_folder, f_img, t_img)) for f_img, t_img in true_context]
            rgb_imgs = [cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) for rgb_img in rgb_imgs]
            rgb_imgs_np_full = np.stack(rgb_imgs, axis=0)

            # 轨迹数据信息
            curr_traj_data = self._get_trajectory(f_curr)

            # 计算动作/目标
            _, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
            goal_pos[:, :3] = normalize_data(goal_pos[:, :3], self.ACTION_STATS)

            # ========== 单历史帧 -> 多目标（逐历史帧可视化） ==========
            his_projected_list = []
            for t_idx, rgb_img in enumerate(rgb_imgs):
                src_time = context_times[t_idx]
                projected_images_o = self._compute_projected_image_o(curr_traj_data, src_time, goal_time, rgb_img)
                his_projected_list.append(projected_images_o)

            # ========== 训练返回的 projected_tensor（保持原接口，用 num_cond_pro=context_size） ==========
            cond_times_default = context_times[-self.num_cond_pro:]
            cond_context_default = [(f_curr, t) for t in cond_times_default]
            cond_rgbs_default = [cv2.imread(get_data_path(self.data_folder, f_img, t_img)) for f_img, t_img in cond_context_default]
            cond_rgbs_default = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in cond_rgbs_default]
            cond_rgbs_np_default = np.stack(cond_rgbs_default, axis=0)
            projected_images_default = self._compute_projected_images(
                curr_traj_data, cond_times_default, cond_rgbs_np_default, goal_time
            )  # (B,H,W,3)
            projected_tensor_list = [self.transform(Image.fromarray(_to_uint8(img))) for img in projected_images_default]
            projected_tensor = torch.stack(projected_tensor_list, dim=0)

            # ========== 多历史帧 -> 多目标：k ∈ {1,2,4,8,16} 只进 grid，不单独存 ==========
            from PIL import ImageDraw, ImageFont
            k_list = [1, 2, 4, 8, 16]
            vis_root = './visualizations-seq2seq'
            sample_dir = os.path.join(vis_root, f'{self.dataset_name}', f'sample_{i}')
            os.makedirs(sample_dir, exist_ok=True)

            # GT（与目标 times 对齐）
            gt_imgs = []
            for j, (_, t_goal) in enumerate(goal_context):
                gt = Image.open(get_data_path(self.data_folder, f_curr, int(t_goal))).convert("RGB")
                gt_np = _to_uint8(np.array(gt))
                gt_imgs.append(gt_np)
                Image.fromarray(gt_np).save(os.path.join(sample_dir, f'gt_{j:03d}_t{int(t_goal)}.png'))
            B = len(gt_imgs)

            # 为 grid 准备：GT 行
            rows = []
            label_size = (64, 48)
            font = ImageFont.load_default()
            label_gt = Image.new("RGB", label_size, (255, 255, 255))
            ImageDraw.Draw(label_gt).text((4, 4), "GT", fill=(0, 0, 0), font=font)
            rows.append([np.array(label_gt)] + gt_imgs)

            mse_report = {}          # {k: {"per_goal": [...], "mean": x}}
            joint_rows_uint8 = []    # 多个 k 的行
            save_per_k = False       # 不单独保存每个 k 的结果

            for k in k_list:
                k_eff = int(min(k, len(context_times)))
                cond_times_k = context_times[-k_eff:]
                cond_context_k = [(f_curr, t) for t in cond_times_k]
                cond_rgbs_k = [cv2.imread(get_data_path(self.data_folder, f_img, t_img)) for f_img, t_img in cond_context_k]
                cond_rgbs_k = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in cond_rgbs_k]
                cond_rgbs_k_np = np.stack(cond_rgbs_k, axis=0)

                proj_k = self._compute_projected_images(curr_traj_data, cond_times_k, cond_rgbs_k_np, goal_time)  # (B,H,W,3)
                joint_uint8_k = []
                per_goal_mse = []
                for j in range(B):
                    pj = _to_uint8(proj_k[j])
                    joint_uint8_k.append(pj)
                    mse_j = _mse_rgb(pj, gt_imgs[j])
                    per_goal_mse.append(mse_j)
                    if save_per_k:
                        Image.fromarray(pj).save(os.path.join(sample_dir, f'proj_joint_k{k_eff}_{j:03d}.png'))

                mse_mean = float(np.mean(per_goal_mse)) if per_goal_mse else float('nan')
                mse_report[str(k_eff)] = {"per_goal": per_goal_mse, "mean": mse_mean}

                # 行左侧标签写上 k 与 MSE
                label_joint = Image.new("RGB", label_size, (255, 255, 255))
                ImageDraw.Draw(label_joint).text((4, 4), f"Joint@k={k_eff}\nMSE={mse_mean:.4f}", fill=(0, 0, 0), font=font)
                joint_rows_uint8.append([np.array(label_joint)] + joint_uint8_k)

            # ========== 逐历史帧单独投影（原逻辑保留） ==========
            T = rgb_imgs_np_full.shape[0]
            per_hist_rows = []
            for t_idx, proj_o in enumerate(his_projected_list):
                axes_equal_B = np.where(np.array(proj_o.shape) == B)[0]
                axis = int(axes_equal_B[0])
                arr = np.moveaxis(proj_o, axis, 0)  # (B,H,W,3)
                row = []
                for j in range(B):
                    im = _to_uint8(arr[j])
                    row.append(im)
                    Image.fromarray(im).save(os.path.join(sample_dir, f'proj_single_t{t_idx:03d}_{j:03d}.png'))
                per_hist_rows.append(row)

            # 历史原图单独保存（便于比对）
            for t_idx in range(T):
                im = _to_uint8(rgb_imgs_np_full[t_idx])
                Image.fromarray(im).save(os.path.join(sample_dir, f'hist_{t_idx:03d}.png'))

            # ========== 组装 grid ==========
            hist_named = []
            for t_idx in range(T):
                pil = Image.fromarray(_to_uint8(rgb_imgs_np_full[t_idx]))
                ImageDraw.Draw(pil).text((6, 6), f"Hist t={context_times[t_idx]}", fill=(255, 255, 255), font=font)
                hist_named.append(np.array(pil))

            # rows: [GT] + [多行 Joint@k] + [每个历史帧的一行]
            rows += joint_rows_uint8
            for t_idx in range(T):
                rows.append([hist_named[t_idx]] + per_hist_rows[t_idx])

            # 统一到相同尺寸（以 GT 第一张为基准）
            H, W = rows[0][1].shape[:2]
            max_cols = B + 1
            pad = 4

            norm_rows = []
            for r in rows:
                resized = []
                for x in r:
                    img = Image.fromarray(x)
                    img = img.resize((W, H), Image.BILINEAR)
                    resized.append(np.array(img))
                while len(resized) < max_cols:
                    resized.append(np.full((H, W, 3), 255, np.uint8))
                norm_rows.append(resized)

            row_width = max_cols * W + (max_cols - 1) * pad
            spacer_w = np.full((H, pad, 3), 255, np.uint8)
            spacer_h = np.full((pad, row_width, 3), 255, np.uint8)

            row_arrays = []
            for r in norm_rows:
                row_img = r[0]
                for x in r[1:]:
                    row_img = np.concatenate((row_img, spacer_w, x), axis=1)
                row_arrays.append(row_img)

            grid = row_arrays[0]
            for rr in row_arrays[1:]:
                grid = np.concatenate((grid, spacer_h, rr), axis=0)

            Image.fromarray(grid).save(os.path.join(sample_dir, 'grid_all.png'))
            with open(os.path.join(sample_dir, 'mse_report.json'), 'w') as f:
                json.dump(mse_report, f, indent=2)

            # ===== 保存深度（历史 + 目标） =====
            if "depth" in curr_traj_data and curr_traj_data["depth"] is not None:
                depth_seq = curr_traj_data["depth"]
                # 历史帧
                for t_idx, t_ctx in enumerate(context_times):
                    if 0 <= t_ctx < len(depth_seq):
                        dm = depth_seq[t_ctx]
                        out16 = os.path.join(sample_dir, f"depth_hist_{t_idx:03d}_t{t_ctx}.png")
                        outvis = os.path.join(sample_dir, f"depth_hist_{t_idx:03d}_t{t_ctx}_vis.png")
                        save_depth_pair(dm, out16, outvis)
                # 目标帧（与 goal_context 对齐）
                for j, (_, t_goal) in enumerate(goal_context):
                    tg = int(t_goal)
                    if 0 <= tg < len(depth_seq):
                        dm = depth_seq[tg]
                        out16 = os.path.join(sample_dir, f"depth_gt_{j:03d}_t{tg}.png")
                        outvis = os.path.join(sample_dir, f"depth_gt_{j:03d}_t{tg}_vis.png")
                        save_depth_pair(dm, out16, outvis)
            # ==================================

            return (
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
                torch.as_tensor(rel_time, dtype=torch.float32),
                torch.as_tensor(projected_tensor, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)


class EvalDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
            len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs)
  
    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, _, _ = self.index_to_data[i]
            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            pred_times = list(range(curr_time + 1, curr_time + self.len_traj_pred + 1))  # 未来 B 帧
            context = [(f_curr, t) for t in context_times]
            pred = [(f_curr, t) for t in pred_times]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
            pred_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in pred])

            curr_traj_data = self._get_trajectory(f_curr)

            # 动作/delta
            actions, _ = self._compute_actions(curr_traj_data, curr_time, np.array(pred_times))
            actions[:, :3] = normalize_data(actions[:, :3], self.ACTION_STATS)
            delta = get_delta_np(actions)

            # 历史帧 RGB（用于单帧可视化）
            rgb_list_full = [cv2.imread(get_data_path(self.data_folder, f_img, t_img)) for f_img, t_img in context]
            rgb_list_full = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in rgb_list_full]
            rgb_np_full = np.stack(rgb_list_full, axis=0)

            # 单历史帧 -> 多目标（逐历史帧可视化）
            his_projected_list = []
            for t_idx, rgb_img in enumerate(rgb_list_full):
                src_time = context_times[t_idx]
                projected_images_o = self._compute_projected_image_o(curr_traj_data, src_time, np.array(pred_times), rgb_img)
                his_projected_list.append(projected_images_o)

            # 评估返回的 projected_tensor：仍用默认 num_cond_pro=context_size（保持接口与含义）
            cond_times_default = context_times[-self.num_cond_pro:]
            cond_context_default = [(f_curr, t) for t in cond_times_default]
            cond_rgbs_default = [cv2.imread(get_data_path(self.data_folder, f_img, t_img)) for f_img, t_img in cond_context_default]
            cond_rgbs_default = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in cond_rgbs_default]
            cond_rgbs_np_default = np.stack(cond_rgbs_default, axis=0)
            projected_images_default = self._compute_projected_images(
                curr_traj_data, cond_times_default, cond_rgbs_np_default, np.array(pred_times)
            )
            projected_tensor_list = [self.transform(Image.fromarray(_to_uint8(img))) for img in projected_images_default]
            projected_tensor = torch.stack(projected_tensor_list, dim=0)

            # ========== 多历史帧 -> 多目标：k ∈ {1,2,4,8,16} 只进 grid，不单独存 ==========
            from PIL import ImageDraw, ImageFont
            k_list = [1, 2, 4, 8, 16]
            vis_root = './visualizations-seq2seq'
            sample_dir = os.path.join(vis_root, f'{self.dataset_name}', f'sample_{i}')
            os.makedirs(sample_dir, exist_ok=True)

            # GT（与 pred_times 对齐）
            gt_imgs = []
            for j, (_, t_pred) in enumerate(pred):
                gt = Image.open(get_data_path(self.data_folder, f_curr, int(t_pred))).convert("RGB")
                gt_np = _to_uint8(np.array(gt))
                gt_imgs.append(gt_np)
                Image.fromarray(gt_np).save(os.path.join(sample_dir, f'gt_{j:03d}_t{int(t_pred)}.png'))
            B = len(gt_imgs)

            # 为 grid 准备：GT 行
            rows = []
            label_size = (64, 48)
            font = ImageFont.load_default()
            label_gt = Image.new("RGB", label_size, (255, 255, 255))
            ImageDraw.Draw(label_gt).text((4, 4), "GT", fill=(0, 0, 0), font=font)
            rows.append([np.array(label_gt)] + gt_imgs)

            mse_report = {}
            joint_rows_uint8 = []
            save_per_k = False

            for k in k_list:
                k_eff = int(min(k, len(context_times)))
                cond_times_k = context_times[-k_eff:]
                cond_context_k = [(f_curr, t) for t in cond_times_k]
                cond_rgbs_k = [cv2.imread(get_data_path(self.data_folder, f_img, t_img)) for f_img, t_img in cond_context_k]
                cond_rgbs_k = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in cond_rgbs_k]
                cond_rgbs_k_np = np.stack(cond_rgbs_k, axis=0)

                proj_k = self._compute_projected_images(curr_traj_data, cond_times_k, cond_rgbs_k_np, np.array(pred_times))
                joint_uint8_k = []
                per_goal_mse = []
                for j in range(B):
                    pj = _to_uint8(proj_k[j])
                    joint_uint8_k.append(pj)
                    mse_j = _mse_rgb(pj, gt_imgs[j])
                    per_goal_mse.append(mse_j)
                    if save_per_k:
                        Image.fromarray(pj).save(os.path.join(sample_dir, f'proj_joint_k{k_eff}_{j:03d}.png'))
                mse_mean = float(np.mean(per_goal_mse)) if per_goal_mse else float('nan')
                mse_report[str(k_eff)] = {"per_goal": per_goal_mse, "mean": mse_mean}

                label_joint = Image.new("RGB", label_size, (255, 255, 255))
                ImageDraw.Draw(label_joint).text((4, 4), f"Joint@k={k_eff}\nMSE={mse_mean:.4f}", fill=(0, 0, 0), font=font)
                joint_rows_uint8.append([np.array(label_joint)] + joint_uint8_k)

            # ========== 逐历史帧单独投影（原逻辑保留） ==========
            T = rgb_np_full.shape[0]
            per_hist_rows = []
            for t_idx, proj_o in enumerate(his_projected_list):
                axes_equal_B = np.where(np.array(proj_o.shape) == B)[0]
                axis = int(axes_equal_B[0])
                arr = np.moveaxis(proj_o, axis, 0)  # (B,H,W,3)
                row = []
                for j in range(B):
                    im = _to_uint8(arr[j])
                    row.append(im)
                    Image.fromarray(im).save(os.path.join(sample_dir, f'proj_single_t{t_idx:03d}_{j:03d}.png'))
                per_hist_rows.append(row)

            # 历史原图单独保存
            for t_idx in range(T):
                im = _to_uint8(rgb_np_full[t_idx])
                Image.fromarray(im).save(os.path.join(sample_dir, f'hist_{t_idx:03d}.png'))

            # ========== 组装 grid ==========
            hist_named = []
            for t_idx in range(T):
                pil = Image.fromarray(_to_uint8(rgb_np_full[t_idx]))
                ImageDraw.Draw(pil).text((6, 6), f"Hist t={context_times[t_idx]}", fill=(255, 255, 255), font=font)
                hist_named.append(np.array(pil))

            rows += joint_rows_uint8
            for t_idx in range(T):
                rows.append([hist_named[t_idx]] + per_hist_rows[t_idx])

            H, W = rows[0][1].shape[:2]
            max_cols = B + 1
            pad = 4

            norm_rows = []
            for r in rows:
                resized = []
                for x in r:
                    img = Image.fromarray(x)
                    img = img.resize((W, H), Image.BILINEAR)
                    resized.append(np.array(img))
                while len(resized) < max_cols:
                    resized.append(np.full((H, W, 3), 255, np.uint8))
                norm_rows.append(resized)

            row_width = max_cols * W + (max_cols - 1) * pad
            spacer_w = np.full((H, pad, 3), 255, np.uint8)
            spacer_h = np.full((pad, row_width, 3), 255, np.uint8)

            row_arrays = []
            for r in norm_rows:
                row_img = r[0]
                for x in r[1:]:
                    row_img = np.concatenate((row_img, spacer_w, x), axis=1)
                row_arrays.append(row_img)

            grid = row_arrays[0]
            for rr in row_arrays[1:]:
                grid = np.concatenate((grid, spacer_h, rr), axis=0)

            Image.fromarray(grid).save(os.path.join(sample_dir, 'grid_all.png'))
            with open(os.path.join(sample_dir, 'mse_report.json'), 'w') as f:
                json.dump(mse_report, f, indent=2)

            # ===== 保存深度（历史 + 未来） =====
            if "depth" in curr_traj_data and curr_traj_data["depth"] is not None:
                depth_seq = curr_traj_data["depth"]
                # 历史帧
                for t_idx, t_ctx in enumerate(context_times):
                    if 0 <= t_ctx < len(depth_seq):
                        dm = depth_seq[t_ctx]
                        out16 = os.path.join(sample_dir, f"depth_hist_{t_idx:03d}_t{t_ctx}.png")
                        outvis = os.path.join(sample_dir, f"depth_hist_{t_idx:03d}_t{t_ctx}_vis.png")
                        save_depth_pair(dm, out16, outvis)
                # 未来帧（与 pred_times 对齐）
                for j, t_pred in enumerate(pred_times):
                    if 0 <= t_pred < len(depth_seq):
                        dm = depth_seq[t_pred]
                        out16 = os.path.join(sample_dir, f"depth_gt_{j:03d}_t{t_pred}.png")
                        outvis = os.path.join(sample_dir, f"depth_gt_{j:03d}_t{t_pred}_vis.png")
                        save_depth_pair(dm, out16, outvis)
            # ==================================

            return (
                torch.tensor([i], dtype=torch.float32),  # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(pred_image, dtype=torch.float32),
                torch.as_tensor(delta, dtype=torch.float32),
                torch.as_tensor(projected_tensor, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)
        
class TrajectoryEvalDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
            len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs)

    def _sample_goal(self, trajectory_name, curr_time, min_goal_dist, max_goal_dist):
        goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1)
        goal_time = curr_time + int(goal_offset)
        return trajectory_name, goal_time, False

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            f_goal, goal_time, _ = self._sample_goal(f_curr, curr_time, min_goal_dist, max_goal_dist)

            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))           
            context = [(f_curr, t) for t in context_times]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
            goal_image = self.transform(Image.open(get_data_path(self.data_folder, f_goal, goal_time))).unsqueeze(0)
            curr_traj_data = self._get_trajectory(f_curr)

            actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, np.array([goal_time]))

            # 联合重投影（保持默认 num_cond_pro=context_size 的返回）
            cond_times = context_times[-self.num_cond_pro:]
            cond_context = [(f_curr, t) for t in cond_times]
            cond_rgbs = [cv2.imread(get_data_path(self.data_folder, f_img, t_img)) for f_img, t_img in cond_context]
            cond_rgbs = [cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) for rgb_img in cond_rgbs]
            cond_rgbs = np.stack(cond_rgbs, axis=0) 
            projected_images = self._compute_projected_images(curr_traj_data, cond_times, cond_rgbs, np.array([goal_time]))
            projected_tensor_list = [self.transform(Image.fromarray(_to_uint8(img))) for img in projected_images]
            projected_tensor = torch.stack(projected_tensor_list, dim=0)

            return (
                torch.tensor([i], dtype=torch.float32),  # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_image, dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
                torch.as_tensor(projected_tensor, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)
