# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------
# Inherited from dataset v4, for visualization
import cv2
import numpy as np
import torch
import os
from PIL import Image
from typing import Tuple
import yaml
import pickle
import tqdm
from torch.utils.data import Dataset
from misc import angle_difference, get_data_path, get_delta_np, normalize_data, to_local_coords
from project_functions import reproject_depth_to_other_pose_2seq, project_to_2d_image_2seq, reproject_depth_to_other_pose_seq2seq, project_to_2d_image_seq2seq, resize_image_half

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
        # use this index to retrieve the dataset name from the data_config.yaml
        self.data_config = all_data_config[self.dataset_name]
        self.transform = transform
        self._load_index(predefined_index)
        self.ACTION_STATS = {}
        for key in all_data_config['action_stats']:
            self.ACTION_STATS[key] = np.expand_dims(all_data_config['action_stats'][key], axis=0)

    def _load_index(self, predefined_index) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
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
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
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
        for k,v in traj_data.items():
            traj_data[k] = v.astype('float')
        return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def _compute_projected_image_o(self, traj_data, curr_time, goal_time, rgb_img):
        pose_src = traj_data["pose"][curr_time]
        pose_dst = traj_data["pose"][goal_time]
        depth_map = traj_data["depth"][curr_time]
        K = traj_data["K"]
        
        projected_images = self.generate_augmented_image_o(K=K, depth_map=depth_map, rgb_img=rgb_img, pose_src=pose_src, pose_dst=pose_dst)
        return projected_images
    
    def generate_augmented_image_o(self, K, depth_map, rgb_img, pose_src, pose_dst) -> np.ndarray:
        """
        基于深度图 + pose 生成从另一个相机视角观察到的图像。
        """
        image_size = depth_map.shape  # (H, W)
        if rgb_img.shape[:2] != image_size:
            rgb_img = resize_image_half(rgb_img)
        points_3d, colors = reproject_depth_to_other_pose_2seq(K, depth_map, rgb_img, pose_src, pose_dst) 
        images = project_to_2d_image_2seq(K, points_3d, colors, image_size) # (H, W, 3, goal_time)
        return images
    # ============ seq2seq ============
    def _compute_projected_images(self, traj_data, context_times, rgb_seq, goal_times_np):
        """
        使用多帧历史 (context_times) 的 depth/rgb/pose;重投影到多个目标位姿 (goal_times_np)。
        返回: np.ndarray, 形状 (B, H, W, 3) ;B = len(goal_times_np)
        """
        K = traj_data["K"]                     # (3,3)
        depth_seq = traj_data["depth"][context_times]   # (T, H, W)
        poses_src_seq = traj_data["pose"][context_times]# (T, 4, 4)
        H, W = depth_seq.shape[-2:]
        poses_dst_seq = traj_data["pose"][goal_times_np]  # (B, 4, 4)

        # 先用 seq2seq 得到每个目标位姿的点云/颜色
        points_3d_all, colors_all = reproject_depth_to_other_pose_seq2seq(
            K=K,
            depth_maps=depth_seq,      # (T,H,W)
            rgb_imgs=rgb_seq,          # (T,H,W,3)
            poses_src=poses_src_seq,   # (T,4,4)
            poses_dst=poses_dst_seq    # (B,4,4)
        )
        # 再做 z-buffer 投成图像
        images = project_to_2d_image_seq2seq(
            K=K,
            points_3d=points_3d_all,   # List[(Ni,3)], 长度 B
            colors=colors_all,         # List[(Ni,3)], 长度 B
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
            rel_time = (goal_offset).astype('float')/(128.)      # TODO: tune this const

            # 历史帧时间序列
            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            true_context = [(f_curr, t) for t in context_times]
            goal_context = [(f_curr, t) for t in goal_time]
            context = true_context + goal_context

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
            # aug
            rgb_imgs = [cv2.imread(get_data_path(self.data_folder, f_img, t_img)) for f_img, t_img in true_context]
            rgb_imgs = [cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) for rgb_img in rgb_imgs]
            rgb_imgs_np = np.stack(rgb_imgs, axis=0) 

            # Load other trajectory data
            curr_traj_data = self._get_trajectory(f_curr)

            # 计算动作/目标
            _, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
            goal_pos[:, :3] = normalize_data(goal_pos[:, :3], self.ACTION_STATS)

            # 使用多历史帧 → 多目标帧重投影（不再传单帧 rgb）
            his_projected_list = []
            for t_idx, rgb_img in enumerate(rgb_imgs):
                src_time = context_times[t_idx] 
                projected_images_o = self._compute_projected_image_o(curr_traj_data, src_time, goal_time, rgb_img)
                his_projected_list.append(projected_images_o)
            projected_images = self._compute_projected_images(curr_traj_data, context_times, rgb_imgs_np, goal_time)  # (B,H,W,3)

            # 转成张量
            projected_tensor_list = [self.transform(Image.fromarray(img)) for img in projected_images]
            projected_tensor = torch.stack(projected_tensor_list, dim=0)

            # ===================== 保存图像（GT一行 / 联合投影一行 / 单独投影T行） =====================

            vis_root = './visualizations-seq2seq'
            sample_dir = os.path.join(vis_root, f'{self.dataset_name}', f'sample_{i}')
            os.makedirs(sample_dir, exist_ok=True)

            # 1) 历史原图（仅单独保存，便于比对）
            T = rgb_imgs_np.shape[0]
            for t_idx in range(T):
                im = rgb_imgs_np[t_idx]
                im = np.clip(im * (255.0 ** (im.max() <= 1)), 0, 255).astype(np.uint8)
                Image.fromarray(im).save(os.path.join(sample_dir, f'hist_{t_idx:03d}.png'))

            # 2) 收集 GT（第一行，列数应为目标数 B）
            gt_imgs = []
            for j, (_, t_goal) in enumerate(goal_context):
                gt = Image.open(get_data_path(self.data_folder, f_curr, int(t_goal))).convert("RGB")
                gt_np = np.array(gt)
                gt_np = np.clip(gt_np * (255.0 ** (gt_np.max() <= 1)), 0, 255).astype(np.uint8)
                gt_imgs.append(gt_np)
                Image.fromarray(gt_np).save(os.path.join(sample_dir, f'gt_{j:03d}_t{int(t_goal)}.png'))

            # 3) 联合投影（第二行，列数应为 B）
            B = projected_images.shape[0]
            joint_imgs = []
            for j in range(B):
                pj = projected_images[j]
                pj = np.clip(pj * (255.0 ** (pj.max() <= 1)), 0, 255).astype(np.uint8)
                joint_imgs.append(pj)
                Image.fromarray(pj).save(os.path.join(sample_dir, f'proj_joint_{j:03d}.png'))

            # 4) 单独投影（后续 T 行）：将 batch 轴移到前面 → (B,H,W,3)，每行应产出 B 张
            per_hist_rows = []
            for t_idx, proj_o in enumerate(his_projected_list):
                axes_equal_B = np.where(np.array(proj_o.shape) == B)[0]
                axis = int(axes_equal_B[0])
                arr = np.moveaxis(proj_o, axis, 0)
                row = []
                for j in range(B):
                    im = arr[j]
                    im = np.clip(im * (255.0 ** (im.max() <= 1)), 0, 255).astype(np.uint8)
                    row.append(im)
                    Image.fromarray(im).save(os.path.join(sample_dir, f'proj_single_t{t_idx:03d}_{j:03d}.png'))
                per_hist_rows.append(row)

            # 5) 组装大图 grid：第0列放行名标签；第1..N列为图像（GT一行、Joint一行、T行历史单独投影）
            from PIL import ImageDraw, ImageFont

            # 计算列数：0列为历史帧，其余为目标列
            B = projected_images.shape[0]
            max_cols = B + 1
            pad = 4

            # 统一像素到 uint8
            T = rgb_imgs_np.shape[0]
            hist_uint8 = []
            for t_idx in range(T):
                im = rgb_imgs_np[t_idx]
                im = np.clip(im * (255.0 ** (im.max() <= 1)), 0, 255).astype(np.uint8)
                hist_uint8.append(im)

            gt_uint8 = []
            for x in gt_imgs:
                im = np.clip(x * (255.0 ** (x.max() <= 1)), 0, 255).astype(np.uint8)
                gt_uint8.append(im)

            joint_uint8 = []
            for x in joint_imgs:
                im = np.clip(x * (255.0 ** (x.max() <= 1)), 0, 255).astype(np.uint8)
                joint_uint8.append(im)

            # 先构造小尺寸的文字贴片（稍后统一 resize）
            label_size = (64, 48)
            font = ImageFont.load_default()
            label_gt = Image.new("RGB", label_size, (255, 255, 255))
            draw_gt = ImageDraw.Draw(label_gt)
            draw_gt.text((4, 4), "GT", fill=(0, 0, 0), font=font)
            label_gt_np = np.array(label_gt)

            label_joint = Image.new("RGB", label_size, (255, 255, 255))
            draw_joint = ImageDraw.Draw(label_joint)
            draw_joint.text((4, 4), "Joint", fill=(0, 0, 0), font=font)
            label_joint_np = np.array(label_joint)

            # 在历史帧图上写行名
            hist_named = []
            for t_idx in range(T):
                pil = Image.fromarray(hist_uint8[t_idx])
                draw = ImageDraw.Draw(pil)
                draw.text((6, 6), f"Hist t={context_times[t_idx]}", fill=(255, 255, 255), font=font)
                hist_named.append(np.array(pil))

            # 组装行：第0列 + 其余B列
            rows = []
            row0 = [label_gt_np] + gt_uint8
            rows.append(row0)
            row1 = [label_joint_np] + joint_uint8
            rows.append(row1)
            for t_idx in range(T):
                row_hist = [hist_named[t_idx]] + per_hist_rows[t_idx]
                rows.append(row_hist)

            # 统一到相同尺寸（以第一张GT为基准）
            H, W = rows[0][1].shape[:2]
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

            # 分隔条
            row_width = max_cols * W + (max_cols - 1) * pad
            spacer_w = np.full((H, pad, 3), 255, np.uint8)
            spacer_h = np.full((pad, row_width, 3), 255, np.uint8)

            # 横向拼每一行
            row_arrays = []
            for r in norm_rows:
                row_img = r[0]
                for x in r[1:]:
                    row_img = np.concatenate((row_img, spacer_w, x), axis=1)
                row_arrays.append(row_img)

            # 纵向拼所有行
            grid = row_arrays[0]
            for rr in row_arrays[1:]:
                grid = np.concatenate((grid, spacer_h, rr), axis=0)

            Image.fromarray(grid).save(os.path.join(sample_dir, 'grid_all.png'))

            # ==========================================================================================


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

            rgb_imgs = [cv2.imread(get_data_path(self.data_folder, f_img, t_img)) for f_img, t_img in context]
            rgb_imgs = [cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) for rgb_img in rgb_imgs]
            rgb_imgs = np.stack(rgb_imgs, axis=0) 

            # 多历史帧 → 多目标帧重投影（B 张投影图）
            projected_images = self._compute_projected_images(curr_traj_data, context_times, rgb_imgs, np.array(pred_times))
            projected_tensor_list = [self.transform(Image.fromarray(img)) for img in projected_images]
            projected_tensor = torch.stack(projected_tensor_list, dim=0)

            print(f"Index {i}, projected_images shape: {projected_images.shape}, projected_tensor shape: {projected_tensor.size()}")

            return (
                torch.tensor([i], dtype=torch.float32), # for logging purposes
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
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
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
            # Compute actions, goal_pos, projected images
            actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, np.array([goal_time]))
            rgb_imgs = [cv2.imread(get_data_path(self.data_folder, f_img, t_img)) for f_img, t_img in context]
            rgb_imgs = [cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) for rgb_img in rgb_imgs]
            rgb_imgs = np.stack(rgb_imgs, axis=0) 
            projected_images = self._compute_projected_images(curr_traj_data, context_times, rgb_imgs, np.array([goal_time]))
            projected_tensor_list = [self.transform(Image.fromarray(img)) for img in projected_images]
            projected_tensor = torch.stack(projected_tensor_list, dim=0)

            return (
                torch.tensor([i], dtype=torch.float32), # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_image, dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
                torch.as_tensor(projected_tensor, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)
