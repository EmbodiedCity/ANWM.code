# -*- coding: utf-8 -*-
"""
PE-Field 数据集（v8）
用途：
- 在不修改 datasets_v3.py 接口与实现的前提下，提供“满血版 PE-Field”训练所需的相机参数与深度。
- 与 v3 独立存在；v3 保持不变，v8 额外返回 K 与当前帧深度，方便在训练侧构建 pix_coords_downs。

主要差异（相对于 v3）：
- TrainingDatasetV8.__getitem__ 返回多两个条目：K(3x3) 和 depth_curr(HxW)
- EvalDatasetV8 / TrajectoryEvalDatasetV8 同样返回 K 与 depth_curr，便于评测阶段也能构建 PE-Field 编码。

推荐用法：
- 训练脚本中用 (K, depth_curr, T_cw) 反投影/变换/投影，得到 (z, u, v) 多尺度网格并传入模型的 pix_coords_downs。
- v3 仍可用于不启用 PE-Field 的基线对比。
"""
from typing import Tuple
import os
import pickle
import yaml
import tqdm
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset

from misc import angle_difference, get_data_path, get_delta_np, normalize_data, to_local_coords
from project_functions import AirsimCoordsConverter, reproject_depth_to_other_pose_2seq, project_to_2d_image_2seq, resize_image_half


class BaseDatasetV8(Dataset):
    """基础数据集（v8）。

    说明：
    - 结构参考 v3，但文件独立；不影响 v3 的任何接口/行为。
    - 提供统一的数据加载、索引与投影辅助函数。
    """
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

        if 'airvln' in dataset_name:
            self.coords_converter = AirsimCoordsConverter()

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

    def _compute_projected_image(self, traj_data, curr_time, goal_time, rgb_img):
        pose_src = traj_data["pose"][curr_time]
        pose_dst = traj_data["pose"][goal_time]

        depth_map = traj_data["depth"][curr_time]
        K = traj_data["K"]

        projected_images = self.generate_augmented_image(K=K, depth_map=depth_map, rgb_img=rgb_img, pose_src=pose_src, pose_dst=pose_dst)
        return projected_images

    def generate_augmented_image(self, K, depth_map, rgb_img, pose_src, pose_dst) -> np.ndarray:
        image_size = depth_map.shape  # (H, W)
        if rgb_img.shape[:2] != image_size:
            rgb_img = resize_image_half(rgb_img)
        points_3d, colors = reproject_depth_to_other_pose_2seq(K, depth_map, rgb_img, pose_src, pose_dst)
        images = project_to_2d_image_2seq(K, points_3d, colors, image_size)  # (H, W, 3, goal_time)
        return images


class TrainingDatasetV8(BaseDatasetV8):
    """训练集（v8）。

    返回（与 v3 基本一致，但多返回 2 项）：
    - obs_image, goal_pos, rel_time, projected_tensor, T_cw
    - K: (3,3)
    - depth_curr: (H, W)
    """
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
        f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
        goal_offset = np.random.randint(min_goal_dist//4, max_goal_dist//4 + 1, size=(self.goals_per_obs))
        goal_time = (curr_time + goal_offset).astype('int')
        rel_time = (goal_offset).astype('float')/(128.)

        context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
        context = [(f_curr, t) for t in context_times] + [(f_curr, t) for t in goal_time]
        context_t = [t for _, t in context]

        obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])

        curr_traj_data = self._get_trajectory(f_curr)

        # aug from current frame
        f_img, t_img = context[self.context_size-1]
        rgb_img = cv2.imread(get_data_path(self.data_folder, f_img, t_img))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # actions/goal and projection
        _, goal_pos, projected_images = self._compute_actions(curr_traj_data, curr_time, goal_time, rgb_img)
        goal_pos[:, :3] = normalize_data(goal_pos[:, :3], self.ACTION_STATS)
        projected_tensor_list = [self.transform(Image.fromarray(img)) for img in projected_images]
        projected_tensor = torch.stack(projected_tensor_list, dim=0)

        # camera mats (world<->camera)
        T_wc = curr_traj_data['pose'][context_t]
        T_wc = torch.as_tensor(T_wc, dtype=torch.float32)
        T_cw = torch.linalg.inv(T_wc)

        # PE-Field essentials
        K = torch.as_tensor(curr_traj_data['K'], dtype=torch.float32)        # (3,3)
        depth_curr = torch.as_tensor(curr_traj_data['depth'][curr_time], dtype=torch.float32)  # (H,W)

        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(rel_time, dtype=torch.float32),
            torch.as_tensor(projected_tensor, dtype=torch.float32),
            torch.as_tensor(T_cw, dtype=torch.float32),
            K,
            depth_curr,
        )

    def _compute_actions(self, traj_data, curr_time, goal_time, rgb_img):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1
        yaw = traj_data["yaw"][start_index:end_index]
        positions = traj_data["point"][start_index:end_index]
        goal_pos = traj_data["point"][goal_time]
        goal_yaw = traj_data["yaw"][goal_time]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)
        if yaw.shape != (self.len_traj_pred + 1,):
            raise ValueError("Unexpected yaw shape in dataset.")

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
        projected_images = self._compute_projected_image(traj_data, curr_time, goal_time, rgb_img)
        return actions, goal_pos, projected_images


class EvalDatasetV8(BaseDatasetV8):
    """评估集（v8）。同样附带 K 与 depth_curr，方便评估时构建 PE-Field。"""
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
        f_curr, curr_time, _, _ = self.index_to_data[i]
        context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
        pred_times = list(range(curr_time + 1, curr_time + self.len_traj_pred + 1))

        context = [(f_curr, t) for t in context_times]
        pred = [(f_curr, t) for t in pred_times]

        obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
        pred_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in pred])

        curr_traj_data = self._get_trajectory(f_curr)

        # PE-Field essentials from current frame
        K = torch.as_tensor(curr_traj_data['K'], dtype=torch.float32)
        depth_curr = torch.as_tensor(curr_traj_data['depth'][curr_time], dtype=torch.float32)

        # last rgb
        f_img, t_img = context[-1]
        rgb_img = cv2.imread(get_data_path(self.data_folder, f_img, t_img))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        actions, _, projected_images = self._compute_actions(curr_traj_data, curr_time, np.array(pred_times), rgb_img)
        actions[:, :3] = normalize_data(actions[:, :3], self.ACTION_STATS)
        delta = get_delta_np(actions)

        projected_tensor_list = [self.transform(Image.fromarray(img)) for img in projected_images]
        projected_tensor = torch.stack(projected_tensor_list, dim=0)

        T_wc_ctx = curr_traj_data['pose'][context_times]
        T_wc_pred = curr_traj_data['pose'][pred_times]
        T_cw_ctx = torch.linalg.inv(torch.as_tensor(T_wc_ctx, dtype=torch.float32))
        T_cw_pred = torch.linalg.inv(torch.as_tensor(T_wc_pred, dtype=torch.float32))

        return (
            torch.tensor([i], dtype=torch.float32),
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(pred_image, dtype=torch.float32),
            torch.as_tensor(delta, dtype=torch.float32),
            torch.as_tensor(projected_tensor, dtype=torch.float32),
            torch.as_tensor(T_cw_ctx, dtype=torch.float32),
            torch.as_tensor(T_cw_pred, dtype=torch.float32),
            K,
            depth_curr,
        )

    def _compute_actions(self, traj_data, curr_time, goal_time, rgb_img):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1
        yaw = traj_data["yaw"][start_index:end_index]
        positions = traj_data["point"][start_index:end_index]
        goal_pos = traj_data["point"][goal_time]
        goal_yaw = traj_data["yaw"][goal_time]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)
        if yaw.shape != (self.len_traj_pred + 1,):
            raise ValueError("Unexpected yaw shape in dataset.")

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
        projected_images = self._compute_projected_image(traj_data, curr_time, goal_time, rgb_img)
        return actions, goal_pos, projected_images


class TrajectoryEvalDatasetV8(BaseDatasetV8):
    """轨迹评估集（v8）。同样附带 K 与 depth_curr。"""
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
        f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
        # sample one goal deterministically
        goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1)
        goal_time = curr_time + int(goal_offset)

        context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
        context = [(f_curr, t) for t in context_times]

        obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
        goal_image = self.transform(Image.open(get_data_path(self.data_folder, f_curr, goal_time))).unsqueeze(0)
        curr_traj_data = self._get_trajectory(f_curr)

        # PE-Field essentials from current frame
        K = torch.as_tensor(curr_traj_data['K'], dtype=torch.float32)
        depth_curr = torch.as_tensor(curr_traj_data['depth'][curr_time], dtype=torch.float32)

        # actions/goal and projection
        f_img, t_img = context[-1]
        rgb_img = cv2.imread(get_data_path(self.data_folder, f_img, t_img))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        actions, goal_pos, projected_images = self._compute_actions(curr_traj_data, curr_time, np.array([goal_time]), rgb_img)
        projected_tensor_list = [self.transform(Image.fromarray(img)) for img in projected_images]
        projected_tensor = torch.stack(projected_tensor_list, dim=0)

        T_wc_ctx = curr_traj_data['pose'][context_times]
        T_cw_ctx = torch.linalg.inv(torch.as_tensor(T_wc_ctx, dtype=torch.float32))
        T_wc_goal = curr_traj_data['pose'][[goal_time]]
        T_cw_goal = torch.linalg.inv(torch.as_tensor(T_wc_goal, dtype=torch.float32))

        return (
            torch.tensor([i], dtype=torch.float32),
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_image, dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.float32),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(projected_tensor, dtype=torch.float32),
            torch.as_tensor(T_cw_ctx, dtype=torch.float32),
            torch.as_tensor(T_cw_goal, dtype=torch.float32),
            K,
            depth_curr,
        )

    def _compute_actions(self, traj_data, curr_time, goal_time, rgb_img):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1
        yaw = traj_data["yaw"][start_index:end_index]
        positions = traj_data["point"][start_index:end_index]
        goal_pos = traj_data["point"][goal_time]
        goal_yaw = traj_data["yaw"][goal_time]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)
        if yaw.shape != (self.len_traj_pred + 1,):
            raise ValueError("Unexpected yaw shape in dataset.")

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
        projected_images = self._compute_projected_image(traj_data, curr_time, goal_time, rgb_img)
        return actions, goal_pos, projected_images

