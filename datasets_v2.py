# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------
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
from project_functions import reproject_depth_to_other_pose_2seq, project_to_2d_image_2seq, resize_image_half

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
            # if traj_len < 12:
            #     continue
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
        # off = 88
        # TIME_KEYS = ("point", "position", "pose", "depth", "yaw")  # 这些第一维是时间
        # # 先确定“时间长度”
        # time_lens = []
        # for k in TIME_KEYS:
        #     if k in traj_data and isinstance(traj_data[k], np.ndarray) and traj_data[k].ndim >= 1:
        #         time_lens.append(traj_data[k].shape[0])
        # time_len = min(time_lens) if len(time_lens) > 0 else 0

        # if time_len > 0 and off > 0:
        #     for k in TIME_KEYS:
        #         if k in traj_data and isinstance(traj_data[k], np.ndarray):
        #             arr = traj_data[k]
        #             # 只切第一维等于 time_len 的数组（按时间展开的）
        #             if arr.ndim >= 1 and arr.shape[0] == time_len:
        #                 traj_data[k] = arr[off:]

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
        """
        基于深度图 + pose 生成从另一个相机视角观察到的图像。
        """
        image_size = depth_map.shape  # (H, W)
        if rgb_img.shape[:2] != image_size:
            rgb_img = resize_image_half(rgb_img)
        points_3d, colors = reproject_depth_to_other_pose_2seq(K, depth_map, rgb_img, pose_src, pose_dst) 
        images = project_to_2d_image_2seq(K, points_3d, colors, image_size) # (H, W, 3, goal_time)
        return images

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
            raise ValueError("is used?")
            # const_len = self.len_traj_pred + 1 - yaw.shape[0]
            # yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            # positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

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
            goal_time = (curr_time + goal_offset).astype('int')
            rel_time = (goal_offset).astype('float')/(128.) # TODO: refactor, currently a fixed const

            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            true_context = [(f_curr, t) for t in context_times]
            goal_context = [(f_curr, t) for t in goal_time]
            context = [(f_curr, t) for t in context_times] + [(f_curr, t) for t in goal_time]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])

            # Load other trajectory data
            curr_traj_data = self._get_trajectory(f_curr)

            # aug
            f_img, t_img = true_context[-1]    # curr_time img
            rgb_img = cv2.imread(get_data_path(self.data_folder, f_img, t_img))
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            
            # Compute actions
            _, goal_pos, projected_images = self._compute_actions(curr_traj_data, curr_time, goal_time, rgb_img)
            goal_pos[:, :3] = normalize_data(goal_pos[:, :3], self.ACTION_STATS)
            
            projected_tensor_list = [self.transform(Image.fromarray(img)) for img in projected_images]
            projected_tensor = torch.stack(projected_tensor_list, dim=0)

            # ===================== 保存图像 =====================
            vis_root = './visualizations'
            sample_dir = os.path.join(vis_root, f'{self.dataset_name}', f'sample_{i}')
            os.makedirs(sample_dir, exist_ok=True)

            # 1. 保存 curr_frame
            curr_img_save_path = os.path.join(sample_dir, 'curr_frame.png')
            Image.fromarray(rgb_img).save(curr_img_save_path)

            # 2. 保存 goal_frame
            for idx, (f_curr, t_goal) in enumerate(goal_context):
                goal_img_path = get_data_path(self.data_folder, f_curr, t_goal)
                goal_img = Image.open(goal_img_path)
                goal_img.save(os.path.join(sample_dir, f'goal_{idx}.png'))

            # 3. 保存 projected goal frame
            for idx, proj_img in enumerate(projected_images):
                proj_img_save_path = os.path.join(sample_dir, f'projected_goal_{idx}.png')
                Image.fromarray(proj_img).save(proj_img_save_path)
            # ====================================================

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
            pred_times = list(range(curr_time + 1, curr_time + self.len_traj_pred + 1))
            
            context = [(f_curr, t) for t in context_times]
            pred = [(f_curr, t) for t in pred_times]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
            pred_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in pred])

            curr_traj_data = self._get_trajectory(f_curr)

            # Compute last rgb image
            f_img, t_img = context[-1]    # curr_time img
            rgb_img = cv2.imread(get_data_path(self.data_folder, f_img, t_img))
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            # Compute actions
            actions, _, projected_images = self._compute_actions(curr_traj_data, curr_time, np.array([curr_time+1]), rgb_img) # last argument is dummy goal
            actions[:, :3] = normalize_data(actions[:, :3], self.ACTION_STATS)
            delta = get_delta_np(actions)
            # Compute projected tensor
            projected_tensor_list = [self.transform(Image.fromarray(img)) for img in projected_images]
            projected_tensor = torch.stack(projected_tensor_list, dim=0)
            # print(f"projected_images shape: {projected_images.shape}, projected_tensor shape: {projected_tensor.size()}")
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
            f_img, t_img = context[-1]    # curr_time img
            rgb_img = cv2.imread(get_data_path(self.data_folder, f_img, t_img))
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            
            actions, goal_pos, projected_images = self._compute_actions(curr_traj_data, curr_time, np.array([goal_time]), rgb_img)
            
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
