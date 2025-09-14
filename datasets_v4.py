# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

# 这个版本为加入了相机编码版本,并且把min_goal_dist,max_goal_dist分别约束到-16, 16
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
from project_functions import AirsimCoordsConverter, reproject_depth_to_other_pose_2seq, project_to_2d_image_2seq, resize_image_half

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
    
    def _compute_projected_image(self, traj_data, curr_time, goal_time, rgb_img):

        pose_src = traj_data["pose"][curr_time]
        pose_dst = traj_data["pose"][goal_time]

        depth_map = traj_data["depth"][curr_time]
        K = traj_data["K"]
        
        # print(f"pose_src shape: {pose_src.shape}, pose_dst shape: {pose_dst.shape}, dep_map shape: {depth_map.shape}")
        
        # projected_images = self.generate_augmented_image(K=K, depth_map=depth_map, rgb_img=rgb_img, pose_src=pose_src, pose_dst=pose_dst)
        projected_images = self.generate_augmented_image_v2(K=K, depth_map=depth_map, rgb_img=rgb_img, pose_src=pose_src, pose_dst=pose_dst)

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

    def generate_augmented_image_v2(self, K, depth_map, rgb_img, pose_src, pose_dst):
        goal_times = pose_dst.shape[0]
        images = []
        for i in range(goal_times):
            p_dst = pose_dst[i]
            images.append(self.warpPerspective(K, depth_map, rgb_img, pose_src, p_dst))
        return images
        
    def warpPerspective(self, K, depth_map, rgb_img, pose_src, pose_dst, out_size=None, fill_value=0) -> np.ndarray:
        """
        重投影 src 图像到 dst 相机视角（同内参 K）。
        参数
        ----
        K : (3,3) numpy.ndarray
            相机内参（src/dst 相同）
        depth_map : (H,W) numpy.ndarray
            与 rgb_img 对齐的深度（单位米）
        rgb_img : (H,W,3) uint8
            源图像（相机 src 拍摄）
        pose_src : (4,4) numpy.ndarray
            相机 src 的 camera-to-world (c2w) 位姿
        pose_dst : (4,4) numpy.ndarray
            相机 dst 的 camera-to-world (c2w) 位姿
        out_size : (W_out, H_out) or None
            目标图像尺寸；None 时用源图像尺寸
        fill_value : int or tuple
            空洞填充值（背景）

        返回
        ----
        img_dst : (H_out, W_out, 3) uint8
            在 dst 视角下渲染的图像
        z_dst : (H_out, W_out) float32
            目标视角的深度（可用于可视化/调试）
        """

        H, W = depth_map.shape
        if out_size is None:
            W_out, H_out = W, H
        else:
            W_out, H_out = out_size

        # 1) 构造像素网格（源图像）
        u, v = np.meshgrid(np.arange(W), np.arange(H))  # (H,W)
        ones = np.ones_like(u, dtype=np.float32)
        pix_src_h = np.stack([u, v, ones], axis=-1).reshape(-1, 3).T  # 3xN
        depth = depth_map.reshape(-1).astype(np.float32)               # N

        # 2) 反投影到 cam_src 坐标： x_src = depth * K^{-1} * [u,v,1]^T
        Kinv = np.linalg.inv(K)
        x_src = (Kinv @ pix_src_h) * depth  # 3xN

        # 3) cam_src -> world -> cam_dst
        #    T_src2dst = (pose_dst)^{-1} @ pose_src
        T_src2dst = np.linalg.inv(pose_dst) @ pose_src
        R = T_src2dst[:3, :3]
        t = T_src2dst[:3, 3:4]  # 3x1

        x_dst = (R @ x_src) + t  # 3xN

        # 4) 投影到 dst 像素
        pix_dst_h = K @ x_dst                      # 3xN
        z = pix_dst_h[2, :] + 1e-6
        u_dst = (pix_dst_h[0, :] / z)
        v_dst = (pix_dst_h[1, :] / z)

        # 5) 前向“光栅化” + Z-buffer（处理遮挡；最近深度覆盖）
        img_dst = np.full((H_out, W_out, 3), fill_value, dtype=rgb_img.dtype)
        z_dst = np.full((H_out, W_out), np.inf, dtype=np.float32)

        # 只保留落在目标画幅内的点
        u_round = np.round(u_dst).astype(np.int64)
        v_round = np.round(v_dst).astype(np.int64)

        valid = (
            (z > 0) &
            (u_round >= 0) & (u_round < W_out) &
            (v_round >= 0) & (v_round < H_out) &
            (depth > 0)
        )

        src_colors = rgb_img.reshape(-1, 3)[valid]
        uu = u_round[valid]
        vv = v_round[valid]
        zz = z[valid].astype(np.float32)

        # Z-buffer：对同一像素，保留深度更小（更近）的样本
        # 用扁平索引实现原子“最小深度写入”
        flat_idx = vv * W_out + uu
        # 为每个像素找到最小深度的索引
        order = np.argsort(zz)  # 从近到远
        flat_idx = flat_idx[order]
        zz = zz[order]
        src_colors = src_colors[order]

        # 只保留每个像素第一次出现（即最小深度）
        _, first_pos = np.unique(flat_idx, return_index=True)
        keep = np.zeros_like(order, dtype=bool)
        keep[first_pos] = True

        flat_idx = flat_idx[keep]
        zz = zz[keep]
        src_colors = src_colors[keep]

        # 写入帧缓冲
        z_dst.flat[flat_idx] = zz
        img_dst.reshape(-1, 3)[flat_idx] = src_colors
        
        # 补全`
        # hole_mask = (img_dst.mean(axis=2) == 0).astype(np.uint8)
        # print(hole_mask.sum())
        # hole_mask = cv2.dilate(hole_mask, np.ones((3,3), np.uint8), iterations=1)
        # img_dst = cv2.inpaint(img_dst, hole_mask*255, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        return img_dst
        
    
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
            # goal_offset = np.random.randint(min_goal_dist//4, max_goal_dist//4 + 1, size=(self.goals_per_obs))
            goal_offset = np.arange(1, 1+self.goals_per_obs)
            
            goal_time = (curr_time + goal_offset).astype('int')
            rel_time = (goal_offset).astype('float')/(128.) # TODO: refactor, currently a fixed const

            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            context = [(f_curr, t) for t in context_times] + [(f_curr, t) for t in goal_time]
            context_t = [t for _, t in context]
            
            # print(f"curr_time: {curr_time}, context_size: {self.context_size}, min_goal_dist: {min_goal_dist}, max_goal_dist: {max_goal_dist}")
            
            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])

            # Load other trajectory data
            curr_traj_data = self._get_trajectory(f_curr)
            # print(f"traj K: {curr_traj_data['K']}, position: {curr_traj_data['position']}, pose: {curr_traj_data['pose']}, point: {curr_traj_data['point']}")
            
            # aug
            f_img, t_img = context[self.context_size-1]    # curr_time img
            # print(f"context: {context}, f_img: {f_img}, t_img: {t_img}, curr time: {curr_time}")

            rgb_img = cv2.imread(get_data_path(self.data_folder, f_img, t_img))
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            
            # Compute actions
            _, goal_pos, projected_images = self._compute_actions(curr_traj_data, curr_time, goal_time, rgb_img)
            goal_pos[:, :3] = normalize_data(goal_pos[:, :3], self.ACTION_STATS)
            
            projected_tensor_list = [self.transform(Image.fromarray(img)) for img in projected_images]
            projected_tensor = torch.stack(projected_tensor_list, dim=0)

            # Compute camera mats
            T_wc = curr_traj_data['pose'][context_t]
            T_wc = torch.as_tensor(T_wc, dtype=torch.float32)
            T_cw = torch.linalg.inv(T_wc)
            
            # print(f"t cw shape: {T_cw.size()}, context_t: {context_t}")
            
            # points = [curr_traj_data['point'][t] for _, t in context]
            # oriens = [[curr_traj_data['pitch'][t], curr_traj_data['roll'][t], curr_traj_data['yaw'][t]] for _, t in context]
            # poses = np.concatenate((np.array(points), np.array(oriens)), axis=-1)
            # t_cws = torch.as_tensor(
            #     np.array([self.coords_converter.trans_cam2world(poses[tt]) for tt in range(len(poses))]), dtype=torch.float32
            # )
            # print("t_cws: ", t_cws - T_cw)
            
            # ori_poses = np.array([curr_traj_data['pose'][t] for _, t in context])
            # print("ori_pose: ", ori_poses)

            
            # ===================== 保存图像 =====================
            vis_root = './visualizations'
            sample_dir = os.path.join(vis_root, f'airvln_16_v2', f'sample_{i}')
            os.makedirs(sample_dir, exist_ok=True)

            # 1. 保存 curr_frame
            curr_img_save_path = os.path.join(sample_dir, f'curr_frame_{curr_time}.png')
            Image.fromarray(rgb_img).save(curr_img_save_path)

            # 2. 保存 goal_frame
            for idx, (f_curr, t_goal) in enumerate(context[self.context_size:]):
                goal_img_path = get_data_path(self.data_folder, f_curr, t_goal)
                goal_img = Image.open(goal_img_path)
                goal_img.save(os.path.join(sample_dir, f'goal_{idx}_{t_goal}.png'))

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
                torch.as_tensor(T_cw, dtype=torch.float32),
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
            # Compute T
            T_wc_ctx = curr_traj_data['pose'][context_times]             # (context_size, 4, 4)
            T_wc_pred = curr_traj_data['pose'][pred_times]               # (len_traj_pred, 4, 4)
            T_cw_ctx = torch.linalg.inv(torch.as_tensor(T_wc_ctx, dtype=torch.float32))   # (context_size, 4, 4)
            T_cw_pred = torch.linalg.inv(torch.as_tensor(T_wc_pred, dtype=torch.float32)) # (len_traj_pred, 4, 4)

            return (
                torch.tensor([i], dtype=torch.float32), # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(pred_image, dtype=torch.float32),
                torch.as_tensor(delta, dtype=torch.float32),
                torch.as_tensor(projected_tensor, dtype=torch.float32),
                torch.as_tensor(T_cw_ctx, dtype=torch.float32), #obs
                torch.as_tensor(T_cw_pred, dtype=torch.float32), #pred
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

            T_wc_ctx = curr_traj_data['pose'][context_times]
            T_cw_ctx = torch.linalg.inv(torch.as_tensor(T_wc_ctx, dtype=torch.float32))
            T_wc_goal = curr_traj_data['pose'][[goal_time]]
            T_cw_goal = torch.linalg.inv(torch.as_tensor(T_wc_goal, dtype=torch.float32))

            return (
                torch.tensor([i], dtype=torch.float32), # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_image, dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
                torch.as_tensor(projected_tensor, dtype=torch.float32),
                torch.as_tensor(T_cw_ctx, dtype=torch.float32),
                torch.as_tensor(T_cw_goal, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)
