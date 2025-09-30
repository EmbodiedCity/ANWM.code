#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
from tqdm import tqdm
import numpy as np

# 原始 split 文件
# split_pkl = "data_splits/airvln_16/test/dataset_dist_8_to_8_n4_len_traj_pred_8_all.pkl"
split_pkl = "data_splits/airvln_16/test/dataset_dist_-64_to_64_n4_len_traj_pred_64.pkl"

with open(split_pkl, "rb") as f:
    split_list = pickle.load(f)[0]
print(f"[info] 原始 split 数量: {len(split_list)}")

no_z_list = []

for traj_id, frame_idx, min_goal_distance, max_goal_distance in tqdm(split_list, desc="Checking trajectories", dynamic_ncols=True):
    print(traj_id, frame_idx, min_goal_distance, max_goal_distance)
    # 假设 traj_data.pkl 路径如下，根据实际情况修改
    traj_dir = os.path.join("data", "airvln_16", traj_id)
    traj_pkl = os.path.join(traj_dir, "traj_data.pkl")

    with open(traj_pkl, "rb") as f:
        traj_data = pickle.load(f)

    # 计算增量 (dx, dy, dz, dyaw)
    diff_traj = np.diff(traj_data["point"], axis=0)        # (T,3)
    diff_yaw  = np.diff(traj_data["yaw"], axis=0).reshape(-1,1)  # (T,1)
    gt_traj   = np.hstack([diff_traj, diff_yaw])           # (T,4)
    dz = diff_traj[:, 2]
    # print(f"{traj_id}: frame_idx={frame_idx}, dz.min={dz.min():.6f}, dz.max={dz.max():.6f}")
    # print("dz:", np.round(dz[:20], 6).tolist())
    start = frame_idx
    end   = frame_idx + max_goal_distance
    gt_slice = gt_traj[start:end]
    print(f"start={start}, end={end}, slice_len={gt_slice.shape[0]}")

    dz_s = gt_slice[:, 2]
    print(f"{traj_id}: frame_idx={frame_idx}, dz.min={dz_s.min():.6f}, dz.max={dz_s.max():.6f}")
    print("dz slice:", np.round(dz_s.tolist(), 6))

    # 判断 z 分量是否全 0（只在切片里）
    if (dz_s == 0).all():
        no_z_list.append((traj_id, frame_idx, min_goal_distance, max_goal_distance))

print(f"[summary] no-z 数量: {len(no_z_list)}")
print("示例:", no_z_list[:5])

# sample
# import random
# random.seed(1120)
# no_z_list = random.sample(no_z_list, k=100)

# 保存新的 split
out_pkl = "data_splits/airvln_16/test/rollout_2d.pkl"
with open(out_pkl, "wb") as f:
    pickle.dump(no_z_list, f)

print(f"[save] 写出新 split 到: {out_pkl}")
