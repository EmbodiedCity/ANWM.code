#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
from tqdm import tqdm
import numpy as np

# 原始 split 文件
# reference_pkl = "data_splits/airvln_16/test/navigation_eval.pkl"
# split_pkl = "data_splits/airvln_16/test/dataset_dist_8_to_8_n16_len_traj_pred_8.pkl"
reference_pkl = "data_splits/airvln_16/test/rollout_turn_135deg.pkl"
# split_pkl = "data_splits/airvln_16/test/dataset_dist_-64_to_64_n16_len_traj_pred_52.pkl"
split_pkl = "data_splits/airvln_16/test/dataset_dist_-64_to_64_n4_len_traj_pred_64.pkl"

with open(split_pkl, "rb") as f:
    split_list = pickle.load(f)[0]
print(f"[info] 原始 split 数量: {len(split_list)}")

# 读取 reference，收集 (traj_id, frame_idx)
with open(reference_pkl, "rb") as f:
    reference_list = pickle.load(f)

ref_keys = set((str(traj_id), int(frame_idx)) for traj_id, frame_idx, *rest in reference_list)
print(f"[info] reference 键数量: {len(ref_keys)}")
print("示例:", reference_list[:5], split_list[:5])

# 按 (traj_id, frame_idx) 过滤 split
matched_list = []
for traj_id, frame_idx, min_goal_distance, max_goal_distance in tqdm(split_list, desc="Filtering by keys", dynamic_ncols=True):
    if (str(traj_id), int(frame_idx) - 12) in ref_keys:
        matched_list.append((traj_id, frame_idx, min_goal_distance, max_goal_distance))

print(f"[summary] 匹配数量: {len(matched_list)} / {len(split_list)}")

# 保存新的 split（仅保留匹配项）
out_pkl = "data_splits/airvln_16/test/rollout_turn_135deg_n4.pkl"
with open(out_pkl, "wb") as f:
    pickle.dump(matched_list, f)

print(f"[save] 写出匹配后的 split 到: {out_pkl}")
