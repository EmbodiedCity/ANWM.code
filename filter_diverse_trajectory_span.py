#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, pickle, random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== 固定参数（按需改） =====
split_pkl = "data_splits/airvln_16/test/dataset_dist_-64_to_64_n16_len_traj_pred_52.pkl"
DATA_ROOT, DATASET, TRAJ_PKL = "data", "airvln_16", "traj_data.pkl"
FALLBACK_HIST_LEN = 16

TARGETS_DEG = [15.0, 30.0, 45.0, 90.0, 135.0]   # 仍然会各自存pkl
DELTA_DEG   = 5.0
SAMPLE_K    = 200
OUT_DIR     = "data_splits/airvln_16/test"
MAX_WORKERS = max(4, os.cpu_count() * 2)

# ===== 读取 split =====
with open(split_pkl, "rb") as f:
    split_list = pickle.load(f)[0]
print(f"[info] split size = {len(split_list)}")

# 预计算目标区间（弧度）
ranges = {deg: (math.radians(deg - DELTA_DEG), math.radians(deg + DELTA_DEG)) for deg in TARGETS_DEG}
hits = {deg: [] for deg in TARGETS_DEG}

# ==== 新增：15°倍数计数器（15,30,...,360） ====
FIFTEEN_BINS = [15.0 * i for i in range(1, 25)]   # 1..24 => 15..360
counts_15deg = {deg: 0 for deg in FIFTEEN_BINS}

def minimal_arc_span_rad(angles_rad: np.ndarray) -> float:
    """
    计算一组角度（弧度）在圆周上的最小覆盖弧长（弧度）。
    做法：将角度映射到 [0, 2π)，排序，找相邻差的最大值(含尾首环)，跨度 = 2π - 最大相邻差。
    """
    angles = np.asarray(angles_rad, np.float64)
    angles = angles[np.isfinite(angles)]
    if angles.size < 2:
        return 0.0

    twopi = 2.0 * math.pi
    a = np.mod(angles, twopi)
    a.sort()
    diffs = np.diff(a)
    wrap_gap = (a[0] + twopi) - a[-1]
    max_gap = np.max(diffs) if diffs.size > 0 else 0.0
    if wrap_gap > max_gap:
        max_gap = wrap_gap
    span = twopi - max_gap
    # 数值稳定：夹到 [0, 2π]
    if span < 0.0:
        span = 0.0
    elif span > twopi:
        span = twopi
    return float(span)

def worker(item):
    traj_id, frame_idx, min_goal_distance, max_goal_distance = item
    p = os.path.join(DATA_ROOT, DATASET, traj_id, TRAJ_PKL)
    if not os.path.isfile(p): 
        return None, None
    with open(p, "rb") as f: 
        traj = pickle.load(f)

    yaw = np.asarray(traj.get("yaw", []), np.float64)
    if yaw.size < 2: 
        return None, None

    # 历史片段：从 end-历史长度 到 end（不含 end）
    hl   = int(min_goal_distance) if (min_goal_distance and min_goal_distance > 0) else FALLBACK_HIST_LEN
    end  = int(frame_idx)
    st   = max(0, end - hl)
    if end - st < 2: 
        return None, None

    seg  = yaw[st:end]

    # ====== 核心变化：不用累计角了，改为“最右—最左”的最小覆盖弧长 ======
    # 这相当于该时间段内朝向的“角度跨度”，自然落在 [0, 2π] 内。
    score = minimal_arc_span_rad(seg)  # 弧度

    return (traj_id, int(frame_idx), min_goal_distance, max_goal_distance), score

# 并行计算
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = [ex.submit(worker, it) for it in split_list]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Filtering", dynamic_ncols=True):
        rec, sc = fut.result()
        if rec is None: 
            continue

        # 原有：分配到指定目标区间并收集（区间为弧度）
        for deg, (lo, hi) in ranges.items():
            if lo <= sc <= hi:
                hits[deg].append(rec)

        # ==== 统计 15° 倍数命中条数（±DELTA_DEG 容差）====
        sc_deg = math.degrees(sc)
        for b in FIFTEEN_BINS:
            if abs(sc_deg - b) <= DELTA_DEG:
                counts_15deg[b] += 1
                break  # 容差不重叠，命中一个桶就退出

# 采样 + 保存
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(1120)
for deg in TARGETS_DEG:
    bucket = hits[deg]
    if len(bucket) > SAMPLE_K:
        bucket = random.sample(bucket, k=SAMPLE_K)
    out_pkl = os.path.join(OUT_DIR, f"rollout_turn_{int(deg)}deg.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(bucket, f)
    print(f"[save] {int(deg)}° 采样 {len(bucket)} / 命中 {len(hits[deg])} -> {out_pkl}")

# ==== 打印 15°倍数统计 ====
print("\n=== 15°倍数角度跨度命中条数（±%.1f°） ===" % DELTA_DEG)
for b in FIFTEEN_BINS:
    print(f"{int(b):>3d}° : {counts_15deg[b]}")
