#!/usr/bin/env python3
"""
提取每个候选轨迹的中间帧，与GT（从数据集加载）的中间帧拼接对比
样式参考 WM_Planning_Evaluator.save_single_sample_panel：
- 左一：GT 中间帧
- 左二：3D 轨迹 (GT + candidates)
- 后面：P1 / P2 / ... 的中间帧
  * label 显示：P1, LPIPS, DreamSim, APE
  * P1 用绿色粗框高亮
"""
import os
import re
import json
import yaml
import pickle
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from matplotlib import patches  # P1 绿色框
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from datasets_v2 import TrajectoryEvalDataset
from misc import transform


def load_image(path):
    """加载图片"""
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Warning: Cannot load {path}: {e}")
        return None


def get_mid_frame_path(metadata_path, frames_dir):
    """从 metadata 获取中间 step 的帧路径（不含 init/goal）"""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    num_frames = meta.get("num_frames", 0)
    if num_frames == 0:
        return None

    # num_frames = 1(init) + step_count + 1(goal)
    step_count = num_frames - 2
    if step_count <= 0:
        return None

    mid_step_idx = step_count // 2  # 中间 step 的索引（0-based）
    mid_frame_name = f"step_{mid_step_idx:03d}.png"
    mid_frame_path = os.path.join(frames_dir, mid_frame_name)
    if os.path.exists(mid_frame_path):
        return mid_frame_path
    return None


def load_gt_mid_frame_from_dataset(run_index, dataset, config):
    """从数据集加载 GT 轨迹的中间帧（时间上居中，对应轨迹的一半）"""
    # 获取该 run 对应的数据
    idxs, obs_image, goal_image, gt_actions, goal_pos, aug_image = dataset[run_index]
    traj_id = int(idxs.item())

    # index_to_data: (f_curr, curr_time, min_goal_dist, max_goal_dist)
    f_curr, curr_time, min_goal_dist, max_goal_dist = dataset.index_to_data[run_index]
    f_goal, goal_time, _ = dataset._sample_goal(f_curr, curr_time, min_goal_dist, max_goal_dist)

    T = gt_actions.shape[0]
    if T == 0:
        return None

    traj_stride = config.get("traj_stride", 1)
    mid_step = T // 2
    mid_time = curr_time + mid_step * traj_stride

    if mid_time > goal_time:
        mid_time = goal_time
    if mid_time < curr_time:
        mid_time = curr_time

    try:
        from misc import get_data_path
        img_path = get_data_path(dataset.data_folder, f_curr, mid_time)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            # resize 到 config 里的 image_size
            img_size = config.get("image_size", None)
            if img_size is not None:
                if isinstance(img_size, (list, tuple)):
                    img = img.resize((img_size[1], img_size[0]), Image.BICUBIC)
                else:
                    img = img.resize((img_size, img_size), Image.BICUBIC)
            return img
    except Exception as e:
        print(f"  Warning: Cannot load GT image from dataset: {e}")
        return None

    return None


def get_gt_traj_xyz_meters(run_index, dataset, config, dataset_name):
    """
    从 dataset 的 gt_actions 里取出 GT 轨迹的 3D 点，并转换为米单位。
    与 WM_Planning_Evaluator.generate_actions 里的逻辑一致：
    - gt_actions[:, :3] 是 waypoint 单位的绝对坐标
    - metric_waypoint_spacing 决定 waypoint -> meter 的映射
    """
    idxs, obs_image, goal_image, gt_actions, goal_pos, aug_image = dataset[run_index]
    data_cfg = config["eval_datasets"][dataset_name]
    spacing = float(data_cfg["metric_waypoint_spacing"])

    gt_xyz_waypoint = gt_actions[:, :3].to("cpu").numpy()  # [T, 3] waypoint 单位
    # 加上起点 (0,0,0)，与 generate_actions 一致：[T+1, 3]
    gt_xyz_waypoint = np.concatenate(
        [np.zeros((1, 3), dtype=np.float32), gt_xyz_waypoint.astype(np.float32)], axis=0
    )
    gt_xyz_meters = gt_xyz_waypoint * spacing  # [T+1, 3]
    return gt_xyz_meters  # (T+1, 3)


def process_run(run_dir, output_dir, dataset, config, candidate_trajs, dataset_name):
    """处理一个 run_xxx 目录，生成 GT + 3D 轨迹 + candidates 的中间帧对比图"""
    run_name = os.path.basename(run_dir)
    print(f"Processing {run_name}...")

    # 解析 run_index / traj_id （与 WM evaluator 中 sid=traj_id 对应）
    try:
        run_index = int(run_name.split('_')[1])
    except Exception:
        print(f"  Warning: Cannot parse run_index from {run_name}")
        return None

    traj_id = run_index  # 与 editor/run_{sid:03d} 中 sid 对齐

    # 从数据集加载 GT 中间帧（对应轨迹的一半）
    gt_mid_img = load_gt_mid_frame_from_dataset(run_index, dataset, config)
    if gt_mid_img is None:
        print(f"  Warning: Cannot load GT mid frame for {run_name}")
        return None

    # GT 轨迹 3D 点（米）
    gt_xyz_meters = get_gt_traj_xyz_meters(run_index, dataset, config, dataset_name)

    # 读取所有 candidates 的中间帧
    candidates_dir = os.path.join(run_dir, "candidates")
    if not os.path.exists(candidates_dir):
        print(f"  Warning: {candidates_dir} not found, skipping")
        return None

    cand_dirs = sorted(glob(os.path.join(candidates_dir, "cand_*")))
    if len(cand_dirs) == 0:
        print(f"  Warning: No candidates found in {run_name}")
        return None

    cand_mid_imgs = []
    cand_infos = []  # 保存 rank / id / lpips / ape / dreamsim，用于绘制标签和 3D 轨迹

    # 该 traj 对应的所有候选轨迹（米单位），形状 [N, T+1, 4]
    if traj_id not in candidate_trajs:
        print(f"  Warning: traj_id {traj_id} not in candidate_trajs, skip 3D plot")
        cand_traj_points_list = None
    else:
        cand_traj_full = np.asarray(candidate_trajs[traj_id], dtype=np.float32)  # [N,T+1,4]
        cand_traj_points_list = cand_traj_full[:, :, :3]  # [N,T+1,3]

    for cand_dir in cand_dirs:
        cand_name = os.path.basename(cand_dir)
        cand_meta_path = os.path.join(cand_dir, "metadata.json")
        cand_frames_dir = os.path.join(cand_dir, "frames")

        if not os.path.exists(cand_meta_path):
            continue

        cand_mid_path = get_mid_frame_path(cand_meta_path, cand_frames_dir)
        if cand_mid_path is None or not os.path.exists(cand_mid_path):
            continue

        cand_mid_img = load_image(cand_mid_path)
        if cand_mid_img is None:
            continue

        # 读取 candidate 的 meta 信息，用于 label 与 dreamsim
        try:
            with open(cand_meta_path, 'r', encoding='utf-8') as f:
                cand_meta = json.load(f)
            cand_rank = cand_meta.get("candidate_rank", -1)
            cand_id = cand_meta.get("candidate_id", -1)
            lpips_loss = float(cand_meta.get("final_lpips", 0.0))
            ape = float(cand_meta.get("cand_ape", 0.0))

            # DreamSim：兼容两种可能字段名，没有就置 0.0
            ds_val = cand_meta.get("cand_dreamsim", cand_meta.get("final_dreamsim", 0.0))
            dreamsim = float(ds_val)
        except Exception:
            cand_rank = -1
            cand_id = -1
            lpips_loss = 0.0
            ape = 0.0
            dreamsim = 0.0

        cand_mid_imgs.append(cand_mid_img)
        cand_infos.append({
            "rank": cand_rank,
            "id": cand_id,
            "lpips": lpips_loss,
            "ape": ape,
            "dreamsim": dreamsim,
        })

    if len(cand_mid_imgs) == 0:
        print(f"  Warning: No valid candidate mid frames for {run_name}")
        return None

    print(f"  Found: GT + {len(cand_mid_imgs)} candidates = {1 + len(cand_mid_imgs)} total images")

    # ===== 可视化：样式参考 save_single_sample_panel =====
    # 布局：GT | Traj(3D) | P1 | P2 | ...
    ncols = 2 + len(cand_mid_imgs)
    fig = plt.figure(figsize=(4 * ncols, 4))

    # 左一：GT 中间帧
    ax_gt = fig.add_subplot(1, ncols, 1)
    ax_gt.imshow(gt_mid_img)
    ax_gt.set_title("GT (mid-frame)", fontsize=14, fontweight='bold')
    ax_gt.axis('off')

    # 左二：3D 轨迹子图
    ax_traj = fig.add_subplot(1, ncols, 2, projection='3d')

    # 颜色与参考代码保持一致
    selected_color = "#2FBF71"  # P1 绿色高亮
    other_colors = ["#F4A259", "#E4572E", "#4C78A8", "#B279A2"]

    # 找到“最佳”candidate：优先用 rank==0，否则按最小 LPIPS
    best_idx = None
    for i, info in enumerate(cand_infos):
        if info["rank"] == 0:
            best_idx = i
            break
    if best_idx is None:
        lpips_list = [info["lpips"] for info in cand_infos]
        best_idx = int(np.argmin(lpips_list))

    # 先画 GT 轨迹（米）
    if gt_xyz_meters is not None and gt_xyz_meters.shape[0] > 0:
        gx, gy, gz = gt_xyz_meters[:, 0], gt_xyz_meters[:, 1], gt_xyz_meters[:, 2]
        ax_traj.plot3D(gx, gy, gz, color="#2066E0", linewidth=3, label="GT")
        # 标记终点为 Goal
        gx_f, gy_f, gz_f = gx[-1], gy[-1], gz[-1]
        ax_traj.scatter(gx_f, gy_f, gz_f, c="#2066E0", s=40, depthshade=True)
        ax_traj.text(gx_f, gy_f, gz_f, "Goal", color="#2066E0", fontsize=9)

    # 再画 candidates 的 3D 轨迹
    if cand_traj_points_list is not None:
        for i, info in enumerate(cand_infos):
            cid = info["id"]
            # 防御：cid 可能为 -1 或超界
            if cid is None or cid < 0 or cid >= cand_traj_points_list.shape[0]:
                continue

            traj_points = cand_traj_points_list[cid]  # (T+1, 3) 米
            xs, ys, zs = traj_points[:, 0], traj_points[:, 1], traj_points[:, 2]

            if i == best_idx:
                traj_color = selected_color
            else:
                traj_color = other_colors[(i - 1) % len(other_colors)]

            ax_traj.plot3D(xs, ys, zs, color=traj_color, linewidth=3)

            # 标签放在 1/3 处，附加一点偏移，避免重合
            if len(xs) > 0:
                mid_idx = max(1, len(xs) // 3)
                x_label, y_label, z_label = xs[mid_idx], ys[mid_idx], zs[mid_idx]

                if len(xs) > 1:
                    traj_dir = np.array([xs[-1] - xs[0], ys[-1] - ys[0], zs[-1] - zs[0]])
                    norm = np.linalg.norm(traj_dir)
                    if norm > 1e-6:
                        traj_dir = traj_dir / norm
                    else:
                        traj_dir = np.array([1, 0, 0], dtype=np.float32)
                else:
                    traj_dir = np.array([1, 0, 0], dtype=np.float32)

                perp_offset = 0.08 * (i + 1)
                perp_vec = np.cross(traj_dir, np.array([0, 0, 1], dtype=np.float32))
                if np.linalg.norm(perp_vec) < 1e-6:
                    perp_vec = np.cross(traj_dir, np.array([0, 1, 0], dtype=np.float32))
                perp_vec = perp_vec / (np.linalg.norm(perp_vec) + 1e-6)
                x_label += perp_vec[0] * perp_offset
                y_label += perp_vec[1] * perp_offset
                z_label += perp_vec[2] * perp_offset

                rank = info["rank"]
                if rank is not None and rank >= 0:
                    p_label = f"P{rank + 1}"
                else:
                    p_label = f"P{i + 1}"

                # 先绘制白色描边，再绘制前景文字
                for dx, dy, dz in [(-0.002, -0.002, -0.002), (-0.002, 0.002, -0.002),
                                   (0.002, -0.002, -0.002), (0.002, 0.002, -0.002),
                                   (-0.002, -0.002, 0.002), (-0.002, 0.002, 0.002),
                                   (0.002, -0.002, 0.002), (0.002, 0.002, 0.002)]:
                    ax_traj.text(
                        x_label + dx, y_label + dy, z_label + dz,
                        p_label, fontsize=12, color="white", weight="bold", alpha=0.8
                    )
                ax_traj.text(
                    x_label, y_label, z_label,
                    p_label, fontsize=12, color=traj_color, weight="bold"
                )

    ax_traj.set_title("Trajectories (3D)")
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.set_zlabel("Z (m)")
    ax_traj.view_init(elev=22, azim=-60)
    ax_traj.grid(True, alpha=0.2)

    # 右侧：Candidates 中间帧 + label (P*, LPIPS, DreamSim, APE)
    axes_cands = [
        fig.add_subplot(1, ncols, 3 + i) for i in range(len(cand_mid_imgs))
    ]

    for i, (img, info, ax) in enumerate(zip(cand_mid_imgs, cand_infos, axes_cands)):
        ax.imshow(img)
        ax.axis('off')

        rank = info["rank"]
        lpips_loss = info["lpips"]
        ape = info["ape"]
        dreamsim = info["dreamsim"]

        if rank is not None and rank >= 0:
            p_label = f"P{rank + 1}"
        else:
            p_label = f"P{i + 1}"

        # 文本框，增加 DreamSim 指标
        text_str = (
            f"{p_label}\n"
            f"LPIPS: {lpips_loss:.3f}\n"
            f"DS: {dreamsim:.3f}\n"
            f"APE: {ape:.2f}"
        )
        ax.text(
            0.5, 0.02, text_str,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            color="black",
            bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.25")
        )

        # P1 (best) 画绿色粗框
        if i == best_idx:
            rect = patches.Rectangle(
                (0, 0), 1, 1,
                transform=ax.transAxes,
                fill=False,
                linewidth=4,
                edgecolor=selected_color
            )
            ax.add_patch(rect)

    plt.tight_layout()

    # 保存
    output_path = os.path.join(output_dir, f"{run_name}_mid_compare.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def get_dataset_eval(config, dataset_name, predefined_index=True):
    """从 sample_trajectories.py 复制的函数"""
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
        traj_names="traj_names.txt"
    )
    return dataset


def load_candidate_trajs(dataset_name, input_dir):
    """
    从 RULE_N*_K*_... 目录名中解析 N，并加载对应的 candidate 轨迹：
    data_splits/{dataset_name}/test/{dataset_name}_{N}_trajectories_long.pkl
    """
    eval_dir = os.path.basename(os.path.dirname(input_dir))
    m = re.search(r"N(\d+)", eval_dir)
    if not m:
        raise RuntimeError(f"Cannot parse N from eval dir name: {eval_dir}")
    num_samples = int(m.group(1))
    pkl_path = f"data_splits/{dataset_name}/test/{dataset_name}_{num_samples}_trajectories_long.pkl"
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"candidate traj pkl not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        candidate_trajs = pickle.load(f)
    print(f"Loaded candidate trajectories from {pkl_path} (num_samples={num_samples})")
    return candidate_trajs


def main():
    # 加载配置和数据集
    with open("config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config

    with open("config/nwm_cdit_airvln_16.yaml", "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    dataset_name = "airvln_16"
    dataset = get_dataset_eval(config, dataset_name, predefined_index=True)
    print(f"Loaded dataset with {len(dataset)} samples")

    # 输入目录：editor/run_xxx
    input_dir = "/data1/tpz/nwm-main/results/nwm_cdit_airvln_16/airvln_16/RULE_N3_K3_RS1_rep1_OPT11141/editor"

    # 加载 candidate 轨迹 (米单位的 x,y,z,yaw)
    candidate_trajs = load_candidate_trajs(dataset_name, input_dir)

    # 输出目录
    output_dir = os.path.join(os.path.dirname(input_dir), "mid_frame_comparison")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    # 处理所有 run 目录
    run_dirs = sorted(glob(os.path.join(input_dir, "run_*")))
    print(f"Found {len(run_dirs)} run directories")

    success_count = 0
    for run_dir in run_dirs:
        result = process_run(run_dir, output_dir, dataset, config, candidate_trajs, dataset_name)
        if result:
            success_count += 1

    print("-" * 60)
    print(f"Completed: {success_count}/{len(run_dirs)} runs processed successfully")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
