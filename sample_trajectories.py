import yaml, numpy as np, pickle, os, torch
from datasets_v4 import TrajectoryEvalDataset
from misc import transform
from rule_based_trajectory_generation import trajectory_generation_rule_based
from noise_trajectory_generation import trajectory_generation_random

def get_dataset_eval(config, dataset_name, predefined_index=True):
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

def sample_trajectories(dim=3, input_fps=4, output_fps=1):
    with open("config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config

    with open("config/nwm_cdit_airvln_16.yaml", "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    # 计算降采样步长
    stride = input_fps // output_fps
    print(f"Downsampling from {input_fps}fps to {output_fps}fps with stride={stride}")

    dataset_name, candidate_number = "airvln_16", 5
    dataset = get_dataset_eval(config, dataset_name, predefined_index=True)
    all_sampled_trajectories = {}
    for i in range(len(dataset)):
        idxs, _, _, gt_actions, _, _ = dataset[i]
        
        # 降采样：将每 stride 个时间步的动作累积起来
        # 与 isolated_nwm_infer_v5.py 中的 generate_rollout 逻辑一致：
        # delta = delta.unflatten(1, (-1, rollout_stride)).sum(2)
        # gt_image = gt_image[:, rollout_stride-1::rollout_stride] (从0.75s开始采样)
        # gt_actions shape: [T, 4]
        if isinstance(gt_actions, torch.Tensor):
            gt_actions_tensor = gt_actions
        else:
            gt_actions_tensor = torch.from_numpy(gt_actions)
        
        T = gt_actions_tensor.shape[0]
        # 将 T 调整为 stride 的倍数
        T_aligned = (T // stride) * stride
        gt_actions_aligned = gt_actions_tensor[:T_aligned]  # [T_aligned, 4]
        
        # 降采样：将每 stride 个时间步的动作累积起来
        # 从索引 0 开始采样，实现秒数对齐（0s, 1s, 2s, ...）
        # 对应的时间点：0s, 1s, 2s, 3s, ... (即 0, stride, 2*stride, 3*stride, ...)
        # 每个时间点的动作是前 stride 个动作的累积：delta[0:4], delta[4:8], delta[8:12], ...
        gt_actions_reshaped = gt_actions_aligned.unflatten(0, (T_aligned // stride, stride))  # [T_down, stride, 4]
        gt_actions_downsampled = gt_actions_reshaped.sum(dim=1)  # [T_down, 4]
        
        # 转换为 numpy
        gt_deltas_xyz = gt_actions_downsampled[:, :3].cpu().numpy()  # [T_down, 3]
        gt_xyz = np.concatenate([np.zeros((1, 3), dtype=np.float32),
                                np.cumsum(gt_deltas_xyz, axis=0).astype(np.float32)], axis=0)  # [T_down+1, 3]
        gt_yaw = np.zeros((gt_xyz.shape[0],), dtype=np.float32)
        GT_traj = [(float(x), float(y), float(z), float(yaw)) for (x, y, z), yaw in zip(gt_xyz, gt_yaw)]

        candidate_trajectories = trajectory_generation_random(GT_traj, candidate_number=1, dim=dim) + \
                                trajectory_generation_rule_based(GT_traj, candidate_number=candidate_number-1, dim=dim)
        candidate_trajectories = np.array(candidate_trajectories, dtype=np.float32)  # [N, T, 4]
        all_sampled_trajectories[int(idxs.item())] = candidate_trajectories
        print(f"Processed {i+1}/{len(dataset)} trajectories", end='\r')

    save_path = f"data_splits/{dataset_name}/test/{dataset_name}_{candidate_number}_trajectories_long.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pickle.dump(all_sampled_trajectories, open(save_path, "wb"))
    print(f"Saved {len(all_sampled_trajectories)} sampled trajectories to {save_path}")

if __name__ == "__main__":
    sample_trajectories(dim=2, input_fps=4, output_fps=1)
