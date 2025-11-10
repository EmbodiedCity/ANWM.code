import yaml, numpy as np, pickle, os, torch
from datasets_v4 import TrajectoryEvalDataset
from misc import transform
from rule_based_trajectory_generation import trajectory_generation_rule_based
from noise_trajectory_generation import trajectory_generation_random

def get_dataset_eval(config, dataset_name, predefined_index=True):
    data_config = config["eval_datasets"][dataset_name]
    if predefined_index:
        predefined_index = f"data_splits/{dataset_name}/test/navigation_eval_16.pkl"
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

with open("config/eval_config.yaml", "r") as f:
    default_config = yaml.safe_load(f)
config = default_config

with open("config/nwm_cdit_airvln_16.yaml", "r") as f:
    user_config = yaml.safe_load(f)
config.update(user_config)

dataset_name, candidate_number = "airvln_16", 5
dataset = get_dataset_eval(config, dataset_name, predefined_index=True)
all_sampled_trajectories = {}
for i in range(len(dataset)):
    idxs, _, _, gt_actions, _, _ = dataset[i]
    gt_deltas_xyz = gt_actions[:, :3].to('cpu').numpy()  # [T, 3]
    gt_xyz = np.concatenate([np.zeros((1, 3), dtype=np.float32),
                             np.cumsum(gt_deltas_xyz, axis=0).astype(np.float32)], axis=0)  # [T+1, 3]
    gt_yaw = np.zeros((gt_xyz.shape[0],), dtype=np.float32)
    GT_traj = [(float(x), float(y), float(z), float(yaw)) for (x, y, z), yaw in zip(gt_xyz, gt_yaw)]

    candidate_trajectories = trajectory_generation_random(GT_traj, candidate_number=1) + \
                             trajectory_generation_rule_based(GT_traj, candidate_number=candidate_number-1)
    candidate_trajectories = np.array(candidate_trajectories, dtype=np.float32)  # [N, T, 4]
    all_sampled_trajectories[int(idxs.item())] = candidate_trajectories

save_path = f"data_splits/{dataset_name}/test/{dataset_name}_{candidate_number}_trajectories.pkl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
pickle.dump(all_sampled_trajectories, open(save_path, "wb"))
print(f"Saved {len(all_sampled_trajectories)} sampled trajectories to {save_path}")

