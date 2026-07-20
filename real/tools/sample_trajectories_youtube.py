#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import pickle
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Dict, List, Sequence, Tuple

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
ANWM_MAIN_DIR = Path(os.environ.get("ANWM_REPO_ROOT", REPO_ROOT))
DEFAULT_SPLITS_ROOT = REPO_ROOT / "data" / "splits"
DEFAULT_PRESET = "sekai_new"
DEFAULT_PRESETS = {
    "sekai_new": REPO_ROOT / "data" / "sekai_new",
}
DEFAULT_TEST_TRAJ_FILES = {
    "sekai_new": REPO_ROOT
    / "data"
    / "splits"
    / "sekai_new"
    / "test"
    / "traj_names.txt",
}
DEFAULT_SOURCE_INDEX_PKLS = {
    "sekai_new": REPO_ROOT
    / "data"
    / "splits"
    / "sekai_new"
    / "test"
    / "dataset_dist_-64_to_64_n4_len_traj_pred_32.pkl",
}
LOCAL_DATASET_DEFAULTS = {
    "sekai_new": {
        "metric_waypoint_spacing": 0.025,
        "source_context_size": 4,
        "source_len_traj_pred": 32,
        "source_min_dist_cat": -64,
        "source_max_dist_cat": 64,
        "source_traj_stride": 80,
        "input_fps": 4.0,
        "output_fps": 1.0,
        "paper_num_candidates": 5,
    },
}

ACTION_SPACE = {
    "forward": (0.025, 0.0, 0.0, 0.0),
    "backward": (-0.025, 0.0, 0.0, 0.0),
    "up": (0.0, 0.0, 0.025, 0.0),
    "down": (0.0, 0.0, -0.025, 0.0),
    "left": (0.0, 0.0, 0.0, np.pi / 192.0),
    "right": (0.0, 0.0, 0.0, -np.pi / 192.0),
}
YOUTUBE_POS_STD = 0.01
YOUTUBE_YAW_STD = 0.01

RULE_SEQUENCES_3D = [
    ["left", "forward", "up", "forward"],
    ["right", "forward", "up", "forward"],
    ["left", "forward", "down", "forward"],
    ["right", "forward", "down", "forward"],
]

RULE_SEQUENCES_2D = [
    ["left", "forward"],
    ["right", "forward"],
    ["left", "left", "forward", "left"],
    ["right", "right", "forward", "right"],
    ["left", "forward", "right", "forward"],
]


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def angle_difference(theta1: float, theta2: np.ndarray) -> np.ndarray:
    delta_theta = theta2 - theta1
    return delta_theta - 2.0 * np.pi * np.floor((delta_theta + np.pi) / (2.0 * np.pi))


def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] != 3:
        raise ValueError(f"Unexpected position dim: {positions.shape}")
    return (positions - curr_pos).dot(rotmat)


def atomic_pickle_dump(obj: object, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, dst)


def read_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_index_entries(path: Path) -> List[Tuple[str, int, int, int]]:
    obj = read_pickle(path)
    if isinstance(obj, tuple):
        if not obj:
            raise ValueError(f"Empty tuple index pkl: {path}")
        obj = obj[0]
    if not isinstance(obj, list):
        raise TypeError(
            f"Expected list or tuple(list, ...), got {type(obj)} from {path}"
        )
    out: List[Tuple[str, int, int, int]] = []
    for item in obj:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise ValueError(f"Bad index item in {path}: {item!r}")
        out.append((str(item[0]), int(item[1]), int(item[2]), int(item[3])))
    return out


def load_traj_names(data_root: Path) -> List[str]:
    traj_names_path = data_root / "traj_names.txt"
    if traj_names_path.exists():
        names = [
            line.strip()
            for line in traj_names_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        names = sorted(
            d.name
            for d in data_root.iterdir()
            if d.is_dir() and (d / "traj_data.pkl").exists()
        )
    if not names:
        raise ValueError(f"No trajectory names found under {data_root}")
    return names


def load_named_traj_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"traj_names file not found: {path}")
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"traj_names file is empty: {path}")
    return names


def load_traj_data(data_root: Path, traj_name: str) -> dict:
    traj_pkl = data_root / traj_name / "traj_data.pkl"
    if not traj_pkl.exists():
        raise FileNotFoundError(f"Missing traj_data.pkl: {traj_pkl}")
    traj_data = read_pickle(traj_pkl)
    if "point" not in traj_data or "yaw" not in traj_data:
        raise KeyError(f"{traj_pkl} must contain `point` and `yaw`")
    return traj_data


def get_traj_length(data_root: Path, traj_name: str) -> int:
    traj_dir = data_root / traj_name
    n_images = sum(
        1 for p in traj_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"
    )
    if n_images > 0:
        return int(n_images)

    traj_data = load_traj_data(data_root, traj_name)
    return int(np.asarray(traj_data["point"]).shape[0])


def ensure_anwm_importable() -> None:
    anwm_main_str = str(ANWM_MAIN_DIR.resolve())
    if anwm_main_str not in sys.path:
        sys.path.insert(0, anwm_main_str)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")


@contextmanager
def pushd(path: Path):
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def import_anwm_module(module_name: str):
    ensure_anwm_importable()
    with pushd(ANWM_MAIN_DIR):
        return importlib.import_module(module_name)


def get_local_dataset_default(dataset_name: str, key: str, fallback):
    return LOCAL_DATASET_DEFAULTS.get(dataset_name, {}).get(key, fallback)


def resolve_source_index_build_params(
    args, dataset_name: str
) -> Tuple[int, int, int, int, int]:
    source_len_traj_pred = int(
        args.source_len_traj_pred
        if args.source_len_traj_pred is not None
        else get_local_dataset_default(
            dataset_name, "source_len_traj_pred", args.len_traj_pred
        )
    )
    source_min_dist_cat = int(
        args.source_min_dist_cat
        if args.source_min_dist_cat is not None
        else get_local_dataset_default(
            dataset_name, "source_min_dist_cat", source_len_traj_pred
        )
    )
    source_max_dist_cat = int(
        args.source_max_dist_cat
        if args.source_max_dist_cat is not None
        else get_local_dataset_default(
            dataset_name, "source_max_dist_cat", source_min_dist_cat
        )
    )
    source_context_size = int(
        args.source_context_size
        if args.source_context_size is not None
        else get_local_dataset_default(dataset_name, "source_context_size", 4)
    )
    source_traj_stride = int(
        args.source_traj_stride
        if args.source_traj_stride is not None
        else get_local_dataset_default(dataset_name, "source_traj_stride", 80)
    )
    if source_max_dist_cat < source_min_dist_cat:
        raise ValueError(
            f"source_max_dist_cat must be >= source_min_dist_cat, got {source_max_dist_cat} < {source_min_dist_cat}"
        )
    return (
        source_min_dist_cat,
        source_max_dist_cat,
        source_context_size,
        source_len_traj_pred,
        source_traj_stride,
    )


def get_source_index_path(split_dir: Path, args, dataset_name: str) -> Path:
    (
        source_min_dist_cat,
        source_max_dist_cat,
        source_context_size,
        source_len_traj_pred,
        _,
    ) = resolve_source_index_build_params(args, dataset_name)
    return split_dir / (
        f"dataset_dist_{source_min_dist_cat}_to_{source_max_dist_cat}"
        f"_n{source_context_size}_len_traj_pred_{source_len_traj_pred}.pkl"
    )


def get_eval_index_filename(args) -> str:
    if args.artifact_naming == "planning_compat":
        return "navigation_eval_16_long.pkl"
    return "navigation_eval.pkl"


def get_candidate_filename(dataset_name: str, args) -> str:
    if args.artifact_naming == "planning_compat":
        return f"{dataset_name}_{int(args.num_samples)}_trajectories_long.pkl"
    return "trajectory_candidates.pkl"


def build_source_index_with_main(
    *,
    dataset_name: str,
    data_root: Path,
    split_dir: Path,
    traj_names: Sequence[str],
    args,
) -> Path:
    dataset_module = import_anwm_module("real.dataset")
    source_index_path = get_source_index_path(split_dir, args, dataset_name)
    (
        source_min_dist_cat,
        source_max_dist_cat,
        source_context_size,
        source_len_traj_pred,
        source_traj_stride,
    ) = resolve_source_index_build_params(args, dataset_name)

    class IndexBuildShim:
        pass

    shim = IndexBuildShim()
    shim.dataset_name = dataset_name
    shim.data_folder = str(data_root)
    shim.data_split_folder = str(split_dir)
    shim.traj_names = list(traj_names)
    shim.min_dist_cat = source_min_dist_cat
    shim.max_dist_cat = source_max_dist_cat
    shim.context_size = source_context_size
    shim.len_traj_pred = source_len_traj_pred
    shim.traj_stride = source_traj_stride
    shim._get_trajectory = MethodType(dataset_module.BaseDataset._get_trajectory, shim)

    samples_index, goals_index = dataset_module.BaseDataset._build_index(
        shim, use_tqdm=True
    )
    atomic_pickle_dump((samples_index, goals_index), source_index_path)
    print(
        f"[{dataset_name}] built source index via real.dataset: {source_index_path} "
        f"(samples={len(samples_index)}, goals={len(goals_index)})"
    )
    return source_index_path


def extract_gt_trajectory_local(
    traj_data: dict,
    *,
    curr_time: int,
    len_traj_pred: int,
) -> np.ndarray:
    start_index = int(curr_time)
    end_index = int(curr_time) + int(len_traj_pred) + 1

    positions = np.asarray(traj_data["point"], dtype=np.float32)[start_index:end_index]
    yaw = np.asarray(traj_data["yaw"], dtype=np.float32)[start_index:end_index].reshape(
        -1
    )
    if (
        positions.shape[0] != int(len_traj_pred) + 1
        or yaw.shape[0] != int(len_traj_pred) + 1
    ):
        raise ValueError(
            f"Not enough future frames: curr_time={curr_time}, len_traj_pred={len_traj_pred}, "
            f"positions={positions.shape}, yaw={yaw.shape}"
        )

    waypoints_pos = to_local_coords(positions, positions[0], float(yaw[0]))
    waypoints_yaw = angle_difference(float(yaw[0]), yaw)
    gt_local = np.concatenate([waypoints_pos, waypoints_yaw[:, None]], axis=-1).astype(
        np.float32
    )
    gt_local[0] = 0.0
    gt_local[:, 3] = 0.0
    return gt_local


def rule_based_trajectory(
    start: np.ndarray,
    steps: int,
    rule_sequence: Sequence[str],
) -> np.ndarray:
    trajectory = [np.asarray(start[:4], dtype=np.float32)]
    for step in range(int(steps)):
        last_pos = trajectory[-1]
        action_choice = rule_sequence[step % len(rule_sequence)]
        action = np.asarray(ACTION_SPACE[action_choice], dtype=np.float32)
        if action_choice in {"left", "right", "up", "down"}:
            current_pos = last_pos + action
        else:
            dx = action[0] * np.cos(float(last_pos[3]))
            dy = action[0] * np.sin(float(last_pos[3]))
            current_pos = last_pos + np.array([dx, dy, 0.0, 0.0], dtype=np.float32)
        trajectory.append(current_pos.astype(np.float32))

    trajectory_np = np.asarray(trajectory, dtype=np.float32)
    trajectory_np -= trajectory_np[0]
    trajectory_np[0] = 0.0
    return trajectory_np


def trajectory_generation_rule_based(
    gt_trajectory: np.ndarray,
    *,
    candidate_number: int,
    dim: int,
    py_rng: random.Random,
) -> List[np.ndarray]:
    if candidate_number <= 0:
        return []

    if dim == 3:
        sequences = list(RULE_SEQUENCES_3D)
    elif dim == 2:
        sequences = list(RULE_SEQUENCES_2D)
    else:
        raise ValueError(f"Invalid dim={dim}, expected 2 or 3")

    py_rng.shuffle(sequences)
    start_pos = np.asarray(gt_trajectory[0], dtype=np.float32).copy()
    start_pos[3] = float(gt_trajectory[-1][3])
    steps = int(len(gt_trajectory) - 1)

    candidates: List[np.ndarray] = []
    seq_idx = 0
    while len(candidates) < candidate_number:
        rule_seq = sequences[seq_idx % len(sequences)]
        candidates.append(rule_based_trajectory(start_pos, steps, rule_seq))
        seq_idx += 1
    return candidates


def random_trajectory_v2(
    gt_traj: np.ndarray,
    *,
    rng: np.random.Generator,
    dim: int,
    pos_std: float = YOUTUBE_POS_STD,
    yaw_std: float = YOUTUBE_YAW_STD,
    eps: float = 1e-6,
) -> np.ndarray:
    gt_traj = np.asarray(gt_traj, dtype=np.float32)
    noise_traj = [gt_traj[0].copy()]

    for i in range(1, len(gt_traj)):
        delta = gt_traj[i] - gt_traj[i - 1]
        delta_noisy = delta.copy()

        for j in range(4):
            if dim == 2 and j == 2:
                continue
            std = pos_std if j < 3 else yaw_std
            delta_noisy[j] += float(rng.normal(0.0, std))

        pos_delta = delta[:3]
        if np.linalg.norm(pos_delta) < eps:
            delta_noisy[0] = 0.0
            delta_noisy[1] = 0.0
            delta_noisy[2] = 0.0
        if abs(float(pos_delta[0])) < eps and abs(float(pos_delta[1])) < eps:
            delta_noisy[0] = 0.0
            delta_noisy[1] = 0.0
        if abs(float(pos_delta[2])) < eps:
            delta_noisy[2] = 0.0

        noise_traj.append(noise_traj[-1] + delta_noisy)

    noise_traj_np = np.asarray(noise_traj, dtype=np.float32)
    noise_traj_np -= noise_traj_np[0]
    noise_traj_np[0] = 0.0
    return noise_traj_np


def trajectory_generation_random(
    gt_trajectory: np.ndarray,
    *,
    candidate_number: int,
    dim: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    return [
        random_trajectory_v2(gt_trajectory, rng=rng, dim=dim)
        for _ in range(int(candidate_number))
    ]


def downsample_candidate_trajectories(
    candidate_trajectories: Sequence[np.ndarray], stride: int
) -> np.ndarray:
    trajectories = np.asarray(candidate_trajectories, dtype=np.float32)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 4:
        raise ValueError(
            f"Expected candidate trajectories with shape [N, T+1, 4], got {trajectories.shape}"
        )

    steps = trajectories.shape[1] - 1
    aligned_steps = (steps // int(stride)) * int(stride)
    if aligned_steps <= 0:
        raise ValueError(
            f"Trajectory too short for downsampling: steps={steps}, stride={stride}"
        )

    aligned = trajectories[:, : aligned_steps + 1, :]
    downsampled: List[np.ndarray] = []
    for traj in aligned:
        deltas = traj[1:] - traj[:-1]
        deltas = deltas.reshape(aligned_steps // int(stride), int(stride), 4).sum(
            axis=1
        )
        traj_start = traj[:1]
        traj_cumsum = np.cumsum(deltas, axis=0)
        traj_ds = np.concatenate([traj_start, traj_cumsum + traj_start], axis=0)
        downsampled.append(traj_ds.astype(np.float32))

    return np.asarray(downsampled, dtype=np.float32)


def downsample_single_trajectory(trajectory: np.ndarray, stride: int) -> np.ndarray:
    return downsample_candidate_trajectories(
        [np.asarray(trajectory, dtype=np.float32)], int(stride)
    )[0]


def generate_candidate_trajectories_for_item(
    *,
    data_root: str,
    traj_name: str,
    curr_time: int,
    len_traj_pred: int,
    dim: int,
    num_samples: int,
    stride: int,
    np_seed: int,
    py_seed: int,
    traj_data: dict | None = None,
) -> np.ndarray:
    if traj_data is None:
        traj_data = load_traj_data(Path(data_root), traj_name)
    gt_traj = extract_gt_trajectory_local(
        traj_data,
        curr_time=int(curr_time),
        len_traj_pred=int(len_traj_pred),
    )
    rng = np.random.default_rng(int(np_seed))
    py_rng = random.Random(int(py_seed))
    candidate_trajectories = trajectory_generation_random(
        gt_traj,
        candidate_number=1,
        dim=int(dim),
        rng=rng,
    ) + trajectory_generation_rule_based(
        gt_traj,
        candidate_number=max(0, int(num_samples) - 1),
        dim=int(dim),
        py_rng=py_rng,
    )
    candidate_trajectories = candidate_trajectories[: int(num_samples)]
    candidate_trajectories = downsample_candidate_trajectories(
        candidate_trajectories, int(stride)
    )

    expected_future_points = int(len_traj_pred) // int(stride)
    expected_total_points = expected_future_points + 1  # include the origin
    if candidate_trajectories.shape[1] != expected_total_points:
        raise ValueError(
            "Unexpected candidate trajectory length after downsampling: "
            f"got {candidate_trajectories.shape[1]} total points, "
            f"expected {expected_total_points} (= 1 start + {expected_future_points} future points)"
        )
    return candidate_trajectories


def resolve_dataset_specs(args) -> List[Tuple[str, Path]]:
    if args.dataset_name or args.data_root:
        if not args.dataset_name or not args.data_root:
            raise ValueError(
                "`--dataset-name` and `--data-root` must be provided together"
            )
        return [(str(args.dataset_name), Path(args.data_root).expanduser().resolve())]

    if args.preset:
        if isinstance(args.preset, str):
            presets = [args.preset]
        else:
            presets = list(args.preset)
    else:
        presets = [DEFAULT_PRESET]
    return [(name, DEFAULT_PRESETS[name].resolve()) for name in presets]


def resolve_test_traj_names(args, dataset_name: str, data_root: Path) -> List[str]:
    if args.test_traj_names:
        traj_names_path = Path(args.test_traj_names).expanduser()
        if not traj_names_path.is_absolute():
            traj_names_path = (data_root / traj_names_path).resolve()
        return load_named_traj_file(traj_names_path)

    auto_test_file = data_root / "test_traj_name.txt"
    if auto_test_file.exists():
        return load_named_traj_file(auto_test_file)

    default_test_file = DEFAULT_TEST_TRAJ_FILES.get(dataset_name)
    if default_test_file is not None and default_test_file.exists():
        return load_named_traj_file(default_test_file)

    if args.use_all_trajectories_as_test:
        return load_traj_names(data_root)

    raise ValueError(
        f"Missing test split for {dataset_name}. "
        f"Expected a file like {auto_test_file}."
    )


def resolve_source_index_pkl(args, data_root: Path) -> Path:
    if not args.source_index_pkl:
        raise ValueError(
            "source index pkl is required for old-chain aligned sampling; pass --source-index-pkl"
        )
    p = Path(args.source_index_pkl).expanduser()
    if not p.is_absolute():
        p = (data_root / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"source index pkl not found: {p}")
    return p


def sample_eval_index_from_source(
    source_index: Sequence[Tuple[str, int, int, int]],
    *,
    allowed_traj_names: Sequence[str],
    sample_size: int,
    seed: int,
    forward_min_goal: int,
) -> List[Tuple[str, int, int, int]]:
    allowed = set(allowed_traj_names)
    filtered: List[Tuple[str, int, int, int]] = []
    for item in source_index:
        traj_name, curr_time, min_goal_distance, max_goal_distance = item
        if traj_name not in allowed:
            continue
        min_goal_distance = int(min_goal_distance)
        max_goal_distance = int(max_goal_distance)
        if int(forward_min_goal) > 0:
            min_goal_distance = max(min_goal_distance, int(forward_min_goal))
        if min_goal_distance > max_goal_distance:
            continue
        filtered.append(
            (str(traj_name), int(curr_time), min_goal_distance, max_goal_distance)
        )
    if not filtered:
        raise ValueError(
            "No valid source index entries remain after filtering by test trajectories"
        )

    if sample_size > 0:
        if sample_size > len(filtered):
            raise ValueError(
                f"Requested sample_size={sample_size}, but only {len(filtered)} entries match the test split"
            )
        rng = random.Random(seed)
        return rng.sample(filtered, k=int(sample_size))
    return list(filtered)


def extract_traj_names_from_index(
    eval_index: Sequence[Tuple[str, int, int, int]],
) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in eval_index:
        traj_name = str(item[0])
        if traj_name in seen:
            continue
        seen.add(traj_name)
        ordered.append(traj_name)
    return ordered


def maybe_keep_first_slice_per_traj(
    eval_index: Sequence[Tuple[str, int, int, int]],
    dataset_name: str,
) -> List[Tuple[str, int, int, int]]:
    if dataset_name != "sekai_huge":
        return [
            (str(item[0]), int(item[1]), int(item[2]), int(item[3]))
            for item in eval_index
        ]

    first_only: List[Tuple[str, int, int, int]] = []
    seen = set()
    for item in eval_index:
        traj_name = str(item[0])
        if traj_name in seen:
            continue
        seen.add(traj_name)
        first_only.append((traj_name, int(item[1]), int(item[2]), int(item[3])))
    return first_only


def maybe_write_traj_names(split_dir: Path, traj_names: Sequence[str]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    content = "".join(f"{name}\n" for name in traj_names)
    traj_names_path = split_dir / "traj_names.txt"
    if (
        not traj_names_path.exists()
        or traj_names_path.read_text(encoding="utf-8") != content
    ):
        traj_names_path.write_text(content, encoding="utf-8")


def process_dataset(spec_idx: int, dataset_name: str, data_root: Path, args) -> None:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    split_dir = Path(args.splits_root).expanduser().resolve() / dataset_name / "test"
    split_dir.mkdir(parents=True, exist_ok=True)
    allowed_traj_names = resolve_test_traj_names(args, dataset_name, data_root)
    maybe_write_traj_names(split_dir, allowed_traj_names)

    index_path = split_dir / get_eval_index_filename(args)
    if args.source_index_pkl:
        source_index_path = resolve_source_index_pkl(args, data_root)
    else:
        source_index_path = get_source_index_path(split_dir, args, dataset_name)
        if args.rebuild_source_index or not source_index_path.exists():
            source_index_path = build_source_index_with_main(
                dataset_name=dataset_name,
                data_root=data_root,
                split_dir=split_dir,
                traj_names=allowed_traj_names,
                args=args,
            )
        else:
            print(f"[{dataset_name}] reuse source index: {source_index_path}")

    if args.source_index_pkl or args.rebuild_index or not index_path.exists():
        source_index = load_index_entries(source_index_path)
        eval_index = sample_eval_index_from_source(
            source_index,
            allowed_traj_names=allowed_traj_names,
            sample_size=int(args.sample_size),
            seed=int(args.seed) + spec_idx,
            forward_min_goal=int(args.forward_min_goal),
        )
        eval_index = maybe_keep_first_slice_per_traj(eval_index, dataset_name)
        atomic_pickle_dump(eval_index, index_path)
        print(
            f"[{dataset_name}] wrote eval index from source pkl: {index_path} "
            f"(source={source_index_path}, selected={len(eval_index)})"
        )
    elif index_path.exists() and not args.rebuild_index:
        eval_index = load_index_entries(index_path)
        eval_index = maybe_keep_first_slice_per_traj(eval_index, dataset_name)
        print(
            f"[{dataset_name}] reuse eval index: {index_path} ({len(eval_index)} samples)"
        )
    else:
        raise ValueError(
            f"[{dataset_name}] missing eval index pkl: {index_path}. "
            "Pass --source-index-pkl <pkl> or let the script build dataset_dist_*.pkl first, "
            "then sample and write the eval index pkl."
        )

    split_traj_names = extract_traj_names_from_index(eval_index)
    maybe_write_traj_names(split_dir, split_traj_names)

    if args.index_only:
        print(f"[{dataset_name}] index-only mode, skip candidate trajectory generation")
        return

    ratio = float(args.input_fps) / float(args.output_fps)
    stride = int(round(ratio))
    if not np.isclose(ratio, stride):
        raise ValueError(
            f"input_fps/output_fps must be an integer ratio, got {args.input_fps}/{args.output_fps}"
        )
    if stride <= 0:
        raise ValueError(f"Invalid downsample stride: {stride}")

    out_path = split_dir / get_candidate_filename(dataset_name, args)
    if args.viz_only:
        if not out_path.exists():
            raise FileNotFoundError(
                f"[{dataset_name}] candidate pkl not found for --viz-only: {out_path}"
            )
        print(
            f"[{dataset_name}] viz-only mode, reuse existing candidate pkl: {out_path}"
        )
    elif out_path.exists() and not args.overwrite:
        print(f"[{dataset_name}] reuse existing candidate pkl: {out_path}")
        if not args.visualize:
            return
    else:
        all_sampled_trajectories: Dict[int, np.ndarray] = {}
        cached_traj_name: str | None = None
        cached_traj_data: dict | None = None
        print(
            f"[{dataset_name}] generating candidate trajectories for {len(eval_index)} eval samples",
            flush=True,
        )
        for sample_id, item in enumerate(eval_index):
            traj_name, curr_time, _, _ = item
            if str(traj_name) != cached_traj_name:
                cached_traj_name = str(traj_name)
                cached_traj_data = load_traj_data(data_root, cached_traj_name)
            np_seed = int(args.seed) + spec_idx * 100000 + sample_id * 2
            py_seed = np_seed + 1
            all_sampled_trajectories[int(sample_id)] = (
                generate_candidate_trajectories_for_item(
                    data_root=str(data_root),
                    traj_name=str(traj_name),
                    curr_time=int(curr_time),
                    len_traj_pred=int(args.len_traj_pred),
                    dim=int(args.dim),
                    num_samples=int(args.num_samples),
                    stride=int(stride),
                    np_seed=np_seed,
                    py_seed=py_seed,
                    traj_data=cached_traj_data,
                )
            )
            print(
                f"[{dataset_name}] processed {sample_id + 1}/{len(eval_index)} "
                f"(sample_id={sample_id}, traj={traj_name})",
                flush=True,
            )

        atomic_pickle_dump(all_sampled_trajectories, out_path)
        print(f"[{dataset_name}] wrote candidates: {out_path}")

    if args.visualize:
        visualize_candidate_trajectories_for_dataset(
            dataset_name=dataset_name,
            data_root=data_root,
            split_dir=split_dir,
            args=args,
            stride=int(stride),
        )


def ensure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_candidate_trajectories(path: Path) -> Dict[int, np.ndarray]:
    obj = read_pickle(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in candidate pkl, got {type(obj)} from {path}")

    candidates: Dict[int, np.ndarray] = {}
    for key, value in obj.items():
        sample_id = int(key)
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 3 or array.shape[-1] != 4:
            raise ValueError(
                f"Expected candidate trajectories with shape [N, T+1, 4], got {array.shape} for sample_id={sample_id}"
            )
        candidates[sample_id] = array
    return candidates


def get_candidate_label(candidate_idx: int) -> str:
    if candidate_idx == 0:
        return "cand0 noise"
    return f"cand{candidate_idx}"


def compute_plot_limits(
    gt_trajectory: np.ndarray,
    candidate_trajectories: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    all_points = np.concatenate(
        [
            gt_trajectory[:, :3],
            candidate_trajectories.reshape(-1, candidate_trajectories.shape[-1])[:, :3],
        ],
        axis=0,
    )
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-3)
    padding = np.maximum(spans * 0.1, 0.05)
    mins = mins - padding
    maxs = maxs + padding
    return (
        (float(mins[0]), float(maxs[0])),
        (float(mins[1]), float(maxs[1])),
        (float(mins[2]), float(maxs[2])),
    )


def render_single_candidate_sample(
    *,
    sample_id: int,
    traj_name: str,
    curr_time: int,
    gt_trajectory: np.ndarray,
    candidate_trajectories: np.ndarray,
    save_path: Path,
) -> List[float]:
    plt = ensure_matplotlib()
    candidate_trajectories = np.asarray(candidate_trajectories, dtype=np.float32)
    gt_trajectory = np.asarray(gt_trajectory, dtype=np.float32)

    end_errors = [
        float(np.linalg.norm(candidate_trajectory[-1, :3] - gt_trajectory[-1, :3]))
        for candidate_trajectory in candidate_trajectories
    ]
    xlim, ylim, zlim = compute_plot_limits(gt_trajectory, candidate_trajectories)

    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

    colors = [
        "tab:blue",
        "tab:red",
        "tab:orange",
        "tab:green",
        "tab:purple",
        "tab:brown",
        "tab:pink",
    ]
    ax_xy.plot(
        gt_trajectory[:, 0],
        gt_trajectory[:, 1],
        "-o",
        color=colors[0],
        linewidth=2.2,
        markersize=5,
        label="GT",
    )
    ax_3d.plot(
        gt_trajectory[:, 0],
        gt_trajectory[:, 1],
        gt_trajectory[:, 2],
        "-o",
        color=colors[0],
        linewidth=2.2,
        markersize=4,
        label="GT",
    )

    for candidate_idx, candidate_trajectory in enumerate(candidate_trajectories):
        color = colors[(candidate_idx + 1) % len(colors)]
        label = get_candidate_label(candidate_idx)
        ax_xy.plot(
            candidate_trajectory[:, 0],
            candidate_trajectory[:, 1],
            "-o",
            color=color,
            linewidth=2.0,
            markersize=5,
            label=label,
        )
        ax_3d.plot(
            candidate_trajectory[:, 0],
            candidate_trajectory[:, 1],
            candidate_trajectory[:, 2],
            "-o",
            color=color,
            linewidth=2.0,
            markersize=4,
            label=label,
        )

    error_text = " | ".join(
        f"{get_candidate_label(candidate_idx)} end err={end_error:.3f}m"
        for candidate_idx, end_error in enumerate(end_errors)
    )
    ax_xy.set_title(f"sample {sample_id} XY\n{error_text}")
    ax_3d.set_title("3D")
    fig.suptitle(f"{traj_name} | curr_time={curr_time}")

    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_xlim(*xlim)
    ax_xy.set_ylim(*ylim)
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best")

    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.set_xlim(*xlim)
    ax_3d.set_ylim(*ylim)
    ax_3d.set_zlim(*zlim)
    ax_3d.view_init(elev=28, azim=-58)

    x_span = max(xlim[1] - xlim[0], 1e-3)
    y_span = max(ylim[1] - ylim[0], 1e-3)
    z_span = max(zlim[1] - zlim[0], 1e-3)
    ax_3d.set_box_aspect((x_span, y_span, z_span))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return end_errors


def render_candidate_overview(
    *,
    dataset_name: str,
    overview_items: Sequence[Tuple[int, np.ndarray, np.ndarray]],
    save_path: Path,
) -> None:
    if not overview_items:
        raise ValueError("No overview items to render")

    plt = ensure_matplotlib()
    n_items = len(overview_items)
    sample_cols = min(4, max(1, n_items))
    sample_rows = int(np.ceil(n_items / sample_cols))
    fig = plt.figure(
        figsize=(7.2 * sample_cols, 3.8 * sample_rows), constrained_layout=True
    )
    grid = fig.add_gridspec(sample_rows, sample_cols * 2)

    colors = [
        "tab:blue",
        "tab:red",
        "tab:orange",
        "tab:green",
        "tab:purple",
        "tab:brown",
        "tab:pink",
    ]
    x_label = "X (m)"
    y_label = "Y (m)"
    z_label = "Z (m)"

    for plot_idx, (sample_id, gt_trajectory, candidate_trajectories) in enumerate(
        overview_items
    ):
        row_idx, col_idx = divmod(plot_idx, sample_cols)
        ax_xy = fig.add_subplot(grid[row_idx, col_idx * 2])
        ax_3d = fig.add_subplot(grid[row_idx, col_idx * 2 + 1], projection="3d")
        xlim, ylim, zlim = compute_plot_limits(gt_trajectory, candidate_trajectories)

        ax_xy.plot(
            gt_trajectory[:, 0],
            gt_trajectory[:, 1],
            "-o",
            color=colors[0],
            linewidth=1.6,
            markersize=3.0,
            label="GT",
        )
        ax_3d.plot(
            gt_trajectory[:, 0],
            gt_trajectory[:, 1],
            gt_trajectory[:, 2],
            "-o",
            color=colors[0],
            linewidth=1.6,
            markersize=2.6,
            label="GT",
        )
        for candidate_idx, candidate_trajectory in enumerate(candidate_trajectories):
            color = colors[(candidate_idx + 1) % len(colors)]
            label = get_candidate_label(candidate_idx)
            ax_xy.plot(
                candidate_trajectory[:, 0],
                candidate_trajectory[:, 1],
                "-o",
                color=color,
                linewidth=1.3,
                markersize=2.6,
                label=label,
            )
            ax_3d.plot(
                candidate_trajectory[:, 0],
                candidate_trajectory[:, 1],
                candidate_trajectory[:, 2],
                "-o",
                color=color,
                linewidth=1.3,
                markersize=2.3,
                label=label,
            )

        ax_xy.set_title(f"sample {sample_id} XY", fontsize=9)
        ax_3d.set_title("3D", fontsize=9)

        ax_xy.set_xlim(*xlim)
        ax_xy.set_ylim(*ylim)
        ax_xy.set_xlabel(x_label, fontsize=8)
        ax_xy.set_ylabel(y_label, fontsize=8)
        ax_xy.tick_params(labelsize=7)
        ax_xy.grid(True, alpha=0.25)

        ax_3d.set_xlim(*xlim)
        ax_3d.set_ylim(*ylim)
        ax_3d.set_zlim(*zlim)
        ax_3d.set_xlabel(x_label, fontsize=8, labelpad=2)
        ax_3d.set_ylabel(y_label, fontsize=8, labelpad=2)
        ax_3d.set_zlabel(z_label, fontsize=8, labelpad=2)
        ax_3d.tick_params(labelsize=7, pad=0)
        ax_3d.view_init(elev=28, azim=-58)

        x_span = max(xlim[1] - xlim[0], 1e-3)
        y_span = max(ylim[1] - ylim[0], 1e-3)
        z_span = max(zlim[1] - zlim[0], 1e-3)
        ax_3d.set_box_aspect((x_span, y_span, z_span))

        if plot_idx == 0:
            ax_xy.legend(loc="best", fontsize=7)

    total_slots = sample_rows * sample_cols
    for empty_idx in range(n_items, total_slots):
        row_idx, col_idx = divmod(empty_idx, sample_cols)
        ax_xy = fig.add_subplot(grid[row_idx, col_idx * 2])
        ax_xy.axis("off")
        ax_3d = fig.add_subplot(grid[row_idx, col_idx * 2 + 1], projection="3d")
        ax_3d.set_axis_off()

    fig.suptitle(f"{dataset_name} trajectory candidates overview")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def resolve_viz_output_dir(args) -> Path:
    if args.viz_output_dir:
        out_dir = Path(args.viz_output_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = (THIS_DIR / out_dir).resolve()
        return out_dir
    return THIS_DIR / "trajectory_candidates_viz"


def write_candidate_summary(
    summary_rows: Sequence[Tuple[int, str, int, Sequence[float]]],
    save_path: Path,
) -> None:
    max_candidates = max((len(row[3]) for row in summary_rows), default=0)
    header = ["sample_id", "traj_name", "curr_time"]
    header.extend(
        f"cand{candidate_idx}_end_err_m" for candidate_idx in range(max_candidates)
    )
    header.append("best_end_err_m")

    lines = ["\t".join(header)]
    for sample_id, traj_name, curr_time, end_errors in summary_rows:
        fields = [str(sample_id), traj_name, str(curr_time)]
        fields.extend(
            f"{float(end_errors[candidate_idx]):.6f}"
            if candidate_idx < len(end_errors)
            else ""
            for candidate_idx in range(max_candidates)
        )
        best_error = min(end_errors) if end_errors else float("nan")
        fields.append(f"{float(best_error):.6f}")
        lines.append("\t".join(fields))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def visualize_candidate_trajectories_for_dataset(
    *,
    dataset_name: str,
    data_root: Path,
    split_dir: Path,
    args,
    stride: int,
) -> None:
    candidate_path = split_dir / get_candidate_filename(dataset_name, args)
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate pkl not found for visualization: {candidate_path}"
        )

    eval_index_path = split_dir / get_eval_index_filename(args)
    if not eval_index_path.exists():
        raise FileNotFoundError(
            f"Eval index pkl not found for visualization: {eval_index_path}"
        )

    candidate_map = load_candidate_trajectories(candidate_path)
    eval_index = load_index_entries(eval_index_path)
    viz_dir = resolve_viz_output_dir(args)
    per_sample_dir = viz_dir / "per_sample"

    traj_data_cache: Dict[str, dict] = {}
    summary_rows: List[Tuple[int, str, int, Sequence[float]]] = []
    overview_items: List[Tuple[int, np.ndarray, np.ndarray]] = []

    for sample_id in sorted(candidate_map.keys()):
        if sample_id >= len(eval_index):
            raise IndexError(
                f"sample_id={sample_id} out of range for eval index length {len(eval_index)}"
            )

        traj_name, curr_time, _, _ = eval_index[sample_id]
        if traj_name not in traj_data_cache:
            traj_data_cache[traj_name] = load_traj_data(data_root, traj_name)
        gt_full = extract_gt_trajectory_local(
            traj_data_cache[traj_name],
            curr_time=int(curr_time),
            len_traj_pred=int(args.len_traj_pred),
        )
        gt_trajectory = downsample_single_trajectory(gt_full, int(stride))
        candidate_trajectories = candidate_map[sample_id]
        if gt_trajectory.shape != candidate_trajectories[0].shape:
            raise ValueError(
                f"GT/candidate shape mismatch for sample_id={sample_id}: "
                f"gt={gt_trajectory.shape}, candidate={candidate_trajectories[0].shape}"
            )

        save_path = per_sample_dir / f"sample_{sample_id:03d}.png"
        end_errors = render_single_candidate_sample(
            sample_id=int(sample_id),
            traj_name=str(traj_name),
            curr_time=int(curr_time),
            gt_trajectory=gt_trajectory,
            candidate_trajectories=candidate_trajectories,
            save_path=save_path,
        )
        summary_rows.append(
            (int(sample_id), str(traj_name), int(curr_time), end_errors)
        )
        overview_items.append((int(sample_id), gt_trajectory, candidate_trajectories))

    write_candidate_summary(summary_rows, viz_dir / "summary.tsv")
    render_candidate_overview(
        dataset_name=dataset_name,
        overview_items=overview_items,
        save_path=viz_dir / "all_samples_overview.png",
    )
    print(f"[{dataset_name}] wrote candidate visualization: {viz_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the final trajectory candidate pkl for the YouTube dataset. "
            "The script will reuse or build the required index pkls automatically."
        )
    )
    parser.add_argument(
        "--preset",
        choices=sorted(DEFAULT_PRESETS.keys()),
        default=DEFAULT_PRESET,
        help=f"Dataset preset. Defaults to {DEFAULT_PRESET}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the generated candidate trajectories pkl.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Render the candidate trajectories into the can_viz folder.",
    )
    parser.add_argument(
        "--viz-only",
        action="store_true",
        help="Skip candidate generation and only render an existing candidate pkl.",
    )
    parser.add_argument(
        "--viz-output-dir",
        default=None,
        help="Output directory for rendered candidate visualizations.",
    )
    return parser


def apply_hidden_defaults(args) -> None:
    dataset_name = str(args.preset)
    default_source_index_pkl = DEFAULT_SOURCE_INDEX_PKLS.get(dataset_name)
    args.dataset_name = None
    args.data_root = None
    args.splits_root = str(DEFAULT_SPLITS_ROOT)
    args.artifact_naming = "neutral"
    args.num_samples = int(
        get_local_dataset_default(dataset_name, "paper_num_candidates", 5)
    )
    args.dim = 3
    args.input_fps = float(get_local_dataset_default(dataset_name, "input_fps", 4.0))
    args.output_fps = float(get_local_dataset_default(dataset_name, "output_fps", 1.0))
    args.len_traj_pred = int(
        get_local_dataset_default(dataset_name, "source_len_traj_pred", 32)
    )
    args.seed = 1120
    args.rebuild_source_index = False
    args.rebuild_index = True
    args.index_only = False
    args.source_index_pkl = (
        str(default_source_index_pkl) if default_source_index_pkl is not None else None
    )
    args.source_min_dist_cat = None
    args.source_max_dist_cat = None
    args.source_context_size = None
    args.source_len_traj_pred = None
    args.source_traj_stride = None
    args.sample_size = 0
    args.forward_min_goal = 1
    args.test_traj_names = None
    args.use_all_trajectories_as_test = False
    if args.viz_only:
        args.visualize = True


def main() -> int:
    args = build_parser().parse_args()
    apply_hidden_defaults(args)
    specs = resolve_dataset_specs(args)
    for spec_idx, (dataset_name, data_root) in enumerate(specs):
        process_dataset(spec_idx, dataset_name, data_root, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
