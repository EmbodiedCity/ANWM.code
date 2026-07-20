#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import re
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


TRAJ_NAME_RE = re.compile(
    r"^(?P<video_id>.+)_(?P<start>\d{7})_(?P<end>\d{7})_processed$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a filtered YouTube trajectory subdataset, remove abnormal viewpoints, "
            "and generate train/test trajectory-name files."
        )
    )
    parser.add_argument("--src", type=Path, required=True, help="Source dataset root.")
    parser.add_argument(
        "--dst", type=Path, required=True, help="Destination dataset root."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Delete destination before writing."
    )
    parser.add_argument(
        "--min_depth_ratio_mid",
        type=float,
        default=4.0,
        help="Reject trajectories whose middle-frame max(depth)/median(depth) is below this threshold.",
    )
    parser.add_argument(
        "--max_abs_pitch",
        type=float,
        default=0.3,
        help="Reject trajectories whose maximum absolute pitch exceeds this threshold in radians.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Approximate fraction of kept trajectories assigned to test per video.",
    )
    parser.add_argument(
        "--candidate_pool_frac",
        type=float,
        default=0.3,
        help="Fraction of the highest-rotation trajectories kept as sampling pool per video.",
    )
    parser.add_argument(
        "--candidate_pool_min",
        type=int,
        default=30,
        help="Minimum candidate-pool size per video before random sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260330,
        help="Random seed for reproducible test split sampling.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write metadata and lists only; skip copying files.",
    )
    return parser.parse_args()


def parse_traj_name(name: str) -> Tuple[str, int, int]:
    match = TRAJ_NAME_RE.match(name)
    if not match:
        raise ValueError(f"Bad trajectory name format: {name}")
    return match.group("video_id"), int(match.group("start")), int(match.group("end"))


def load_traj_metrics(traj_pkl: Path) -> Dict[str, float]:
    with traj_pkl.open("rb") as f:
        traj_data = pickle.load(f)

    pose = np.asarray(traj_data.get("pose", []), dtype=np.float64)
    depth = np.asarray(traj_data.get("depth", []), dtype=np.float32)
    if pose.ndim != 3 or pose.shape[1:] != (4, 4):
        raise ValueError(f"bad pose shape {pose.shape}")
    if depth.ndim != 3:
        raise ValueError(f"bad depth shape {depth.shape}")
    if pose.shape[0] <= 0 or depth.shape[0] != pose.shape[0]:
        raise ValueError(
            f"invalid pose/depth lengths: pose={pose.shape}, depth={depth.shape}"
        )

    rotation = pose[:, :3, :3]
    pitch = np.arctan2(
        -rotation[:, 2, 0], np.sqrt(rotation[:, 2, 1] ** 2 + rotation[:, 2, 2] ** 2)
    )
    pitch = (pitch + math.pi) % (2.0 * math.pi) - math.pi

    rel = np.einsum(
        "nij,njk->nik", np.transpose(rotation[:-1], (0, 2, 1)), rotation[1:]
    )
    tr = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    step_rot_deg = (
        np.arccos(tr) * (180.0 / math.pi)
        if tr.size
        else np.zeros((0,), dtype=np.float64)
    )

    mid_depth = depth[pose.shape[0] // 2]
    max_depth_mid = float(np.nanmax(mid_depth))
    med_depth_mid = float(np.nanmedian(mid_depth))
    depth_ratio_mid = max_depth_mid / (med_depth_mid + 1e-9)

    return {
        "n_frames": float(pose.shape[0]),
        "max_abs_pitch": float(np.nanmax(np.abs(pitch))),
        "max_depth_mid": max_depth_mid,
        "med_depth_mid": med_depth_mid,
        "depth_ratio_mid": float(depth_ratio_mid),
        "total_rot_deg": float(np.sum(step_rot_deg)),
        "max_step_rot_deg": float(np.max(step_rot_deg)) if step_rot_deg.size else 0.0,
    }


def write_lines(path: Path, lines: Sequence[str]) -> None:
    content = "".join(f"{line}\n" for line in lines)
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def weighted_sample_without_replacement(
    items: Sequence[Tuple[str, float]],
    *,
    sample_size: int,
    rng: random.Random,
) -> List[str]:
    if sample_size <= 0:
        return []
    keys: List[Tuple[float, str]] = []
    for name, weight in items:
        safe_weight = max(float(weight), 1e-6)
        keys.append((rng.random() ** (1.0 / safe_weight), name))
    return [name for _, name in sorted(keys, reverse=True)[:sample_size]]


def select_test_names(
    kept_names: Sequence[str],
    metrics_by_name: Dict[str, Dict[str, float]],
    *,
    test_ratio: float,
    candidate_pool_frac: float,
    candidate_pool_min: int,
    seed: int,
) -> List[str]:
    per_video: Dict[str, List[str]] = defaultdict(list)
    for name in kept_names:
        video_id, _, _ = parse_traj_name(name)
        per_video[video_id].append(name)

    rng = random.Random(int(seed))
    selected: List[str] = []
    for video_id in sorted(per_video.keys()):
        names = sorted(per_video[video_id], key=lambda item: parse_traj_name(item)[1])
        n_total = len(names)
        n_test = max(1, int(round(n_total * float(test_ratio))))
        ranked = sorted(
            names,
            key=lambda item: (
                -float(metrics_by_name[item]["total_rot_deg"]),
                -float(metrics_by_name[item]["max_step_rot_deg"]),
                item,
            ),
        )
        pool_size = max(
            int(math.ceil(n_total * float(candidate_pool_frac))),
            int(candidate_pool_min),
            n_test,
        )
        pool_size = min(pool_size, n_total)
        pool = [
            (name, metrics_by_name[name]["total_rot_deg"])
            for name in ranked[:pool_size]
        ]
        picked = weighted_sample_without_replacement(pool, sample_size=n_test, rng=rng)
        selected.extend(picked)

    selected = sorted(
        set(selected),
        key=lambda item: (parse_traj_name(item)[0], parse_traj_name(item)[1], item),
    )
    return selected


def copy_selected(src_root: Path, dst_root: Path, names: Sequence[str]) -> None:
    rsync_dirs = dst_root / "rsync_dirs.txt"
    write_lines(rsync_dirs, [f"{name}/" for name in names])
    subprocess.run(
        [
            "rsync",
            "-a",
            "--files-from",
            str(rsync_dirs),
            f"{src_root}/",
            f"{dst_root}/",
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()

    src_root = args.src.expanduser().resolve()
    dst_root = args.dst.expanduser().resolve()
    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {src_root}")

    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Destination already exists: {dst_root}")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=False)

    traj_dirs = sorted(
        [
            path
            for path in src_root.iterdir()
            if path.is_dir() and (path / "traj_data.pkl").exists()
        ],
        key=lambda path: path.name,
    )
    if not traj_dirs:
        raise ValueError(f"No trajectories found in {src_root}")

    kept_names: List[str] = []
    rejected_rows: List[Dict[str, str]] = []
    metrics_by_name: Dict[str, Dict[str, float]] = {}
    start_time = time.time()
    for index, traj_dir in enumerate(traj_dirs, start=1):
        traj_name = traj_dir.name
        traj_pkl = traj_dir / "traj_data.pkl"
        metrics: Dict[str, float] = {}
        reasons: List[str] = []
        try:
            metrics = load_traj_metrics(traj_pkl)
            if not math.isfinite(metrics["depth_ratio_mid"]) or metrics[
                "depth_ratio_mid"
            ] < float(args.min_depth_ratio_mid):
                reasons.append("low_depth_ratio_mid")
            if not math.isfinite(metrics["max_abs_pitch"]) or metrics[
                "max_abs_pitch"
            ] > float(args.max_abs_pitch):
                reasons.append("high_pitch")
        except Exception as exc:
            reasons = [f"error:{type(exc).__name__}"]

        if reasons:
            rejected_rows.append(
                {
                    "traj_name": traj_name,
                    "reasons": ";".join(reasons),
                    "depth_ratio_mid": f"{metrics.get('depth_ratio_mid', float('nan')):.6g}",
                    "max_depth_mid": f"{metrics.get('max_depth_mid', float('nan')):.6g}",
                    "med_depth_mid": f"{metrics.get('med_depth_mid', float('nan')):.6g}",
                    "max_abs_pitch": f"{metrics.get('max_abs_pitch', float('nan')):.6g}",
                    "total_rot_deg": f"{metrics.get('total_rot_deg', float('nan')):.6g}",
                    "max_step_rot_deg": f"{metrics.get('max_step_rot_deg', float('nan')):.6g}",
                    "n_frames": f"{metrics.get('n_frames', float('nan')):.6g}",
                }
            )
            continue

        metrics_by_name[traj_name] = metrics
        kept_names.append(traj_name)

        if index % 100 == 0 or index == len(traj_dirs):
            elapsed = time.time() - start_time
            print(
                f"[filter] {index:>4}/{len(traj_dirs):<4} kept={len(kept_names):<4} rejected={len(rejected_rows):<4} ({elapsed:.1f}s)",
                flush=True,
            )

    test_names = select_test_names(
        kept_names,
        metrics_by_name,
        test_ratio=float(args.test_ratio),
        candidate_pool_frac=float(args.candidate_pool_frac),
        candidate_pool_min=int(args.candidate_pool_min),
        seed=int(args.seed),
    )
    train_names = [name for name in kept_names if name not in set(test_names)]

    write_lines(dst_root / "traj_names.txt", kept_names)
    write_lines(dst_root / "test_traj_name.txt", test_names)
    write_lines(dst_root / "train_traj_names.txt", train_names)
    write_csv(dst_root / "rejected_traj_names.csv", rejected_rows)

    meta = {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "filters": {
            "min_depth_ratio_mid": float(args.min_depth_ratio_mid),
            "max_abs_pitch": float(args.max_abs_pitch),
        },
        "split": {
            "test_ratio": float(args.test_ratio),
            "candidate_pool_frac": float(args.candidate_pool_frac),
            "candidate_pool_min": int(args.candidate_pool_min),
            "seed": int(args.seed),
        },
        "counts": {
            "total": len(traj_dirs),
            "kept": len(kept_names),
            "rejected": len(rejected_rows),
            "train": len(train_names),
            "test": len(test_names),
        },
        "generated_at_unix": time.time(),
    }
    (dst_root / "filter_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if not args.dry_run:
        copy_selected(src_root, dst_root, kept_names)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
