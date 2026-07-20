#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


TRAJ_NAME_RE = re.compile(r"^(?P<vid>.+)_(?P<s>\d{7})_(?P<e>\d{7})_processed$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export per-trajectory statistics from processed and raw Sekai trajectories."
    )
    ap.add_argument(
        "--output_root",
        type=Path,
        default=Path("data/sekai_new"),
        help="Processed trajectory output root containing traj_names.txt.",
    )
    ap.add_argument(
        "--traj_npz_dir",
        type=Path,
        default=Path("data/sekai_raw/camera_trajectories/sekai-real-drone"),
        help="Directory containing raw Sekai trajectory .npz files.",
    )
    ap.add_argument(
        "--output_csv",
        type=Path,
        default=Path("outputs/sekai_trajectory_stats.csv"),
        help="CSV file to write.",
    )
    ap.add_argument(
        "--output_summary_json",
        type=Path,
        default=Path("outputs/sekai_trajectory_stats_summary.json"),
        help="Summary JSON file to write.",
    )
    ap.add_argument(
        "--source_fps", type=float, default=30.0, help="Raw trajectory fps."
    )
    ap.add_argument(
        "--source_num_frames", type=int, default=300, help="Raw frames per clip."
    )
    ap.add_argument(
        "--out_fps", type=float, default=4.0, help="Sampled trajectory fps."
    )
    ap.add_argument(
        "--out_num_frames",
        type=int,
        default=0,
        help="Override sampled frame count; 0 means auto.",
    )
    return ap.parse_args()


def read_traj_names(traj_names_file: Path) -> List[str]:
    if not traj_names_file.exists():
        raise FileNotFoundError(f"traj_names.txt not found: {traj_names_file}")
    names: List[str] = []
    for line in traj_names_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            names.append(line)
    if not names:
        raise ValueError(f"No trajectory names found in {traj_names_file}")
    return names


def compute_frame_sample_indices(
    src_len: int,
    *,
    source_fps: float,
    out_fps: float,
    out_num_frames: int,
) -> np.ndarray:
    if int(out_num_frames) > 0:
        out_len = int(out_num_frames)
    else:
        out_len = int(
            math.floor(float(src_len) * float(out_fps) / float(source_fps) + 1e-9)
        )
    out_len = max(1, out_len)

    ratio = float(source_fps) / float(out_fps)
    idx = np.floor(np.arange(out_len, dtype=np.float64) * ratio + 1e-9).astype(np.int64)
    idx = idx[idx < int(src_len)]
    if idx.size == 0:
        idx = np.array([0], dtype=np.int64)
    return idx


def compute_path_stats(points: np.ndarray) -> Dict[str, float]:
    points = np.asarray(points, dtype=np.float64)
    centered = points - points[0]
    step = (
        np.linalg.norm(points[1:] - points[:-1], axis=1)
        if len(points) > 1
        else np.zeros((0,), dtype=np.float64)
    )
    return {
        "end_disp": float(np.linalg.norm(centered[-1])),
        "max_disp_from_start": float(np.linalg.norm(centered, axis=1).max()),
        "path_length": float(step.sum()),
    }


def summarize_rows(
    rows: Sequence[Dict[str, object]], numeric_keys: Iterable[str]
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "count": len(rows),
        "numeric": {},
    }
    numeric = out["numeric"]
    assert isinstance(numeric, dict)
    for key in numeric_keys:
        col = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        numeric[key] = {
            "min": float(col.min()),
            "p05": float(np.percentile(col, 5)),
            "median": float(np.median(col)),
            "p95": float(np.percentile(col, 95)),
            "max": float(col.max()),
        }
    return out


def main() -> int:
    args = parse_args()

    names = read_traj_names(args.output_root / "traj_names.txt")
    sample_idx = compute_frame_sample_indices(
        int(args.source_num_frames),
        source_fps=float(args.source_fps),
        out_fps=float(args.out_fps),
        out_num_frames=int(args.out_num_frames),
    )

    rows: List[Dict[str, object]] = []
    for name in names:
        m = TRAJ_NAME_RE.match(name)
        if not m:
            raise ValueError(f"Unexpected trajectory name format: {name}")
        video_id = m.group("vid")
        start_frame = int(m.group("s"))
        end_frame = int(m.group("e"))
        frame_span = int(end_frame - start_frame)
        npz_path = (
            args.traj_npz_dir / f"{video_id}_{start_frame:07d}_{end_frame:07d}.npz"
        )
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing raw trajectory npz: {npz_path}")

        with np.load(npz_path) as data:
            pose_full = np.asarray(data["extrinsic"], dtype=np.float64)

        if pose_full.shape != (int(args.source_num_frames), 4, 4):
            raise ValueError(
                f"Unexpected extrinsic shape {pose_full.shape} in {npz_path}"
            )

        raw_points = pose_full[:, :3, 3]
        sampled_points = raw_points[sample_idx]
        raw_stats = compute_path_stats(raw_points)
        sampled_stats = compute_path_stats(sampled_points)

        rows.append(
            {
                "traj_name": name,
                "video_id": video_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_span": frame_span,
                "source_num_frames": int(args.source_num_frames),
                "sampled_num_frames": int(sample_idx.shape[0]),
                "source_fps": float(args.source_fps),
                "out_fps": float(args.out_fps),
                "clip_window_sec": float(frame_span / float(args.source_fps)),
                "raw_timestamp_span_sec": float(
                    (int(args.source_num_frames) - 1) / float(args.source_fps)
                ),
                "sampled_timestamp_span_sec": float(
                    (int(sample_idx.shape[0]) - 1) / float(args.out_fps)
                ),
                "raw_end_disp_unit": raw_stats["end_disp"],
                "raw_max_disp_unit": raw_stats["max_disp_from_start"],
                "raw_path_length_unit": raw_stats["path_length"],
                "sampled_end_disp_unit": sampled_stats["end_disp"],
                "sampled_max_disp_unit": sampled_stats["max_disp_from_start"],
                "sampled_path_length_unit": sampled_stats["path_length"],
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "output_root": str(args.output_root),
        "traj_npz_dir": str(args.traj_npz_dir),
        "source_fps": float(args.source_fps),
        "source_num_frames": int(args.source_num_frames),
        "out_fps": float(args.out_fps),
        "sampled_num_frames": int(sample_idx.shape[0]),
        "sample_indices_first10": [int(x) for x in sample_idx[:10]],
        "sample_indices_last10": [int(x) for x in sample_idx[-10:]],
    }
    summary.update(
        summarize_rows(
            rows,
            numeric_keys=[
                "clip_window_sec",
                "raw_timestamp_span_sec",
                "sampled_timestamp_span_sec",
                "raw_end_disp_unit",
                "raw_max_disp_unit",
                "raw_path_length_unit",
                "sampled_end_disp_unit",
                "sampled_max_disp_unit",
                "sampled_path_length_unit",
            ],
        )
    )

    with args.output_summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote csv: {args.output_csv}")
    print(f"[ok] wrote summary: {args.output_summary_json}")
    print(f"[ok] trajectories: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
