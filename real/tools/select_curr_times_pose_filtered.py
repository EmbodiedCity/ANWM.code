from __future__ import annotations

import argparse
import math
import os
import pickle
from typing import Iterable, List, Optional

import numpy as np


def _rotation_step_degrees(poses: np.ndarray) -> np.ndarray:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (T,4,4), got {poses.shape}")
    if poses.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)

    R = poses[:, :3, :3].astype(np.float64, copy=False)
    Rt = R[:-1]
    Rnext = R[1:]
    Rrel = np.einsum("nij,njk->nik", np.transpose(Rt, (0, 2, 1)), Rnext)
    tr = (np.trace(Rrel, axis1=1, axis2=2) - 1.0) / 2.0
    tr = np.clip(tr, -1.0, 1.0)
    ang = np.arccos(tr) * (180.0 / math.pi)
    return ang.astype(np.float32, copy=False)


def _translation_step(poses: np.ndarray) -> np.ndarray:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (T,4,4), got {poses.shape}")
    if poses.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)
    trans = poses[:, :3, 3].astype(np.float64, copy=False)
    step = np.linalg.norm(np.diff(trans, axis=0), axis=1)
    return step.astype(np.float32, copy=False)


def _evenly_spaced_indices(n: int, k: int) -> List[int]:
    if k <= 0:
        return []
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]

    step = (n - 1) / float(k - 1)
    idxs: List[int] = []
    last = -1
    for i in range(k):
        j = int(round(i * step))
        j = max(0, min(n - 1, j))
        if j != last:
            idxs.append(j)
            last = j
    if idxs[-1] != n - 1:
        idxs[-1] = n - 1
    return idxs


def _range_with_step(start: int, end_inclusive: int, step: int) -> List[int]:
    if step <= 0:
        raise ValueError(f"frame_step must be >0, got {step}")
    if end_inclusive < start:
        return []
    return list(range(int(start), int(end_inclusive) + 1, int(step)))


def _filter_by_pose_jumps(
    *,
    candidates: Iterable[int],
    poses: np.ndarray,
    context_size: int,
    len_traj_pred: int,
    step_pctl: float,
    rot_pctl: float,
    step_mult: float,
    rot_mult: float,
    step_max: Optional[float],
    rot_max_deg: Optional[float],
) -> List[int]:
    T = int(poses.shape[0])
    if T <= 1:
        return list(map(int, candidates))

    step = _translation_step(poses)  # (T-1,)
    rot_deg = _rotation_step_degrees(poses)  # (T-1,)

    step_q = float(np.percentile(step, step_pctl)) if step.size else 0.0
    rot_q = float(np.percentile(rot_deg, rot_pctl)) if rot_deg.size else 0.0

    step_thr = (
        float(step_max) if step_max is not None else float(step_q * float(step_mult))
    )
    rot_thr = (
        float(rot_max_deg)
        if rot_max_deg is not None
        else float(rot_q * float(rot_mult))
    )

    if not math.isfinite(step_thr) or step_thr <= 0:
        step_thr = float("inf")
    if not math.isfinite(rot_thr) or rot_thr <= 0:
        rot_thr = float("inf")

    bad = (
        (~np.isfinite(step))
        | (step > step_thr)
        | (~np.isfinite(rot_deg))
        | (rot_deg > rot_thr)
    )
    pref = np.concatenate([[0], np.cumsum(bad.astype(np.int64))])  # (T,)

    out: List[int] = []
    for t in candidates:
        t = int(t)
        a = t - int(context_size) + 1
        b = t + int(len_traj_pred)
        if a < 0 or b >= T:
            continue
        if int(pref[b] - pref[a]) == 0:
            out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select curr_time indices for YouTube traj_data.pkl, with optional pose jump filtering.",
    )
    ap.add_argument("traj_pkl", help="Path to traj_data.pkl")
    ap.add_argument("--context_size", type=int, default=8)
    ap.add_argument("--len_traj_pred", type=int, default=16)
    ap.add_argument("--frame_step", type=int, default=200)
    ap.add_argument("--curr_time", type=int, default=None)
    ap.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If >0, cap selected times per trajectory.",
    )

    ap.add_argument(
        "--filter_pose_jumps",
        action="store_true",
        help="Filter windows with large pose jumps.",
    )
    ap.add_argument(
        "--step_pctl",
        type=float,
        default=95.0,
        help="Translation step percentile used to set the jump threshold.",
    )
    ap.add_argument(
        "--rot_pctl",
        type=float,
        default=95.0,
        help="Rotation step percentile used to set the jump threshold (degrees).",
    )
    ap.add_argument(
        "--step_mult",
        type=float,
        default=1.0,
        help="Threshold = percentile(step, step_pctl) * step_mult (unless --step_max).",
    )
    ap.add_argument(
        "--rot_mult",
        type=float,
        default=1.0,
        help="Threshold = percentile(rot_deg, rot_pctl) * rot_mult (unless --rot_max_deg).",
    )
    ap.add_argument(
        "--step_max",
        type=float,
        default=None,
        help="Absolute translation-step threshold override.",
    )
    ap.add_argument(
        "--rot_max_deg",
        type=float,
        default=None,
        help="Absolute rotation-step threshold override (degrees).",
    )

    args = ap.parse_args()

    pkl_path = str(args.traj_pkl)
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(pkl_path)

    with open(pkl_path, "rb") as f:
        traj_data = pickle.load(f)

    import numpy as np

    poses = np.asarray(traj_data.get("pose", []), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"`pose` must be (T,4,4), got {poses.shape}")

    T_total = int(poses.shape[0])
    start_t = int(args.context_size) - 1
    end_t = T_total - int(args.len_traj_pred) - 1
    if end_t < start_t:
        print("")
        return

    if args.curr_time is not None:
        candidates = [int(args.curr_time)]
    else:
        candidates = _range_with_step(start_t, end_t, int(args.frame_step))

    # Keep only those within the legal range (in case --curr_time is out of range)
    candidates = [t for t in candidates if start_t <= int(t) <= end_t]

    if args.filter_pose_jumps:
        candidates = _filter_by_pose_jumps(
            candidates=candidates,
            poses=poses,
            context_size=int(args.context_size),
            len_traj_pred=int(args.len_traj_pred),
            step_pctl=float(args.step_pctl),
            rot_pctl=float(args.rot_pctl),
            step_mult=float(args.step_mult),
            rot_mult=float(args.rot_mult),
            step_max=args.step_max,
            rot_max_deg=args.rot_max_deg,
        )

    candidates = sorted(set(map(int, candidates)))

    max_samples = int(args.max_samples)
    if max_samples > 0 and len(candidates) > max_samples:
        idxs = _evenly_spaced_indices(len(candidates), max_samples)
        candidates = [candidates[i] for i in idxs]

    print(" ".join(map(str, candidates)))


if __name__ == "__main__":
    main()
