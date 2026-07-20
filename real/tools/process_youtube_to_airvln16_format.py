#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pickle
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
CAMERA_NPZ_RE = re.compile(
    r"^(?P<vid>[A-Za-z0-9_-]{11})_(?P<s>\d{7})_(?P<e>\d{7})\.npz$"
)


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _resolve_sekai_real_drone_csv(
    input_root: Path, explicit: Optional[str]
) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p

    candidates = [
        input_root / "sekai-real-drone.csv",
        input_root.parent / "sekai-real-drone.csv",
    ]

    # Common workspace layout: <workspace>/sekai-codebase/dataset_downloading/sekai-real-drone.csv
    for parent in [input_root.parent.parent, THIS_DIR.parent.parent]:
        candidates.append(
            parent / "sekai-codebase" / "dataset_downloading" / "sekai-real-drone.csv"
        )

    for p in candidates:
        if p.exists():
            return p
    return None


def _load_allowed_camera_files(meta_csv: Path, scene: str) -> set[str]:
    allowed = set()
    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"cameraFile", "scene"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"meta_csv missing columns: {sorted(missing)}")
        for row in reader:
            if (row.get("scene") or "").strip() != scene:
                continue
            cam = (row.get("cameraFile") or "").strip()
            if not cam:
                continue
            allowed.add(Path(cam).name)
    return allowed


def _guess_youtube_id_from_filename(name: str) -> Optional[str]:
    # Common yt-dlp patterns:
    #   <id>.<ext>
    #   <id>.f299.mp4 / <id>.f251.webm
    #   <id>.<ext>.part
    if name.endswith(".part"):
        name = name[: -len(".part")]

    first = name.split(".", 1)[0]
    if YOUTUBE_ID_RE.fullmatch(first):
        return first

    stem = Path(name).stem
    if YOUTUBE_ID_RE.fullmatch(stem):
        return stem

    return None


def _run(cmd: Sequence[str]) -> None:
    subprocess.run(list(cmd), check=True)


def _ffprobe_video_dims(video_path: Path) -> Tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    p = subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    data = json.loads(p.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe returned no video streams: {video_path}")
    width = int(streams[0]["width"])
    height = int(streams[0]["height"])
    return width, height


def _choose_best_video(candidates: Sequence[Path]) -> Path:
    def score(p: Path) -> Tuple[int, int, str]:
        name = p.name.lower()
        # Prefer format 299 (1080p60) if available, then mp4, then webm.
        if ".f299." in name:
            pri = 0
        elif name.endswith(".mp4"):
            pri = 1
        elif name.endswith(".webm"):
            pri = 2
        else:
            pri = 3
        return (pri, len(name), name)

    return sorted(candidates, key=score)[0]


def _compute_intrinsics_pixel(
    K_norm: np.ndarray,
    *,
    in_w: int,
    in_h: int,
    crop_square: bool,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    K_norm = np.asarray(K_norm, dtype=np.float64)
    if K_norm.shape != (3, 3):
        raise ValueError(f"Expected intrinsic shape (3,3), got {K_norm.shape}")

    fx_px = float(K_norm[0, 0]) * float(in_w)
    fy_px = float(K_norm[1, 1]) * float(in_h)
    cx_px = float(K_norm[0, 2]) * float(in_w)
    cy_px = float(K_norm[1, 2]) * float(in_h)

    crop_w, crop_h = in_w, in_h
    x0, y0 = 0.0, 0.0
    if crop_square:
        crop_size = float(min(in_w, in_h))
        x0 = (float(in_w) - crop_size) / 2.0
        y0 = (float(in_h) - crop_size) / 2.0
        crop_w = int(round(crop_size))
        crop_h = int(round(crop_size))
        cx_px -= x0
        cy_px -= y0

    sx = float(out_w) / float(crop_w)
    sy = float(out_h) / float(crop_h)

    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx_px * sx
    K[1, 1] = fy_px * sy
    K[0, 2] = cx_px * sx
    K[1, 2] = cy_px * sy
    return K


def _compute_intrinsic_matrix_from_fov(
    width: int, height: int, fov_x_degree: float
) -> np.ndarray:
    fov_x = np.radians(float(fov_x_degree))
    fov_y = 2.0 * np.arctan((float(height) / float(width)) * np.tan(fov_x / 2.0))

    fx = float(width) / (2.0 * np.tan(fov_x / 2.0))
    fy = float(height) / (2.0 * np.tan(fov_y / 2.0))
    cx = float(width) / 2.0
    cy = float(height) / 2.0

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


def _ffmpeg_extract_frames(
    *,
    video_path: Path,
    start_sec: float,
    fps: float,
    num_frames: int,
    out_dir: Path,
    crop_square: bool,
    out_w: int,
    out_h: int,
    jpeg_q: int,
    start_number: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    filters: List[str] = [f"fps={fps}"]
    if crop_square:
        # Center-crop to square, then resize.
        filters.append(
            "crop="
            "min(iw\\,ih):"
            "min(iw\\,ih):"
            "(iw-min(iw\\,ih))/2:"
            "(ih-min(iw\\,ih))/2"
        )
    filters.append(f"scale={out_w}:{out_h}:flags=lanczos")
    vf = ",".join(filters)

    out_pat = str(out_dir / "%d.jpg")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        vf,
        "-frames:v",
        str(int(num_frames)),
        "-start_number",
        str(int(start_number)),
        "-q:v",
        str(int(jpeg_q)),
        out_pat,
    ]
    _run(cmd)


def _count_jpgs(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    return sum(
        1 for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"
    )


@dataclass(frozen=True)
class ClipSpec:
    video_id: str
    s_frame: int
    e_frame: int
    npz_path: Path

    @property
    def num_frames(self) -> int:
        return int(self.e_frame - self.s_frame)

    @property
    def traj_name(self) -> str:
        return f"{self.video_id}_{self.s_frame:07d}_{self.e_frame:07d}_processed"


def _discover_clips(traj_dir: Path) -> List[ClipSpec]:
    clips: List[ClipSpec] = []
    for p in sorted(traj_dir.glob("*.npz")):
        m = CAMERA_NPZ_RE.match(p.name)
        if not m:
            continue
        vid = m.group("vid")
        s = int(m.group("s"))
        e = int(m.group("e"))
        clips.append(ClipSpec(video_id=vid, s_frame=s, e_frame=e, npz_path=p))
    return clips


def _group_clips_by_video(clips: Sequence[ClipSpec]) -> Dict[str, List[ClipSpec]]:
    groups: Dict[str, List[ClipSpec]] = {}
    for c in clips:
        groups.setdefault(c.video_id, []).append(c)
    for vid in groups:
        groups[vid].sort(key=lambda x: (x.s_frame, x.e_frame))
    return groups


def _rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    R_rel = (R1.T @ R2).astype(np.float64)
    tr = float(np.trace(R_rel))
    cos = (tr - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    ang = float(np.degrees(np.arccos(cos)))
    return ang


def _split_video_clips_into_trajectories(
    video_id: str,
    clips: Sequence[ClipSpec],
    *,
    args,
) -> Tuple[List[List[ClipSpec]], Dict[str, int]]:
    """
    Split clips from the same video into continuous trajectory groups.

    A boundary is considered a "cut" if:
      - the gap between clips is too large, OR
      - pose jump implies an unrealistic speed/rotation rate.
    """
    if not clips:
        return [], {}

    source_fps = float(args.source_fps)
    source_num_frames = int(args.source_num_frames)
    max_gap_sec = float(args.traj_max_gap_sec)
    max_speed = float(args.traj_max_speed)
    max_rot_deg_s = float(args.traj_max_rot_deg_s)

    # Preload boundary poses for each clip once (npz is small).
    boundaries = []
    for c in clips:
        try:
            with np.load(c.npz_path) as d:
                pose_full = np.asarray(d["extrinsic"], dtype=np.float64)
            if pose_full.shape != (source_num_frames, 4, 4):
                boundaries.append(None)
                continue
            if str(args.pose_convention) == "w2c":
                pose_full = np.linalg.inv(pose_full)
            boundaries.append(
                {
                    "R0": pose_full[0, :3, :3],
                    "t0": pose_full[0, :3, 3],
                    "R1": pose_full[-1, :3, :3],
                    "t1": pose_full[-1, :3, 3],
                }
            )
        except Exception:
            boundaries.append(None)

    groups: List[List[ClipSpec]] = []
    reasons: Dict[str, int] = {}

    curr: List[ClipSpec] = [clips[0]]
    for i in range(1, len(clips)):
        prev = clips[i - 1]
        nxt = clips[i]

        dt_frames = int(nxt.s_frame - prev.e_frame)
        if dt_frames < 0:
            reason = "overlap"
            reasons[reason] = reasons.get(reason, 0) + 1
            groups.append(curr)
            curr = [nxt]
            continue

        dt_sec = max(float(dt_frames) / float(source_fps), 1.0 / float(source_fps))
        if dt_sec > max_gap_sec:
            reason = "gap"
            reasons[reason] = reasons.get(reason, 0) + 1
            groups.append(curr)
            curr = [nxt]
            continue

        b_prev = boundaries[i - 1]
        b_nxt = boundaries[i]
        if b_prev is None or b_nxt is None:
            reason = "pose_missing"
            reasons[reason] = reasons.get(reason, 0) + 1
            groups.append(curr)
            curr = [nxt]
            continue

        trans_jump = float(np.linalg.norm(b_nxt["t0"] - b_prev["t1"]))
        rot_jump_deg = _rotation_angle_deg(b_prev["R1"], b_nxt["R0"])
        speed = trans_jump / dt_sec
        rot_speed = rot_jump_deg / dt_sec

        if speed > max_speed:
            reason = "speed"
            reasons[reason] = reasons.get(reason, 0) + 1
            groups.append(curr)
            curr = [nxt]
            continue

        if rot_speed > max_rot_deg_s:
            reason = "rot"
            reasons[reason] = reasons.get(reason, 0) + 1
            groups.append(curr)
            curr = [nxt]
            continue

        curr.append(nxt)

    groups.append(curr)
    return groups, reasons


def _build_video_map(video_dir: Path) -> Dict[str, List[Path]]:
    m: Dict[str, List[Path]] = {}
    for p in video_dir.iterdir():
        if not p.is_file():
            continue
        vid = _guess_youtube_id_from_filename(p.name)
        if vid is None:
            continue
        m.setdefault(vid, []).append(p)
    return m


def _normalize_video_id(s: str) -> str:
    s = str(s).strip()
    if not s:
        return ""

    s = Path(s).name
    if s.endswith(".part"):
        s = s[: -len(".part")]

    for ext in (".npz", ".mp4", ".webm"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
            break

    s = re.sub(r"\.f\d+$", "", s)
    if s.endswith("_processed"):
        s = s[: -len("_processed")]

    m = re.match(r"^(?P<vid>[A-Za-z0-9_-]{11})_\d{7}_\d{7}$", s)
    if m:
        s = m.group("vid")
    return s


def _parse_only_video_ids(args) -> Optional[set[str]]:
    tokens: List[str] = []

    if getattr(args, "only_videos_file", None):
        p = Path(str(args.only_videos_file))
        if not p.exists():
            raise FileNotFoundError(f"--only_videos_file not found: {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens.append(line)

    if getattr(args, "only_videos", None):
        tokens.extend([str(x) for x in args.only_videos])

    if not tokens:
        return None

    out: set[str] = set()
    for t in tokens:
        for part in str(t).split(","):
            vid = _normalize_video_id(part)
            if vid:
                out.add(vid)
    return out or None


def _compute_frame_sample_indices(
    src_len: int,
    *,
    source_fps: float,
    out_fps: float,
    out_num_frames: int,
) -> np.ndarray:
    if src_len <= 0:
        raise ValueError(f"src_len must be > 0, got {src_len}")
    if source_fps <= 0:
        raise ValueError(f"source_fps must be > 0, got {source_fps}")
    if out_fps <= 0:
        raise ValueError(f"out_fps must be > 0, got {out_fps}")
    if out_fps > source_fps + 1e-9:
        raise ValueError(
            f"out_fps ({out_fps}) > source_fps ({source_fps}) is not supported"
        )

    if int(out_num_frames) > 0:
        out_len = int(out_num_frames)
    else:
        out_len = int(
            math.floor(float(src_len) * float(out_fps) / float(source_fps) + 1e-9)
        )
    out_len = max(1, out_len)

    # For out_fps=1 and source_fps=30: 0,30,60,...,270.
    ratio = float(source_fps) / float(out_fps)
    idx = np.floor(np.arange(out_len, dtype=np.float64) * ratio + 1e-9).astype(np.int64)
    idx = idx[idx < int(src_len)]
    if idx.size == 0:
        idx = np.array([0], dtype=np.int64)
    return idx


def _process_one_video(
    video_id: str,
    clips: Sequence[ClipSpec],
    *,
    out_name: Optional[str] = None,
    video_map: Dict[str, List[Path]],
    video_dims_cache: Dict[Path, Tuple[int, int]],
    args,
) -> Tuple[str, str]:
    out_name = out_name or f"{video_id}_processed"
    out_dir = Path(args.output_root) / out_name
    traj_pkl = out_dir / "traj_data.pkl"

    source_fps = float(args.source_fps)
    out_fps = float(args.out_fps)
    source_num_frames = int(args.source_num_frames)

    sample_idx = _compute_frame_sample_indices(
        source_num_frames,
        source_fps=source_fps,
        out_fps=out_fps,
        out_num_frames=int(args.out_num_frames),
    )
    frames_per_clip = int(sample_idx.shape[0])
    expected_frames = int(frames_per_clip * len(clips))

    if traj_pkl.exists():
        n_jpg = _count_jpgs(out_dir)
        if (not bool(args.overwrite)) and n_jpg == expected_frames:
            return ("skip", out_name)
        if not bool(args.overwrite):
            return ("exists_mismatch", out_name)

    candidates = video_map.get(video_id) or []
    if not candidates:
        return ("missing_video", out_name)
    video_path = _choose_best_video(candidates)

    if video_path not in video_dims_cache:
        video_dims_cache[video_path] = _ffprobe_video_dims(video_path)
    in_w, in_h = video_dims_cache[video_path]

    for c in clips:
        if c.num_frames != source_num_frames:
            return ("bad_len", out_name)

    if out_dir.exists():
        if bool(args.overwrite) or not traj_pkl.exists():
            shutil.rmtree(out_dir)
        else:
            return ("exists_mismatch", out_name)
    out_dir.mkdir(parents=True, exist_ok=False)

    try:
        if str(args.K_mode) == "fov":
            K = _compute_intrinsic_matrix_from_fov(
                int(args.out_w), int(args.out_h), float(args.fov_x_degree)
            )
        elif str(args.K_mode) == "sekai":
            # In video mode, K must be constant across the whole trajectory; use the first clip.
            with np.load(clips[0].npz_path) as d0:
                K_norm0 = np.asarray(d0["intrinsic"], dtype=np.float64)
            K = _compute_intrinsics_pixel(
                K_norm0,
                in_w=in_w,
                in_h=in_h,
                crop_square=bool(args.crop_square),
                out_w=int(args.out_w),
                out_h=int(args.out_h),
            )
        else:
            raise ValueError(f"Unknown K_mode: {args.K_mode}")

        pose_list: List[np.ndarray] = []
        point_list: List[np.ndarray] = []
        position_list: List[np.ndarray] = []
        yaw_list: List[np.ndarray] = []
        pitch_list: List[np.ndarray] = []
        roll_list: List[np.ndarray] = []
        abs_timestamps: List[np.ndarray] = []

        frame_counter = 0
        for clip_idx, clip in enumerate(clips, start=1):
            start_sec = float(clip.s_frame) / float(source_fps)
            _ffmpeg_extract_frames(
                video_path=video_path,
                start_sec=start_sec,
                fps=float(out_fps),
                num_frames=int(frames_per_clip),
                out_dir=out_dir,
                crop_square=bool(args.crop_square),
                out_w=int(args.out_w),
                out_h=int(args.out_h),
                jpeg_q=int(args.jpeg_q),
                start_number=int(frame_counter),
            )

            for j in range(frame_counter, frame_counter + frames_per_clip):
                if not (out_dir / f"{j}.jpg").exists():
                    raise RuntimeError(
                        f"Missing extracted frame: {out_dir / f'{j}.jpg'}"
                    )

            with np.load(clip.npz_path) as d:
                pose_full = np.asarray(d["extrinsic"], dtype=np.float64)
                if str(args.K_mode) == "sekai":
                    K_norm = np.asarray(d["intrinsic"], dtype=np.float64)
                    K_i = _compute_intrinsics_pixel(
                        K_norm,
                        in_w=in_w,
                        in_h=in_h,
                        crop_square=bool(args.crop_square),
                        out_w=int(args.out_w),
                        out_h=int(args.out_h),
                    )
                    if not np.allclose(K_i, K, rtol=0.0, atol=1e-5):
                        print(
                            f"[warn] {video_id}: K varies across clips; using the first clip K.",
                            file=sys.stderr,
                        )

            if pose_full.shape != (source_num_frames, 4, 4):
                raise ValueError(
                    f"Unexpected pose shape {pose_full.shape} for {clip.npz_path}"
                )

            if str(args.pose_convention) == "w2c":
                pose_full = np.linalg.inv(pose_full)

            pose = pose_full[sample_idx]

            R = pose[:, :3, :3]
            yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0]) + float(args.yaw_offset)
            yaw = _wrap_pi(yaw)

            if str(args.pitch_roll) == "pose":
                pitch = np.arctan2(
                    -R[:, 2, 0], np.sqrt(R[:, 2, 1] ** 2 + R[:, 2, 2] ** 2)
                )
                pitch = _wrap_pi(pitch)

                roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
                roll = _wrap_pi(roll)
            elif str(args.pitch_roll) == "zero":
                pitch = np.zeros_like(yaw, dtype=np.float64)
                roll = np.zeros_like(yaw, dtype=np.float64)
            else:
                raise ValueError(f"Unknown pitch_roll: {args.pitch_roll}")

            point = pose[:, :3, 3].astype(np.float64)
            position = point[:, :2].astype(np.float64)

            pose_list.append(pose.astype(np.float64))
            point_list.append(point.astype(np.float64))
            position_list.append(position.astype(np.float64))
            yaw_list.append(yaw.astype(np.float64))
            pitch_list.append(pitch.astype(np.float64))
            roll_list.append(roll.astype(np.float64))
            abs_timestamps.append(
                (
                    (float(clip.s_frame) + sample_idx.astype(np.float64))
                    / float(source_fps)
                ).astype(np.float64)
            )

            frame_counter += frames_per_clip
            if int(args.inner_log_every) and int(args.inner_log_every) > 0:
                if clip_idx % int(args.inner_log_every) == 0:
                    print(
                        f"  - clips {clip_idx:>4}/{len(clips):<4} ({video_id})",
                        file=sys.stderr,
                    )

        pose_cat = np.concatenate(pose_list, axis=0)
        point_cat = np.concatenate(point_list, axis=0)
        position_cat = np.concatenate(position_list, axis=0)
        yaw_cat = np.concatenate(yaw_list, axis=0)
        pitch_cat = np.concatenate(pitch_list, axis=0)
        roll_cat = np.concatenate(roll_list, axis=0)
        abs_ts_cat = np.concatenate(abs_timestamps, axis=0)

        if pose_cat.shape[0] != expected_frames:
            raise RuntimeError(
                f"Expected {expected_frames} frames, got {pose_cat.shape[0]}"
            )

        if str(args.timestamps_mode) == "relative":
            timestamps = (
                np.arange(expected_frames, dtype=np.float64) / float(out_fps)
            ).astype(np.float64)
        elif str(args.timestamps_mode) == "absolute":
            timestamps = abs_ts_cat.astype(np.float64)
        else:
            raise ValueError(f"Unknown timestamps_mode: {args.timestamps_mode}")

        images = np.asarray([str(i) for i in range(expected_frames)], dtype=str)

        traj_data = {
            "K": K.astype(np.float64),
            "position": position_cat.astype(np.float64),
            "yaw": yaw_cat.astype(np.float64),
            "timestamps": timestamps.astype(np.float64),
            "images": images,
            "pitch": pitch_cat.astype(np.float64),
            "roll": roll_cat.astype(np.float64),
            "point": point_cat.astype(np.float64),
            "pose": pose_cat.astype(np.float64),
        }

        with traj_pkl.open("wb") as f:
            pickle.dump(traj_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return ("ok", out_name)
    except Exception:
        if not args.keep_failed:
            shutil.rmtree(out_dir, ignore_errors=True)
        raise


def _process_one_clip(
    clip: ClipSpec,
    *,
    video_map: Dict[str, List[Path]],
    video_dims_cache: Dict[Path, Tuple[int, int]],
    args,
) -> Tuple[str, str]:
    out_dir = Path(args.output_root) / clip.traj_name
    traj_pkl = out_dir / "traj_data.pkl"

    source_fps = float(args.source_fps)
    out_fps = float(args.out_fps)
    source_num_frames = int(args.source_num_frames)
    sample_idx = _compute_frame_sample_indices(
        source_num_frames,
        source_fps=source_fps,
        out_fps=out_fps,
        out_num_frames=int(args.out_num_frames),
    )
    out_num_frames = int(sample_idx.shape[0])

    if traj_pkl.exists():
        n_jpg = _count_jpgs(out_dir)
        if (not bool(args.overwrite)) and n_jpg == out_num_frames:
            return ("skip", clip.traj_name)
        if not bool(args.overwrite):
            return ("exists_mismatch", clip.traj_name)

    candidates = video_map.get(clip.video_id) or []
    if not candidates:
        return ("missing_video", clip.traj_name)
    video_path = _choose_best_video(candidates)

    if video_path not in video_dims_cache:
        video_dims_cache[video_path] = _ffprobe_video_dims(video_path)
    in_w, in_h = video_dims_cache[video_path]

    if clip.num_frames != source_num_frames:
        return ("bad_len", clip.traj_name)

    if out_dir.exists():
        if bool(args.overwrite) or not traj_pkl.exists():
            # remove partial output
            shutil.rmtree(out_dir)
        else:
            return ("exists_mismatch", clip.traj_name)
    out_dir.mkdir(parents=True, exist_ok=False)

    try:
        start_sec = float(clip.s_frame) / float(source_fps)

        _ffmpeg_extract_frames(
            video_path=video_path,
            start_sec=start_sec,
            fps=float(out_fps),
            num_frames=int(out_num_frames),
            out_dir=out_dir,
            crop_square=bool(args.crop_square),
            out_w=int(args.out_w),
            out_h=int(args.out_h),
            jpeg_q=int(args.jpeg_q),
        )

        n_jpg = _count_jpgs(out_dir)
        if n_jpg != int(out_num_frames):
            raise RuntimeError(f"Extracted {n_jpg} jpgs, expected {out_num_frames}")

        with np.load(clip.npz_path) as d:
            K_norm = np.asarray(d["intrinsic"], dtype=np.float64)
            pose_full = np.asarray(d["extrinsic"], dtype=np.float64)

        if pose_full.shape != (source_num_frames, 4, 4):
            raise ValueError(
                f"Unexpected pose shape {pose_full.shape} for {clip.npz_path}"
            )

        if str(args.pose_convention) == "w2c":
            pose_full = np.linalg.inv(pose_full)
        pose = pose_full[sample_idx]

        if str(args.K_mode) == "fov":
            K = _compute_intrinsic_matrix_from_fov(
                int(args.out_w), int(args.out_h), float(args.fov_x_degree)
            )
        elif str(args.K_mode) == "sekai":
            K = _compute_intrinsics_pixel(
                K_norm,
                in_w=in_w,
                in_h=in_h,
                crop_square=bool(args.crop_square),
                out_w=int(args.out_w),
                out_h=int(args.out_h),
            )
        else:
            raise ValueError(f"Unknown K_mode: {args.K_mode}")

        R = pose[:, :3, :3]
        yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0]) + float(args.yaw_offset)
        yaw = _wrap_pi(yaw)

        if str(args.pitch_roll) == "pose":
            pitch = np.arctan2(-R[:, 2, 0], np.sqrt(R[:, 2, 1] ** 2 + R[:, 2, 2] ** 2))
            pitch = _wrap_pi(pitch)

            roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
            roll = _wrap_pi(roll)
        elif str(args.pitch_roll) == "zero":
            pitch = np.zeros_like(yaw, dtype=np.float64)
            roll = np.zeros_like(yaw, dtype=np.float64)
        else:
            raise ValueError(f"Unknown pitch_roll: {args.pitch_roll}")

        point = pose[:, :3, 3].astype(np.float64)
        position = point[:, :2].astype(np.float64)

        if str(args.timestamps_mode) == "relative":
            timestamps = (
                np.arange(out_num_frames, dtype=np.float64) / float(out_fps)
            ).astype(np.float64)
        elif str(args.timestamps_mode) == "absolute":
            timestamps = (
                (float(clip.s_frame) + sample_idx.astype(np.float64))
                / float(source_fps)
            ).astype(np.float64)
        else:
            raise ValueError(f"Unknown timestamps_mode: {args.timestamps_mode}")

        images = np.asarray([str(i) for i in range(out_num_frames)], dtype=str)

        traj_data = {
            "K": K.astype(np.float64),
            "position": position.astype(np.float64),
            "yaw": yaw.astype(np.float64),
            "timestamps": timestamps.astype(np.float64),
            "images": images,
            "pitch": pitch.astype(np.float64),
            "roll": roll.astype(np.float64),
            "point": point.astype(np.float64),
            "pose": pose.astype(np.float64),
        }

        with traj_pkl.open("wb") as f:
            pickle.dump(traj_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return ("ok", clip.traj_name)
    except Exception:
        if not args.keep_failed:
            shutil.rmtree(out_dir, ignore_errors=True)
        raise


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert Sekai-Real-Drone clips to the trajectory format used by NWM v5_3."
    )
    ap.add_argument("--input_root", default="data/sekai_raw")
    ap.add_argument(
        "--traj_dir",
        default=None,
        help="Default: <input_root>/camera_trajectories/sekai-real-drone",
    )
    ap.add_argument("--video_dir", default=None, help="Default: <input_root>/videos")
    ap.add_argument(
        "--output_root",
        default=None,
        help="Default: <input_root>/outputs.",
    )

    ap.add_argument(
        "--scene",
        choices=["outdoor-urban", "outdoor-natural", "all"],
        default="outdoor-urban",
        help="Only process clips whose metadata 'scene' matches this value.",
    )
    ap.add_argument(
        "--meta_csv",
        default=None,
        help="Path to sekai-real-drone.csv (for --scene filtering). If omitted, auto-detected from the workspace.",
    )

    ap.add_argument(
        "--source_fps",
        "--fps",
        type=float,
        default=30.0,
        help="Trajectory fps in sekai .npz (default 30).",
    )
    ap.add_argument(
        "--source_num_frames",
        "--num_frames",
        type=int,
        default=300,
        help="Expected number of frames (poses) in each sekai .npz clip (default 300).",
    )
    ap.add_argument(
        "--out_fps",
        type=float,
        default=1.0,
        help="Output image fps (default 1.0 => 1 image per second).",
    )
    ap.add_argument(
        "--out_num_frames",
        type=int,
        default=0,
        help="Override output frames per clip (0 = auto from source_num_frames/source_fps/out_fps).",
    )
    ap.add_argument(
        "--timestamps_mode",
        choices=["relative", "absolute"],
        default="relative",
        help="relative: timestamps start at 0; absolute: derived from clip start frame and source_fps.",
    )

    ap.add_argument(
        "--crop_square",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Center-crop to square before resize (default: enabled).",
    )
    ap.add_argument("--out_w", type=int, default=512)
    ap.add_argument("--out_h", type=int, default=512)
    ap.add_argument(
        "--jpeg_q", type=int, default=2, help="ffmpeg -q:v (lower is better)."
    )

    ap.add_argument(
        "--yaw_offset",
        type=float,
        default=0.0,
        help="Yaw = atan2(R10,R00) + yaw_offset.",
    )
    ap.add_argument(
        "--pitch_roll",
        choices=["zero", "pose"],
        default="zero",
        help="How to fill pitch/roll. 'zero' matches the released preprocessing; 'pose' derives from rotation.",
    )

    ap.add_argument(
        "--pose_convention",
        choices=["c2w", "w2c"],
        default="c2w",
        help="Expected convention for npz 'extrinsic'. Output 'pose' is always c2w (camera-to-world).",
    )

    ap.add_argument(
        "--K_mode",
        choices=["fov", "sekai"],
        default="fov",
        help="How to write K in traj_data.pkl. 'fov' uses a fixed FOV; 'sekai' uses per-clip intrinsics.",
    )
    ap.add_argument(
        "--fov_x_degree", type=float, default=90.0, help="Used when --K_mode=fov."
    )

    ap.add_argument(
        "--group_by",
        choices=["trajectory", "video", "clip"],
        default="video",
        help=(
            "trajectory: split each video into continuous trajectories (recommended for edited videos); "
            "video: force-merge all clips of a video; "
            "clip: one trajectory per 10s clip (.npz)."
        ),
    )
    ap.add_argument(
        "--only_videos",
        nargs="*",
        default=None,
        help="Only process these video IDs (accepts raw IDs or names like '<id>_processed').",
    )
    ap.add_argument(
        "--only_videos_file",
        type=str,
        default=None,
        help="Text file with one video ID per line. Lines starting with # are ignored.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N items (trajectories/videos/clips depending on --group_by; 0 = all).",
    )
    ap.add_argument(
        "--log_every",
        type=int,
        default=0,
        help="Progress logging frequency for outer loop (0 = auto).",
    )
    ap.add_argument(
        "--inner_log_every",
        type=int,
        default=0,
        help="(trajectory/video mode) Progress logging frequency inside a clip-sequence over its clips (0 = auto).",
    )
    ap.add_argument(
        "--traj_max_gap_sec",
        type=float,
        default=1.0,
        help="(trajectory mode) Max allowed gap (seconds) between adjacent clips to be merged.",
    )
    ap.add_argument(
        "--traj_max_speed",
        type=float,
        default=1.0,
        help="(trajectory mode) Max allowed translation speed (units/sec) at clip boundary.",
    )
    ap.add_argument(
        "--traj_max_rot_deg_s",
        type=float,
        default=90.0,
        help="(trajectory mode) Max allowed rotation rate (deg/sec) at clip boundary.",
    )
    ap.add_argument(
        "--keep_failed",
        action="store_true",
        help="Keep partial outputs on failure for debugging.",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Re-create already processed clips."
    )
    ap.add_argument(
        "--write_traj_names",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write <output_root>/traj_names.txt.",
    )

    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    input_root = Path(args.input_root)
    traj_dir = (
        Path(args.traj_dir)
        if args.traj_dir
        else (input_root / "camera_trajectories" / "sekai-real-drone")
    )
    video_dir = Path(args.video_dir) if args.video_dir else (input_root / "videos")
    output_root = (
        Path(args.output_root) if args.output_root else (input_root / "outputs")
    )
    args.output_root = str(output_root)

    if not traj_dir.exists():
        print(f"[error] traj_dir not found: {traj_dir}", file=sys.stderr)
        return 2
    if not video_dir.exists():
        print(f"[error] video_dir not found: {video_dir}", file=sys.stderr)
        return 2

    if float(args.out_fps) > float(args.source_fps) + 1e-9:
        print(
            f"[error] out_fps ({args.out_fps}) must be <= source_fps ({args.source_fps})",
            file=sys.stderr,
        )
        return 2

    clips = _discover_clips(traj_dir)
    if str(args.scene) != "all":
        meta_csv = _resolve_sekai_real_drone_csv(input_root, args.meta_csv)
        if meta_csv is None or not meta_csv.exists():
            print(
                "[error] --scene filtering requires sekai-real-drone.csv. Provide --meta_csv or place it under input_root.",
                file=sys.stderr,
            )
            return 2
        allowed = _load_allowed_camera_files(meta_csv, str(args.scene))
        before = len(clips)
        clips = [c for c in clips if c.npz_path.name in allowed]
        print(
            f"[ok] scene={args.scene}: {len(clips)}/{before} clips (meta_csv: {meta_csv})"
        )

    try:
        only_video_ids = _parse_only_video_ids(args)
    except Exception as e:
        print(
            f"[error] Failed to parse --only_videos/--only_videos_file: {e}",
            file=sys.stderr,
        )
        return 2

    if only_video_ids:
        before = len(clips)
        clips = [c for c in clips if c.video_id in only_video_ids]
        present = {c.video_id for c in clips}
        missing = sorted(only_video_ids - present)
        print(
            f"[ok] only_videos: {len(clips)}/{before} clips (videos: {len(present)}/{len(only_video_ids)})"
        )
        if missing:
            print(f"[warn] only_videos missing: {', '.join(missing)}", file=sys.stderr)

    if not clips:
        print(f"[error] No clips discovered under: {traj_dir}", file=sys.stderr)
        return 2

    video_map = _build_video_map(video_dir)
    if not video_map:
        print(f"[error] No videos discovered under: {video_dir}", file=sys.stderr)
        return 2

    output_root.mkdir(parents=True, exist_ok=True)

    source_fps = float(args.source_fps)
    out_fps = float(args.out_fps)
    source_num_frames = int(args.source_num_frames)
    sample_idx = _compute_frame_sample_indices(
        source_num_frames,
        source_fps=source_fps,
        out_fps=out_fps,
        out_num_frames=int(args.out_num_frames),
    )
    frames_per_clip = int(sample_idx.shape[0])

    if int(args.inner_log_every) > 0:
        args.inner_log_every = int(args.inner_log_every)
    else:
        args.inner_log_every = (
            50 if str(args.group_by) in {"video", "trajectory"} else 0
        )

    clips_by_video = _group_clips_by_video(clips)

    items_total = 0
    selected_clips: List[ClipSpec] = []
    seq_items: List[Tuple[str, str, List[ClipSpec]]] = []
    split_reason_counts: Dict[str, int] = {}
    multi_traj_videos_total = 0
    video_total = len(clips_by_video)
    traj_total = 0

    if str(args.group_by) == "clip":
        selected_clips = clips
        if args.limit and int(args.limit) > 0:
            selected_clips = selected_clips[: int(args.limit)]
        items_total = len(selected_clips)
    elif str(args.group_by) == "video":
        video_ids = sorted(clips_by_video.keys())
        if args.limit and int(args.limit) > 0:
            video_ids = video_ids[: int(args.limit)]
        for vid in video_ids:
            c = list(clips_by_video[vid])
            selected_clips.extend(c)
            seq_items.append((vid, f"{vid}_processed", c))
        items_total = len(seq_items)
    elif str(args.group_by) == "trajectory":
        for vid in sorted(clips_by_video.keys()):
            traj_groups, reasons = _split_video_clips_into_trajectories(
                vid, clips_by_video[vid], args=args
            )
            if len(traj_groups) > 1:
                multi_traj_videos_total += 1
            for k, v in reasons.items():
                split_reason_counts[k] = split_reason_counts.get(k, 0) + int(v)
            for g in traj_groups:
                start_f = int(g[0].s_frame)
                end_f = int(g[-1].e_frame)
                out_name = f"{vid}_{start_f:07d}_{end_f:07d}_processed"
                seq_items.append((vid, out_name, list(g)))

        traj_total = len(seq_items)
        seq_items.sort(key=lambda x: (x[0], x[2][0].s_frame, x[2][-1].e_frame))
        if args.limit and int(args.limit) > 0:
            seq_items = seq_items[: int(args.limit)]

        for _, _, c in seq_items:
            selected_clips.extend(c)
        items_total = len(seq_items)
    else:
        raise ValueError(f"Unknown group_by: {args.group_by}")

    if int(args.log_every) > 0:
        log_every = int(args.log_every)
    else:
        if str(args.group_by) == "video":
            log_every = 1
        elif str(args.group_by) == "clip":
            log_every = 50
        else:
            target_lines = 50
            log_every = max(1, int(items_total) // int(target_lines))

    print("=" * 80)
    print("Sekai-Real-Drone YouTube preprocessing")
    print(f"input_root  : {input_root}")
    print(f"traj_dir    : {traj_dir}")
    print(f"video_dir   : {video_dir}")
    print(f"output_root : {output_root}")
    print(f"scene       : {args.scene}")
    print(f"group_by    : {args.group_by}")
    print(f"source      : {source_fps:g} fps, {source_num_frames} frames/clip")
    print(f"output      : {out_fps:g} fps, {frames_per_clip} frames/clip")
    print(
        f"size        : {int(args.out_w)}x{int(args.out_h)} (crop_square={bool(args.crop_square)})"
    )
    if str(args.group_by) == "clip":
        print(f"trajectories: {items_total} (each = 1 clip)")
    elif str(args.group_by) == "video":
        print(f"videos      : {items_total}")
    else:
        uniq_v = {vid for vid, _, _ in seq_items}
        per_vid_counts: Dict[str, int] = {}
        for vid, _, _ in seq_items:
            per_vid_counts[vid] = per_vid_counts.get(vid, 0) + 1
        multi_traj_videos_sel = sum(1 for v in per_vid_counts.values() if v > 1)

        v_total_str = str(video_total) if video_total else "0"
        t_total_str = str(traj_total) if traj_total else str(items_total)
        print(
            f"videos      : {len(uniq_v)}/{v_total_str} (multi-traj: {multi_traj_videos_sel}/{multi_traj_videos_total})"
        )
        print(f"trajectories: {items_total}/{t_total_str}")
        if split_reason_counts:
            parts = [f"{k}={v}" for k, v in sorted(split_reason_counts.items())]
            print(f"splits(all) : {', '.join(parts)}")
    print(f"clips       : {len(selected_clips)}")
    print("=" * 80)

    video_dims_cache: Dict[Path, Tuple[int, int]] = {}
    ok: List[str] = []
    skipped = 0
    exists_mismatch = 0
    missing_video = 0
    bad_len = 0
    failed = 0

    if str(args.group_by) == "clip":
        for idx, clip in enumerate(selected_clips, start=1):
            try:
                status, name = _process_one_clip(
                    clip,
                    video_map=video_map,
                    video_dims_cache=video_dims_cache,
                    args=args,
                )
            except Exception as e:
                failed += 1
                print(f"[fail] {clip.traj_name}: {e}", file=sys.stderr)
                continue

            if status == "ok":
                ok.append(name)
            elif status == "skip":
                skipped += 1
            elif status == "exists_mismatch":
                exists_mismatch += 1
            elif status == "missing_video":
                missing_video += 1
            elif status == "bad_len":
                bad_len += 1

            if log_every and (idx % log_every == 0 or idx == items_total):
                print(
                    f"[{idx:>5}/{items_total:<5}] ok={len(ok):>5} skip={skipped:>5} exist={exists_mismatch:>5} miss={missing_video:>4} bad={bad_len:>4} fail={failed:>4}",
                    file=sys.stderr,
                )
    else:
        for idx, (vid, out_name, clips_for_vid) in enumerate(seq_items, start=1):
            try:
                status, name = _process_one_video(
                    vid,
                    clips_for_vid,
                    out_name=out_name,
                    video_map=video_map,
                    video_dims_cache=video_dims_cache,
                    args=args,
                )
            except Exception as e:
                failed += 1
                print(f"[fail] {out_name}: {e}", file=sys.stderr)
                continue

            if status == "ok":
                ok.append(name)
            elif status == "skip":
                skipped += 1
            elif status == "exists_mismatch":
                exists_mismatch += 1
            elif status == "missing_video":
                missing_video += 1
            elif status == "bad_len":
                bad_len += 1

            if log_every and (idx % log_every == 0 or idx == items_total):
                label = "vid" if str(args.group_by) == "video" else "traj"
                print(
                    f"[{idx:>3}/{items_total:<3}] {label}={out_name} clips={len(clips_for_vid):>4} frames={len(clips_for_vid)*frames_per_clip:>6} status={status}",
                    file=sys.stderr,
                )

    print(f"[done] output_root: {output_root.resolve()}")
    print(
        f"[done] ok: {len(ok)}, skipped: {skipped}, exists_mismatch: {exists_mismatch}, missing_video: {missing_video}, bad_len: {bad_len}, failed: {failed}"
    )
    if exists_mismatch and not bool(args.overwrite):
        print(
            "[hint] Some outputs already exist but don't match the current settings. Re-run with --overwrite or change --output_root.",
            file=sys.stderr,
        )

    if bool(args.write_traj_names):
        traj_names_path = output_root / "traj_names.txt"
        with traj_names_path.open("w", encoding="utf-8") as f:
            names: List[str] = []
            if str(args.group_by) == "clip":
                for clip in selected_clips:
                    p = output_root / clip.traj_name
                    if not (p / "traj_data.pkl").exists():
                        continue
                    if _count_jpgs(p) != frames_per_clip:
                        continue
                    names.append(clip.traj_name)
            else:
                for _, out_name, clips_for_item in seq_items:
                    expected_frames = int(len(clips_for_item) * frames_per_clip)
                    p = output_root / out_name
                    if not (p / "traj_data.pkl").exists():
                        continue
                    if _count_jpgs(p) != expected_frames:
                        continue
                    names.append(out_name)

            for n in sorted(set(names)):
                f.write(n + "\n")
        print(f"[done] traj_names: {traj_names_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
