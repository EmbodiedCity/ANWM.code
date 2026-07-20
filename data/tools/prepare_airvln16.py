#!/usr/bin/env python3
"""Convert AirVLN scene-16 observations to the trajectory format used by ANWM."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMERA_FRAME_TRANSFORM = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert AirVLN scene-16 RGB, depth, and waypoint annotations to "
            "ANWM trajectory folders."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="AirVLN root containing traj_obs/ and waypoints/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "airvln_16",
        help="Destination for <trajectory_id>_processed folders.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "airvln_16",
        help="Released split root used to select trajectories.",
    )
    parser.add_argument(
        "--episode-files",
        nargs="+",
        type=Path,
        help=(
            "Episode JSON files. Relative paths are resolved under "
            "<source-root>/waypoints; defaults to train.json and val_seen.json."
        ),
    )
    parser.add_argument("--scene-id", type=int, default=16)
    parser.add_argument("--skip-frames", type=int, default=11)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--horizontal-fov", type=float, default=90.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace trajectory folders that already exist.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Process available released trajectories without failing on missing annotations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate annotations and released split membership without writing data.",
    )
    parser.add_argument(
        "--limit", type=int, help="Process at most this many trajectories."
    )
    return parser


def intrinsic_matrix(width: int, height: int, horizontal_fov: float) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")
    if not 0.0 < horizontal_fov < 180.0:
        raise ValueError("Horizontal field of view must be between 0 and 180 degrees")

    fov_x = np.radians(horizontal_fov)
    fov_y = 2.0 * np.arctan((height / width) * np.tan(fov_x / 2.0))
    return np.array(
        [
            [width / (2.0 * np.tan(fov_x / 2.0)), 0.0, width / 2.0],
            [0.0, height / (2.0 * np.tan(fov_y / 2.0)), height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def pose_matrix(
    x: float, y: float, z: float, roll: float, pitch: float, yaw: float
) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_euler("zyx", [yaw, pitch, roll]).as_matrix()
    pose[:3, 3] = [x, y, z]
    return pose @ CAMERA_FRAME_TRANSFORM


def resolve_episode_files(
    source_root: Path, values: Sequence[Path] | None
) -> List[Path]:
    values = list(values or (Path("train.json"), Path("val_seen.json")))
    waypoint_root = source_root / "waypoints"
    return [path if path.is_absolute() else waypoint_root / path for path in values]


def load_episodes(paths: Iterable[Path]) -> List[dict]:
    episodes: List[dict] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Episode file does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        file_episodes = (
            payload.get("episodes") if isinstance(payload, dict) else payload
        )
        if not isinstance(file_episodes, list):
            raise ValueError(f"Expected an episode list in {path}")
        episodes.extend(file_episodes)
    return episodes


def read_released_names(split_root: Path) -> List[str]:
    names: List[str] = []
    for split in ("train", "test"):
        path = split_root / split / "traj_names.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Released split file does not exist: {path}")
        names.extend(
            line.strip() for line in path.read_text().splitlines() if line.strip()
        )
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate trajectory names found under {split_root}")
    return names


def select_episodes(
    episodes: Iterable[dict], released_names: Sequence[str], scene_id: int
) -> Tuple[List[Tuple[str, dict]], List[str]]:
    by_output_name: Dict[str, dict] = {}
    for episode in episodes:
        if int(episode.get("scene_id", -1)) != scene_id:
            continue
        trajectory_id = str(episode.get("trajectory_id", "")).strip()
        reference_path = episode.get("reference_path")
        if not trajectory_id or not isinstance(reference_path, list):
            raise ValueError(
                "Each selected episode needs trajectory_id and reference_path"
            )
        output_name = f"{trajectory_id}_processed"
        if output_name in by_output_name:
            raise ValueError(f"Duplicate trajectory annotation: {trajectory_id}")
        by_output_name[output_name] = episode

    selected = [
        (name, by_output_name[name])
        for name in released_names
        if name in by_output_name
    ]
    missing = [name for name in released_names if name not in by_output_name]
    return selected, missing


def parse_waypoint(
    value: object, trajectory_name: str, frame: int
) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) < 6:
        raise ValueError(
            f"{trajectory_name} frame {frame}: expected [x, y, z, roll, pitch, yaw]"
        )
    return tuple(float(component) for component in value[:6])


def convert_episode(
    source_root: Path,
    output_name: str,
    episode: dict,
    camera_matrix: np.ndarray,
    skip_frames: int,
) -> Tuple[dict, List[Tuple[Path, str]]]:
    trajectory_id = str(episode["trajectory_id"])
    trajectory_root = source_root / "traj_obs" / trajectory_id
    reference_path = episode["reference_path"]
    if len(reference_path) <= skip_frames:
        raise ValueError(
            f"{output_name}: only {len(reference_path)} waypoints, cannot skip {skip_frames}"
        )

    positions = []
    points = []
    yaws = []
    pitches = []
    rolls = []
    timestamps = []
    image_names = []
    depths = []
    poses = []
    image_copies: List[Tuple[Path, str]] = []

    for output_frame, source_frame in enumerate(
        range(skip_frames, len(reference_path))
    ):
        x, y, z, roll, pitch, yaw = parse_waypoint(
            reference_path[source_frame], output_name, source_frame
        )
        image_path = trajectory_root / "rgb" / f"rgb_obs_front_{source_frame}.png"
        depth_path = trajectory_root / "dep" / f"dep_obs_front_{source_frame}.npy"
        if not image_path.is_file():
            raise FileNotFoundError(f"{output_name}: missing RGB frame {image_path}")
        if not depth_path.is_file():
            raise FileNotFoundError(f"{output_name}: missing depth frame {depth_path}")

        depth = np.load(depth_path, allow_pickle=False)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(
                f"{output_name}: expected 2D depth at {depth_path}, got {depth.shape}"
            )

        positions.append([x, y])
        points.append([x, y, z])
        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw)
        timestamps.append(float(source_frame))
        image_names.append(str(output_frame))
        depths.append(depth)
        poses.append(pose_matrix(x, y, z, roll, pitch, yaw))
        image_copies.append((image_path, f"{output_frame}.jpg"))

    trajectory = {
        "K": camera_matrix,
        "position": np.asarray(positions),
        "yaw": np.asarray(yaws),
        "timestamps": np.asarray(timestamps),
        "images": np.asarray(image_names),
        "pitch": np.asarray(pitches),
        "roll": np.asarray(rolls),
        "point": np.asarray(points),
        "depth": np.stack(depths),
        "pose": np.stack(poses),
    }
    return trajectory, image_copies


def write_trajectory(
    output_root: Path,
    output_name: str,
    trajectory: dict,
    image_copies: Sequence[Tuple[Path, str]],
    overwrite: bool,
) -> bool:
    target = output_root / output_name
    if target.exists() and not overwrite:
        return False
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_name}.", dir=output_root))
    try:
        for source, filename in image_copies:
            # Preserve the released pipeline: PNG bytes are copied to numeric .jpg names.
            shutil.copy2(source, temporary / filename)
        with (temporary / "traj_data.pkl").open("wb") as handle:
            pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return True


def main() -> int:
    args = build_parser().parse_args()
    if args.skip_frames < 0:
        raise ValueError("--skip-frames must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    split_root = args.split_root.expanduser().resolve()
    episode_files = resolve_episode_files(source_root, args.episode_files)
    episodes = load_episodes(episode_files)
    released_names = read_released_names(split_root)
    selected, missing = select_episodes(episodes, released_names, args.scene_id)
    if missing and not args.allow_missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Missing annotations for {len(missing)} released trajectories: {preview}. "
            "Use --allow-missing only for partial data preparation."
        )
    if args.limit is not None:
        selected = selected[: args.limit]

    print(
        f"Selected {len(selected)} trajectories for scene {args.scene_id} "
        f"({len(missing)} released annotations missing)."
    )
    if args.dry_run:
        return 0

    camera_matrix = intrinsic_matrix(args.width, args.height, args.horizontal_fov)
    written = 0
    skipped = 0
    distances_2d: List[float] = []
    distances_3d: List[float] = []
    for index, (output_name, episode) in enumerate(selected, start=1):
        trajectory, image_copies = convert_episode(
            source_root, output_name, episode, camera_matrix, args.skip_frames
        )
        positions = trajectory["position"]
        points = trajectory["point"]
        distances_2d.extend(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        distances_3d.extend(np.linalg.norm(np.diff(points, axis=0), axis=1))
        if write_trajectory(
            output_root, output_name, trajectory, image_copies, args.overwrite
        ):
            written += 1
        else:
            skipped += 1
        print(f"[{index}/{len(selected)}] {output_name}")

    print(f"Wrote {written} trajectories to {output_root}; skipped {skipped} existing.")
    if distances_2d:
        print(f"Mean 2D waypoint spacing: {float(np.mean(distances_2d)):.6f} m")
        print(f"Mean 3D waypoint spacing: {float(np.mean(distances_3d)):.6f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
