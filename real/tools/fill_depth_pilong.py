#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import pickle
import re
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def _read_lines(p: Path) -> List[str]:
    out: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _discover_traj_names(
    data_folder: Path, traj_names_file: Optional[str], *, image_ext: str
) -> List[str]:
    def scan_dirs() -> List[str]:
        names: List[str] = []
        for d in sorted(data_folder.iterdir()):
            if not d.is_dir():
                continue
            if not (d / "traj_data.pkl").exists():
                continue
            # Minimal sanity: require at least one image.
            if not (d / f"0{image_ext}").exists():
                continue
            names.append(d.name)
        return names

    if traj_names_file:
        p = Path(traj_names_file)
        if not p.is_absolute():
            p = data_folder / p
        if not p.exists():
            raise FileNotFoundError(f"traj_names_file not found: {p}")
        names = _read_lines(p)
        names = [n for n in names if n]
        if names:
            return names
        # Fallback: some pipelines accidentally write an empty traj_names.txt (e.g., due to fps mismatch).
        return scan_dirs()

    default_names = data_folder / "traj_names.txt"
    if default_names.exists():
        names = _read_lines(default_names)
        names = [n for n in names if n]
        if names:
            return names
        scanned = scan_dirs()
        # If traj_names.txt exists but is empty, regenerate it for convenience.
        if scanned:
            try:
                default_names.write_text("\n".join(scanned) + "\n", encoding="utf-8")
            except Exception:
                pass
        return scanned

    return scan_dirs()


def _iter_image_paths(traj_dir: Path, n_frames: int, image_ext: str) -> Iterable[Path]:
    for t in range(int(n_frames)):
        yield traj_dir / f"{t}{image_ext}"


def _compute_target_hw_for_pilong(
    w_orig: int, h_orig: int, pixel_limit: int
) -> Tuple[int, int]:
    w_orig = int(w_orig)
    h_orig = int(h_orig)
    pixel_limit = int(pixel_limit)
    if w_orig <= 0 or h_orig <= 0:
        raise ValueError(f"Invalid image size: {w_orig}x{h_orig}")
    if pixel_limit <= 0:
        raise ValueError(f"pixel_limit must be > 0, got {pixel_limit}")

    scale = math.sqrt(pixel_limit / float(w_orig * h_orig))
    w_target = float(w_orig) * scale
    h_target = float(h_orig) * scale
    k, m = round(w_target / 14.0), round(h_target / 14.0)
    while (k * 14) * (m * 14) > pixel_limit:
        if (k / max(m, 1e-6)) > (w_target / max(h_target, 1e-6)):
            k -= 1
        else:
            m -= 1
    target_w, target_h = max(1, k) * 14, max(1, m) * 14
    return int(target_h), int(target_w)


def _resize_depth_stack(depth: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 3:
        raise ValueError(f"depth must be (N,H,W), got {depth.shape}")
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if depth.shape[1] == out_h and depth.shape[2] == out_w:
        return depth

    resized = np.empty((depth.shape[0], out_h, out_w), dtype=np.float32)
    try:
        import cv2  # type: ignore

        for i in range(depth.shape[0]):
            resized[i] = cv2.resize(
                depth[i], (out_w, out_h), interpolation=cv2.INTER_LINEAR
            )
    except Exception:
        for i in range(depth.shape[0]):
            im = Image.fromarray(depth[i], mode="F")
            im = im.resize((out_w, out_h), resample=Image.BILINEAR)
            resized[i] = np.asarray(im, dtype=np.float32)
    return resized


@dataclass
class PiLongConfig:
    pilong_dir: str
    pilong_weights: str
    device: str
    pixel_limit: int
    chunk_size: int
    depth_scale: float


class PiLongDepthEstimator:
    def __init__(self, cfg: PiLongConfig):
        import torch
        from safetensors.torch import load_file

        pilong_repo = os.path.abspath(cfg.pilong_dir)
        if pilong_repo not in sys.path:
            sys.path.insert(0, pilong_repo)

        from pi3.models.pi3 import Pi3

        device = str(cfg.device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(
                "[warn] --device=cuda but CUDA is not available; falling back to cpu.",
                file=sys.stderr,
            )
            device = "cpu"

        torch_device = torch.device(device)
        model = Pi3().to(torch_device).eval()
        state = load_file(cfg.pilong_weights)
        model.load_state_dict(state, strict=False)

        if torch_device.type == "cuda":
            major, _minor = torch.cuda.get_device_capability()
            self._autocast_dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self._autocast_dtype = None

        self._torch = torch
        self._torch_device = torch_device
        self._model = model

    def infer_depth(
        self,
        image_paths: Sequence[Path],
        *,
        target_hw: Tuple[int, int],
    ) -> np.ndarray:
        import torch
        from torchvision import transforms

        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        to_tensor = transforms.ToTensor()
        resampling = getattr(Image, "Resampling", Image)

        tensor_list = []
        for p in image_paths:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im = im.resize((target_w, target_h), resample=resampling.LANCZOS)
                tensor_list.append(to_tensor(im))
        images = torch.stack(tensor_list, dim=0).to(self._torch_device)

        if self._torch_device.type == "cuda":
            autocast_ctx = self._torch.cuda.amp.autocast(dtype=self._autocast_dtype)
        else:
            autocast_ctx = nullcontext()

        with torch.no_grad():
            with autocast_ctx:
                pred = self._model(images[None])

        local_points = pred["local_points"][0]
        depth = local_points[..., 2].detach().float().cpu().numpy()
        return np.asarray(depth, dtype=np.float32)


def _load_traj_meta(traj_pkl: Path) -> Tuple[dict, int]:
    with traj_pkl.open("rb") as f:
        traj_data = pickle.load(f)
    if "pose" not in traj_data:
        raise KeyError(f"`pose` not found in {traj_pkl}")
    poses = np.asarray(traj_data["pose"])
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected pose shape {poses.shape} in {traj_pkl}")
    return traj_data, int(poses.shape[0])


def _atomic_pickle_dump(obj: object, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, dst)


def _parse_devices_list(devices: str) -> List[str]:
    parts = [p.strip() for p in str(devices).split(",")]
    return [p for p in parts if p]


def _expand_devices(device: str, num_procs: int) -> List[str]:
    device = str(device)
    num_procs = int(num_procs)
    if num_procs <= 1:
        return [device]
    if not device.startswith("cuda"):
        return [device for _ in range(num_procs)]
    if device == "cuda":
        start = 0
    else:
        m = re.match(r"^cuda:(\d+)$", device)
        start = int(m.group(1)) if m else 0
    return [f"cuda:{start + i}" for i in range(num_procs)]


def _spawn_workers(argv: List[str], *, devices: Sequence[str]) -> int:
    script = str(Path(__file__).resolve())
    py = sys.executable
    num_workers = len(devices)
    if num_workers <= 1:
        return 0

    procs: List[subprocess.Popen] = []
    try:
        for wid, dev in enumerate(devices):
            cmd = [
                py,
                "-u",
                script,
                *argv,
                "--worker_id",
                str(wid),
                "--num_workers",
                str(num_workers),
                "--device",
                str(dev),
            ]
            procs.append(subprocess.Popen(cmd))
        rc = 0
        for p in procs:
            p.wait()
            if p.returncode and rc == 0:
                rc = int(p.returncode)
        return rc
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fill missing `depth` in traj_data.pkl using Pi-Long (Pi3)."
    )
    ap.add_argument(
        "--data_folder",
        required=True,
        help="Dataset folder containing <traj_name>/traj_data.pkl and images.",
    )
    ap.add_argument(
        "--traj_names_file",
        default=None,
        help="Optional traj_names.txt path (relative to data_folder is allowed). If omitted, auto-detect.",
    )
    ap.add_argument("--image_ext", default=".jpg")
    ap.add_argument(
        "--only", nargs="*", default=None, help="Only process these trajectory names."
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="Process at most N trajectories (0 = all)."
    )
    ap.add_argument(
        "--force", action="store_true", help="Recompute even if depth exists."
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="List what would be processed, without writing anything.",
    )

    ap.add_argument(
        "--num_procs",
        type=int,
        default=0,
        help="Parallel worker processes (recommended for multi-GPU). 0/1 disables parallelism.",
    )
    ap.add_argument(
        "--devices",
        default=None,
        help="Optional comma-separated devices, e.g. 'cuda:0,cuda:1,cuda:2,cuda:3' or 'cpu,cpu'. Overrides --num_procs.",
    )
    ap.add_argument("--worker_id", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--num_workers", type=int, default=0, help=argparse.SUPPRESS)

    ap.add_argument("--pilong_dir", default="third_party/Pi-Long")
    ap.add_argument(
        "--pilong_weights",
        default=None,
        help="Defaults to <pilong_dir>/weights/model.safetensors",
    )
    ap.add_argument("--device", default="cuda", help="cuda/cpu")
    ap.add_argument("--pixel_limit", type=int, default=255000)
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=16,
        help="Frames per forward pass (batch size; tune for memory).",
    )
    ap.add_argument(
        "--depth_scale",
        type=float,
        default=1.0,
        help="Multiply predicted depth by this constant.",
    )

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = ap.parse_args(raw_argv)

    is_worker = (int(args.worker_id) >= 0) and (int(args.num_workers) > 0)
    log_prefix = (
        f"[w{int(args.worker_id):>2}/{int(args.num_workers):<2}] " if is_worker else ""
    )

    if (not is_worker) and (args.devices or (int(args.num_procs) > 1)):
        if args.devices:
            devices = _parse_devices_list(str(args.devices))
        else:
            devices = _expand_devices(str(args.device), int(args.num_procs))
        if len(devices) > 1:
            print(
                f"{log_prefix}[info] spawning {len(devices)} workers: {', '.join(devices)}",
                file=sys.stderr,
            )
            rc = _spawn_workers(raw_argv, devices=devices)
            if rc != 0:
                print(
                    f"{log_prefix}[error] one or more workers failed (rc={rc})",
                    file=sys.stderr,
                )
            return int(rc)

    data_folder = Path(args.data_folder)
    if not data_folder.exists():
        print(
            f"{log_prefix}[error] data_folder not found: {data_folder}", file=sys.stderr
        )
        return 2

    names = _discover_traj_names(
        data_folder, args.traj_names_file, image_ext=str(args.image_ext)
    )
    if args.only:
        only = set(args.only)
        names = [n for n in names if n in only]
    if args.limit and int(args.limit) > 0:
        names = names[: int(args.limit)]

    if is_worker:
        wid = int(args.worker_id)
        nworkers = int(args.num_workers)
        names = names[wid::nworkers]

    if not names:
        print(
            f"{log_prefix}[error] No trajectories found under: {data_folder}",
            file=sys.stderr,
        )
        return 2

    pilong_dir = os.path.abspath(str(args.pilong_dir))
    weights_path = (
        str(args.pilong_weights)
        if args.pilong_weights
        else os.path.join(pilong_dir, "weights", "model.safetensors")
    )
    if not os.path.exists(weights_path):
        print(
            f"{log_prefix}[error] Pi-Long weights not found: {weights_path}",
            file=sys.stderr,
        )
        return 2

    cfg = PiLongConfig(
        pilong_dir=pilong_dir,
        pilong_weights=os.path.abspath(weights_path),
        device=str(args.device),
        pixel_limit=int(args.pixel_limit),
        chunk_size=max(1, int(args.chunk_size)),
        depth_scale=float(args.depth_scale),
    )
    estimator: Optional[PiLongDepthEstimator] = None

    total = len(names)
    ok = 0
    skipped = 0
    failed = 0

    print(f"{log_prefix}[info] trajectories: {total}", file=sys.stderr)
    print(
        f"{log_prefix}[info] device={cfg.device} chunk_size={cfg.chunk_size} pixel_limit={cfg.pixel_limit} depth_scale={cfg.depth_scale}",
        file=sys.stderr,
    )

    for idx, name in enumerate(names, start=1):
        traj_dir = data_folder / name
        traj_pkl = traj_dir / "traj_data.pkl"
        marker = traj_dir / ".pilong_depth_done"

        if not traj_pkl.exists():
            failed += 1
            print(
                f"{log_prefix}[{idx:>4}/{total:<4}] [fail] {name}: missing traj_data.pkl",
                file=sys.stderr,
            )
            continue

        if marker.exists() and (not args.force):
            skipped += 1
            print(f"{log_prefix}[{idx:>4}/{total:<4}] [skip] {name}: marker exists")
            continue

        try:
            traj_data, n_frames = _load_traj_meta(traj_pkl)
            if n_frames <= 0:
                raise ValueError("n_frames=0")

            if (not args.force) and ("depth" in traj_data):
                depth = np.asarray(traj_data["depth"])
                if depth.ndim == 3 and int(depth.shape[0]) == int(n_frames):
                    if not args.dry_run:
                        try:
                            marker.write_text("1\n", encoding="utf-8")
                        except Exception:
                            pass
                    skipped += 1
                    print(
                        f"{log_prefix}[{idx:>4}/{total:<4}] [skip] {name}: depth already present"
                    )
                    continue

            img0 = traj_dir / f"0{args.image_ext}"
            if not img0.exists():
                raise FileNotFoundError(f"Missing first image: {img0}")
            with Image.open(img0) as im:
                w0, h0 = im.size

            target_hw = _compute_target_hw_for_pilong(w0, h0, int(cfg.pixel_limit))

            if args.dry_run:
                print(
                    f"{log_prefix}[{idx:>4}/{total:<4}] [plan] {name}: frames={n_frames} img={w0}x{h0} pilong_in={target_hw[1]}x{target_hw[0]} chunk={cfg.chunk_size}"
                )
                ok += 1
                continue

            if estimator is None:
                estimator = PiLongDepthEstimator(cfg)

            depth_out = np.empty((n_frames, h0, w0), dtype=np.float32)

            paths = list(_iter_image_paths(traj_dir, n_frames, str(args.image_ext)))
            for p in paths:
                if not p.exists():
                    raise FileNotFoundError(f"Missing image: {p}")

            for s in range(0, n_frames, int(cfg.chunk_size)):
                e = min(n_frames, s + int(cfg.chunk_size))
                chunk = paths[s:e]
                depth_pred = estimator.infer_depth(chunk, target_hw=target_hw)
                depth_pred = _resize_depth_stack(depth_pred, (h0, w0))
                depth_pred = np.clip(depth_pred, 1e-6, None)
                depth_out[s:e] = depth_pred

                if (s == 0) or (e == n_frames) or (s // int(cfg.chunk_size)) % 10 == 0:
                    print(
                        f"{log_prefix}  - {name}: {e:>6}/{n_frames:<6} frames",
                        file=sys.stderr,
                    )

            if cfg.depth_scale != 1.0:
                depth_out *= float(cfg.depth_scale)

            traj_data["depth"] = depth_out
            _atomic_pickle_dump(traj_data, traj_pkl)
            try:
                marker.write_text("1\n", encoding="utf-8")
            except Exception:
                pass

            ok += 1
            print(
                f"{log_prefix}[{idx:>4}/{total:<4}] [ok] {name}: depth={depth_out.shape} ({depth_out.dtype})"
            )
        except Exception as e:
            failed += 1
            print(
                f"{log_prefix}[{idx:>4}/{total:<4}] [fail] {name}: {e}", file=sys.stderr
            )

    print(f"{log_prefix}[done] data_folder: {data_folder.resolve()}")
    print(f"{log_prefix}[done] ok={ok} skipped={skipped} failed={failed} total={total}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
