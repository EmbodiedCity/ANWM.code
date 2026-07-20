#!/usr/bin/env python3
"""Check local dependencies and release metadata without importing GPU libraries."""

import argparse
import importlib
import importlib.util
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_MODULES = ("torch", "torchvision", "timm", "diffusers", "cv2", "yaml", "scipy")
METRIC_MODULES = ("lpips", "dreamsim", "torcheval")
REAL_MODULES = ("evo", "lpips")


def check_modules(names, verify_imports=False):
    missing = []
    for name in names:
        installed = importlib.util.find_spec(name) is not None
        error = None
        if installed and verify_imports:
            try:
                importlib.import_module(name)
            except Exception as exc:
                installed = False
                error = f"{type(exc).__name__}: {exc}"
        print(f"[{'ok' if installed else 'missing'}] python module: {name}")
        if error:
            print(f"  import failed: {error}")
        if not installed:
            missing.append(name)
    return missing


def check_file(relative_path):
    path = ROOT / relative_path
    present = path.is_file()
    print(f"[{'ok' if present else 'missing'}] file: {relative_path}")
    return present


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=("core", "real", "all"),
        default="all",
        help="dependency set to check",
    )
    parser.add_argument(
        "--require-checkpoint",
        action="store_true",
        help="require the released 0200000 checkpoint at the documented path",
    )
    parser.add_argument(
        "--verify-imports",
        action="store_true",
        help="import each dependency to detect binary and transitive dependency errors",
    )
    args = parser.parse_args()

    failed = sys.version_info < (3, 9)
    print(f"[{'ok' if not failed else 'unsupported'}] Python: {sys.version.split()[0]}")

    modules = list(CORE_MODULES)
    if args.component == "all":
        modules.extend(METRIC_MODULES)
    if args.component in ("real", "all"):
        modules.extend(REAL_MODULES)
    failed = bool(check_modules(dict.fromkeys(modules), args.verify_imports)) or failed

    required_files = [
        "config/anwm.yaml",
        "config/eval_config.yaml",
        "data_splits/airvln_16/test/rollout_16.pkl",
    ]
    if args.component in ("real", "all"):
        required_files.extend(
            [
                "real/config/eval_config.yaml",
                "real/data_splits/sekai_new/test/navigation_eval.pkl",
                "real/data_splits/sekai_new/test/trajectory_candidates.pkl",
            ]
        )
        ffmpeg_present = shutil.which("ffmpeg") is not None
        print(f"[{'ok' if ffmpeg_present else 'missing'}] executable: ffmpeg")
        failed = not ffmpeg_present or failed

    for relative_path in required_files:
        failed = not check_file(relative_path) or failed

    if args.require_checkpoint:
        failed = (
            not check_file("logs/anwm_cdit_airvln/checkpoints/0200000.pth.tar")
            or failed
        )

    if failed:
        print("Environment check failed. Install the missing dependencies or files.")
        return 1
    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
