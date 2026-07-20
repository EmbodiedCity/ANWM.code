#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="${RAW_ROOT:-${REPO_ROOT}/data/sekai_raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/sekai_new}"
PILONG_DIR="${PILONG_DIR:-${REPO_ROOT}/third_party/Pi-Long}"
PILONG_WEIGHTS="${PILONG_WEIGHTS:-${PILONG_DIR}/weights/model.safetensors}"

python "${REPO_ROOT}/real/tools/process_youtube_to_airvln16_format.py" \
  --input_root "${RAW_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --scene outdoor-urban \
  --group_by trajectory \
  --K_mode sekai \
  --out_fps 4 \
  --timestamps_mode relative \
  --write_traj_names

python "${REPO_ROOT}/real/tools/fill_depth_pilong.py" \
  --data_folder "${OUTPUT_ROOT}" \
  --pilong_dir "${PILONG_DIR}" \
  --pilong_weights "${PILONG_WEIGHTS}"

python "${REPO_ROOT}/real/tools/sample_trajectories_youtube.py" \
  --preset sekai_new \
  --overwrite
