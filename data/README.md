# Data preparation

The repository tracks preprocessing code and split metadata under `data/`.
Raw and processed image/depth data remain ignored by Git.

```text
data/
  tools/prepare_airvln16.py    AirVLN scene-16 converter
  splits/airvln_16/            released ANWM train/test metadata
  splits/sekai_new/            released Sekai evaluation metadata
  airvln_16/                   generated AirVLN trajectories (ignored)
  sekai_new/                   generated Sekai trajectories (ignored)
```

## AirVLN-16

The converter replaces the machine-specific
`Data_preprocessing_airvln_16.ipynb` used during development. It expects this
source layout:

```text
<source-root>/
  waypoints/train.json
  waypoints/val_seen.json
  traj_obs/<trajectory_id>/rgb/rgb_obs_front_<frame>.png
  traj_obs/<trajectory_id>/dep/dep_obs_front_<frame>.npy
```

Run it after installing ANWM:

```bash
python data/tools/prepare_airvln16.py \
  --source-root /path/to/airvln \
  --output-root data/airvln_16
```

By default, the converter filters `scene_id == 16`, skips the first 11 frames,
uses 512 x 512 images with a 90-degree horizontal field of view, applies the
camera-frame transform used for the released checkpoint, and processes only
trajectories named by `data/splits/airvln_16`. Each output folder contains
numeric image files and a `traj_data.pkl` with camera intrinsics, position,
orientation, depth, and pose arrays.

Use `--dry-run` to check annotation coverage before conversion and `--help` for
all options. The original notebooks and their hard-coded workstation paths are
not required.

## Sekai

Sekai preparation remains isolated in `real/tools/`. See
[`real/README.md`](../real/README.md) for the complete pipeline.
