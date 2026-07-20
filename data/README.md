# Data

This directory contains the preprocessing code and released split metadata used
by ANWM. Raw RGB-D data and generated trajectories are not tracked by Git.

```text
data/
  preprocessing/Data_preprocessing_airvln_16.ipynb
  splits/airvln_16/
  splits/sekai_new/
  airvln_16/       processed simulation trajectories (not tracked)
  sekai_new/       processed real-world trajectories (not tracked)
```

`Data_preprocessing_airvln_16.ipynb` is the original AirVLN scene-16 data
preparation notebook used for the released pipeline. Set its `root_dir` to the
local AirVLN directory before running it. The expected source layout is:

```text
<root_dir>/
  waypoints/train.json
  waypoints/val_seen.json
  traj_obs/<trajectory_id>/rgb/rgb_obs_front_<frame>.png
  traj_obs/<trajectory_id>/dep/dep_obs_front_<frame>.npy
```

The generated trajectory folders contain numeric image files and
`traj_data.pkl`. Place them under `data/airvln_16/` for training and inference.

Sekai preparation remains under `real/tools/`; see
[`real/README.md`](../real/README.md).
