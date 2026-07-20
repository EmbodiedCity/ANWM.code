# NWM Real-World Extension

This directory isolates the Sekai-Real-Drone workflow from the AirVLN-16 NWM
core. It reuses the released `nwm` model and diffusion package but owns all
real-world-specific configuration, split metadata, preprocessing, and planning
code.

## Layout

```text
config/                 Sekai data and planning configuration
data_splits/sekai_new/  released navigation index and candidate trajectories
dataset.py              Sekai trajectory dataset adapter
planning_eval.py        real-world planning evaluator
tools/                   conversion, depth completion, filtering, and sampling
```

## Installation

Install the real-world dependency group from the repository root:

```bash
python -m pip install -e ".[real]"
```

## External data

Obtain Sekai-Real-Drone from the
[official repository](https://github.com/Lixsp11/sekai-codebase) and place the
processed trajectories under `data/sekai_new`. Raw video conversion requires
`ffmpeg`.

Depth completion uses [Pi-Long](https://github.com/DengKaiCQ/Pi-Long). Keep its
source and weights outside this repository or under the ignored path
`third_party/Pi-Long`.

## Preprocessing

```bash
RAW_ROOT=data/sekai_raw \
OUTPUT_ROOT=data/sekai_new \
PILONG_DIR=third_party/Pi-Long \
bash real/tools/prepare_dataset.sh
```

Individual tools expose their options through `--help`:

```bash
python real/tools/process_youtube_to_airvln16_format.py --help
python real/tools/fill_depth_pilong.py --help
python real/tools/sample_trajectories_youtube.py --help
```

## Planning evaluation

The evaluator uses the root v5_3 checkpoint and the candidate trajectories in
`real/data_splits/sekai_new/test`:

```bash
torchrun --nproc_per_node=1 planning_eval.py \
  --exp config/v5_3.yaml \
  --ckp 0200000 \
  --datasets sekai_new \
  --output_dir outputs/planning
```
