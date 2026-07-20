<div align="center">

# ANWM

## Aerial World Model for Long-horizon Visual Generation and Navigation in 3D Space

**A physics-informed, action-conditioned world model for aerial visual generation and navigation in large-scale 3D environments.**

[![arXiv](https://img.shields.io/badge/arXiv-2512.21887-b31b1b.svg)](https://arxiv.org/abs/2512.21887)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Coming%20Soon-yellow.svg)]()

</div>

## Overview

**ANWM** (Aerial Navigation World Model) is an action-conditioned world model that predicts future egocentric observations for UAV navigation in large-scale **3D environments**. Conditioned on recent visual history and a 4-DoF UAV action `(delta x, delta y, delta z, delta yaw)`, it autoregressively imagines observations along candidate trajectories and ranks each path by visual similarity to the goal.

Highlights:

- **TB-scale pre-training** on action-conditioned world models with 4-DoF UAV trajectories, including large-scale data curation and high-throughput distributed training.
- **Physics-inspired** Future Frame Projection (**FFP**) that projects historical frames to future viewpoints, injecting coarse geometric priors and stabilizing long-horizon visual generation.
- **Long-horizon** visual forecasting and stronger navigation success: extending the effective prediction horizon from under 10 m (indoor) to the **hundreds of meters** scale in outdoor open-space environments.

Paper: [Aerial World Model](https://arxiv.org/abs/2512.21887).

## Dataset

Coming soon on [Hugging Face]().

The simulated benchmark is built from [AerialVLN](https://github.com/AirVLN/AirVLN), [OpenFly](https://github.com/SHAILAB-IPEC/OpenFly-Platform), and [OpenUAV](https://github.com/buaa-colalab/TravelUAV) trajectories across more than 40 Unreal Engine environments. Each segment has 48 observations and 47 relative actions. The same flight is recorded from front, left, right, and rear orientations to reduce forward-motion bias.

| Split | Trajectory segments | Frames per segment | Actions per segment |
|---|---:|---:|---:|
| Train | 350K | 48 | 47 |
| Test | 2.2K | 48 | 47 |

The test set contains 1.1K planar and 1.1K 3D trajectories. Horizontal speed is 5 m/s, vertical speed is 2 m/s, yaw speed is 15 deg/s, and the average trajectory length is approximately 80.7 m. Real-world evaluation uses the drone-view subset of [Sekai](https://github.com/Lixsp11/sekai-codebase) with estimated depth.

## Main Results

### 32-second visual generation

| Setting | Method | LPIPS (lower) | DreamSim (lower) | FID (lower) |
|---|---|---:|---:|---:|
| 2D simulation | NWM | 0.524 | 0.400 | 61.0 |
| 2D simulation | **ANWM** | **0.433** | **0.294** | **32.5** |
| 3D simulation | NWM | 0.535 | 0.377 | 47.6 |
| 3D simulation | **ANWM** | **0.389** | **0.271** | **36.1** |
| 3D real world | NWM | 0.388 | 0.229 | 149.6 |
| 3D real world | **ANWM** | **0.371** | **0.176** | **138.9** |

### Visual navigation

| Setting | Method | ATE (lower) | RPE (lower) | SR (higher) | NE (lower) |
|---|---|---:|---:|---:|---:|
| 2D simulation | NWM | 7.72 | 0.89 | 63.0 | 12.71 |
| 2D simulation | **ANWM** | **6.30** | **0.78** | **73.0** | **10.30** |
| 3D simulation | NWM | 8.52 | **1.03** | 58.0 | 14.51 |
| 3D simulation | **ANWM** | **8.13** | 1.06 | **60.0** | **14.12** |
| 3D real world | NWM | 17.13 | 3.79 | 28.0 | 27.02 |
| 3D real world | **ANWM** | **15.56** | **3.51** | **33.0** | **24.41** |

## Installation

Python 3.9 or newer and a CUDA-enabled PyTorch installation are required.
Install the PyTorch build matching the local CUDA runtime first, then install
ANWM:

```bash
conda create -n anwm python=3.9 -y
conda activate anwm
python -m pip install -e ".[metrics]"
python scripts/check_environment.py --component metrics --verify-imports
```

The Stable Diffusion VAE `stabilityai/sd-vae-ft-ema` is downloaded by
Diffusers on first use.

## Data and Checkpoint

Datasets and checkpoints will be released on [Hugging Face]().

The repository includes the released split manifests under `data/splits/`.
Place processed trajectories and the ANWM checkpoint at:

```text
data/airvln_16/<trajectory>/...
logs/anwm_cdit_airvln/checkpoints/0200000.pth.tar
```

The original AirVLN-16 preparation code is provided at
`data/preprocessing/Data_preprocessing_airvln_16.ipynb`. Set `root_dir` in the
notebook to the local AirVLN directory. The notebook writes trajectory folders
containing numeric image files and `traj_data.pkl`.

## Released Configuration

The paper model is defined by `config/anwm.yaml`:

| Parameter | Value |
|---|---:|
| Backbone | CDiT-XL/2 |
| Parameters | ~1.1B |
| Image size | 224 x 224 |
| Context frames | 4 |
| Prediction steps | 16 |
| Goals per observation | 4 |
| Batch size per GPU | 1 |
| Learning rate | 8e-5 |
| Gradient clipping | 10.0 |
| Training diffusion steps | 1000 |
| Inference sampling steps | 250 |
| Released checkpoint | 200,000 steps |

## Inference

Generate ANWM rollout predictions:

```bash
torchrun --standalone --nproc_per_node=1 infer.py \
  --exp config/anwm.yaml \
  --ckp 0200000 \
  --datasets airvln_16 \
  --eval_type rollout \
  --rollout_fps_values 1,4
```

Generate the matching ground truth:

```bash
torchrun --standalone --nproc_per_node=1 infer.py \
  --exp config/anwm.yaml \
  --datasets airvln_16 \
  --eval_type rollout \
  --rollout_fps_values 1,4 \
  --gt 1
```

Evaluate LPIPS, DreamSim, and FID:

```bash
python evaluate.py \
  --datasets airvln_16 \
  --gt_dir outputs/inference/gt \
  --exp_dir outputs/inference/anwm \
  --eval_types rollout \
  --rollout_fps_values 1,4
```

## Training

The released run used six GPUs. `config/anwm.yaml` contains the paper model and
optimization parameters:

```bash
torchrun --standalone --nproc_per_node=6 train.py \
  --config config/anwm.yaml \
  --global-seed 0 \
  --bfloat16 1 \
  --torch-compile 1
```

Checkpoints are written to `logs/anwm_cdit_airvln/checkpoints/`. The numbered
checkpoint used for the reported model is `0200000.pth.tar`.

## Real-world Planning

Real-world evaluation follows the same overall workflow as simulation, but uses
different data, depth, and entry scripts:

| | Simulation | Real-world |
|---|---|---|
| Data | AirVLN-16 | [Sekai](https://github.com/Lixsp11/sekai-codebase) drone-view |
| Depth | simulator GT | [Pi-Long](https://github.com/DengKaiCQ/Pi-Long) |
| Eval | [`infer.py`](infer.py) / [`evaluate.py`](evaluate.py) | [`planning_eval.py`](planning_eval.py) |
| Config | [`config/`](config/) | [`real/config/`](real/config/) |

Key files (other scripts under `real/tools/` are optional):

| Role | File |
|---|---|
| Convert videos → AirVLN-16 | [`real/tools/process_youtube_to_airvln16_format.py`](real/tools/process_youtube_to_airvln16_format.py) |
| Fill depth with Pi-Long | [`real/tools/fill_depth_pilong.py`](real/tools/fill_depth_pilong.py) |
| One-shot preprocess | [`real/tools/prepare_dataset.sh`](real/tools/prepare_dataset.sh) |
| Dataset adapter | [`real/dataset.py`](real/dataset.py) |
| Planning evaluator | [`real/planning_eval.py`](real/planning_eval.py) |
| Configs | [`real/config/`](real/config/) |
| Split + candidates | [`data/splits/sekai_new/`](data/splits/sekai_new/) |

Install deps, preprocess (needs `ffmpeg`; keep Pi-Long under `third_party/Pi-Long`
or elsewhere), then evaluate. Point `RAW_ROOT` / `OUTPUT_ROOT` at your own
directories to replace or re-index data.

```bash
python -m pip install -e ".[real]"
python scripts/check_environment.py --component real --verify-imports

RAW_ROOT=data/sekai_raw \
OUTPUT_ROOT=data/sekai_new \
PILONG_DIR=third_party/Pi-Long \
bash real/tools/prepare_dataset.sh

torchrun --standalone --nproc_per_node=1 planning_eval.py \
  --exp config/anwm.yaml \
  --ckp 0200000 \
  --datasets sekai_new \
  --output_dir outputs/planning \
  --num_samples 5
```

## Repository Layout

```text
anwm/             model, diffusion, projection, dataset, and rollout code
config/           paper training and simulation evaluation configuration
data/             original preprocessing notebook and released split metadata
real/             Sekai data adapters, depth fill, and planning evaluation
train.py          distributed training entry point
infer.py          visual generation entry point
evaluate.py       LPIPS, DreamSim, and FID evaluation
planning_eval.py  real-world trajectory-ranking entry point
```

## Limitations

- Generation may drift or collapse as trajectories approach 200 m.
- Fine structures such as windows and facades can become distorted.
- FFP depends on depth quality and camera calibration.
- Real-world results are offline evaluations, not autonomous UAV deployment.

## Acknowledgements

ANWM builds on [AerialVLN](https://github.com/AirVLN/AirVLN), [OpenFly](https://github.com/SHAILAB-IPEC/OpenFly-Platform), [OpenUAV](https://github.com/buaa-colalab/TravelUAV), [Sekai](https://github.com/Lixsp11/sekai-codebase), [NWM](https://github.com/facebookresearch/nwm), [Matrix-Game](https://github.com/SkyworkAI/Matrix-Game), and [YUME](https://github.com/stdstu12/YUME).

## Citation

```bibtex
@article{zhang2025anwm,
  title={Aerial World Model for Long-horizon Visual Generation and Navigation in 3D Space},
  author={Zhang, Weichen and Tang, Peizhi and Zeng, Xin and Man, Fanhang and Yu, Shiquan and Dai, Zichao and Zhao, Baining and Chen, Hongjin and Shang, Yu and Wu, Wei and Gao, Chen and Chen, Xinlei and Wang, Xin and Li, Yong and Zhu, Wenwu},
  journal={arXiv preprint arXiv:2512.21887},
  year={2025}
}
```
