<div align="center">

# ANWM

## Aerial World Model for Long-horizon Visual Generation and Navigation in 3D Space

**A physics-informed, action-conditioned world model for aerial visual generation and navigation in large-scale 3D environments.**

</div>

## Overview

ANWM predicts future egocentric observations from recent visual history and a
4-DoF UAV action `(delta x, delta y, delta z, delta yaw)`. During navigation,
it autoregressively imagines observations along candidate trajectories and
ranks each path by the LPIPS distance between its predicted endpoint and the
visual goal.

ANWM introduces two components for long-horizon aerial generation:

- **Future Frame Projection (FFP)** projects historical RGB-D observations into
  the future camera view and provides a physically grounded image prior.
- **Independent Latent Modulation (ILM)** conditions historical observations
  and projected future frames separately so that projection holes and depth
  errors do not dominate generation.

## Method

FFP back-projects historical depth into 3D, transforms visible points to the
future camera pose, projects them onto the image plane, and resolves overlapping
points by depth. The resulting coarse future view is encoded alongside the
visual history. ANWM uses independent modulation and shared-weight
cross-attention branches to combine these two conditioning sources in a
Conditional Diffusion Transformer.

The model supports forward/backward, left/right, up/down, and yaw motion. For
navigation, candidate trajectories are supplied by an external planner; ANWM
imagines each trajectory and selects the endpoint most similar to the goal.

## Dataset

The simulated benchmark is built from AerialVLN, OpenFly, and OpenUAV
trajectories across more than 40 Unreal Engine environments. Each segment has
48 observations and 47 relative actions. The same flight is recorded from
front, left, right, and rear orientations to reduce forward-motion bias.

| Split | Trajectory segments | Frames per segment | Actions per segment |
|---|---:|---:|---:|
| Train | 350K | 48 | 47 |
| Test | 2.2K | 48 | 47 |

The test set contains 1.1K planar and 1.1K 3D trajectories. Horizontal speed is
5 m/s, vertical speed is 2 m/s, yaw speed is 15 deg/s, and the average
trajectory length is approximately 80.7 m. Real-world evaluation uses the
drone-view subset of Sekai with estimated depth.

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
| Parameters | 1,134,100,256 |
| Image size | 224 x 224 |
| Context frames | 16 |
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

Install the Sekai planning dependencies:

```bash
python -m pip install -e ".[real]"
python scripts/check_environment.py --component real --verify-imports
```

Place processed Sekai trajectories under `data/sekai_new`, then run:

```bash
torchrun --standalone --nproc_per_node=1 planning_eval.py \
  --exp config/anwm.yaml \
  --ckp 0200000 \
  --datasets sekai_new \
  --output_dir outputs/planning \
  --num_samples 5
```

Sekai preprocessing and configuration remain isolated under `real/`; see
[`real/README.md`](real/README.md).

## Repository Layout

```text
anwm/             model, diffusion, projection, dataset, and rollout code
config/           paper training and simulation evaluation configuration
data/             original preprocessing notebook and released split metadata
real/             Sekai preprocessing and planning evaluation
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

ANWM builds on AerialVLN, OpenFly, OpenUAV, Sekai, NWM, Matrix-Game, and YUME.
