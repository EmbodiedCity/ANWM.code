# NWM v5_3

This repository contains the minimal Navigation World Model (NWM) v5_3
training and evaluation pipeline. The core package targets AirVLN-16. The
optional [`real/`](real/) extension contains the Sekai real-world preprocessing
and planning workflow as a separate module.

The release excludes raw datasets, model weights, experiment outputs, older
model variants, and vendored third-party repositories. The repository state
before this cleanup is preserved on the
`archive/pre-open-source-20260720` branch.

## Layout

```text
nwm/                         reusable model, diffusion, data, and utility code
config/                      v5_3 training and AirVLN evaluation configuration
data_splits/airvln_16/       released AirVLN split metadata
train.py                     distributed training entry point
infer.py                     image prediction entry point
evaluate.py                  image metric evaluation
real/                        isolated Sekai real-world extension
tests/                       release and configuration integrity tests
scripts/check_environment.py dependency and metadata checker
```

## Installation

Python 3.9 or newer and a CUDA-capable PyTorch installation are required for
training and inference. Install the PyTorch build appropriate for the local
CUDA runtime first, then install this repository:

```bash
python -m pip install -e ".[metrics,real]"
python scripts/check_environment.py --component all --verify-imports
```

For development:

```bash
python -m pip install -r requirements-dev.txt
ruff check .
python -m unittest discover -s tests -v
```

## Data and checkpoints

Place external artifacts at these repository-relative paths:

```text
data/airvln_16/<trajectory>/...
logs/nwm_cdit_airvln_v5_3/checkpoints/0200000.pth.tar
```

Data and checkpoints are ignored by Git. The committed `data_splits` directory
contains trajectory names and evaluation indexes only.

## Training

The released v5_3 checkpoint uses 16 context frames, 16 prediction steps, and
the `CDiT-XL/2` model:

```bash
torchrun --nproc_per_node=8 train.py --config config/v5_3.yaml
```

## Image prediction and metrics

```bash
torchrun --nproc_per_node=1 infer.py \
  --exp config/v5_3.yaml \
  --ckp 0200000 \
  --datasets airvln_16 \
  --eval_type rollout
```

Generate ground truth with the same command plus `--gt 1`, then evaluate:

```bash
python evaluate.py \
  --datasets airvln_16 \
  --gt_dir outputs/inference/gt \
  --exp_dir outputs/inference/v5_3 \
  --eval_types rollout
```

## Real-world extension

Sekai preprocessing, split metadata, dataset code, and planning evaluation are
kept under [`real/`](real/). The compatibility entry point remains:

```bash
torchrun --nproc_per_node=1 planning_eval.py \
  --exp config/v5_3.yaml \
  --ckp 0200000 \
  --datasets sekai_new \
  --output_dir outputs/planning
```

See [`real/README.md`](real/README.md) for data preparation and external
dependencies.

## Reproducibility notes

- The v5_3 model contains exactly `1,134,100,256` parameters.
- Runtime paths are resolved from the repository root, independent of the
  caller's current working directory.
- PyTorch checkpoint files use pickle. Only load checkpoints from a trusted
  source.
- Full training and planning evaluation require GPUs, external datasets, and
  separately distributed model weights.
