# ANWM

This repository contains the official training and evaluation pipeline for
**ANWM: Aerial World Model for Long-horizon Visual Generation and Navigation
in 3D Space**. The core package targets AirVLN-16. The optional
[`real/`](real/) extension contains the Sekai real-world preprocessing and
planning workflow as a separate module.

The release excludes raw datasets, model weights, experiment outputs, older
model variants, and vendored third-party repositories. The repository state
before this cleanup is preserved on the
`archive/pre-open-source-20260720` branch.

## Layout

```text
anwm/                        reusable model, diffusion, data, and utility code
config/                      ANWM training and AirVLN evaluation configuration
data/README.md               dataset layout and preparation instructions
data/tools/                  reproducible AirVLN-16 preprocessing
data/splits/airvln_16/       released AirVLN split metadata
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
python -m pip install -e ".[metrics,real,dev]"
ruff check .
python -m unittest discover -s tests -v
```

## Data and checkpoints

Place external artifacts at these repository-relative paths:

```text
data/airvln_16/<trajectory>/...
logs/anwm_cdit_airvln/checkpoints/0200000.pth.tar
```

Data and checkpoints are ignored by Git. The committed `data/splits` directory
contains trajectory names and evaluation indexes only. To convert the source
AirVLN observations into the released ANWM format, run:

```bash
python data/tools/prepare_airvln16.py \
  --source-root /path/to/airvln \
  --output-root data/airvln_16
```

See [`data/README.md`](data/README.md) for the expected source layout and
conversion details.

## Training

The released ANWM checkpoint uses 16 context frames, 16 prediction steps, and
the `CDiT-XL/2` model:

```bash
torchrun --nproc_per_node=8 train.py --config config/anwm.yaml
```

## Image prediction and metrics

```bash
torchrun --nproc_per_node=1 infer.py \
  --exp config/anwm.yaml \
  --ckp 0200000 \
  --datasets airvln_16 \
  --eval_type rollout
```

Generate ground truth with the same command plus `--gt 1`, then evaluate:

```bash
python evaluate.py \
  --datasets airvln_16 \
  --gt_dir outputs/inference/gt \
  --exp_dir outputs/inference/anwm \
  --eval_types rollout
```

## Real-world extension

Sekai preprocessing, split metadata, dataset code, and planning evaluation are
kept under [`real/`](real/). The compatibility entry point remains:

```bash
torchrun --nproc_per_node=1 planning_eval.py \
  --exp config/anwm.yaml \
  --ckp 0200000 \
  --datasets sekai_new \
  --output_dir outputs/planning
```

See [`real/README.md`](real/README.md) for data preparation and external
dependencies.

## Reproducibility notes

- The ANWM model contains exactly `1,134,100,256` parameters.
- Runtime paths are resolved from the repository root, independent of the
  caller's current working directory.
- PyTorch checkpoint files use pickle. Only load checkpoints from a trusted
  source.
- Full training and planning evaluation require GPUs, external datasets, and
  separately distributed model weights.
