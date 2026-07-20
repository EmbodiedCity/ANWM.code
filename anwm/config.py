"""Configuration and repository path helpers."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Union

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PathLike = Union[str, Path]


def repo_path(path: PathLike) -> Path:
    """Resolve a repository-relative path without depending on the current directory."""
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def load_yaml(path: PathLike) -> Dict[str, Any]:
    resolved = repo_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {resolved}")
    return value


def load_runtime_config(
    experiment: PathLike,
    defaults: PathLike = "config/eval_config.yaml",
) -> Dict[str, Any]:
    """Load an experiment config and resolve runtime paths from the repo root."""
    config = load_yaml(defaults)
    config.update(load_yaml(experiment))
    config = deepcopy(config)

    required = {"run_name", "results_dir", "model", "image_size", "context_size"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    config["results_dir"] = str(repo_path(config["results_dir"]))
    for section in ("datasets", "eval_datasets"):
        for dataset in config.get(section, {}).values():
            for key in ("data_folder", "train", "test"):
                if key in dataset:
                    dataset[key] = str(repo_path(dataset[key]))
    return config


def split_path(
    dataset_name: str, filename: str, root: PathLike = "data/splits"
) -> Path:
    return repo_path(Path(root) / dataset_name / "test" / filename)
