"""Shared helpers: config loading, seeding, run folders and run metadata.

Used by every entry point (train / predict / evaluate) so that all models share
one implementation of these rules (CLAUDE.md, Rules 4 and 7).
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
from importlib import metadata as _metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- config
def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | os.PathLike) -> dict:
    """Load a YAML config. A top-level ``inherits: other.yaml`` key (path relative to
    the file) is loaded first and then overridden key by key (nested dicts merge,
    everything else replaces)."""
    path = Path(path)
    if not path.is_absolute():
        path = (Path.cwd() / path) if (Path.cwd() / path).exists() else (PROJECT_ROOT / path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherits", None)
    if parent is not None:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    """Stable hash of a resolved config; stored in checkpoints so a run is never
    resumed with a different config."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def project_path(p: str | os.PathLike) -> Path:
    """Resolve a config path (relative paths are relative to the project root)."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def portable_path(p: str | os.PathLike) -> str:
    """A path as written into saved result files: relative to the project root when it is
    inside the project, otherwise with $WORK or $HOME abbreviated, so that no personal
    absolute path ends up in the (public) results."""
    p = Path(p)
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        pass
    for var in ("WORK", "HOME"):
        base = os.environ.get(var)
        if base:
            try:
                return f"${var}/{p.relative_to(base).as_posix()}"
            except ValueError:
                pass
    return str(p)


def run_dir_for(cfg: dict, seed: int) -> Path:
    """outputs/<run_name>/seed<seed>; run_name defaults to model.name."""
    name = cfg.get("run_name") or cfg["model"]["name"]
    return project_path(cfg["paths"]["outputs_dir"]) / name / f"seed{seed}"


def prob_archive_path(run_dir: Path, split: str) -> Path:
    """Saved probability maps of one run and split: <run_dir>/<split>/probs.npz
    (float16 sigmoid outputs at the model's output size, one member per image_id)."""
    return Path(run_dir) / split / "probs.npz"


# --------------------------------------------------------------------------- seeding
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and every CUDA device)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # also seeds all CUDA devices
    torch.cuda.manual_seed_all(seed)


def configure_cudnn(cfg: dict) -> None:
    import torch

    rep = cfg.get("reproducibility", {})
    torch.backends.cudnn.deterministic = bool(rep.get("cudnn_deterministic", True))
    torch.backends.cudnn.benchmark = bool(rep.get("cudnn_benchmark", False))


def epoch_seed(seed: int, epoch: int) -> int:
    """Seed for the training DataLoader's generator in a given epoch.

    The shuffle order and the worker seeds of epoch ``e`` are a function of
    (seed, e) only, so they are identical whether or not the run was resumed."""
    return int(seed) * 100_003 + int(epoch)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: seed NumPy and ``random`` in each worker from
    the per-worker torch seed (which comes from the loader's generator)."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- metadata
def git_info() -> dict:
    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                                  text=True, timeout=10, check=True).stdout.strip()
        except Exception:
            return None

    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(dirty) if commit else None}


def package_versions() -> dict[str, str]:
    return {d.metadata["Name"]: d.version for d in _metadata.distributions()
            if d.metadata["Name"]}


def environment_info() -> dict:
    info: dict[str, Any] = {
        "time": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "git": git_info(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name()
    except ImportError:
        pass
    return info


def write_run_metadata(run_dir: Path, cfg: dict, args: dict, extra: dict | None = None) -> None:
    """Save the resolved config once, and append one JSON line per session
    (first start and every resume) with environment, git commit and package versions."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = run_dir / "config.yaml"
    if not cfg_file.exists():
        with open(cfg_file, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    record = {"args": args, **environment_info(), **(extra or {})}
    record["packages"] = package_versions()
    with open(run_dir / "run_info.jsonl", "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
