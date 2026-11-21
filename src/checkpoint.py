"""Checkpoint save / load with everything needed for an exact resume.

Every checkpoint holds (CLAUDE.md, Phase 0, step 5):
  * model, optimizer and mixed-precision GradScaler state;
  * the number of finished epochs, the best validation score (and its epoch),
    and the early-stopping counter;
  * the random-number states of Python, NumPy, torch CPU and every CUDA device;
  * the weights of the best epoch so far, so that ``last.pt`` alone is a
    complete, consistent record (``best.pt`` is an exported copy of them).

Writes are atomic (write to a temporary file, then rename), so a job killed at
its time limit never leaves a half-written ``last.pt``.

All values are tensors / plain Python types, so files load with
``torch.load(weights_only=True)`` (no arbitrary code execution on load).
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT_VERSION = 1


# --------------------------------------------------------------------------- RNG
def capture_rng_state() -> dict:
    np_state = np.random.get_state(legacy=False)
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": np_state["bit_generator"],
            "key": torch.from_numpy(np_state["state"]["key"].astype(np.int64)),
            "pos": int(np_state["state"]["pos"]),
            "has_gauss": int(np_state["has_gauss"]),
            "gauss": float(np_state["gauss"]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    """Restore all RNG states. Tensors may be on any device (``torch.load`` with
    ``map_location="cuda"`` moves them to the GPU); they are moved back to the CPU here."""
    py = state["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    n = state["numpy"]
    np.random.set_state({
        "bit_generator": n["bit_generator"],
        "state": {"key": n["key"].cpu().numpy().astype(np.uint32), "pos": n["pos"]},
        "has_gauss": n["has_gauss"],
        "gauss": n["gauss"],
    })
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = [s.cpu() for s in (state.get("torch_cuda") or [])]
    if cuda_states and torch.cuda.is_available():
        n_dev = torch.cuda.device_count()
        if len(cuda_states) != n_dev:
            print(f"[checkpoint] warning: saved {len(cuda_states)} CUDA RNG states, "
                  f"{n_dev} device(s) visible; restoring the first {min(n_dev, len(cuda_states))}")
        for i, s in enumerate(cuda_states[:n_dev]):
            torch.cuda.set_rng_state(s, device=i)


# --------------------------------------------------------------------------- save / load
def _atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:   # make sure the bytes are on disk before the rename
        os.fsync(f.fileno())
    os.replace(tmp, path)


def cpu_state_dict(module: torch.nn.Module) -> dict:
    return {k: v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}


def save_checkpoint(path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler | None, epoch: int,
                    best_score: float | None, best_epoch: int | None,
                    epochs_without_improvement: int, best_model_state: dict | None,
                    finished: bool, finish_reason: str | None, seed: int,
                    config_hash: str, extra: dict | None = None) -> None:
    """``epoch`` = number of finished epochs (training resumes at this epoch index)."""
    _atomic_save({
        "format_version": FORMAT_VERSION,
        "model": cpu_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_score": None if best_score is None else float(best_score),
        "best_epoch": best_epoch,
        "epochs_without_improvement": int(epochs_without_improvement),
        "best_model": best_model_state,
        "finished": bool(finished),
        "finish_reason": finish_reason,
        "seed": int(seed),
        "config_hash": config_hash,
        "rng": capture_rng_state(),
        "extra": extra or {},
    }, Path(path))


def save_best_export(path: Path, *, best_model_state: dict, best_epoch: int,
                     best_score: float, seed: int, config_hash: str) -> None:
    """``best.pt``: the best epoch's weights only (what predict.py loads)."""
    _atomic_save({"format_version": FORMAT_VERSION, "model": best_model_state,
                  "epoch": int(best_epoch), "best_score": float(best_score),
                  "seed": int(seed), "config_hash": config_hash}, Path(path))


def load_checkpoint(path: Path, *, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    scaler: torch.amp.GradScaler | None = None,
                    map_location: str | torch.device = "cpu",
                    restore_rng: bool = True) -> dict:
    """Load a checkpoint into ``model`` (and optimizer / scaler if given) and,
    by default, restore all RNG states. Returns the checkpoint dict."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if ckpt.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unknown checkpoint format {ckpt.get('format_version')}")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng and "rng" in ckpt:
        restore_rng_state(ckpt["rng"])
    return ckpt
