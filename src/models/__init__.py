"""Model registry: ``build_model(cfg)`` builds the network named in ``cfg["model"]["name"]``.

Models 1-4 (2x2 ablation: attention no/yes x multi-scale no/yes) share the
baseline U-Net backbone; only the added module differs (designs approved in Phase 4).
"""
from __future__ import annotations

import importlib

import torch

# name -> (module, class)
_REGISTRY = {
    "unet": ("src.models.unet", "UNET"),
    "attention_unet": ("src.models.attention_unet", "AttentionUNET"),
    "multiscale_unet": ("src.models.multiscale_unet", "MultiScaleUNET"),
    "ms_attention_unet": ("src.models.ms_attention_unet", "MultiScaleAttentionUNET"),
    "sam_lora": ("src.models.sam_lora", "SamLoRA"),
}


def build_model(cfg: dict) -> torch.nn.Module:
    m = dict(cfg["model"])
    name = m.pop("name")
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(_REGISTRY)}")
    module, cls = _REGISTRY[name]
    return getattr(importlib.import_module(module), cls)(**m)


def build_optimizer(cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    o = cfg["train"]["optimizer"]
    params = [p for p in model.parameters() if p.requires_grad]
    if o["name"].lower() == "adam":
        return torch.optim.Adam(params, lr=float(o["lr"]))
    raise KeyError(f"unknown optimizer {o['name']!r}")
