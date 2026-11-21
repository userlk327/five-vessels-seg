"""Model 5b: SAM ViT-B fine-tuned with LoRA (SAMed-style), no prompts. Fully automatic.

* Official ``segment-anything`` (1.0) ViT-B, checkpoint sam_vit_b_01ec64.pth.
* Image encoder frozen. LoRA (rank 4) on the q and v parts of the fused ``qkv`` projection
  of every image-encoder attention block (12 blocks), as in SAMed:
      qkv(x) -> [q + B_q A_q x, k, v + B_v A_v x]
  A: kaiming-uniform (a = sqrt 5), B: zeros, so training starts from the pretrained SAM and
  A gets a zero gradient on the very first step (B = 0); see the Phase 5 gradient check.
* Prompt encoder frozen; no prompts (empty sparse embedding + its learned no-mask embedding).
* Mask decoder trained, ``multimask_output=False`` (mask token 0).
* Gradient path: ``Sam.forward`` and ``SamPredictor`` run under ``torch.no_grad()``, so the
  image encoder, prompt encoder and mask decoder are called directly here. The mask decoder
  is written for one image per call (it repeats the image embedding per prompt set), so it
  is called once per sample of the batch.

Input: (N, 3, 1024, 1024) RGB in 0..255 (SAM's own normalization is applied here, via
``Sam.preprocess``). Output: (N, 1, 256, 256) mask logits.

Model 5a (zero-shot, oracle point prompts) uses the plain SAM: src/sam_oracle.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRAQKV(nn.Module):
    """Frozen fused qkv projection + low-rank updates of its q and v parts."""

    def __init__(self, qkv: nn.Linear, rank: int):
        super().__init__()
        self.qkv, self.dim = qkv, qkv.in_features
        if qkv.out_features != 3 * self.dim:
            raise ValueError("expected a fused qkv projection (out = 3 x in)")
        self.a_q = nn.Linear(self.dim, rank, bias=False)
        self.b_q = nn.Linear(rank, self.dim, bias=False)
        self.a_v = nn.Linear(self.dim, rank, bias=False)
        self.b_v = nn.Linear(rank, self.dim, bias=False)
        for a, b in ((self.a_q, self.b_q), (self.a_v, self.b_v)):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
            nn.init.zeros_(b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        d = self.dim
        return torch.cat([qkv[..., :d] + self.b_q(self.a_q(x)), qkv[..., d:2 * d],
                          qkv[..., 2 * d:] + self.b_v(self.a_v(x))], dim=-1)


class SamLoRA(nn.Module):
    def __init__(self, checkpoint: str | None = None, lora_rank: int = 4, lora_targets=("q", "v"),
                 model_type: str = "vit_b", sam: nn.Module | None = None):
        super().__init__()
        if tuple(lora_targets) != ("q", "v"):
            raise ValueError("LoRA is applied to the q and v parts of qkv (SAMed)")
        if sam is None:
            from segment_anything import sam_model_registry

            from src.utils import project_path
            sam = sam_model_registry[model_type](checkpoint=str(project_path(checkpoint)))
        self.sam = sam
        for p in self.sam.parameters():                  # freeze everything ...
            p.requires_grad_(False)
        for p in self.sam.mask_decoder.parameters():     # ... except the mask decoder
            p.requires_grad_(True)
        for blk in self.sam.image_encoder.blocks:        # LoRA modules are new -> trainable
            blk.attn.qkv = LoRAQKV(blk.attn.qkv, lora_rank)

    def lora_parameters(self):
        return [p for n, p in self.named_parameters() if ".a_q." in n or ".b_q." in n or ".a_v." in n or ".b_v." in n]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.sam.image_encoder(self.sam.preprocess(x))
        pe = self.sam.prompt_encoder
        with torch.no_grad():
            sparse, dense = pe(points=None, boxes=None, masks=None)
            image_pe = pe.get_dense_pe()
        out = []
        for i in range(emb.shape[0]):
            low, _ = self.sam.mask_decoder(image_embeddings=emb[i:i + 1], image_pe=image_pe,
                                           sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                           multimask_output=False)
            out.append(low)
        return torch.cat(out, dim=0)
