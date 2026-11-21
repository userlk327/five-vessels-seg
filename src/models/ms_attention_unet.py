"""Model 4: Multi-Scale Attention U-Net (attention yes, multi-scale yes): the attention
gates of model 2 and the multi-scale blocks of model 3 together, completing the 2x2
ablation. Design: src/models/blocks.py.
"""
from src.models.blocks import UNetVariant


class MultiScaleAttentionUNET(UNetVariant):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__(in_channels, out_channels, attention=True, multiscale=True)
