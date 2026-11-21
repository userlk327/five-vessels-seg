"""Model 3: Multi-Scale U-Net (attention no, multi-scale yes).

The baseline U-Net backbone with the second 3x3 conv of every block (encoder 1-4,
bottleneck, decoder 4-1) replaced by parallel 3x3 convolutions with dilation 1, 2 and 4,
output channels split across the branches, fused by 1x1 conv + BN + ReLU.
Design: src/models/blocks.py (MultiScaleConv, UNetVariant).
"""
from src.models.blocks import UNetVariant


class MultiScaleUNET(UNetVariant):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__(in_channels, out_channels, attention=False, multiscale=True)
