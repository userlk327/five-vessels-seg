"""Model 2: Attention U-Net (attention yes, multi-scale no).

The baseline U-Net backbone (same depth, channels and upsampling) plus additive attention
gates (Oktay et al., 2018) on each of the 4 skip connections, gated by the decoder
features one level coarser. Design: src/models/blocks.py (AttentionGate, UNetVariant).
"""
from src.models.blocks import UNetVariant


class AttentionUNET(UNetVariant):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__(in_channels, out_channels, attention=True, multiscale=False)
