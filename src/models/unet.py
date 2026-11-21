"""Model 1: standard U-Net (baseline).

``convolution_block`` and ``UNET`` are copied UNCHANGED from Baseline_UNET.ipynb
(cell 7); only this docstring and the two imports were added.
Phase 3 confirms that it matches the standard design in CLAUDE.md Part A:
4 levels down / 4 up, 64 -> 1024 channels, BatchNorm, ConvTranspose upsampling,
1x1 output conv producing logits.
"""
import torch
import torch.nn as nn


# Standard UNET implementation
def convolution_block(in_channels, out_channels):
  return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
  )


class UNET(nn.Module):
  def __init__(self, in_channels=3, out_channels=1):
    super().__init__()
    self.pool = nn.MaxPool2d(2)

    # 1. Encoder (Downsampling path)
    self.enc1 = convolution_block(in_channels, 64)
    self.enc2 = convolution_block(64, 128)
    self.enc3 = convolution_block(128, 256)
    self.enc4 = convolution_block(256, 512)

    # 2. Bottleneck
    self.bottleneck = convolution_block(512, 1024)

    # 3. Decoder (Upsampling path + Convolutions)
    self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
    self.dec4 = convolution_block(1024, 512)   # 512 (up4) + 512 (enc4) = 1024 in

    self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.dec3 = convolution_block(512, 256)    # 256 (up3) + 256 (enc3) = 512 in

    self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.dec2 = convolution_block(256, 128)    # 128 (up2) + 128 (enc2) = 256 in

    self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.dec1 = convolution_block(128, 64)     # 64 (up1) + 64 (enc1) = 128 in

    # 4. Final 1x1 projection to raw logits
    self.out = nn.Conv2d(64, out_channels, kernel_size=1)

  def forward(self, x):
      # Encoder passes
      e1 = self.enc1(x)
      e2 = self.enc2(self.pool(e1))
      e3 = self.enc3(self.pool(e2))
      e4 = self.enc4(self.pool(e3))

      # Bottleneck
      b = self.bottleneck(self.pool(e4))

      # Decoder passes with skip connections
      d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
      d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
      d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
      d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

      return self.out(d1)
