"""Phase 4 models: the backbone is the notebook U-Net; only the added modules differ."""
import pytest
import torch

from src.models import build_model
from src.models.blocks import AttentionGate, MultiScaleConv, UNetVariant, split_channels, upsample2x_bilinear
from src.models.unet import UNET
from src.utils import load_config

CONFIGS = ["configs/unet.yaml", "configs/attention_unet.yaml", "configs/multiscale_unet.yaml",
           "configs/ms_attention_unet.yaml"]


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_plain_variant_is_the_notebook_unet():
    """Same state-dict keys and shapes; same seed -> same weights; same outputs."""
    torch.manual_seed(0)
    a = UNET()
    torch.manual_seed(0)
    b = UNetVariant(attention=False, multiscale=False)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(sa[k].shape == sb[k].shape and torch.equal(sa[k], sb[k]) for k in sa)
    a.eval(); b.eval()
    x = torch.randn(1, 3, 64, 64)
    assert torch.equal(a(x), b(x))


def test_attention_unet_starts_from_the_unet_backbone_weights():
    torch.manual_seed(0)
    u = UNET().state_dict()
    torch.manual_seed(0)
    a = UNetVariant(attention=True).state_dict()
    assert all(torch.equal(u[k], a[k]) for k in u)
    assert {k.split(".")[0] for k in set(a) - set(u)} == {"att1", "att2", "att3", "att4"}


@pytest.mark.parametrize("cfg", CONFIGS)
def test_models_build_from_config_and_keep_the_shape(cfg):
    m = build_model(load_config(cfg)).eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 64, 64))
    assert y.shape == (2, 1, 64, 64)


def test_attention_gate():
    gate = AttentionGate(64, 128)
    x, g = torch.randn(2, 64, 32, 32), torch.randn(2, 128, 16, 16)
    alpha = gate.coefficients(x, g)
    assert alpha.shape == (2, 1, 32, 32) and alpha.min() > 0 and alpha.max() < 1
    assert torch.allclose(gate(x, g), x * alpha)
    # theta 64*32*4 (no bias) + phi 128*32 + 32 + psi 32 + 1
    assert n_params(gate) == 64 * 32 * 4 + 128 * 32 + 32 + 32 + 1


def test_multiscale_conv_splits_channels_and_keeps_size():
    assert split_channels(64, 3) == [22, 21, 21] and sum(split_channels(1024, 3)) == 1024
    m = MultiScaleConv(64, 64)
    assert [b.dilation for b in m.branches] == [(1, 1), (2, 2), (4, 4)]
    assert [b.out_channels for b in m.branches] == [22, 21, 21]
    assert m(torch.randn(1, 64, 40, 40)).shape == (1, 64, 40, 40)
    # branches cost one ordinary 3x3 conv (9 * 64 * 64); added: 1x1 fusion + one BN
    assert n_params(m) == 9 * 64 * 64 + 2 * 64 + 64 * 64 + 2 * 64


def test_multiscale_within_ten_percent_of_the_baseline():
    base = n_params(UNET())
    for att, ms in [(False, True), (True, False), (True, True)]:
        assert abs(n_params(UNetVariant(attention=att, multiscale=ms)) / base - 1) < 0.10


def test_deterministic_upsampling_equals_interpolate():
    torch.manual_seed(0)
    x = torch.randn(2, 1, 9, 16, dtype=torch.float64, requires_grad=True)
    ref = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    out = upsample2x_bilinear(x)
    assert out.shape == (2, 1, 18, 32) and torch.allclose(out, ref, atol=1e-12)
    w = torch.randn_like(ref)
    g_ref, = torch.autograd.grad((ref * w).sum(), x)
    g_out, = torch.autograd.grad((out * w).sum(), x)
    assert torch.allclose(g_out, g_ref, atol=1e-12)


@pytest.mark.parametrize("cfg", CONFIGS[1:])
def test_added_modules_receive_gradients(cfg):
    from src.losses import BCEDiceLoss
    torch.manual_seed(0)
    m = build_model(load_config(cfg)).train()
    t = (torch.rand(2, 1, 64, 64) < 0.2).float() * 0.5
    BCEDiceLoss()(m(torch.randn(2, 3, 64, 64)), t)[0].backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, missing
    added = [p for n, p in m.named_parameters() if n.startswith("att") or ".fuse." in n or ".branches." in n]
    assert added and all(p.grad.abs().sum() > 0 for p in added)
