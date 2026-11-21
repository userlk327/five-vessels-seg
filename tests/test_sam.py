"""Fine-tuned SAM (5b) and oracle SAM (5a) on a tiny randomly initialized SAM (CPU)."""
from functools import partial

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

from src.data import stitch_tiles, tile_slices
from src.losses import BCEDiceLoss
from src.models.blocks import upsample_bilinear_int
from src.models.sam_lora import LoRAQKV, SamLoRA
from src.sam_oracle import prompt_rng, sample_prompts, to_input_frame
from src.sam_train import _group, _group_weights, check_tile_batch, logits_at_input, prob_maps
from src.utils import load_config


def tiny_sam():
    torch.manual_seed(0)
    enc = ImageEncoderViT(depth=2, embed_dim=32, img_size=1024, mlp_ratio=2,
                          norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_heads=2, patch_size=16,
                          qkv_bias=True, use_rel_pos=True, global_attn_indexes=(1,), window_size=14, out_chans=256)
    prompt = PromptEncoder(embed_dim=256, image_embedding_size=(64, 64), input_image_size=(1024, 1024), mask_in_chans=16)
    dec = MaskDecoder(num_multimask_outputs=3, transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=512,
                                                                             num_heads=8),
                      transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=64)
    return Sam(image_encoder=enc, prompt_encoder=prompt, mask_decoder=dec,
               pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375])


@pytest.fixture(scope="module")
def model():
    return SamLoRA(sam=tiny_sam(), lora_rank=4)


def test_freezing_and_lora_placement(model):
    groups = {}
    for n, p in model.named_parameters():
        groups.setdefault(_group(n), set()).add(p.requires_grad)
    assert groups["image_encoder (frozen)"] == {False} and groups["prompt_encoder (frozen)"] == {False}
    assert groups["mask_decoder"] == {True} and groups["lora_A"] == {True} and groups["lora_B"] == {True}
    assert all(isinstance(b.attn.qkv, LoRAQKV) for b in model.sam.image_encoder.blocks)
    assert len(model.lora_parameters()) == 4 * len(model.sam.image_encoder.blocks)


def test_lora_starts_as_identity():
    sam = tiny_sam()
    qkv = sam.image_encoder.blocks[0].attn.qkv
    x = torch.randn(1, 14, 14, 32)
    ref = qkv(x)
    assert torch.equal(LoRAQKV(qkv, 4)(x), ref)                       # B = 0 at init


def test_forward_batch_equals_single_images(model):
    model.eval()
    x = torch.rand(2, 3, 1024, 1024) * 255
    with torch.no_grad():
        both = model(x)
        one = torch.cat([model(x[:1]), model(x[1:])])
    assert both.shape == (2, 1, 256, 256) and torch.allclose(both, one, atol=1e-5)


def test_gradient_pattern_first_step(model):
    """B = 0 at init: B gets gradients, A gets exactly zero. Decoder: hypernetwork MLPs of the
    unused mask tokens get zero gradients, the IoU head none; frozen parts none."""
    model.train()
    model.zero_grad(set_to_none=True)
    x = torch.rand(1, 3, 1024, 1024) * 255
    t = (torch.rand(1, 1, 1024, 1024) < 0.1).float()
    BCEDiceLoss()(upsample_bilinear_int(model(x), 4), t)[0].backward()
    for n, p in model.named_parameters():
        g = _group(n)
        if g == "lora_B":
            assert p.grad is not None and p.grad.abs().sum() > 0, n
        elif g == "lora_A":
            assert p.grad is not None and not p.grad.any(), n
        elif g.endswith("(frozen)"):
            assert p.grad is None, n
        elif "iou_prediction_head" in n:
            assert p.grad is None, n
        elif "output_hypernetworks_mlps.0." in n:
            assert p.grad.abs().sum() > 0, n
        elif "output_hypernetworks_mlps." in n:
            assert not p.grad.any(), n


def test_x4_upsampling_equals_sam_postprocessing():
    x = torch.randn(2, 1, 16, 16, dtype=torch.float64, requires_grad=True)
    ref = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
    out = upsample_bilinear_int(x, 4)
    assert torch.allclose(out, ref, atol=1e-12)
    w = torch.randn_like(ref)
    assert torch.allclose(torch.autograd.grad((out * w).sum(), x)[0], torch.autograd.grad((ref * w).sum(), x)[0], atol=1e-12)


def test_tiles_and_stitching_are_inverse():
    img = np.arange(16 * 16).reshape(16, 16)
    tiles = np.stack([img[tile_slices(16, 8, t)] for t in range(4)])
    assert np.array_equal(stitch_tiles(tiles, 2), img)
    tt = torch.from_numpy(tiles)[:, None]
    assert torch.equal(stitch_tiles(tt, 2)[0], torch.from_numpy(img))


def test_prob_maps_per_variant():
    cfg = load_config("configs/sam_lora_tiled.yaml")
    low = torch.randn(8, 1, 256, 256)                                    # 2 images x 4 tiles
    p = prob_maps(low, cfg)
    assert p.shape == (2, 1, 512, 512)
    assert torch.allclose(p[1, 0, :256, 256:], torch.sigmoid(low[5, 0]))  # image 2, tile 1 = top right
    w = load_config("configs/sam_lora_whole.yaml")
    assert prob_maps(low[:2], w).shape == (2, 1, 1024, 1024)
    assert torch.allclose(prob_maps(low[:1], w), torch.sigmoid(logits_at_input(low[:1], w)))


def test_group_weights_average_each_step():
    w = _group_weights(5, 2, 2)                  # micro-batches 2, 2, 1 -> groups (2, 2), (1)
    assert w == [(0.5, False), (0.5, True), (1.0, True)]
    assert sum(x for x, _ in _group_weights(1908, 2, 2)) == pytest.approx(1908 / 4)


def test_oracle_prompts():
    gt = np.zeros((2048, 2048), bool)
    gt[1000:1004, 200:1800] = True
    fov = np.zeros_like(gt)
    fov[100:1900, 100:1900] = True
    p5, n5 = sample_prompts(gt, fov, 5, prompt_rng(0, "train_1_A"), 10)
    p10, n10 = sample_prompts(gt, fov, 10, prompt_rng(0, "train_1_A"), 10)
    assert np.array_equal(p10[:5], p5) and np.array_equal(n10[:5], n5)          # nested
    x, y = p10[:, 0].astype(int), p10[:, 1].astype(int)
    assert gt[y, x].all()
    xn, yn = n10[:, 0].astype(int), n10[:, 1].astype(int)
    assert fov[yn, xn].all() and (np.abs(yn - 1001.5) > 10).all()
    assert np.allclose(to_input_frame(np.array([[0.0, 2047.0]]), 2048, 1024), [[-0.25, 1023.25]])


def test_val_loader_yields_complete_tile_sets(tmp_path):
    """SamTileDataset + DataLoader(batch = 4, no shuffle), as in validation: every batch is
    one image's 4 tiles in order, so the stitched map is the whole image."""
    import cv2
    from torch.utils.data import DataLoader

    from src.data import SamTileDataset
    cfg = load_config("configs/sam_lora_tiled.yaml")
    cfg["data"]["native_resolution"], cfg["data"]["sam_resolution"] = 64, 32
    cfg["paths"]["cache_dir"] = str(tmp_path)
    ids = ["train_1_A", "train_2_D", "train_3_G"]
    (tmp_path / "images_clahe_64").mkdir()
    full = {}
    for n, i in enumerate(ids):
        full[i] = np.random.RandomState(n).randint(0, 255, (64, 64, 3), np.uint8)
        cv2.imwrite(str(tmp_path / "images_clahe_64" / f"{i}.png"), full[i][..., ::-1])
    loader = DataLoader(SamTileDataset(cfg, ids, train=False, with_masks=False), batch_size=4, shuffle=False)
    seen = []
    for batch in loader:
        got = check_tile_batch(batch, 4)
        img = stitch_tiles(batch["image"], 2).permute(1, 2, 0).numpy().astype(np.uint8)
        assert np.array_equal(img, full[got[0]])                      # stitched tiles = the original image
        seen += got
    assert seen == ids
    bad = {"image_id": ["a", "a", "b", "b"], "tile": torch.tensor([0, 1, 0, 1])}
    with pytest.raises(RuntimeError):
        check_tile_batch(bad, 4)
