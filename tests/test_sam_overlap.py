import numpy as np

from src.sam_overlap import blend_tiles, tent, tile_origins


def _origins():
    s = tile_origins(2048, 1024, 512)
    return [(y, x) for y in s for x in s]


def test_tile_grid_is_3x3_and_contains_the_main_2x2_tiles():
    o = _origins()
    assert len(o) == 9
    assert {(0, 0), (0, 1024), (1024, 0), (1024, 1024)} <= set(o)


def test_tents_sum_to_one_inside_the_image():
    w = tent(256)
    total = np.zeros(512)
    for g0 in (0, 128, 256):
        total[g0:g0 + 256] += w
    assert np.allclose(total[128:384], 1.0)
    assert w.min() > 0 and np.isclose(w.max(), 1 - 0.5 / 128)


def test_blending_crops_of_one_map_reproduces_it_exactly():
    rng = np.random.RandomState(0)
    full = rng.randn(512, 512).astype(np.float32)
    o = _origins()
    tiles = np.stack([full[y // 4:y // 4 + 256, x // 4:x // 4 + 256] for y, x in o])
    assert np.allclose(blend_tiles(tiles, o, 512, 4), full, atol=1e-5)


def test_blend_is_a_convex_combination_of_overlapping_tiles():
    o = _origins()
    tiles = np.stack([np.full((256, 256), float(k)) for k in range(9)])
    out = blend_tiles(tiles, o, 512, 4)
    assert out.min() >= 0 and out.max() <= 8
    assert np.isclose(out[0, 0], 0) and np.isclose(out[-1, -1], 8)     # corners: one tile only
    assert np.isclose(out[256, 256], 4, atol=0.05)                    # image centre: centre tile dominates
