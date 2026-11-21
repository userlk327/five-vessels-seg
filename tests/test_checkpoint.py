"""Checkpoint / resume tests (CPU, tiny synthetic model; safe on the login node).

The key test trains a tiny model whose every epoch consumes Python, NumPy and torch
randomness (dropout, random data, DataLoader workers with augmentation-like noise),
interrupts it, resumes it in a "fresh process" (different seed, different initial
weights, new optimizer), and checks that the result is bit-identical to an
uninterrupted run.
"""
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS for the GPU test

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.checkpoint import capture_rng_state, load_checkpoint, restore_rng_state
from src.train import fit
from src.utils import epoch_seed, set_seed, seed_worker

SEED, MAX_EPOCHS, PATIENCE = 0, 10, 4


class NoisyData(Dataset):
    """Each item draws from NumPy, random and torch, like random augmentation does."""

    def __len__(self):
        return 16

    def __getitem__(self, i):
        x = torch.full((8,), float(i)) + torch.rand(8) + np.random.rand() + random.random()
        return x.float(), torch.tensor([float(i % 2)])


class Interrupted(Exception):
    pass


def make_run(seed_for_init: int, interrupt_at: int | None = None, device: str = "cpu"):
    set_seed(seed_for_init)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Dropout(0.5),
                                torch.nn.Linear(16, 1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler(device)
    gen = torch.Generator()
    loader = DataLoader(NoisyData(), batch_size=4, shuffle=True, num_workers=2,
                        worker_init_fn=seed_worker, generator=gen)
    val_x = torch.linspace(-1, 1, 64).reshape(8, 8).to(device)

    def train_epoch(epoch):
        if interrupt_at is not None and epoch == interrupt_at:
            raise Interrupted
        gen.manual_seed(epoch_seed(SEED, epoch))
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)   # on cuda, randn_like and dropout use the CUDA RNG
            x = x + 0.01 * torch.randn_like(x) + 0.01 * np.random.randn() + 0.01 * random.random()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        return {"train_loss": total}

    def val(epoch):
        model.eval()
        with torch.no_grad():
            out = model(val_x).mean().item()
        return {"val_loss": abs(out), "val_dice": -abs(out - 0.3) + 0.01 * random.random()}

    return model, opt, scaler, train_epoch, val


def run_fit(run_dir, seed_for_init, interrupt_at=None, device="cpu"):
    model, opt, scaler, tr, va = make_run(seed_for_init, interrupt_at, device)
    try:
        res = fit(model=model, optimizer=opt, scaler=scaler, run_dir=run_dir, seed=SEED,
                  cfg_hash="test", max_epochs=MAX_EPOCHS, patience=PATIENCE,
                  train_epoch_fn=tr, val_fn=va, device=device)
    except Interrupted:
        res = None
    return model, res


def read_log(run_dir):
    import csv
    with open(run_dir / "train_log.csv") as f:
        return [{k: v for k, v in r.items() if k != "epoch_time_s"} for r in csv.DictReader(f)]


def test_rng_roundtrip():
    set_seed(123)
    state = capture_rng_state()
    a = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    random.random(); np.random.rand(); torch.rand(5)       # move every generator on
    restore_rng_state(state)
    b = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    assert a == b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (runs in jobs/env_check.sbatch)")
def test_cuda_rng_roundtrip(tmp_path):
    set_seed(123)
    state = capture_rng_state()
    assert len(state["torch_cuda"]) == torch.cuda.device_count() >= 1
    a = torch.rand(5, device="cuda").tolist()
    torch.rand(7, device="cuda")
    torch.save({"rng": state}, tmp_path / "rng.pt")            # through a file, like a checkpoint
    loaded = torch.load(tmp_path / "rng.pt", map_location="cuda", weights_only=True)["rng"]
    loaded["torch_cpu"] = loaded["torch_cpu"].cpu()
    loaded["torch_cuda"] = [s.cpu() for s in loaded["torch_cuda"]]
    restore_rng_state(loaded)
    assert torch.rand(5, device="cuda").tolist() == a


def test_checkpoint_has_required_fields(tmp_path):
    run_fit(tmp_path, seed_for_init=SEED, interrupt_at=3)
    ckpt = torch.load(tmp_path / "last.pt", weights_only=True)   # loads without pickle code
    for key in ["model", "optimizer", "scaler", "epoch", "best_score", "best_epoch",
                "epochs_without_improvement", "best_model", "rng", "seed", "config_hash"]:
        assert key in ckpt, key
    assert set(ckpt["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert ckpt["epoch"] == 3 and ckpt["scaler"]["scale"] > 0 and ckpt["optimizer"]["state"]


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU (runs in jobs/env_check.sbatch)"))]


@pytest.fixture
def deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("interrupt_at", [1, 4])
def test_resume_is_bit_identical(tmp_path, interrupt_at, device, deterministic):
    """On cuda this is the real GPU path: fit() loads last.pt with map_location="cuda"."""
    ref_dir, res_dir = tmp_path / "ref", tmp_path / "resumed"
    ref_model, ref = run_fit(ref_dir, seed_for_init=SEED, device=device)

    run_fit(res_dir, seed_for_init=SEED, interrupt_at=interrupt_at, device=device)
    (res_dir / "best.pt").unlink()           # simulate a kill before best.pt was exported
    # "fresh process": different seed and init; everything must come from last.pt
    res_model, res = run_fit(res_dir, seed_for_init=999, device=device)

    assert res["resumed_from"] == interrupt_at
    assert (ref["best_epoch"], ref["best_score"], ref["finish_reason"]) == \
           (res["best_epoch"], res["best_score"], res["finish_reason"])
    for (k, a), (_, b) in zip(ref_model.state_dict().items(), res_model.state_dict().items()):
        assert torch.equal(a, b), k
    ref_best = torch.load(ref_dir / "best.pt", weights_only=True)
    res_best = torch.load(res_dir / "best.pt", weights_only=True)
    assert ref_best["epoch"] == res_best["epoch"]
    assert all(torch.equal(ref_best["model"][k], res_best["model"][k]) for k in ref_best["model"])
    assert read_log(ref_dir) == read_log(res_dir)


def test_finished_run_does_nothing(tmp_path):
    run_fit(tmp_path, seed_for_init=SEED)
    before = (tmp_path / "last.pt").stat().st_mtime_ns
    _, res = run_fit(tmp_path, seed_for_init=SEED, interrupt_at=0)   # would raise if it trained
    assert res["finished"] and (tmp_path / "last.pt").stat().st_mtime_ns == before


def test_config_change_is_refused(tmp_path):
    run_fit(tmp_path, seed_for_init=SEED, interrupt_at=2)
    model, opt, scaler, tr, va = make_run(SEED)
    with pytest.raises(RuntimeError, match="different config"):
        fit(model=model, optimizer=opt, scaler=scaler, run_dir=tmp_path, seed=SEED,
            cfg_hash="changed", max_epochs=MAX_EPOCHS, patience=PATIENCE,
            train_epoch_fn=tr, val_fn=va)


def test_load_checkpoint_restores_optimizer_and_scaler(tmp_path):
    run_fit(tmp_path, seed_for_init=SEED, interrupt_at=3)
    model, opt, scaler, _, _ = make_run(seed_for_init=999)
    ckpt = load_checkpoint(tmp_path / "last.pt", model=model, optimizer=opt, scaler=scaler)
    assert scaler.get_scale() == ckpt["scaler"]["scale"]
    saved = ckpt["optimizer"]["state"][0]["exp_avg"]
    assert torch.equal(opt.state_dict()["state"][0]["exp_avg"], saved)
