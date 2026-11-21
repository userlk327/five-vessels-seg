#!/bin/bash
#SBATCH --job-name=fives-envcheck
#SBATCH --partition=ltarunl
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#
# One-off check that the pinned env sees the GPU and runs mixed precision.
# Usage: sbatch jobs/env_check.sbatch   (to test a V100: add --constraint=gpu_v100)

source jobs/common.sh

python - <<'EOF' || { echo 'ENV CHECK FAILED'; exit 1; }
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version())
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU visible"
cap = torch.cuda.get_device_capability()
arch = f"sm_{cap[0]}{cap[1]}"
archs = torch.cuda.get_arch_list()
print("device:", torch.cuda.get_device_name(), "| capability", arch)
print("torch arch list:", archs)
# A binary built for sm_XY runs on sm_XZ with Z >= Y (same major version), e.g. sm_86 code on an
# L40S (sm_89). Check that this torch build has native code the device can run.
runnable = [a for a in archs if a.startswith("sm_") and int(a[3:-1]) == cap[0] and int(a[-1]) <= cap[1]]
print("runnable on this device:", runnable)
assert runnable, f"no compiled code in this torch build runs on {arch}"
print("compiled arch flags:", torch._C._cuda_getArchFlags())
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
x = torch.randn(1024, 1024, device="cuda")
with torch.autocast("cuda", dtype=torch.float16):
    y = x @ x
print("fp16 autocast matmul ok:", y.dtype)
# one fp16 training step through cuDNN conv + BatchNorm + ConvTranspose with a GradScaler
net = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.BatchNorm2d(16),
                          torch.nn.ReLU(), torch.nn.ConvTranspose2d(16, 1, 2, stride=2)).cuda()
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda")
img, tgt = torch.randn(4, 3, 64, 64, device="cuda"), torch.rand(4, 1, 128, 128, device="cuda").round()
with torch.autocast("cuda", dtype=torch.float16):
    logits = net(img)
loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), tgt)
scaler.scale(loss).backward()
scaler.step(opt)
scaler.update()
torch.cuda.synchronize()
print(f"fp16 conv train step ok: loss={loss.item():.4f}, logits {logits.dtype}, "
      f"grad scale {scaler.get_scale():.0f}, peak mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
print("ENV CHECK PASSED")
EOF

echo "== unit tests on the GPU node (includes the CUDA RNG round trip)"
python -m pytest -v
