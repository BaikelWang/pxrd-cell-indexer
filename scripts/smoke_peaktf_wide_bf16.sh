#!/usr/bin/env bash
# Preflight for v4 wide + bf16. Run on the training box before the full job.
#
# Checks:
#   1) gstar6_to_lattice under bf16 (the v3 killer)
#   2) build peaktf-wide + flow 8x1024, forward + sample under autocast
#   3) optional: one micro-batch train step if a free GPU is visible
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
from __future__ import annotations

import argparse
import sys
import traceback

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from pxrd_cell_indexing.geometry import gstar6_to_lattice
from train_flow_seedgen import SeedGenerator, peaktf_encoder_cfg

ok = True

def check(name, fn):
    global ok
    try:
        fn()
        print(f"[PASS] {name}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()

# 1) the exact v3 failure mode
def test_gstar6_bf16():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.randn(32, 6, device=device, dtype=torch.bfloat16)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        # mimic eval: stay in autocast while decoding
        cells = gstar6_to_lattice(z)
    assert cells.shape == (32, 6)
    assert torch.isfinite(cells.float()).all()

check("gstar6_to_lattice under bf16/autocast", test_gstar6_bf16)

# 2) wide model build + sample
def test_wide_sample():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = argparse.Namespace(
        peaktf_scale="wide", flow_hidden=1024, flow_layers=8
    )
    cfg = peaktf_encoder_cfg("wide")
    assert cfg["peak_transformer_d_model"] == 512
    assert cfg["peak_transformer_num_layers"] == 8
    model = SeedGenerator(args).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"       wide params = {n:,}")
    # synthetic peak batch
    B, P = 4, 24
    x = torch.linspace(10, 80, P, device=device).view(1, P, 1).expand(B, P, 1).contiguous()
    y = torch.rand(B, P, 1, device=device) * 100
    npeaks = torch.full((B,), P, device=device, dtype=torch.long)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        emb = model.encode(x, y, npeaks)
        z = model.sample(emb, num_samples=8, steps=10)
        cells = gstar6_to_lattice(z.reshape(-1, 6))
    assert cells.shape[-1] == 6
    assert torch.isfinite(cells.float()).all()

check("peaktf-wide forward+sample+decode under bf16", test_wide_sample)

# 3) train-step smoke only if GPU has headroom
def test_train_step():
    if not torch.cuda.is_available():
        print("       skip train-step (no CUDA)")
        return
    free, total = torch.cuda.mem_get_info()
    free_gb = free / (1024**3)
    print(f"       GPU free ≈ {free_gb:.1f} GiB / {total/(1024**3):.1f} GiB")
    if free_gb < 10:
        print("       skip train-step (GPU occupied; run this on the empty 4-GPU box)")
        return
    from pxrd_cell_indexing.data.normalization import GStar6Normalizer
    args = argparse.Namespace(
        peaktf_scale="wide", flow_hidden=1024, flow_layers=8
    )
    model = SeedGenerator(args).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # fake unit-normal targets in gstar6 space
    B, P = 8, 32
    x = torch.linspace(10, 80, P, device="cuda").view(1, P, 1).expand(B, P, 1).contiguous()
    y = torch.rand(B, P, 1, device="cuda") * 100
    npeaks = torch.full((B,), P, device="cuda", dtype=torch.long)
    z1 = torch.randn(B, 6, device="cuda")
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, y, npeaks, z1)
    loss.backward()
    opt.step()
    print(f"       train loss={float(loss.detach()):.4f}")

check("optional train micro-step at free VRAM", test_train_step)

print()
print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
PY
