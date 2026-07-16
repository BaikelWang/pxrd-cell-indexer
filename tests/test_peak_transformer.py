"""Tests for A2 PeakGeometryTransformerEncoder."""

from __future__ import annotations

import torch

from pxrd_cell_indexing.model.encoder.peak_transformer import PeakGeometryTransformerEncoder
from pxrd_cell_indexing.model.heads import build_indexing_model


def _fake_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Two samples: 3 peaks and 2 peaks (concatenated).
    pxrd_x = torch.tensor([[10.0], [25.0], [40.0], [12.0], [30.0]], dtype=torch.float32)
    pxrd_y = torch.tensor([[80.0], [40.0], [20.0], [90.0], [30.0]], dtype=torch.float32)
    peak_num = torch.tensor([3, 2], dtype=torch.long)
    return pxrd_x, pxrd_y, peak_num


def test_peak_transformer_shapes_and_mask_modes() -> None:
    pxrd_x, pxrd_y, peak_num = _fake_batch()
    for mode, npeaks in (("pos", 20), ("pos_i", 20), ("geom", 48)):
        for pool in ("cls", "mean", "cls_mean"):
            enc = PeakGeometryTransformerEncoder(
                {
                    "output_dim": 512,
                    "peak_transformer_token_mode": mode,
                    "peak_transformer_max_peaks": npeaks,
                    "peak_transformer_dropout": 0.0,
                    "peak_transformer_num_layers": 2,
                    "peak_transformer_d_model": 64,
                    "peak_transformer_num_heads": 4,
                    "peak_transformer_ffn_dim": 128,
                    "peak_transformer_fourier_freqs": 8,
                    "peak_transformer_pool": pool,
                }
            )
            enc.eval()
            with torch.no_grad():
                out = enc(pxrd_x, pxrd_y, peak_num)
            assert out.shape == (2, 512)
            assert torch.isfinite(out).all()


def test_peak_transformer_fourier_and_pool_variants() -> None:
    pxrd_x, pxrd_y, peak_num = _fake_batch()
    for fmode in ("linear", "log", "loglinear"):
        for pool in ("cls_mean", "attn", "cls_mean_max"):
            enc = PeakGeometryTransformerEncoder(
                {
                    "output_dim": 128,
                    "peak_transformer_token_mode": "geom",
                    "peak_transformer_max_peaks": 20,
                    "peak_transformer_dropout": 0.0,
                    "peak_transformer_num_layers": 2,
                    "peak_transformer_d_model": 64,
                    "peak_transformer_num_heads": 4,
                    "peak_transformer_ffn_dim": 128,
                    "peak_transformer_fourier_freqs": 16,
                    "peak_transformer_fourier_mode": fmode,
                    "peak_transformer_pool": pool,
                }
            )
            enc.eval()
            with torch.no_grad():
                out = enc(pxrd_x, pxrd_y, peak_num)
            assert out.shape == (2, 128)
            assert torch.isfinite(out).all()
            # empty batch must not NaN either
            with torch.no_grad():
                out0 = enc(torch.zeros(0, 1), torch.zeros(0, 1), torch.tensor([0, 0]))
            assert torch.isfinite(out0).all()


def test_peak_transformer_legacy_no_fourier() -> None:
    pxrd_x, pxrd_y, peak_num = _fake_batch()
    enc = PeakGeometryTransformerEncoder(
        {
            "output_dim": 64,
            "peak_transformer_token_mode": "pos",
            "peak_transformer_max_peaks": 20,
            "peak_transformer_dropout": 0.0,
            "peak_transformer_num_layers": 2,
            "peak_transformer_d_model": 64,
            "peak_transformer_num_heads": 4,
            "peak_transformer_ffn_dim": 128,
            "peak_transformer_fourier_freqs": 0,
            "peak_transformer_pool": "cls",
        }
    )
    enc.eval()
    with torch.no_grad():
        out = enc(pxrd_x, pxrd_y, peak_num)
    assert out.shape == (2, 64)


def test_peak_transformer_empty_and_short() -> None:
    enc = PeakGeometryTransformerEncoder(
        {
            "output_dim": 128,
            "peak_transformer_token_mode": "geom",
            "peak_transformer_max_peaks": 20,
            "peak_transformer_dropout": 0.0,
            "peak_transformer_num_layers": 2,
            "peak_transformer_d_model": 64,
            "peak_transformer_num_heads": 4,
            "peak_transformer_ffn_dim": 128,
        }
    )
    enc.eval()
    # N=0
    with torch.no_grad():
        out0 = enc(
            torch.zeros(0, 1),
            torch.zeros(0, 1),
            torch.tensor([0, 0], dtype=torch.long),
        )
    assert out0.shape == (2, 128)
    # N=1 < 20
    with torch.no_grad():
        out1 = enc(
            torch.tensor([[15.0]], dtype=torch.float32),
            torch.tensor([[50.0]], dtype=torch.float32),
            torch.tensor([1], dtype=torch.long),
        )
    assert out1.shape == (1, 128)
    # N>48 truncated
    n = 60
    tt = torch.linspace(5, 70, n).reshape(-1, 1)
    inten = torch.ones(n, 1)
    enc48 = PeakGeometryTransformerEncoder(
        {
            "output_dim": 128,
            "peak_transformer_token_mode": "geom",
            "peak_transformer_max_peaks": 48,
            "peak_transformer_dropout": 0.0,
            "peak_transformer_num_layers": 2,
            "peak_transformer_d_model": 64,
            "peak_transformer_num_heads": 4,
            "peak_transformer_ffn_dim": 128,
        }
    )
    enc48.eval()
    with torch.no_grad():
        out = enc48(tt, inten, torch.tensor([n], dtype=torch.long))
    assert out.shape == (1, 128)


def test_build_indexing_model_peak_transformer() -> None:
    model = build_indexing_model(
        checkpoint_path=None,
        encoder_config={
            "encoder_type": "peak_transformer",
            "peak_transformer_token_mode": "pos",
            "peak_transformer_max_peaks": 20,
            "peak_transformer_dropout": 0.0,
            "peak_transformer_num_layers": 2,
            "peak_transformer_d_model": 64,
            "peak_transformer_num_heads": 4,
            "peak_transformer_ffn_dim": 128,
            "output_dim": 512,
        },
        freeze_encoder=False,
        normalize_embedding=False,
    )
    model.eval()
    pxrd_x = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float32)
    pxrd_y = torch.tensor([[100.0], [50.0], [25.0]], dtype=torch.float32)
    peak_num = torch.tensor([3], dtype=torch.long)
    with torch.no_grad():
        out = model(pxrd_x, pxrd_y, peak_num)
    assert out["lattice_norm"].shape == (1, 6)
