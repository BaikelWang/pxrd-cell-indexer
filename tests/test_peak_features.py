"""Unit tests for physical peak features (R1)."""

from __future__ import annotations

import numpy as np
import torch

from pxrd_cell_indexing.data.peak_features import (
    PeakFeatureConfig,
    build_inverse_d2_histogram_features,
    build_per_peak_features,
    histogram_feature_dim,
    inverse_d2_from_two_theta,
    padding_mask_from_peak_num,
    peak_feature_dim,
    reciprocal_d_from_two_theta,
)
from pxrd_cell_indexing.model.encoder.bert import BertModel
from pxrd_cell_indexing.model.encoder.histogram import InverseD2HistogramEncoder
from pxrd_cell_indexing.model.encoder.loader import DEFAULT_ENCODER_CONFIG
from pxrd_cell_indexing.model.heads import build_indexing_model


def test_reciprocal_and_inverse_d2_roundtrip_scale() -> None:
    tt = np.array([10.0, 20.0, 40.0, 80.0], dtype=np.float64)
    s = reciprocal_d_from_two_theta(tt, wavelength_angstrom=1.54184)
    g = inverse_d2_from_two_theta(tt, wavelength_angstrom=1.54184)
    assert np.allclose(g, np.asarray(s) ** 2, rtol=1e-5)
    # Bragg: d = λ/(2sinθ); reciprocal_d = 1/d
    theta = np.deg2rad(tt / 2.0)
    d = 1.54184 / (2.0 * np.sin(theta))
    assert np.allclose(s, 1.0 / d, rtol=1e-5)


def test_per_peak_feature_shapes() -> None:
    tt = np.array([10.4, 20.1, 35.7], dtype=np.float32)
    inten = np.array([100.0, 40.0, 10.0], dtype=np.float32)
    for mode, dim in [
        ("legacy", 1),
        ("continuous_2theta_i", 2),
        ("reciprocal_d_i", 2),
        ("inverse_d2_i", 2),
        ("inverse_d2_logi", 2),
        ("inverse_d2_only", 1),
    ]:
        cfg = PeakFeatureConfig(feature_mode=mode)  # type: ignore[arg-type]
        feats = build_per_peak_features(tt, inten, config=cfg)
        assert feats.shape == (3, dim)
        assert peak_feature_dim(mode) == dim  # type: ignore[arg-type]
        assert np.isfinite(feats).all()


def test_histogram_feature_dim_and_invariance() -> None:
    cfg = PeakFeatureConfig(hist_bins=256, sorted_peak_count=24)
    tt = np.array([12.0, 24.0, 36.0], dtype=np.float32)
    inten = np.array([100.0, 50.0, 25.0], dtype=np.float32)
    f1 = build_inverse_d2_histogram_features(tt, inten, config=cfg)
    f2 = build_inverse_d2_histogram_features(tt, inten * 3.0, config=cfg)
    assert f1.shape == (histogram_feature_dim(cfg),)
    assert np.allclose(f1, f2, atol=1e-5)


def test_padding_mask_from_peak_num() -> None:
    peak_num = torch.tensor([3, 1, 2], dtype=torch.long)
    mask = padding_mask_from_peak_num(peak_num, include_cls=True)
    assert mask.shape == (3, 4)
    assert not mask[:, 0].any()
    assert mask[1, 2:].all()
    assert not mask[0, 1:4].any()


def test_physical_bert_forward() -> None:
    cfg = dict(DEFAULT_ENCODER_CONFIG)
    cfg.update(
        {
            "position_encoding": "physical",
            "peak_feature_mode": "inverse_d2_i",
        }
    )
    model = BertModel(**cfg)
    model.eval()
    pxrd_x = torch.tensor([[10.4], [20.1], [30.0], [15.2], [25.3]], dtype=torch.float32)
    pxrd_y = torch.tensor([[12.0], [45.0], [80.0], [60.0], [30.0]], dtype=torch.float32)
    peak_num = torch.tensor([3, 2], dtype=torch.long)
    with torch.no_grad():
        out = model(pxrd_x, pxrd_y, peak_num)
    assert out.shape == (2, 512)
    assert torch.isfinite(out).all()


def test_histogram_encoder_forward() -> None:
    enc = InverseD2HistogramEncoder({"output_dim": 512, "histogram_dropout": 0.0})
    enc.eval()
    pxrd_x = torch.tensor([[10.4], [20.1], [30.0], [15.2], [25.3]], dtype=torch.float32)
    pxrd_y = torch.tensor([[12.0], [45.0], [80.0], [60.0], [30.0]], dtype=torch.float32)
    peak_num = torch.tensor([3, 2], dtype=torch.long)
    with torch.no_grad():
        out = enc(pxrd_x, pxrd_y, peak_num)
    assert out.shape == (2, 512)
    assert torch.isfinite(out).all()


def test_build_indexing_model_histogram() -> None:
    model = build_indexing_model(
        checkpoint_path=None,
        encoder_config={"encoder_type": "histogram", "histogram_dropout": 0.0},
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


def test_histogram_spectrum_fusion_forward() -> None:
    from pxrd_cell_indexing.model.encoder.spectrum_fusion import HistogramSpectrumFusionEncoder

    enc = HistogramSpectrumFusionEncoder(
        {
            "output_dim": 512,
            "histogram_dropout": 0.0,
            "histogram_hidden_dim": 256,
            "histogram_num_blocks": 2,
            "spectrum_bins": 256,
            "spectrum_cnn_channels": (32, 64),
        }
    )
    enc.eval()
    pxrd_x = torch.tensor([[10.4], [20.1], [30.0], [15.2], [25.3]], dtype=torch.float32)
    pxrd_y = torch.tensor([[12.0], [45.0], [80.0], [60.0], [30.0]], dtype=torch.float32)
    peak_num = torch.tensor([3, 2], dtype=torch.long)
    with torch.no_grad():
        out = enc(pxrd_x, pxrd_y, peak_num)
    assert out.shape == (2, 512)
    assert torch.isfinite(out).all()


def test_spectrum_only_forward() -> None:
    from pxrd_cell_indexing.model.encoder.spectrum_fusion import SpectrumOnlyEncoder

    enc = SpectrumOnlyEncoder(
        {
            "output_dim": 512,
            "histogram_dropout": 0.0,
            "spectrum_bins": 256,
            "spectrum_cnn_channels": (32, 64),
        }
    )
    enc.eval()
    pxrd_x = torch.tensor([[10.4], [20.1], [30.0], [15.2], [25.3]], dtype=torch.float32)
    pxrd_y = torch.tensor([[12.0], [45.0], [80.0], [60.0], [30.0]], dtype=torch.float32)
    peak_num = torch.tensor([3, 2], dtype=torch.long)
    with torch.no_grad():
        out = enc(pxrd_x, pxrd_y, peak_num)
    assert out.shape == (2, 512)
    assert torch.isfinite(out).all()


def test_build_indexing_model_histogram_fusion() -> None:
    model = build_indexing_model(
        checkpoint_path=None,
        encoder_config={
            "encoder_type": "histogram_fusion",
            "histogram_dropout": 0.0,
            "histogram_hidden_dim": 256,
            "histogram_num_blocks": 2,
            "spectrum_bins": 256,
            "spectrum_cnn_channels": (32, 64),
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
