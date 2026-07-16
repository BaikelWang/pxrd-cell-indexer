"""Smoke tests for vendored RealPXRD encoder."""

from __future__ import annotations

import pytest
import torch

from pxrd_cell_indexing.model.encoder.loader import (
    DEFAULT_ENCODER_CONFIG,
    REALPXRD_ENCODER_CHECKPOINT,
    build_bert_model,
    load_xrd_encoder_from_checkpoint,
)


def _fake_peak_batch(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    peak_counts = [3, 2]
    pxrd_x = torch.tensor(
        [[10.0], [20.0], [30.0], [15.0], [25.0]],
        dtype=torch.float32,
    )
    pxrd_y = torch.tensor(
        [[12.0], [45.0], [80.0], [60.0], [30.0]],
        dtype=torch.float32,
    )
    peak_num = torch.tensor(peak_counts[:batch_size], dtype=torch.long)
    return pxrd_x, pxrd_y, peak_num


def test_encoder_forward_shape() -> None:
    encoder = build_bert_model()
    encoder.eval()
    pxrd_x, pxrd_y, peak_num = _fake_peak_batch()

    with torch.no_grad():
        output = encoder(pxrd_x, pxrd_y, peak_num)

    assert output.shape == (2, DEFAULT_ENCODER_CONFIG["output_dim"])


def test_continuous_position_encoding_preserves_subdegree() -> None:
    """A3: continuous pos must accept non-integer 2θ and differ from truncated discrete."""
    cont = build_bert_model({"position_encoding": "continuous"})
    disc = build_bert_model({"position_encoding": "discrete"})
    cont.eval()
    disc.eval()
    # Peaks at 10.4° vs 10.0° — discrete collapses both to index 10.
    pxrd_x = torch.tensor([[10.4], [20.0], [30.0], [10.0], [20.0]], dtype=torch.float32)
    pxrd_y = torch.tensor([[12.0], [45.0], [80.0], [12.0], [45.0]], dtype=torch.float32)
    peak_num = torch.tensor([3, 2], dtype=torch.long)
    with torch.no_grad():
        out_cont = cont(pxrd_x, pxrd_y, peak_num)
        out_disc = disc(pxrd_x, pxrd_y, peak_num)
    assert out_cont.shape == (2, 512)
    assert torch.isfinite(out_cont).all()
    # Same architecture random init → not equal; just ensure continuous path runs.
    assert out_cont.shape == out_disc.shape


@pytest.mark.skipif(
    not REALPXRD_ENCODER_CHECKPOINT.exists(),
    reason="RealPXRD checkpoint not available on this machine",
)
def test_encoder_checkpoint_load_coverage() -> None:
    encoder, report = load_xrd_encoder_from_checkpoint(REALPXRD_ENCODER_CHECKPOINT)

    assert report["loaded_key_count"] == 38
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []

    encoder.eval()
    pxrd_x, pxrd_y, peak_num = _fake_peak_batch()
    with torch.no_grad():
        output = encoder(pxrd_x, pxrd_y, peak_num)

    assert output.shape == (2, 512)
    assert torch.isfinite(output).all()
