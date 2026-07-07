"""Tests for model heads and IndexingModel."""

from __future__ import annotations

import torch

from pxrd_cell_indexing.model.encoder.loader import build_bert_model
from pxrd_cell_indexing.model.heads import HeadConfig, IndexingModel


def _fake_batch(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pxrd_x = torch.tensor([[10.0], [20.0], [30.0], [15.0], [25.0]], dtype=torch.float32)
    pxrd_y = torch.tensor([[12.0], [45.0], [80.0], [60.0], [30.0]], dtype=torch.float32)
    peak_num = torch.tensor([3, 2][:batch_size], dtype=torch.long)
    return pxrd_x, pxrd_y, peak_num


def test_indexing_model_forward_shapes() -> None:
    encoder = build_bert_model()
    model = IndexingModel(encoder, head_config=HeadConfig())
    model.eval()
    pxrd_x, pxrd_y, peak_num = _fake_batch()
    with torch.no_grad():
        outputs = model(pxrd_x, pxrd_y, peak_num)
    assert outputs["crystal_system_logits"].shape == (2, 7)
    assert outputs["lattice_norm"].shape == (2, 6)
    assert outputs["embedding"].shape == (2, 512)


def test_freeze_encoder_disables_grad() -> None:
    encoder = build_bert_model()
    model = IndexingModel(encoder, freeze_encoder=True)
    assert model.encoder_frozen
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.crystal_system_head.parameters())
    assert any(p.requires_grad for p in model.lattice_head.parameters())


def test_normalize_embedding_flag_controls_l2_norm() -> None:
    encoder = build_bert_model()
    model = IndexingModel(encoder, normalize_embedding=False)
    model.eval()
    pxrd_x, pxrd_y, peak_num = _fake_batch(batch_size=1)
    with torch.no_grad():
        embedding = model.encode(pxrd_x, pxrd_y, peak_num)
    norm = torch.linalg.vector_norm(embedding, dim=-1)
    assert not torch.allclose(norm, torch.ones_like(norm), atol=1e-3)

    model_norm = IndexingModel(encoder, normalize_embedding=True)
    model_norm.eval()
    with torch.no_grad():
        embedding_norm = model_norm.encode(pxrd_x, pxrd_y, peak_num)
    norm_on = torch.linalg.vector_norm(embedding_norm, dim=-1)
    assert torch.allclose(norm_on, torch.ones_like(norm_on), atol=1e-5)
