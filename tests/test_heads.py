"""Tests for model heads and IndexingModel."""

from __future__ import annotations

import torch

from pxrd_cell_indexing.data.normalization import MatrixLatticeNormalizer
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
    assert outputs["lattice_norm"].shape == (2, 6)
    assert outputs["embedding"].shape == (2, 512)
    assert "crystal_system_logits" not in outputs


def test_indexing_model_forward_shapes_matrix9_output_dim() -> None:
    encoder = build_bert_model()
    model = IndexingModel(encoder, head_config=HeadConfig(output_dim=9))
    model.eval()
    pxrd_x, pxrd_y, peak_num = _fake_batch()
    with torch.no_grad():
        outputs = model(pxrd_x, pxrd_y, peak_num)
    assert outputs["lattice_norm"].shape == (2, 9)


def test_freeze_encoder_disables_grad() -> None:
    encoder = build_bert_model()
    model = IndexingModel(encoder, freeze_encoder=True)
    assert model.encoder_frozen
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.head_parameters())


def test_cs_conditional_oracle_routing() -> None:
    encoder = build_bert_model()
    model = IndexingModel(
        encoder,
        head_config=HeadConfig(head_type="cs_conditional", use_cs_classifier=False),
        normalize_embedding=False,
    )
    model.eval()
    pxrd_x, pxrd_y, peak_num = _fake_batch()
    cs = torch.tensor([0, 3], dtype=torch.long)
    with torch.no_grad():
        outputs = model(pxrd_x, pxrd_y, peak_num, crystal_system_idx=cs, cs_route="oracle")
    assert outputs["lattice_norm"].shape == (2, 6)
    assert outputs["lattice_norm_all"].shape == (2, 7, 6)
    assert torch.equal(outputs["routed_cs_idx"], cs)


def test_cs_conditional_predicted_routing() -> None:
    encoder = build_bert_model()
    model = IndexingModel(
        encoder,
        head_config=HeadConfig(
            head_type="cs_conditional",
            use_cs_classifier=True,
            default_cs_route="predicted",
        ),
        normalize_embedding=False,
    )
    model.eval()
    pxrd_x, pxrd_y, peak_num = _fake_batch()
    with torch.no_grad():
        outputs = model(pxrd_x, pxrd_y, peak_num, cs_route="predicted")
    assert outputs["crystal_system_logits"].shape == (2, 7)
    assert outputs["routed_cs_idx"].shape == (2,)
    assert outputs["lattice_norm"].shape == (2, 6)


def test_cubic_bravais_split_oracle_setting() -> None:
    encoder = build_bert_model()
    model = IndexingModel(
        encoder,
        head_config=HeadConfig(
            head_type="cs_conditional",
            cubic_bravais_split=True,
            use_cs_classifier=False,
        ),
        normalize_embedding=False,
    )
    model.eval()
    pxrd_x, pxrd_y, peak_num = _fake_batch()
    cs = torch.tensor([0, 0], dtype=torch.long)
    # F-like 60° and P-like 90°
    phys = torch.tensor(
        [
            [4.0, 4.0, 4.0, 60.0, 60.0, 60.0],
            [4.0, 4.0, 4.0, 90.0, 90.0, 90.0],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        outputs = model(
            pxrd_x,
            pxrd_y,
            peak_num,
            crystal_system_idx=cs,
            cs_route="oracle",
            lattice_phys=phys,
            setting_route="oracle",
        )
    assert outputs["cubic_setting_norm_all"].shape == (2, 3, 6)
    assert outputs["routed_cubic_setting_idx"].tolist() == [1, 0]
    assert outputs["lattice_norm"].shape == (2, 6)


def test_cubic_setting_classifier_route() -> None:
    encoder = build_bert_model()
    model = IndexingModel(
        encoder,
        head_config=HeadConfig(
            head_type="cs_conditional",
            cubic_bravais_split=True,
            use_cs_classifier=True,
            use_cubic_setting_classifier=True,
            default_setting_route="classifier",
        ),
        normalize_embedding=False,
    )
    model.eval()
    pxrd_x, pxrd_y, peak_num = _fake_batch()
    cs = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        outputs = model(
            pxrd_x,
            pxrd_y,
            peak_num,
            crystal_system_idx=cs,
            cs_route="oracle",
            setting_route="classifier",
        )
    assert outputs["cubic_setting_logits"].shape == (2, 3)
    assert "routed_cubic_setting_idx" in outputs
    # Non-cubic sample keeps CS head; cubic uses setting head.
    assert outputs["lattice_norm"].shape == (2, 6)


def test_multi_hypothesis_shapes_and_cubic_unaffected() -> None:
    encoder = build_bert_model()
    model = IndexingModel(
        encoder,
        head_config=HeadConfig(
            head_type="cs_conditional",
            multi_hypothesis=True,
            num_hypotheses=3,
        ),
        normalize_embedding=False,
    )
    model.eval()
    pxrd_x, pxrd_y, peak_num = _fake_batch()
    cs = torch.tensor([0, 3], dtype=torch.long)  # cubic, hexagonal
    with torch.no_grad():
        outputs = model(pxrd_x, pxrd_y, peak_num, crystal_system_idx=cs, cs_route="oracle")
    assert outputs["lattice_norm"].shape == (2, 6)
    assert outputs["lattice_norm_all"].shape == (2, 7, 6)
    assert outputs["lattice_hyp_all"].shape == (2, 7, 3, 6)
    assert outputs["lattice_hyp_selected"].shape == (2, 3, 6)
    # Cubic (K=1) is padded by repeating hyp-0: all 3 slots identical.
    cubic_hyp = outputs["lattice_hyp_all"][0, 0]
    assert torch.allclose(cubic_hyp[0], cubic_hyp[1])
    assert torch.allclose(cubic_hyp[0], cubic_hyp[2])
    # Hexagonal (non-cubic) head should generally produce distinct hypotheses
    # (random init, no training) -- not required to be identical.
    hex_hyp = outputs["lattice_hyp_all"][1, 3]
    assert hex_hyp.shape == (3, 6)


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


def test_shared_head_optional_cs_classifier() -> None:
    encoder = build_bert_model()
    model = IndexingModel(
        encoder,
        head_config=HeadConfig(
            head_type="shared",
            use_cs_classifier=True,
            head_num_layers=4,
            hidden_dim=128,
        ),
        normalize_embedding=False,
    )
    model.eval()
    pxrd_x, pxrd_y, peak_num = _fake_batch()
    with torch.no_grad():
        outputs = model(pxrd_x, pxrd_y, peak_num)
    assert outputs["lattice_norm"].shape == (2, 6)
    assert outputs["crystal_system_logits"].shape == (2, 7)


def test_film_head_oracle_and_predicted() -> None:
    encoder = build_bert_model()
    model = IndexingModel(
        encoder,
        head_config=HeadConfig(
            head_type="film",
            use_cs_classifier=True,
            head_num_layers=4,
            hidden_dim=128,
            default_cs_route="oracle",
        ),
        normalize_embedding=False,
    )
    model.eval()
    pxrd_x, pxrd_y, peak_num = _fake_batch()
    cs = torch.tensor([0, 3], dtype=torch.long)
    with torch.no_grad():
        out_oracle = model(
            pxrd_x, pxrd_y, peak_num, crystal_system_idx=cs, cs_route="oracle"
        )
        out_pred = model(pxrd_x, pxrd_y, peak_num, cs_route="predicted")
    assert out_oracle["lattice_norm"].shape == (2, 6)
    assert torch.equal(out_oracle["routed_cs_idx"], cs)
    assert out_pred["lattice_norm"].shape == (2, 6)
    assert out_pred["crystal_system_logits"].shape == (2, 7)
    assert out_pred["routed_cs_idx"].shape == (2,)


def test_cubic_exact_head_ties_lengths() -> None:
    encoder = build_bert_model()
    model = IndexingModel(
        encoder,
        head_config=HeadConfig(
            head_type="film",
            use_cs_classifier=True,
            head_num_layers=2,
            hidden_dim=64,
            default_cs_route="oracle",
            default_setting_route="oracle",
            cubic_exact=True,
        ),
        normalize_embedding=False,
    )
    # Identity normalizer so decoded phys == constructed matrix6 exactly.
    norm = MatrixLatticeNormalizer(
        component_mean=(0.0,) * 6, component_std=(1.0,) * 6
    )
    model.set_normalizer(norm)
    model.eval()
    pxrd_x, pxrd_y, peak_num = _fake_batch()
    cs = torch.tensor([0, 3], dtype=torch.long)  # one cubic, one hexagonal
    lat = torch.tensor(
        [[4.0, 4.0, 4.0, 90.0, 90.0, 90.0], [3.0, 3.0, 5.0, 90.0, 90.0, 120.0]]
    )
    with torch.no_grad():
        out = model(
            pxrd_x,
            pxrd_y,
            peak_num,
            crystal_system_idx=cs,
            cs_route="oracle",
            lattice_phys=lat,
            setting_route="oracle",
        )
    assert out["cubic_setting_logits"].shape == (2, 3)
    phys = norm.denormalize(out["lattice_norm"])
    a, b, c, al, be, ga = phys[0].tolist()
    assert abs(a - b) < 1e-3 and abs(b - c) < 1e-3  # cubic a=b=c exact
    assert abs(al - 90.0) < 1e-2 and abs(ga - 90.0) < 1e-2  # oracle setting angle
