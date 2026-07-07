"""Data contracts for PXRD cell indexing."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

InputAxis = Literal["two_theta", "d_spacing"]
CRYSTAL_SYSTEMS = (
    "cubic",
    "tetragonal",
    "orthorhombic",
    "hexagonal",
    "trigonal",
    "monoclinic",
    "triclinic",
)
CRYSTAL_SYSTEM_TO_IDX: dict[str, int] = {
    name: idx for idx, name in enumerate(CRYSTAL_SYSTEMS)
}
TOP_K_DEFAULT = 20


class PXRDPeakTable(BaseModel):
    """Variable-length peak table aligned with RealPXRD BertModel input."""

    two_theta: list[float] = Field(..., description="2θ angles in degrees (baseline axis)")
    intensity: list[float] = Field(..., description="Peak intensities (0-100 scale)")
    wavelength_angstrom: float = Field(..., gt=0, description="X-ray wavelength metadata (Å)")
    peak_num: int = Field(..., ge=0, description="Number of peaks; equals len(two_theta)")
    input_axis: InputAxis = Field(
        default="two_theta",
        description="Position axis for encoder; d_spacing reserved for ablation",
    )


class PXRDProfile(BaseModel):
    """Legacy normalized PXRD intensity profile (kept for backward-compatible stubs)."""

    two_theta: list[float] = Field(..., description="2θ angles in degrees")
    intensity: list[float] = Field(..., description="Normalized intensity values")
    wavelength_angstrom: float = Field(..., gt=0, description="X-ray wavelength (Å)")


class CellParameters(BaseModel):
    """Unit cell parameters."""

    a: float = Field(..., gt=0)
    b: float = Field(..., gt=0)
    c: float = Field(..., gt=0)
    alpha: float = Field(..., gt=0, lt=180)
    beta: float = Field(..., gt=0, lt=180)
    gamma: float = Field(..., gt=0, lt=180)
    space_group: str | None = Field(None, description="International space group symbol")


class LatticeCandidate(BaseModel):
    """One Top-K lattice indexing candidate."""

    crystal_system: str = Field(..., description="One of the seven Bravais lattice families")
    a: float = Field(..., gt=0)
    b: float = Field(..., gt=0)
    c: float = Field(..., gt=0)
    alpha: float = Field(..., gt=0, lt=180)
    beta: float = Field(..., gt=0, lt=180)
    gamma: float = Field(..., gt=0, lt=180)
    confidence: float = Field(..., ge=0, description="Ranking confidence for this candidate")


class IndexingInput(BaseModel):
    """Model input for a single indexing request."""

    sample_id: str
    peak_table: PXRDPeakTable | None = None
    profile: PXRDProfile | None = None
    # TODO(Step 5): add optional metadata, multi-phase flags per requirements.


class IndexingResult(BaseModel):
    """Model output for a single indexing request."""

    sample_id: str
    candidates: list[LatticeCandidate] = Field(
        default_factory=list,
        description="Top-K lattice candidates sorted by confidence (D19 K=20)",
    )
    cell: CellParameters | None = Field(
        None,
        description="Deprecated Top-1 shortcut; prefer candidates[0] after M1.5",
    )
    confidence: float | None = Field(None, ge=0, le=1)


class EvaluationMetrics(BaseModel):
    """Aggregate evaluation metrics."""

    n_samples: int
    cell_param_mae: float | None = None
    space_group_accuracy: float | None = None
    crystal_system_accuracy: float | None = None
    lattice_match_rate: float | None = None
    top_k_recall: float | None = None
    # TODO(Step 5): align metrics with docs/00-requirements.md acceptance criteria.
