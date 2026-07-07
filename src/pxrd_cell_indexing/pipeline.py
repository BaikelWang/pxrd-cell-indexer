"""Pipeline stubs for PXRD cell indexing.

Step 3 constraint: interfaces only — no business logic until PM confirms skeleton.
"""

from __future__ import annotations

from typing import Protocol

from pxrd_cell_indexing.types import (
    TOP_K_DEFAULT,
    CellParameters,
    IndexingInput,
    IndexingResult,
    LatticeCandidate,
    PXRDPeakTable,
)


class IndexingModel(Protocol):
    """Pluggable cell indexing model backend."""

    def predict(self, sample: IndexingInput) -> IndexingResult:
        """Run indexing for one sample."""
        ...


def preprocess(sample: IndexingInput) -> IndexingInput:
    """Validate and normalize peak-table input.

    TODO(Step 5): implement y>5 filtering, optional max_peaks, and collate helpers.
    """
    return sample


def run_indexing(sample: IndexingInput, model: IndexingModel | None = None) -> IndexingResult:
    """End-to-end indexing entry point (walking skeleton stub).

    TODO(Step 5): wire encoder + heads + Top-K generator.
    """
    normalized = preprocess(sample)
    if model is not None:
        return model.predict(normalized)

    placeholder = __placeholder_candidate()
    return IndexingResult(
        sample_id=normalized.sample_id,
        candidates=[placeholder],
        cell=CellParameters(
            a=placeholder.a,
            b=placeholder.b,
            c=placeholder.c,
            alpha=placeholder.alpha,
            beta=placeholder.beta,
            gamma=placeholder.gamma,
            space_group=None,
        ),
        confidence=placeholder.confidence,
    )


def __placeholder_candidate() -> LatticeCandidate:
    return LatticeCandidate(
        crystal_system="cubic",
        a=1.0,
        b=1.0,
        c=1.0,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        confidence=0.0,
    )


def peak_table_from_input(sample: IndexingInput) -> PXRDPeakTable | None:
    """Return the peak-table contract when present."""
    return sample.peak_table


def top_k_limit() -> int:
    """Configured Top-K size for inference outputs."""
    return TOP_K_DEFAULT
