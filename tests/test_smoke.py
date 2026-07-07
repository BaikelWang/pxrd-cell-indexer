"""Smoke tests for project scaffold."""

from pxrd_cell_indexing import run_indexing
from pxrd_cell_indexing.types import IndexingInput, LatticeCandidate, PXRDPeakTable, PXRDProfile


def test_import_package() -> None:
    import pxrd_cell_indexing

    assert pxrd_cell_indexing.__version__ == "0.0.0"


def test_peak_table_instantiation() -> None:
    peak_table = PXRDPeakTable(
        two_theta=[10.0, 20.0],
        intensity=[12.0, 100.0],
        wavelength_angstrom=1.5406,
        peak_num=2,
    )
    assert peak_table.input_axis == "two_theta"
    assert peak_table.peak_num == 2


def test_types_instantiation() -> None:
    profile = PXRDProfile(two_theta=[10.0, 20.0], intensity=[0.1, 1.0], wavelength_angstrom=1.5406)
    sample = IndexingInput(sample_id="smoke-001", profile=profile)
    assert sample.sample_id == "smoke-001"


def test_run_indexing_stub() -> None:
    peak_table = PXRDPeakTable(
        two_theta=[10.0],
        intensity=[100.0],
        wavelength_angstrom=1.5406,
        peak_num=1,
    )
    sample = IndexingInput(sample_id="smoke-002", peak_table=peak_table)
    result = run_indexing(sample)
    assert result.sample_id == "smoke-002"
    assert len(result.candidates) == 1
    assert isinstance(result.candidates[0], LatticeCandidate)
    assert result.cell is not None
    assert result.cell.a > 0
