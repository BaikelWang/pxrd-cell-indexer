"""Tests for MP100 CIF simulation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

from pxrd_cell_indexing.data.mp100 import (
    load_mp100_dataset,
    load_mp100_sample,
    peaks_to_model_tensors,
    simulate_pxrd_from_structure,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MP100_DIR = PROJECT_ROOT / "data" / "MP-100samples-benchmark"


def test_simulate_pxrd_from_structure_filters_intensity() -> None:
    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    two_theta, intensity = simulate_pxrd_from_structure(structure)
    assert two_theta.ndim == 1
    assert intensity.ndim == 1
    assert two_theta.shape[0] == intensity.shape[0]
    assert two_theta.shape[0] > 0
    assert np.all(intensity > 5.0)
    assert np.all(two_theta >= 5.0)
    assert np.all(two_theta <= 80.0)


def test_load_mp100_sample_from_real_cif() -> None:
    cif_files = sorted(MP100_DIR.glob("*.cif"))
    assert len(cif_files) == 100
    sample = load_mp100_sample(cif_files[0])
    assert sample.peak_num > 0
    assert sample.truth_lattice.shape == (6,)
    assert sample.truth_lattice[0] > 0
    pxrd_x, pxrd_y, peak_num = peaks_to_model_tensors(sample.two_theta, sample.intensity)
    assert pxrd_x.shape == (peak_num, 1)
    assert pxrd_y.shape == (peak_num, 1)


def test_load_mp100_dataset_count() -> None:
    samples = load_mp100_dataset(MP100_DIR)
    assert len(samples) == 100
