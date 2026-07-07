"""MP100 benchmark helpers: CIF → simulated peaks + primitive truth lattice."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pxrd_cell_indexing.data.dataset import filter_peaks
from pxrd_cell_indexing.types import CRYSTAL_SYSTEM_TO_IDX

DEFAULT_WAVELENGTH_ANGSTROM = 1.54184  # Cu Kα, pymatgen XRDCalculator default
DEFAULT_TWO_THETA_MIN = 5.0
DEFAULT_TWO_THETA_MAX = 80.0
DEFAULT_INTENSITY_MIN = 5.0
DEFAULT_SYMPREC = 0.01


@dataclass(frozen=True)
class MP100Sample:
    sample_id: str
    cif_path: Path
    two_theta: np.ndarray
    intensity: np.ndarray
    peak_num: int
    truth_lattice: np.ndarray  # [a,b,c,alpha,beta,gamma] primitive
    crystal_system: str
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM


def simulate_pxrd_from_structure(
    structure: Structure,
    *,
    two_theta_min: float = DEFAULT_TWO_THETA_MIN,
    two_theta_max: float = DEFAULT_TWO_THETA_MAX,
    intensity_min: float = DEFAULT_INTENSITY_MIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate PXRD using the same pipeline as ``241113_save_pxrd_data.py``."""
    analyzer = SpacegroupAnalyzer(structure, symprec=DEFAULT_SYMPREC)
    conventional = analyzer.get_conventional_standard_structure()
    reduced = conventional.get_reduced_structure()
    pattern = XRDCalculator().get_pattern(
        reduced,
        scaled=True,
        two_theta_range=(two_theta_min, two_theta_max),
    )
    two_theta = np.asarray(pattern.x, dtype=np.float32)
    intensity = np.asarray(pattern.y, dtype=np.float32)
    return filter_peaks(two_theta, intensity, intensity_min=intensity_min)


def primitive_lattice_params_from_structure(
    structure: Structure,
    *,
    symprec: float = DEFAULT_SYMPREC,
) -> np.ndarray:
    """Extract primitive six-parameter lattice labels (D1)."""
    primitive = SpacegroupAnalyzer(structure, symprec=symprec).find_primitive()
    lattice = primitive.lattice
    return np.array(
        [lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma],
        dtype=np.float32,
    )


def load_mp100_sample(
    cif_path: str | Path,
    *,
    two_theta_min: float = DEFAULT_TWO_THETA_MIN,
    two_theta_max: float = DEFAULT_TWO_THETA_MAX,
    intensity_min: float = DEFAULT_INTENSITY_MIN,
    symprec: float = DEFAULT_SYMPREC,
) -> MP100Sample:
    """Load one MP100 CIF and produce peaks + primitive truth lattice."""
    path = Path(cif_path)
    structure = Structure.from_file(path)
    two_theta, intensity = simulate_pxrd_from_structure(
        structure,
        two_theta_min=two_theta_min,
        two_theta_max=two_theta_max,
        intensity_min=intensity_min,
    )
    truth = primitive_lattice_params_from_structure(structure, symprec=symprec)
    crystal_system = SpacegroupAnalyzer(structure, symprec=symprec).get_crystal_system()
    if crystal_system not in CRYSTAL_SYSTEM_TO_IDX:
        raise ValueError(f"Unsupported crystal system from CIF {path}: {crystal_system}")
    return MP100Sample(
        sample_id=path.stem,
        cif_path=path,
        two_theta=two_theta,
        intensity=intensity,
        peak_num=int(two_theta.shape[0]),
        truth_lattice=truth,
        crystal_system=crystal_system,
    )


def load_mp100_dataset(
    cif_dir: str | Path,
    *,
    two_theta_min: float = DEFAULT_TWO_THETA_MIN,
    two_theta_max: float = DEFAULT_TWO_THETA_MAX,
    intensity_min: float = DEFAULT_INTENSITY_MIN,
    symprec: float = DEFAULT_SYMPREC,
) -> list[MP100Sample]:
    """Load all ``*.cif`` files under ``cif_dir`` sorted by filename."""
    directory = Path(cif_dir)
    samples: list[MP100Sample] = []
    for cif_path in sorted(directory.glob("*.cif")):
        samples.append(
            load_mp100_sample(
                cif_path,
                two_theta_min=two_theta_min,
                two_theta_max=two_theta_max,
                intensity_min=intensity_min,
                symprec=symprec,
            )
        )
    return samples


def peaks_to_model_tensors(
    two_theta: np.ndarray,
    intensity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Convert peak arrays to BertModel-compatible flattened tensors."""
    two_theta = np.asarray(two_theta, dtype=np.float32).reshape(-1, 1)
    intensity = np.asarray(intensity, dtype=np.float32).reshape(-1, 1)
    peak_num = int(two_theta.shape[0])
    return two_theta, intensity, peak_num
