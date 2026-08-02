"""MP100 benchmark helpers: CIF → simulated peaks + truth lattice."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.random import Generator
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pxrd_cell_indexing.data.canonical import CanonicalConvention, canonicalize_lattice
from pxrd_cell_indexing.data.dataset import SpectrumAugmentConfig, augment_spectrum, filter_peaks
from pxrd_cell_indexing.types import CRYSTAL_SYSTEM_TO_IDX

DEFAULT_WAVELENGTH_ANGSTROM = 1.54184  # Cu Kα, pymatgen XRDCalculator default
DEFAULT_TWO_THETA_MIN = 5.0
DEFAULT_TWO_THETA_MAX = 80.0
DEFAULT_INTENSITY_MIN = 5.0
DEFAULT_SYMPREC = 0.01
DEFAULT_MP100_AUGMENT_SEED = 42


@dataclass(frozen=True)
class MP100Sample:
    sample_id: str
    cif_path: Path
    two_theta: np.ndarray
    intensity: np.ndarray
    peak_num: int
    truth_lattice: np.ndarray  # [a,b,c,alpha,beta,gamma] under ``convention``
    crystal_system: str
    convention: CanonicalConvention = "primitive"
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
    """Extract primitive six-parameter lattice labels (legacy D1)."""
    return truth_lattice_params_from_structure(
        structure, convention="primitive", symprec=symprec
    )


def truth_lattice_params_from_structure(
    structure: Structure,
    *,
    convention: CanonicalConvention = "niggli",
    symprec: float = DEFAULT_SYMPREC,
) -> np.ndarray:
    """Extract six-parameter truth lattice under a crystallographic convention.

    Always starts from the primitive cell (same as historical MP100 labels),
    then applies ``canonicalize_lattice`` for ``reduced`` / ``niggli``.
    """
    primitive = SpacegroupAnalyzer(structure, symprec=symprec).find_primitive()
    if convention == "primitive":
        lattice = primitive.lattice
        return np.array(
            [lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma],
            dtype=np.float32,
        )
    can = canonicalize_lattice(primitive.lattice.matrix, convention=convention)
    return can.as_array().astype(np.float32)


def _rng_for_sample(base_seed: int, sample_id: str) -> Generator:
    """Deterministic per-sample RNG (order-independent across dataset shuffles)."""
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(seed)


def apply_realpxrd_xrd_augment(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    rng: Generator,
    config: SpectrumAugmentConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Online RealPXRD ``augment_spectrum`` on already I>5-filtered peaks."""
    cfg = config or SpectrumAugmentConfig()
    tt = np.asarray(two_theta, dtype=np.float64)
    ii = np.asarray(intensity, dtype=np.float64)
    return augment_spectrum(tt, ii, config=cfg, rng=rng, pre_filter_intensity=ii)


def load_mp100_sample(
    cif_path: str | Path,
    *,
    two_theta_min: float = DEFAULT_TWO_THETA_MIN,
    two_theta_max: float = DEFAULT_TWO_THETA_MAX,
    intensity_min: float = DEFAULT_INTENSITY_MIN,
    symprec: float = DEFAULT_SYMPREC,
    convention: CanonicalConvention = "primitive",
    xrd_augment: bool = False,
    augment_seed: int = DEFAULT_MP100_AUGMENT_SEED,
    augment_config: SpectrumAugmentConfig | None = None,
) -> MP100Sample:
    """Load one MP100 CIF and produce peaks + truth lattice.

    Default ``convention='primitive'`` preserves historical MP100 numbers.
    For R9+ canonical models, pass ``convention='niggli'``.

    When ``xrd_augment=True``, apply RealPXRD-style online noise/shift/scale
    (same defaults as training ``SpectrumAugmentConfig``). Ideal peaks remain
    the default so historical MP100 numbers stay comparable.
    """
    path = Path(cif_path)
    structure = Structure.from_file(path)
    two_theta, intensity = simulate_pxrd_from_structure(
        structure,
        two_theta_min=two_theta_min,
        two_theta_max=two_theta_max,
        intensity_min=intensity_min,
    )
    if xrd_augment:
        rng = _rng_for_sample(augment_seed, path.stem)
        two_theta, intensity = apply_realpxrd_xrd_augment(
            two_theta, intensity, rng=rng, config=augment_config
        )
    truth = truth_lattice_params_from_structure(
        structure, convention=convention, symprec=symprec
    )
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
        convention=convention,
    )


def load_mp100_dataset(
    cif_dir: str | Path,
    *,
    two_theta_min: float = DEFAULT_TWO_THETA_MIN,
    two_theta_max: float = DEFAULT_TWO_THETA_MAX,
    intensity_min: float = DEFAULT_INTENSITY_MIN,
    symprec: float = DEFAULT_SYMPREC,
    convention: CanonicalConvention = "primitive",
    xrd_augment: bool = False,
    augment_seed: int = DEFAULT_MP100_AUGMENT_SEED,
    augment_config: SpectrumAugmentConfig | None = None,
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
                convention=convention,
                xrd_augment=xrd_augment,
                augment_seed=augment_seed,
                augment_config=augment_config,
            )
        )
    return samples


def with_realpxrd_xrd_augment(
    sample: MP100Sample,
    *,
    augment_seed: int = DEFAULT_MP100_AUGMENT_SEED,
    augment_config: SpectrumAugmentConfig | None = None,
) -> MP100Sample:
    """Return a copy of ``sample`` with RealPXRD online augment applied to peaks."""
    rng = _rng_for_sample(augment_seed, sample.sample_id)
    two_theta, intensity = apply_realpxrd_xrd_augment(
        sample.two_theta, sample.intensity, rng=rng, config=augment_config
    )
    return replace(
        sample,
        two_theta=two_theta,
        intensity=intensity,
        peak_num=int(two_theta.shape[0]),
    )


def peaks_to_model_tensors(
    two_theta: np.ndarray,
    intensity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Convert peak arrays to BertModel-compatible flattened tensors."""
    two_theta = np.asarray(two_theta, dtype=np.float32).reshape(-1, 1)
    intensity = np.asarray(intensity, dtype=np.float32).reshape(-1, 1)
    peak_num = int(two_theta.shape[0])
    return two_theta, intensity, peak_num
