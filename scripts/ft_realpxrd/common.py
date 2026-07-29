"""Shared data / lattice / checkpoint helpers for RealPXRD FT arms."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT = Path(__file__).resolve().parents[2]
REALPXRD = Path("/nanolab/users/wyx/archive/RealPXRD-Solver")
CKPT = REALPXRD / "pretrained/weight/2501/pxrd-all/last_one.ckpt"
TRAIN_LMDB = Path("/nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_train.lmdb")
VALID_LMDB = Path("/nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_valid.lmdb")
MP100_DIR = PROJECT / "data/MP-100samples-benchmark"
SUBSET_JSON = PROJECT / "results/ft_subset_10k_seed42.json"

sys.path.insert(0, str(REALPXRD))
sys.path.insert(0, str(PROJECT / "scripts"))

from app.data.dataset import augment_spectrum  # noqa: E402
from app.data.utils import MultiLMDBDataset  # noqa: E402
from pymatgen.core import Lattice  # noqa: E402


def matrix_to_six(mat) -> list[float]:
    lat = Lattice(np.asarray(mat, dtype=float).reshape(3, 3))
    return [float(lat.a), float(lat.b), float(lat.c), float(lat.alpha), float(lat.beta), float(lat.gamma)]


def six_to_matrix(six) -> np.ndarray:
    a, b, c, al, be, ga = [float(x) for x in six]
    return Lattice.from_parameters(a, b, c, al, be, ga).matrix.astype(np.float32)


def volume_from_six(six) -> float:
    return float(Lattice.from_parameters(*[float(x) for x in six]).volume)


class LatticeNormalizer:
    """log lengths + angle/90; fitted on train six-params."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = None if std is None else np.asarray(std, dtype=np.float64)

    @staticmethod
    def encode_raw(six) -> np.ndarray:
        a, b, c, al, be, ga = [float(x) for x in six]
        return np.array(
            [math.log(max(a, 1e-3)), math.log(max(b, 1e-3)), math.log(max(c, 1e-3)), al / 90.0, be / 90.0, ga / 90.0],
            dtype=np.float64,
        )

    def fit(self, sixes: list[list[float]]) -> "LatticeNormalizer":
        arr = np.stack([self.encode_raw(s) for s in sixes], axis=0)
        self.mean = arr.mean(axis=0)
        self.std = arr.std(axis=0)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def encode(self, six) -> np.ndarray:
        assert self.mean is not None
        return ((self.encode_raw(six) - self.mean) / self.std).astype(np.float32)

    def decode(self, z) -> list[float]:
        assert self.mean is not None
        z = np.asarray(z, dtype=np.float64)
        r = z * self.std + self.mean
        return [
            float(math.exp(r[0])),
            float(math.exp(r[1])),
            float(math.exp(r[2])),
            float(r[3] * 90.0),
            float(r[4] * 90.0),
            float(r[5] * 90.0),
        ]

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "LatticeNormalizer":
        return cls(mean=d["mean"], std=d["std"])


def build_or_load_subset(seed: int = 42, n_train: int = 10000, n_valid: int = 1000) -> dict:
    if SUBSET_JSON.exists():
        return json.loads(SUBSET_JSON.read_text())
    rng = np.random.RandomState(seed)
    train_ds = MultiLMDBDataset(str(TRAIN_LMDB), key_to_id=False, gzip=True)
    valid_ds = MultiLMDBDataset(str(VALID_LMDB), key_to_id=False, gzip=True)
    train_idx = sorted(rng.choice(len(train_ds), size=n_train, replace=False).tolist())
    valid_idx = sorted(rng.choice(len(valid_ds), size=n_valid, replace=False).tolist())
    out = {
        "seed": seed,
        "n_train": n_train,
        "n_valid": n_valid,
        "train_lmdb": str(TRAIN_LMDB),
        "valid_lmdb": str(VALID_LMDB),
        "train_indices": train_idx,
        "valid_indices": valid_idx,
    }
    SUBSET_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUBSET_JSON.write_text(json.dumps(out, indent=2))
    return out


def _load_peaks(data_dict, xrd_augment: bool):
    pxrd_x = np.asarray(data_dict["pxrd_x"], dtype=np.float32)
    pxrd_y = np.asarray(data_dict["pxrd_y"], dtype=np.float32)
    mask = pxrd_y > 5
    pxrd_x, pxrd_y = pxrd_x[mask], pxrd_y[mask]
    if len(pxrd_x) == 0:
        # fallback: keep strongest peak
        i = int(np.argmax(np.asarray(data_dict["pxrd_y"])))
        pxrd_x = np.asarray([data_dict["pxrd_x"][i]], dtype=np.float32)
        pxrd_y = np.asarray([max(float(data_dict["pxrd_y"][i]), 1.0)], dtype=np.float32)
    if xrd_augment:
        pxrd_x, pxrd_y = augment_spectrum(pxrd_x, pxrd_y)
        mask = pxrd_y > 5
        if mask.any():
            pxrd_x, pxrd_y = pxrd_x[mask].astype(np.float32), pxrd_y[mask].astype(np.float32)
    return pxrd_x.reshape(-1, 1), pxrd_y.reshape(-1, 1)


class PeakLatticeSubset(Dataset):
    """LMDB subset → peaks + primitive lattice six-params (+ matrix for arm A)."""

    def __init__(
        self,
        lmdb_path: str | Path,
        indices: list[int],
        *,
        xrd_augment: bool,
        normalizer: LatticeNormalizer | None = None,
        need_matrix: bool = False,
    ):
        self.ds = MultiLMDBDataset(str(lmdb_path), key_to_id=False, gzip=True)
        self.indices = list(indices)
        self.xrd_augment = xrd_augment
        self.normalizer = normalizer
        self.need_matrix = need_matrix

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        data_dict, _ = self.ds[self.indices[i]]
        px, py = _load_peaks(data_dict, self.xrd_augment)
        six = matrix_to_six(data_dict["p_lattice_matrix"])
        item = {
            "pxrd_x": torch.from_numpy(px),
            "pxrd_y": torch.from_numpy(py),
            "peak_num": int(len(px)),
            "lattice_six": torch.tensor(six, dtype=torch.float32),
            "index": int(self.indices[i]),
        }
        if self.normalizer is not None:
            item["lattice_norm"] = torch.from_numpy(self.normalizer.encode(six))
        if self.need_matrix:
            item["lattice_matrix"] = torch.from_numpy(
                np.asarray(data_dict["p_lattice_matrix"], dtype=np.float32).reshape(3, 3)
            )
        return item


def collate_peaks(batch: list[dict]) -> dict:
    """Flatten peaks like RealPXRD BertModel expects; pad-free concat."""
    xs, ys, nums = [], [], []
    for b in batch:
        xs.append(b["pxrd_x"])
        ys.append(b["pxrd_y"])
        nums.append(b["peak_num"])
    out = {
        "pxrd_x": torch.cat(xs, dim=0),
        "pxrd_y": torch.cat(ys, dim=0),
        "peak_num": torch.tensor(nums, dtype=torch.long),
        "lattice_six": torch.stack([b["lattice_six"] for b in batch], dim=0),
    }
    if "lattice_norm" in batch[0]:
        out["lattice_norm"] = torch.stack([b["lattice_norm"] for b in batch], dim=0)
    if "lattice_matrix" in batch[0]:
        out["lattice_matrix"] = torch.stack([b["lattice_matrix"] for b in batch], dim=0)
    return out


def fit_normalizer(lmdb_path: Path, indices: list[int], max_fit: int = 10000) -> LatticeNormalizer:
    ds = MultiLMDBDataset(str(lmdb_path), key_to_id=False, gzip=True)
    sixes = []
    for idx in indices[:max_fit]:
        d, _ = ds[idx]
        sixes.append(matrix_to_six(d["p_lattice_matrix"]))
    return LatticeNormalizer().fit(sixes)


def load_bert_from_ckpt(device: torch.device, continuous_pos: bool = False):
    """Build BertModel from pxrd-all checkpoint. Optionally swap to Fourier pos."""
    from app.model.bert import BertModel

    from .models import BertFourierPos

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]["xrd_encoder"]
    kw = {k: v for k, v in hp.items() if k != "_target_"}
    if continuous_pos:
        model = BertFourierPos(**kw)
    else:
        model = BertModel(**kw)
    sd = {
        k[len("xrd_encoder.") :]: v
        for k, v in ck["state_dict"].items()
        if k.startswith("xrd_encoder.")
    }
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Fourier arm expects missing embed_positions
    return model.to(device), missing, unexpected


def load_cspflow_from_ckpt(
    device: torch.device,
    continuous_pos: bool = False,
    encoder_override: dict | None = None,
):
    """Instantiate CSPFlow + load full weights (for arm A).

    ``continuous_pos`` swaps the integer-degree ``embed_positions`` lookup for a
    continuous Fourier(sin^2 theta) encoding, so sub-degree peak shifts survive
    the encoder. ``pos_proj`` is then randomly initialized and must be trained.

    ``encoder_override`` resizes the XRD encoder (e.g. wider/deeper). Every
    pretrained tensor then has an incompatible shape, so encoder weights are
    **not** loaded and the encoder trains from scratch -- this is no longer
    transfer learning, it is a scratch control that reuses the architecture.
    """
    import hydra
    from omegaconf import OmegaConf

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    # Build via hydra instantiate using saved hparams
    # CSPFlow.__init__ expects nested hparams as attributes via Lightning
    from app.model.flow import CSPFlow

    # Reconstruct a minimal namespace-like object
    class NS:
        pass

    hp = ck["hyper_parameters"]
    # CSPFlow uses self.hparams from Lightning save_hyperparameters
    # Easiest path: construct modules manually
    from app.model.bert import BertModel
    from app.model.cspnet_xrd import CSPNet
    from app.model.flow import SinusoidalTimeEmbeddings

    xrd_kw = {k: v for k, v in hp["xrd_encoder"].items() if k != "_target_"}
    dec_kw = {k: v for k, v in hp["decoder"].items() if k != "_target_"}
    # CSPNet latent_dim in ctor is hidden+time via flow: atom_latent_emb = Linear(hidden + latent_dim)
    # In flow.__init__: decoder = instantiate(..., latent_dim = latent_dim + time_dim)
    # So decoder.latent_dim in yaml is 512 but actual Linear is 512+256=768
    decoder = CSPNet(
        hidden_dim=dec_kw["hidden_dim"],
        latent_dim=hp["latent_dim"] + hp["time_dim"],
        max_atoms=dec_kw.get("max_atoms", 100),
        num_layers=dec_kw["num_layers"],
        act_fn=dec_kw.get("act_fn", "silu"),
        dis_emb=dec_kw.get("dis_emb", "sin"),
        num_freqs=dec_kw.get("num_freqs", 128),
        ln=dec_kw.get("ln", True),
        ip=dec_kw.get("ip", True),
    )
    if encoder_override:
        xrd_kw.update(encoder_override)

    if continuous_pos:
        from .models import BertFourierPos

        encoder = BertFourierPos(**xrd_kw)
    else:
        encoder = BertModel(**xrd_kw)

    class Bundle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.xrd_encoder = encoder
            self.decoder = decoder
            self.time_embedding = SinusoidalTimeEmbeddings(hp["time_dim"])
            self.timesteps = hp["timesteps"]
            self.time_dim = hp["time_dim"]
            self.latent_dim = hp["latent_dim"]

    model = Bundle()
    sd = ck["state_dict"]
    # filter optim keys; Lightning prefixes with nothing for modules
    prefixes = ("decoder.",) if encoder_override else ("xrd_encoder.", "decoder.")
    filtered = {k: v for k, v in sd.items() if k.startswith(prefixes)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return model.to(device), hp, missing, unexpected
