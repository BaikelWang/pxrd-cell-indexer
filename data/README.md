# Data

## In repository (tracked)

| Path | Description |
|---|---|
| [`MP-100samples-benchmark/`](MP-100samples-benchmark/) | 100 stratified CIF files for MP100 evaluation |
| `processed/*.json` | Small sampling stats manifests (regeneratable metadata) |

## Generated locally (gitignored)

| Path | Size (approx.) | How to regenerate |
|---|---|---|
| `processed/train10k_seed42.jsonl` | ~4 MB | `scripts/investigate_10k_sample.py` |
| `processed/valid1400_seed42.jsonl` | ~560 KB | `scripts/investigate_valid_sample.py` |
| `processed/lattice_stats_seed42.json` | ~1 KB | `scripts/compute_lattice_stats.py` |

## External (read-only, not in this repo)

| Path | Description |
|---|---|
| `alex_aflow_oqmd_mp/datasets/pxrd_241113_train.lmdb` | Main training set (~6M) |
| `alex_aflow_oqmd_mp/datasets/pxrd_241113_valid.lmdb` | Validation set (25,551) |
| `pretrained/weight/2501/pxrd-all/last_one.ckpt` | RealPXRD encoder weights (~145 MB) |

LMDB fields: `pxrd_x`, `pxrd_y`, `p_lattice_matrix`, `p_atom_type`, `p_atom_pos`.

See [`alex_aflow_oqmd_mp/datasets/数据集说明.txt`](../../../alex_aflow_oqmd_mp/datasets/数据集说明.txt).

## Git policy

- **Do not commit**: `.lmdb`, `.jsonl`, checkpoints, large eval dumps
- **Do commit**: benchmark CIFs, small stats json, README
