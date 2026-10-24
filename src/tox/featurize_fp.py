"""A1 - Morgan fingerprints + RDKit 2D descriptors, computed once and cached
row-aligned with the clean parquet (row i of each .npy == row i of C1)."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from tox.constants import CLEAN, CONFIGS, DESC_CACHE, FP_CACHE

RDLogger.DisableLog("rdApp.*")
CFG = yaml.safe_load((CONFIGS / "aml.yaml").read_text())
gen = rdFingerprintGenerator.GetMorganGenerator(radius=CFG["fp_radius"], fpSize=CFG["fp_size"])
DESC_NAMES = [n for n, _ in Descriptors.descList]
IPC = DESC_NAMES.index("Ipc")


def morgan(smi):
    return gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi)).astype(np.uint8)


def rdkit_desc(smi):
    m = Chem.MolFromSmiles(smi)
    d = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
    d[IPC] = np.log1p(d[IPC])        # Ipc goes up to ~1e41, overflows float32 inside sklearn trees
    return d


def load_features():
    """-> (fp uint8 [n, 2048], desc float64 [n, n_desc]); computes the caches if missing.
    Fails loudly if a cache no longer lines up with C1 (e.g. the parquet was rebuilt after featurizing),
    because a stale cache would silently pair molecules with the wrong labels."""
    if not (FP_CACHE.exists() and DESC_CACHE.exists()):
        main()
    fp, desc = np.load(FP_CACHE), np.load(DESC_CACHE)
    n = pq.read_metadata(CLEAN).num_rows
    if not fp.shape[0] == desc.shape[0] == n:
        raise ValueError(f"feature caches out of date: fp has {fp.shape[0]} rows, desc {desc.shape[0]}, "
                         f"{CLEAN.name} {n}. Rerun python -m tox.featurize_fp")
    return fp, desc


def main():
    smiles = pd.read_parquet(CLEAN).smiles
    fp = np.stack(smiles.map(morgan))
    desc = np.stack(smiles.map(rdkit_desc))
    np.save(FP_CACHE, fp)
    np.save(DESC_CACHE, desc)
    print(f"fp {fp.shape}, desc {desc.shape} -> {FP_CACHE.parent}")


if __name__ == "__main__":
    main()
