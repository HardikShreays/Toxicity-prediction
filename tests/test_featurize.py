import numpy as np
import pandas as pd
import pytest

from tox import featurize_fp
from tox.featurize_fp import DESC_NAMES, morgan, rdkit_desc


def test_morgan_shape_and_dtype():
    fp = morgan("c1ccccc1O")
    assert fp.shape == (2048,) and fp.dtype == np.uint8
    assert set(np.unique(fp)) <= {0, 1} and fp.sum() > 0


def test_same_molecule_same_fingerprint():
    # two spellings of phenol
    assert np.array_equal(morgan("c1ccccc1O"), morgan("Oc1ccccc1"))


def test_descriptors_line_up_with_names():
    d = rdkit_desc("CCO")
    assert d.shape == (len(DESC_NAMES),)
    assert np.isclose(d[DESC_NAMES.index("HeavyAtomCount")], 3)


def test_ipc_is_logged():
    big = "CCCCCCCCCCCCCCCCCCCC(CCCCCCCCCC)CCCCCCCCCCCCCCC(CCCCCCCCCC)CCCCCCCCCC"
    assert rdkit_desc(big)[DESC_NAMES.index("Ipc")] < 1e3


# load_features cache checks, against a tiny synthetic C1 in a temp dir
@pytest.fixture
def tiny_c1(tmp_path, monkeypatch):
    clean = tmp_path / "clean.parquet"
    pd.DataFrame({"smiles": ["CCO", "c1ccccc1O", "CC(=O)O"]}).to_parquet(clean)
    monkeypatch.setattr(featurize_fp, "CLEAN", clean)
    monkeypatch.setattr(featurize_fp, "FP_CACHE", tmp_path / "fp.npy")
    monkeypatch.setattr(featurize_fp, "DESC_CACHE", tmp_path / "desc.npy")
    return tmp_path


def test_missing_caches_are_built_row_aligned(tiny_c1):
    fp, desc = featurize_fp.load_features()
    assert fp.shape == (3, 2048) and desc.shape == (3, len(DESC_NAMES))
    assert np.array_equal(fp[1], morgan("c1ccccc1O"))


def test_matching_caches_load(tiny_c1):
    np.save(tiny_c1 / "fp.npy", np.zeros((3, 2048), np.uint8))
    np.save(tiny_c1 / "desc.npy", np.zeros((3, 5)))
    fp, desc = featurize_fp.load_features()
    assert fp.shape[0] == desc.shape[0] == 3


@pytest.mark.parametrize("n_fp, n_desc", [(4, 4), (3, 2)])
def test_stale_cache_fails_loudly(tiny_c1, n_fp, n_desc):
    np.save(tiny_c1 / "fp.npy", np.zeros((n_fp, 2048), np.uint8))
    np.save(tiny_c1 / "desc.npy", np.zeros((n_desc, 5)))
    with pytest.raises(ValueError, match="out of date"):
        featurize_fp.load_features()
