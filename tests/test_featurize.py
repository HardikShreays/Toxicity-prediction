import numpy as np

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
