import pandas as pd
import pytest

from tox.constants import CLEAN, SPLIT_NAMES, SPLITS, TASKS
from tox.splits import load, random_split, scaffold_balanced, scaffold_det

# small fake scaffold list: a few big groups and many singletons
FAKE = ["A"] * 30 + ["B"] * 20 + ["C"] * 10 + [f"s{i}" for i in range(40)]


@pytest.mark.parametrize("parts", [scaffold_det(FAKE), scaffold_balanced(FAKE, 0), random_split(len(FAKE), 0)])
def test_parts_disjoint_and_cover_everything(parts):
    tr, va, te = map(set, parts)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(range(len(FAKE)))


@pytest.mark.parametrize("parts", [scaffold_det(FAKE), scaffold_balanced(FAKE, 1)])
def test_no_scaffold_in_two_parts(parts):
    scafs = [{FAKE[i] for i in p} for p in parts]
    assert not (scafs[0] & scafs[1]) and not (scafs[0] & scafs[2]) and not (scafs[1] & scafs[2])


def test_balanced_split_changes_with_seed():
    assert scaffold_balanced(FAKE, 0) != scaffold_balanced(FAKE, 1)


# the committed split files (skipped if the data hasn't been built yet)
needs_data = pytest.mark.skipif(not CLEAN.exists(), reason="run python -m tox.data first")


@needs_data
@pytest.mark.parametrize("name", SPLIT_NAMES)
def test_real_split_files(name):
    df = pd.read_parquet(CLEAN).set_index("smiles")
    s = load(name)
    tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(df.index)
    if name.startswith("scaffold"):
        sc = [set(df.loc[list(p), "scaffold"]) for p in (tr, va, te)]
        assert not (sc[0] & sc[1]) and not (sc[0] & sc[2]) and not (sc[1] & sc[2])
    for role in ("val", "test"):
        y = df.loc[s[role], TASKS]
        missing = [t for t in TASKS if (y[t] == 1).sum() == 0]
        if missing:
            print(f"WARNING {name}: no positives in {role} for {missing}")
