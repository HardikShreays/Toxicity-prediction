"""S3 - tests for standardize, the dedup rules and the C1 contract.

The pure functions (`standardize`, `scaffold`, `dedup`) are tested on hand-written inputs, so
they run without the download. The contract checks read the committed C1 parquet and are
skipped when it has not been generated yet (it is git-ignored); `python -m tox.data` creates it.
"""
import numpy as np
import pandas as pd
import pytest

from tox.constants import CLEAN, TASKS
from tox.data import dedup, scaffold, standardize

needs_data = pytest.mark.skipif(not CLEAN.exists(), reason="run python -m tox.data first")


# --- standardize ---------------------------------------------------------------------------

def test_standardize_returns_canonical_smiles():
    """The same molecule written two ways must collapse to one string - dedup depends on it."""
    assert standardize("OCC") == standardize("CCO") == "CCO"
    assert standardize("c1ccccc1") == standardize("C1=CC=CC=C1")


def test_standardize_is_idempotent():
    for smi in ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1"]:
        assert standardize(standardize(smi)) == standardize(smi)


def test_standardize_keeps_only_the_largest_fragment():
    """The counter-ion is dropped, so a hydrochloride and its free base become the same row."""
    assert standardize("c1ccccc1C(=O)O.Cl") == standardize("c1ccccc1C(=O)O")
    assert standardize("CC(=O)[O-].[Na+]") == "CC(=O)[O-]"


def test_stripping_a_salt_does_not_neutralize_what_is_left():
    """Documented decision: largest fragment only, charges untouched. So the sodium salt and
    the free acid stay DIFFERENT molecules - they are not merged by dedup."""
    assert standardize("CC(=O)[O-].[Na+]") != standardize("CC(=O)O")


def test_standardize_returns_none_for_unparseable_smiles():
    """main() drops exactly these rows into dropped.csv; nothing else may return None."""
    assert standardize("not_a_molecule") is None
    assert standardize("C(C") is None
    assert standardize("[CH5]") is None


def test_empty_smiles_survives_standardization_and_then_breaks_scaffold():
    """Edge case, recorded rather than fixed: RDKit parses "" into an empty molecule, so
    standardize passes it through instead of returning None, and scaffold() then raises.
    Tox21 contains no such row (asserted below), so the committed pipeline is unaffected."""
    assert standardize("") == ""
    with pytest.raises(ValueError):
        scaffold("")


@needs_data
def test_the_dataset_contains_no_empty_smiles():
    assert (pd.read_parquet(CLEAN).smiles == "").sum() == 0


def test_standardize_preserves_stereochemistry_and_charge():
    """The documented decision (open question 2): only salts are stripped."""
    assert "@" in standardize("C[C@H](N)C(=O)O")
    assert standardize("C[C@H](N)C(=O)O") != standardize("C[C@@H](N)C(=O)O")
    assert "+" in standardize("C[N+](C)(C)C.[Cl-]")


# --- scaffold ------------------------------------------------------------------------------

def test_scaffold_of_a_ring_system_is_its_murcko_skeleton():
    """Side chains are removed, the ring system is kept, so analogues group together."""
    assert scaffold("c1ccccc1CCO") == scaffold("c1ccccc1CCCN") == "c1ccccc1"


def test_acyclic_molecules_share_the_empty_scaffold():
    """The plan notes these form one group and land in train; that must stay true."""
    assert scaffold("CCO") == ""
    assert scaffold("CC(=O)O") == ""


# --- dedup ---------------------------------------------------------------------------------

def _frame(rows):
    return pd.DataFrame(rows, columns=["mol_id", "smiles"] + TASKS)


def _row(mol_id, smi, **labels):
    r = {"mol_id": mol_id, "smiles": smi, **{t: np.nan for t in TASKS}}
    r.update(labels)
    return r


def test_dedup_keeps_agreeing_labels_and_the_first_mol_id():
    df = _frame([_row("a", "CCO", **{TASKS[0]: 1.0}), _row("b", "CCO", **{TASKS[0]: 1.0})])
    clean, conflicts = dedup(df)
    assert len(clean) == 1 and clean.mol_id.iloc[0] == "a"
    assert clean[TASKS[0]].iloc[0] == 1.0
    assert conflicts.empty


def test_dedup_sets_conflicting_labels_to_nan_and_logs_them():
    """A 0 and a 1 for the same assay is assay noise; it must become missing, never a guess."""
    df = _frame([_row("a", "CCO", **{TASKS[0]: 1.0}), _row("b", "CCO", **{TASKS[0]: 0.0})])
    clean, conflicts = dedup(df)
    assert len(clean) == 1
    assert np.isnan(clean[TASKS[0]].iloc[0])
    assert len(conflicts) == 1
    assert conflicts.iloc[0].smiles == "CCO" and conflicts.iloc[0].task == TASKS[0]
    assert conflicts.iloc[0].mol_ids == "a;b"


def test_dedup_merges_tasks_independently():
    """One conflicting endpoint must not discard the agreeing ones in the same group."""
    df = _frame([_row("a", "CCO", **{TASKS[0]: 1.0, TASKS[1]: 0.0}),
                 _row("b", "CCO", **{TASKS[0]: 0.0, TASKS[1]: 0.0})])
    clean, conflicts = dedup(df)
    assert np.isnan(clean[TASKS[0]].iloc[0])        # conflicting
    assert clean[TASKS[1]].iloc[0] == 0.0           # agreeing, kept
    assert list(conflicts.task) == [TASKS[0]]


def test_dedup_keeps_missing_labels_missing():
    """All-NaN stays NaN: a merge must never invent a label."""
    df = _frame([_row("a", "CCO"), _row("b", "CCO")])
    clean, conflicts = dedup(df)
    assert clean[TASKS].isna().all(axis=None) and conflicts.empty


def test_dedup_fills_a_gap_from_the_other_duplicate():
    df = _frame([_row("a", "CCO"), _row("b", "CCO", **{TASKS[2]: 1.0})])
    clean, _ = dedup(df)
    assert clean[TASKS[2]].iloc[0] == 1.0


def test_dedup_leaves_distinct_molecules_alone():
    df = _frame([_row("a", "CCO", **{TASKS[0]: 1.0}), _row("b", "CCN", **{TASKS[0]: 0.0})])
    clean, conflicts = dedup(df)
    assert len(clean) == 2 and conflicts.empty
    assert clean.smiles.is_unique


# --- contract C1 ---------------------------------------------------------------------------

@needs_data
def test_c1_has_the_contract_columns_in_order():
    df = pd.read_parquet(CLEAN)
    assert list(df.columns) == ["mol_id", "smiles", "scaffold"] + TASKS


@needs_data
def test_c1_labels_are_floats_restricted_to_1_0_and_nan():
    df = pd.read_parquet(CLEAN)
    for t in TASKS:
        assert df[t].dtype == float
        assert set(np.unique(df[t].dropna())) <= {0.0, 1.0}


@needs_data
def test_c1_smiles_are_unique_canonical_and_deduplicated():
    """Dedup ran before the split; downstream code keys on these strings."""
    df = pd.read_parquet(CLEAN)
    assert df.smiles.is_unique
    assert df.smiles.notna().all()
    sample = df.smiles.head(200)
    assert (sample.map(standardize) == sample).all()


@needs_data
def test_c1_scaffolds_match_the_scaffold_function():
    df = pd.read_parquet(CLEAN).head(200)
    assert (df.smiles.map(scaffold) == df.scaffold).all()


@needs_data
def test_c1_every_endpoint_has_both_classes_and_some_missing_labels():
    """The dataset the whole project assumes: sparse, imbalanced, never fully observed."""
    df = pd.read_parquet(CLEAN)
    for t in TASKS:
        assert set(np.unique(df[t].dropna())) == {0.0, 1.0}
        assert df[t].isna().any()
