"""S1-S3 - download, standardize, dedup. Writes contract C1.

Decision (open question 2): we keep charges and stereochemistry as they are.
Only salts/counter-ions are stripped (largest fragment). Stereo isomers are
genuinely different compounds and Tox21 tested them separately.
"""
import hashlib
import urllib.request

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from tox.constants import CLEAN, CONFLICTS, DROPPED, PROCESSED, RAW_CSV, RAW_URL, TASKS

RDLogger.DisableLog("rdApp.*")
lfc = rdMolStandardize.LargestFragmentChooser()


def standardize(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m = lfc.choose(m)                         # drop salts and counter-ions
    return Chem.MolToSmiles(m, canonical=True)


def scaffold(smi):
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)


def download():
    if not RAW_CSV.exists():
        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAW_URL, RAW_CSV)
    return hashlib.md5(RAW_CSV.read_bytes()).hexdigest()


def dedup(df):
    """Merge rows with the same canonical smiles. Per task: agreeing labels are kept,
    conflicting ones (a 0 and a 1) become NaN. First mol_id wins."""
    conflicts = []
    rows = []
    for smi, g in df.groupby("smiles", sort=False):
        row = {"mol_id": g.mol_id.iloc[0], "smiles": smi}
        for t in TASKS:
            vals = g[t].dropna().unique()
            if len(vals) > 1:
                conflicts.append({"smiles": smi, "task": t, "mol_ids": ";".join(g.mol_id)})
                row[t] = np.nan
            else:
                row[t] = vals[0] if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(conflicts, columns=["smiles", "task", "mol_ids"])


def main():
    md5 = download()
    raw = pd.read_csv(RAW_CSV)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    df = raw.copy()
    df["smiles"] = df.smiles.map(standardize)
    dropped = df[df.smiles.isna()]
    dropped[["mol_id"]].to_csv(DROPPED, index=False)
    df = df.dropna(subset=["smiles"])

    n_before = len(df)
    n_dup_groups = int((df.smiles.value_counts() > 1).sum())
    clean, conflicts = dedup(df)
    conflicts.to_csv(CONFLICTS, index=False)

    clean["scaffold"] = clean.smiles.map(scaffold)
    clean[TASKS] = clean[TASKS].astype(float)
    clean = clean[["mol_id", "smiles", "scaffold"] + TASKS]
    clean.to_parquet(CLEAN, index=False)

    (PROCESSED / "README.md").write_text(
        "# data/processed\n\n"
        f"- raw file: `{RAW_URL}`\n"
        f"- raw md5: `{md5}`\n"
        f"- raw rows: {len(raw)}, columns: {', '.join(raw.columns)}\n"
        f"- unparseable smiles dropped: {len(dropped)} (see dropped.csv)\n"
        f"- rows after standardization: {n_before}\n"
        f"- duplicate canonical smiles groups: {n_dup_groups} "
        f"({n_before - len(clean)} rows merged)\n"
        f"- conflicting labels set to NaN: {len(conflicts)} (task, molecule) pairs "
        f"(see conflicts.csv)\n"
        f"- clean rows: {len(clean)}, unique scaffolds: {clean.scaffold.nunique()}\n"
        "- standardization: largest fragment only, charges and stereo untouched\n"
    )
    print((PROCESSED / "README.md").read_text())


if __name__ == "__main__":
    main()
