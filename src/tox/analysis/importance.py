"""P3 (AML side): XGBoost gain importances -> substructures and descriptor figures.

Reads results/aml_importance/ (Arjun's A4 output, scaffold_det) and the clean parquet, recovers
the Morgan substructure behind each top bit via bitInfo, and saves PNGs to results/figures/.
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw

from tox.constants import AML_IMPORTANCE, CLEAN, FIGURES, TASKS

RDLogger.DisableLog("rdApp.*")
FOCUS = ["SR-MMP", "NR-AhR", "SR-ARE"]
TOP = 10
DPI = 300
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def bit_examples(smiles, bits, radius=2, n_bits=2048):
    """For each wanted bit: first molecule containing it (index, atom, radius) and how many molecules have it."""
    found, count = {}, {b: 0 for b in bits}
    for i, smi in enumerate(smiles):
        info = {}
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius, n_bits, bitInfo=info)
        for b in bits:
            if b in info:
                count[b] += 1
                found.setdefault(b, (i, *info[b][0]))
    return {b: dict(mol=i, atom=a, radius=r, n_mols=count[b]) for b, (i, a, r) in found.items()}


def plot_bits(task, imp, smiles):
    top = imp[imp.task == task].head(TOP).copy()
    bits = [int(f[3:]) for f in top.feature]
    ex = bit_examples(smiles, bits)
    mols, legends, hl = [], [], []
    for b, g in zip(bits, top.gain):
        e = ex[b]
        m = Chem.MolFromSmiles(smiles[e["mol"]])
        env = Chem.FindAtomEnvironmentOfRadiusN(m, e["radius"], e["atom"])
        atoms = {e["atom"]}
        for bi in env:
            bd = m.GetBondWithIdx(bi)
            atoms |= {bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()}
        mols.append(m); hl.append(sorted(atoms))
        legends.append(f"bit {b} | gain {g:.0f} | r={e['radius']} | in {e['n_mols']} mols")
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 260), legends=legends,
                               highlightAtomLists=hl)
    path = FIGURES / f"importance_bits_{task}.png"
    img.save(path)
    return path


def plot_desc(imp, top=10):
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    for ax, t in zip(axes.ravel(), TASKS):
        d = imp[imp.task == t].head(top).iloc[::-1]
        ax.barh(d.feature, d.gain, color="#2b6cb0")
        ax.set_title(t, fontsize=10)
        ax.tick_params(labelsize=7)
    fig.suptitle("XGBoost gain importance (FP+descriptors, scaffold_det): top 10 per endpoint")
    fig.tight_layout()
    path = FIGURES / "importance_desc_top10.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_group_share(imp):
    imp = imp.copy()
    imp["kind"] = imp.feature.str.startswith("bit").map({True: "Morgan bits", False: "RDKit descriptors"})
    s = imp.groupby(["task", "kind"]).gain.sum().unstack().reindex(TASKS)
    s = s.div(s.sum(axis=1), axis=0)
    ax = s.plot.bar(stacked=True, figsize=(9, 4), color=["#a0aec0", "#2b6cb0"], width=0.8)
    ax.set_ylabel("share of total gain"); ax.set_xlabel("")
    ax.set_title("Where the gain comes from in xgb_fpdesc (scaffold_det)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    path = FIGURES / "importance_gain_share.png"
    plt.savefig(path, dpi=DPI); plt.close()
    return s, path


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    smiles = pd.read_parquet(CLEAN).smiles.tolist()
    fp = pd.read_csv(AML_IMPORTANCE / "xgb_fp__scaffold_det.csv")
    fd = pd.read_csv(AML_IMPORTANCE / "xgb_fpdesc__scaffold_det.csv")
    for t in FOCUS:
        print(plot_bits(t, fp, smiles))
    print(plot_desc(fd))
    s, p = plot_group_share(fd)
    print(p); print(s.round(2).to_string())
    print(fd.groupby("task").head(3).to_string(index=False))


if __name__ == "__main__":
    main()
