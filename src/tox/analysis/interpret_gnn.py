"""P3 (GNN side) - per-atom attributions for the graph model, rendered as similarity maps.

    python -m tox.analysis.interpret_gnn                        # mpnn, SR-MMP, scaffold_det seed 0
    python -m tox.analysis.interpret_gnn --model gcn --task NR-AhR
    python -m tox.analysis.interpret_gnn --n 3                  # fewer molecules, quicker

What it does, for one endpoint of one trained model:

  1. reads the COMMITTED C3 predictions (results/preds) for that model/split/seed, so the
     molecules shown are the ones the reported numbers were computed from - nothing is
     re-predicted for the figure;
  2. picks the `n` most confident true positives and the `n` most confident false positives on
     the TEST part (the errors worth looking at are the confident ones);
  3. attributes each prediction to atoms with `torch_geometric.explain.GNNExplainer`, which
     learns a mask over nodes that preserves the model's output for THAT molecule and THAT
     endpoint;
  4. renders the attribution over the structure with `rdkit.Chem.Draw.SimilarityMaps`.

The model is loaded from the run's checkpoint when it exists (results/gnn/ckpt/..., git-ignored)
and trained with the frozen config otherwise, seeded exactly as in `train_gnn.run`, so the
explained model is the one that produced the committed predictions.

What a map does and does not prove (stated in the report, asked in the viva): a high weight
means the model's score for this molecule drops when that atom is masked. It is evidence about
THIS model on THIS molecule, not a claim about chemistry, and not a causal statement about
toxicity. Two different explainers can disagree; attributions are a debugging aid, not a result.
"""
from __future__ import annotations

import argparse
import io

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

from tox.constants import CLEAN, FIGURES, PREDS, RESULTS, SPLITS, TASKS
from tox.featurize_graph import smiles_to_graph
from tox.models_gnn import build_model, canonical_name
from tox.train_gnn import GNN_RESULTS, fit, frozen_config, load_split_graphs, make_pos_weight, set_seed

OUT = RESULTS / "interpret"
ARCH_KEYS = ("hidden", "n_layers", "pooling", "dropout", "use_bond_features")


class _TaskView(torch.nn.Module):
    """Explainer-facing view of a trained model: (x, edge_index, ...) -> one endpoint's logit.

    `Explainer` calls `model(x, edge_index, **kwargs)`, while our models take a PyG Batch and
    return 12 logits. This adapter bridges both, so neither the models nor the explainer need
    to change.
    """

    def __init__(self, model: torch.nn.Module, task: int):
        super().__init__()
        self.model, self.task = model, task

    def forward(self, x, edge_index, edge_attr=None, batch=None, **_):
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.batch = torch.zeros(x.size(0), dtype=torch.long) if batch is None else batch
        return self.model(data)[:, [self.task]]


def trained_model(model_name: str, split: str, seed: int):
    """The trained model behind the committed predictions: from its checkpoint, or retrained."""
    model_name = canonical_name(model_name)
    cfg = frozen_config(model_name)
    model = build_model(model_name, **{k: cfg[k] for k in ARCH_KEYS})
    ckpt = GNN_RESULTS / "ckpt" / f"{model_name}__{split}__seed{seed}" / "best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=False)["model"])
        print(f"loaded {ckpt}")
    else:
        print(f"{ckpt} not found (checkpoints are git-ignored); retraining with the frozen config")
        graphs = load_split_graphs(split, roles=("train", "val"))
        set_seed(seed)                                   # same order as train_gnn.run
        model = build_model(model_name, **{k: cfg[k] for k in ARCH_KEYS})
        fit(model, graphs["train"], graphs["val"], cfg, seed, make_pos_weight(cfg, graphs["train"]))
    return model.eval()


def pick_molecules(model_name: str, split: str, seed: int, task: str, n: int = 5) -> pd.DataFrame:
    """The n most confident true positives and false positives on test, from the C3 file."""
    k = TASKS.index(task)
    preds = pd.read_parquet(PREDS / f"{model_name}__{split}__seed{seed}.parquet")
    preds = preds[preds.role == "test"]
    labels = pd.read_parquet(CLEAN).set_index("smiles")[task]
    df = pd.DataFrame({"smiles": preds.smiles.values, "prob": preds[task].values})
    df["y"] = labels.reindex(df.smiles).values
    df = df[df.y.notna()]
    tp = df[df.y == 1.0].nlargest(n, "prob").assign(kind="tp")
    fp = df[df.y == 0.0].nlargest(n, "prob").assign(kind="fp")
    return pd.concat([tp, fp], ignore_index=True)


def atom_attributions(model: torch.nn.Module, smiles: str, task: int, epochs: int = 200) -> np.ndarray:
    """One weight per atom, in RDKit atom order (the order `featurize_graph` builds x in)."""
    g = smiles_to_graph(smiles)
    if g is None:
        raise ValueError(f"RDKit cannot parse {smiles!r}")
    explainer = Explainer(
        model=_TaskView(model, task),
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="object",                 # one scalar per atom
        edge_mask_type="object",
        model_config=dict(mode="binary_classification", task_level="graph", return_type="raw"),
    )
    batch = torch.zeros(g.x.size(0), dtype=torch.long)
    ex = explainer(g.x, g.edge_index, edge_attr=g.edge_attr, batch=batch)
    return ex.node_mask.detach().squeeze(-1).cpu().numpy()


def draw(smiles: str, weights: np.ndarray, path, title: str = "", size: int = 500) -> None:
    """Attribution over the structure, saved as a 300-dpi PNG.

    Weights are centred and scaled to [-1, 1] so one colour scale reads the same across
    molecules: green is "the model leans on this atom", pink is the opposite.
    """
    mol = Chem.MolFromSmiles(smiles)
    w = np.asarray(weights, dtype=float) - np.mean(weights)
    scale = np.abs(w).max()
    if scale > 0:
        w = w / scale
    d = rdMolDraw2D.MolDraw2DCairo(size, size)
    SimilarityMaps.GetSimilarityMapFromWeights(mol, [float(v) for v in w], draw2d=d)
    d.FinishDrawing()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.2, 3.4))
    ax.imshow(plt.imread(io.BytesIO(d.GetDrawingText()), format="png"))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=7)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="GNN atom attributions (plan P3, GNN side).")
    ap.add_argument("--model", default="mpnn")
    ap.add_argument("--split", default="scaffold_det")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="SR-MMP", choices=TASKS)
    ap.add_argument("--n", type=int, default=5, help="molecules per group (plan asks for 5)")
    ap.add_argument("--epochs", type=int, default=200, help="GNNExplainer optimisation steps")
    a = ap.parse_args()

    model_name = canonical_name(a.model)
    k = TASKS.index(a.task)
    chosen = pick_molecules(model_name, a.split, a.seed, a.task, a.n)
    model = trained_model(model_name, a.split, a.seed)

    rows = []
    for i, r in chosen.iterrows():
        w = atom_attributions(model, r.smiles, k, a.epochs)
        stem = f"{model_name}_{a.task}_{r.kind}{list(chosen[chosen.kind == r.kind].index).index(i) + 1}"
        draw(r.smiles, w, FIGURES / f"interpret_{stem}.png",
             f"{r.kind.upper()}  p={r.prob:.2f}  y={int(r.y)}  {a.task}")
        mol = Chem.MolFromSmiles(r.smiles)
        top = np.argsort(-w)[:3]
        rows.append({"model": model_name, "split": a.split, "seed": a.seed, "task": a.task,
                     "kind": r.kind, "smiles": r.smiles, "y": int(r.y), "prob": float(r.prob),
                     "n_atoms": int(mol.GetNumAtoms()),
                     "top_atoms": ";".join(f"{int(j)}:{mol.GetAtomWithIdx(int(j)).GetSymbol()}"
                                           f":{w[j]:.3f}" for j in top)})
        print(f"{stem}: p={r.prob:.3f} y={int(r.y)} {rows[-1]['top_atoms']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"{model_name}__{a.split}__seed{a.seed}__{a.task}.csv", index=False)
    (OUT / "README.md").write_text(
        "# results/interpret\n\n"
        "Per-atom attributions for the graph model (plan P3), written by "
        "`python -m tox.analysis.interpret_gnn`.\n\n"
        "One CSV per (model, split, seed, endpoint) listing the molecules explained and their "
        "three highest-weighted atoms; the maps themselves are "
        "`results/figures/interpret_*.png`.\n\n"
        "A weight is the GNNExplainer node mask: how much the model's logit for that endpoint "
        "relies on that atom, for that molecule. It describes the model, not the chemistry.\n")
    print(f"\n{len(out)} molecules -> {OUT} and results/figures/interpret_*.png")


if __name__ == "__main__":
    main()
