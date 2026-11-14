"""Phase 2 hybrid runs (Hardik): the three variants on the same splits and seeds as their
parents, plus the hybrid ablations (plan p. 18-19), all scored with the shared tox.metrics.

CLI:
  python -m tox.train_hybrid --grid                  # hyb_fusion / hyb_embxgb / hyb_stack
                                                     # on every RUNS (split, seed) pair
  python -m tox.train_hybrid --variant hyb_fusion --split scaffold_det --seed 0
  python -m tox.train_hybrid --ablations             # scaffold_det x 3 seeds, one factor changed
  python -m tox.train_hybrid --list                  # print the ablation arms
  python -m tox.train_hybrid --score                 # score hyb* C3 files -> C4 (shared scorer)
  python -m tox.train_hybrid --summary               # ablation arms vs their baselines

Prerequisites: ./run_all.sh (fingerprint C3 files, feature caches) and ./run_gnn.sh (frozen
tuned MPNN config + the mpnn__{split}__seed{s}.pt checkpoints). Nothing here computes a
final metric: C3 files are written for val + test and tox.metrics does the scoring.

Ablation arms (scaffold_det, 3 seeds, baseline = the main run of the same variant):
  hyb_fusion-nofp        fusion without the fingerprint branch
  hyb_fusion-nograph     fusion without the graph branch
  hyb_embxgb-finetune    the GNN encoder fine-tuned (fine_tune_epochs extra epochs at
                         fine_tune_lr) before its embeddings are frozen into XGBoost - the
                         plan's "frozen vs fine-tuned encoder" question, with the XGBoost
                         stage held fixed
The other two hybrid ablations of the plan (sparsest vs densest endpoints, Tanimoto
similarity bins) are analyses of the saved predictions, in tox.hybrid_analysis.
"""
from __future__ import annotations

import argparse

# import-order guard (macOS, see conftest.py): torch must be imported BEFORE xgboost, or a
# later XGBoost fit segfaults. train_aml (xgboost) is imported below, after hybrid/train_gnn
# have imported torch - keep that order.
import numpy as np
import pandas as pd
import yaml

from tox.constants import CLEAN, CONFIGS, HYBRID, METRICS, PREDS, RESULTS, SEEDS, TASKS
from tox.featurize_fp import load_features
from tox.hybrid import (Fusion, encoder_embeddings, load_mpnn_encoder, stack_features,
                        stack_one, warm_start, with_fps, write_meta)
from tox.splits import load
from tox.train_aml import fit_task
from tox.train_gnn import (RUNS, fit, frozen_config, load_split_graphs, make_pos_weight,
                           predict_probs, score_preds, set_seed, write_c3)

VARIANTS = ("hyb_fusion", "hyb_embxgb", "hyb_stack")
SPLIT = "scaffold_det"                       # ablations, like every other ablation in the repo
OUT = RESULTS / "ablations" / "hybrid"       # preds / logs / metrics / summary.csv
RUNNERS = {"hyb_fusion": None, "hyb_embxgb": None, "hyb_stack": None}   # filled below


def load_config(overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load((CONFIGS / "hybrid.yaml").read_text())
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    # branch switches of the fusion ablation arms (hyb_fusion-nofp / -nograph); on by default
    cfg.setdefault("use_fp", True)
    cfg.setdefault("use_graph", True)
    return cfg


def _split_features(split: str) -> tuple[dict, np.ndarray, dict]:
    """(role -> smiles) from the committed C2 split, the C1-aligned Morgan cache, and
    (role -> labels [n, 12] with NaN kept). Nothing here touches any other split."""
    s = load(split)
    fp, _desc = load_features()
    labels = pd.read_parquet(CLEAN).set_index("smiles")[TASKS]
    y = {r: labels.loc[s[r], TASKS].to_numpy(dtype=float) for r in ("train", "val", "test")}
    return s, fp, y


def run_fusion(split: str, seed: int, cfg: dict, out=PREDS, log_dir=HYBRID, graphs=None,
               tag: str | None = None) -> dict:
    """hyb_fusion (Hardik): MPNN branch warm-started from this split+seed's grid checkpoint,
    fingerprint branch, shared head, end-to-end masked BCE with the GNN training recipe.

    Early stopping, checkpointing and seeding are exactly tox.train_gnn.fit, so the hybrid
    is trained under the identical protocol as its GNN parent. With use_fp/use_graph
    switched off the module becomes one of the plan's ablation arms.
    """
    from tox.train_gnn import GNN_RESULTS
    tag = tag or "hyb_fusion"
    stem = f"{tag}__{split}__seed{seed}"
    graphs = graphs or load_split_graphs(split)
    if cfg["use_fp"]:
        smiles = pd.read_parquet(CLEAN).smiles
        fp, _ = load_features()
        graphs = {r: with_fps(graphs[r], fp, smiles) for r in graphs}
    enc_cfg = frozen_config(cfg["encoder"])
    set_seed(seed)
    model = Fusion(hidden=enc_cfg["hidden"], n_layers=enc_cfg["n_layers"],
                   pooling=enc_cfg["pooling"], dropout=enc_cfg["dropout"],
                   use_bond_features=enc_cfg["use_bond_features"], fp_hidden=cfg["fp_hidden"],
                   use_fp=cfg["use_fp"], use_graph=cfg["use_graph"])
    if cfg["use_graph"]:
        warm_start(model, GNN_RESULTS / f"mpnn__{split}__seed{seed}.pt")
    pw = make_pos_weight(cfg, graphs["train"])
    info = fit(model, graphs["train"], graphs["val"], cfg, seed, pw, ckpt_dir=log_dir / "ckpt" / stem)
    probs = {r: predict_probs(model, graphs[r]) for r in ("val", "test")}
    write_c3(graphs, probs, out / f"{stem}.parquet")
    write_meta(log_dir / f"{stem}.json", {"variant": "hyb_fusion", "tag": tag, "split": split,
               "seed": seed, "config": cfg, "encoder_config": enc_cfg,
               "pos_weight": None if pw is None else pw.tolist(),
               "early_stopping": {"metric": "val_mean_roc_auc", "patience": cfg["patience"]},
               **info})
    print(f"{stem}: best epoch {info['best_epoch']}, val mean ROC-AUC {info['best_val_roc_auc']:.3f}, "
          f"PR-AUC {info['best_val_pr_auc']:.3f}", flush=True)
    return info


def run_embxgb(split: str, seed: int, cfg: dict, out=PREDS, log_dir=HYBRID, graphs=None,
               tag: str | None = None) -> dict:
    """hyb_embxgb (Arjun): frozen pooled MPNN embeddings || Morgan fingerprint -> one XGBoost
    per endpoint (the tox.train_aml recipe: small grid, per-endpoint selection on val PR-AUC,
    early stopping on val). The encoder is a feature extractor; with
    tag=hyb_embxgb-finetune it is first fine-tuned (fine_tune_epochs at fine_tune_lr) on this
    split's train rows - the plan's frozen-vs-fine-tuned ablation arm.
    """
    tag = tag or "hyb_embxgb"
    fine_tuned = tag.endswith("-finetune")
    stem = f"{tag}__{split}__seed{seed}"
    s, fp, y = _split_features(split)
    graphs = graphs or load_split_graphs(split)
    enc_cfg = frozen_config(cfg["encoder"])
    model = load_mpnn_encoder(split, seed, enc_cfg)
    ft_info = None
    if fine_tuned:
        ft_cfg = {**cfg, "lr": cfg["embxgb"]["fine_tune_lr"],
                  "max_epochs": cfg["embxgb"]["fine_tune_epochs"],
                  "patience": cfg["embxgb"]["fine_tune_epochs"]}
        ft_info = fit(model, graphs["train"], graphs["val"], ft_cfg, seed,
                      make_pos_weight(ft_cfg, graphs["train"]))
    emb = {r: encoder_embeddings(model, graphs[r]) for r in ("train", "val", "test")}
    pos = pd.Series(range(fp.shape[0]), index=pd.read_parquet(CLEAN).smiles)
    X = {r: np.hstack([fp[pos[s[r]]].astype(np.float32), emb[r].astype(np.float32)])
         for r in ("train", "val", "test")}
    probs = {r: np.zeros((len(s[r]), len(TASKS))) for r in ("val", "test")}
    run_log = {}
    for j, t in enumerate(TASKS):
        xgb_model, info = fit_task("xgb", X, {k: v[:, j] for k, v in y.items()}, seed)
        for r in probs:
            probs[r][:, j] = xgb_model.predict_proba(X[r])[:, 1]
        run_log[t] = info
    write_c3(graphs, probs, out / f"{stem}.parquet")
    write_meta(log_dir / f"{stem}.json", {"variant": "hyb_embxgb", "tag": tag, "split": split,
               "seed": seed, "config": cfg, "encoder_config": enc_cfg, "fine_tuned": fine_tuned,
               "fine_tune": None if ft_info is None else
               {"best_epoch": ft_info["best_epoch"], "epochs_run": ft_info["epochs_run"]},
               "tasks": run_log})
    mean_ap = float(np.nanmean([v["val_pr_auc"] if v["val_pr_auc"] is not None else np.nan
                                for v in run_log.values()]))
    print(f"{stem}: XGBoost on fp||embeddings, val mean PR-AUC {mean_ap:.3f}", flush=True)
    return {"val_pr_auc": mean_ap}


def run_stack(split: str, seed: int, cfg: dict, out=PREDS, log_dir=HYBRID, tag: str | None = None,
              graphs=None) -> dict:
    """hyb_stack (Arjun): per-endpoint logistic regression on the val probabilities of the two
    parents (xgb_fpdesc + mpnn, the same split and seed), fitted on val, applied to test.

    The parents must already exist as C3 files (run_all.sh wrote the xgb_fpdesc ones, the GNN
    grid the mpnn ones). Nothing here is fitted on test (plan rule 2).
    """
    from tox.metrics import load_preds
    tag = tag or "hyb_stack"
    stem = f"{tag}__{split}__seed{seed}"
    a = PREDS / f"{cfg['stack']['xgb_parent']}__{split}__seed{seed}.parquet"
    b = PREDS / f"{cfg['stack']['gnn_parent']}__{split}__seed{seed}.parquet"
    for p in (a, b):
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing; the stack needs both parents' C3 files")
    labels = pd.read_parquet(CLEAN).set_index("smiles")[TASKS]
    feats = {r: stack_features(a, b, r, labels) for r in ("val", "test")}
    X = {r: feats[r][0] for r in feats}
    y = {r: feats[r][1] for r in feats}
    probs, fallback = stack_one(X, y, seed)
    df = pd.read_parquet(a)                       # parent row order (asserted in stack_features)
    for j, t in enumerate(TASKS):
        for r in ("val", "test"):
            df.loc[df.role == r, t] = probs[r][:, j]
    assert df[TASKS].to_numpy().min() >= 0 and df[TASKS].to_numpy().max() <= 1
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{stem}.parquet", index=False)
    write_meta(log_dir / f"{stem}.json", {"variant": "hyb_stack", "tag": tag, "split": split,
               "seed": seed, "config": cfg,
               "parents": {"xgb": cfg["stack"]["xgb_parent"], "gnn": cfg["stack"]["gnn_parent"]},
               "fallback_tasks": fallback})
    print(f"{stem}: logistic stack on val probabilities"
          + (f" (fallback on {len(fallback)} endpoints: {', '.join(fallback)})" if fallback else ""),
          flush=True)
    return {"fallback": fallback}


RUNNERS.update({"hyb_fusion": run_fusion, "hyb_embxgb": run_embxgb, "hyb_stack": run_stack})

ABLATION_TAGS = {
    "hyb_fusion-nofp":     {"variant": "hyb_fusion", "cfg": {"use_fp": False}},
    "hyb_fusion-nograph":  {"variant": "hyb_fusion", "cfg": {"use_graph": False}},
    "hyb_embxgb-finetune": {"variant": "hyb_embxgb", "cfg": {}},
}


def run_ablations(seeds=SEEDS, force: bool = False, out_dir=None, graphs=None,
                  max_epochs: int | None = None) -> list:
    """Every ablation arm on scaffold_det for each seed; resumable (skips existing C3 files)."""
    out_dir = OUT if out_dir is None else out_dir
    cfg = load_config()
    if max_epochs is not None:
        cfg["max_epochs"] = max_epochs
    graphs = graphs or load_split_graphs(SPLIT)
    done = []
    for tag, spec in ABLATION_TAGS.items():
        for seed in seeds:
            if not force and (out_dir / "preds" / f"{tag}__{SPLIT}__seed{seed}.parquet").exists():
                continue
            RUNNERS[spec["variant"]](SPLIT, seed, {**cfg, **spec["cfg"]},
                                     out=out_dir / "preds", log_dir=out_dir / "logs",
                                     graphs=graphs, tag=tag)
            done.append((tag, seed))
    return done


def summarize(metrics_dir=None, baseline_dir=None) -> pd.DataFrame:
    """One row per run token: 12-task mean per seed, then mean/std over seeds - the same
    recipe as tox.aggregate and tox.ablate_gnn.summarize, so numbers are comparable to the
    main table. Baselines (the main hyb_fusion / hyb_embxgb runs) come from results/metrics.
    """
    out_dir = OUT if metrics_dir is None else metrics_dir
    baseline_dir = METRICS if baseline_dir is None else baseline_dir
    frames = []
    for d, kind in ((out_dir / "metrics", "ablation"), (baseline_dir, "main")):
        if not d.exists():
            continue
        for f in sorted(d.glob("hyb*__scaffold_det__seed*.csv")):
            tag, _split, seed = f.stem.split("__")
            variant = tag.split("-")[0]
            arm = "baseline" if kind == "main" else tag.split("-", 1)[1]
            frames.append(pd.read_csv(f).assign(variant=variant, arm=arm, kind=kind,
                                                seed=int(seed[4:])))
    if not frames:
        raise FileNotFoundError("no scored hybrid files; run tox.train_hybrid --score first")
    df = pd.concat(frames, ignore_index=True)
    per_run = df.groupby(["variant", "kind", "arm", "seed"], sort=False)[["roc_auc", "pr_auc"]].mean()
    return (per_run.groupby(["variant", "kind", "arm"], sort=False)
                   .agg(n_runs=("roc_auc", "size"),
                        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"))
                   .reset_index())


def main():
    ap = argparse.ArgumentParser(description="Hybrid models (plan p. 18-19).")
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", action="store_true", help="all three variants on every RUNS pair")
    ap.add_argument("--ablations", action="store_true",
                    help="the hybrid ablation arms on scaffold_det")
    ap.add_argument("--score", action="store_true", help="score hyb* C3 files with the shared scorer")
    ap.add_argument("--summary", action="store_true", help="ablation arms vs baselines -> summary.csv")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, help="override max_epochs (smoke tests only)")
    args = ap.parse_args()

    if args.score:
        main_tokens = {f.stem.split("__")[0] for f in PREDS.glob("hyb*__*.parquet")}
        ab_tokens = ({f.stem.split("__")[0] for f in (OUT / "preds").glob("*.parquet")}
                     if (OUT / "preds").exists() else set())
        n_main = len(score_preds(tokens=main_tokens)) if main_tokens else 0
        n_ab = len(score_preds(OUT / "preds", OUT / "metrics", "test", ab_tokens)) if ab_tokens else 0
        n_ab_val = (len(score_preds(OUT / "preds", OUT / "val_metrics", "val", ab_tokens))
                    if ab_tokens else 0)
        print(f"scored {n_main} main (test) + {n_ab} ablation (test) + {n_ab_val} ablation (val) files")
    elif args.summary:
        s = summarize()
        s.to_csv(OUT / "summary.csv", index=False)
        print(s.round(3).to_string(index=False))
    elif args.ablations:
        run_ablations(force=args.force, max_epochs=args.epochs)
    elif args.grid:
        cfg = load_config({"max_epochs": args.epochs})
        for variant in VARIANTS:
            for split, seed in RUNS:
                if not args.force and (PREDS / f"{variant}__{split}__seed{seed}.parquet").exists():
                    continue
                RUNNERS[variant](split, seed, cfg)
    else:
        if not (args.variant and args.split):
            ap.error("--variant and --split are required unless --grid is given")
        RUNNERS[args.variant](args.split, args.seed,
                              load_config({"max_epochs": args.epochs}))


if __name__ == "__main__":
    main()
