"""A2 - the whole AML grid from one command:
6 model-feature configs x 12 endpoints x 7 splits (scaffold_det run with all 3 seeds,
the seeded splits with their own seed). Writes C3 prediction files for val + test rows.

python -m tox.train_aml                     # everything (skips files that already exist)
python -m tox.train_aml --models xgb_fp --force
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from tox.constants import AML_IMPORTANCE, AML_PARAMS, CLEAN, PREDS, SEEDS, TASKS
from tox.featurize_fp import DESC_NAMES, load_features
from tox.models_aml import desc_pipeline, grid, make_model
from tox.splits import load

MODELS = [f"{a}_{f}" for a in ("lr", "rf", "xgb") for f in ("fp", "fpdesc")]
RUNS = ([("scaffold_det", s) for s in SEEDS]
        + [(f"scaffold_s{s}", s) for s in SEEDS]
        + [(f"random_s{s}", s) for s in SEEDS])


def build_X(feats, fp, desc, idx):
    """Feature matrices for train/val/test. Descriptor preprocessing is fit on train rows only."""
    X = {k: fp[v].astype(np.float32) for k, v in idx.items()}
    names = [f"bit{i}" for i in range(fp.shape[1])]
    if feats == "fpdesc":
        pipe = desc_pipeline().fit(desc[idx["train"]])
        for k, v in idx.items():
            X[k] = np.hstack([X[k], pipe.transform(desc[v]).astype(np.float32)])
        names += list(pipe.get_feature_names_out(DESC_NAMES))
    return X, names


def fit_task(algo, X, y, seed):
    """Fit one endpoint: try the grid, keep the params with the best val PR-AUC."""
    mtr, mva = ~np.isnan(y["train"]), ~np.isnan(y["val"])
    Xtr, ytr = X["train"][mtr], y["train"][mtr]
    Xva, yva = X["val"][mva], y["val"][mva]
    val_ok = len(np.unique(yva)) == 2          # val needs both classes for early stopping / selection
    best = None
    for params in grid(algo):
        model = make_model(algo, params, seed, ytr, early_stop=val_ok)
        if algo == "xgb" and val_ok:
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        else:
            model.fit(Xtr, ytr)
        ap = average_precision_score(yva, model.predict_proba(Xva)[:, 1]) if val_ok else np.nan
        if best is None or ap > best[1]:
            best = (model, ap, params)
        if not val_ok:
            break                                   # nothing to select on -> first grid point, logged
    model, ap, params = best
    info = {"params": params, "val_pr_auc": None if np.isnan(ap) else round(float(ap), 4),
            "val_has_both_classes": val_ok}
    if algo == "xgb" and val_ok:
        info["best_iteration"] = int(model.best_iteration)
    return model, info


def run(model_name, split, seed, df, fp, desc, log):
    algo, feats = model_name.split("_")
    s = load(split)
    pos = pd.Series(range(len(df)), index=df.smiles)
    idx = {k: pos[s[k]].to_numpy() for k in ("train", "val", "test")}
    X, names = build_X(feats, fp, desc, idx)
    Y = df[TASKS].to_numpy(dtype=float)
    y_all = {k: Y[v] for k, v in idx.items()}

    probs = {k: np.zeros((len(idx[k]), len(TASKS))) for k in ("val", "test")}
    run_log, importance = {}, []
    for j, t in enumerate(TASKS):
        model, info = fit_task(algo, X, {k: v[:, j] for k, v in y_all.items()}, seed)
        for k in probs:
            probs[k][:, j] = model.predict_proba(X[k])[:, 1]
        run_log[t] = info
        if algo == "xgb":
            gain = model.get_booster().get_score(importance_type="gain")
            importance += [{"task": t, "feature": names[int(f[1:])], "gain": g} for f, g in gain.items()]

    out = pd.concat([pd.DataFrame({"smiles": s[k], "role": k, **dict(zip(TASKS, probs[k].T))})
                     for k in ("val", "test")], ignore_index=True)
    out.to_parquet(PREDS / f"{model_name}__{split}__seed{seed}.parquet", index=False)
    log.setdefault(model_name, {})[f"{split}__seed{seed}"] = run_log
    if algo == "xgb" and split == "scaffold_det" and seed == 0:
        AML_IMPORTANCE.mkdir(parents=True, exist_ok=True)
        (pd.DataFrame(importance).sort_values(["task", "gain"], ascending=[True, False])
         .to_csv(AML_IMPORTANCE / f"{model_name}__scaffold_det.csv", index=False))
    mean_ap = np.nanmean([v["val_pr_auc"] if v["val_pr_auc"] is not None else np.nan
                          for v in run_log.values()])
    print(f"{model_name:11s} {split:13s} seed{seed}  val mean PR-AUC {mean_ap:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--force", action="store_true", help="redo runs whose prediction file exists")
    args = ap.parse_args()

    df = pd.read_parquet(CLEAN)
    fp, desc = load_features()
    PREDS.mkdir(parents=True, exist_ok=True)
    log = json.loads(AML_PARAMS.read_text()) if AML_PARAMS.exists() else {}
    for m in args.models:
        for split, seed in RUNS:
            if not args.force and (PREDS / f"{m}__{split}__seed{seed}.parquet").exists():
                continue
            run(m, split, seed, df, fp, desc, log)
            AML_PARAMS.write_text(json.dumps(log, indent=1))   # save after every run (resumable)


if __name__ == "__main__":
    main()
