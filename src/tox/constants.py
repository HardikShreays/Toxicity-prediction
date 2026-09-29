"""Shared constants. Every path in the project comes from here, never hardcode one."""
from pathlib import Path

TASKS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
         "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
SEEDS = [0, 1, 2]

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"

DATA = ROOT / "data"
RAW_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
RAW_CSV = DATA / "raw" / "tox21.csv.gz"
PROCESSED = DATA / "processed"
CLEAN = PROCESSED / "tox21_clean.parquet"            # contract C1
DROPPED = PROCESSED / "dropped.csv"
CONFLICTS = PROCESSED / "conflicts.csv"
FP_CACHE = PROCESSED / "fp_morgan2048.npy"
DESC_CACHE = PROCESSED / "desc_rdkit.npy"
SPLITS = DATA / "splits"                             # contract C2

RESULTS = ROOT / "results"
PREDS = RESULTS / "preds"                            # contract C3
METRICS = RESULTS / "metrics"                        # contract C4
FIGURES = RESULTS / "figures"
AML_PARAMS = RESULTS / "aml_params.json"
AML_IMPORTANCE = RESULTS / "aml_importance"
SUMMARY = RESULTS / "summary.csv"
HEADLINE = RESULTS / "headline.csv"

SPLIT_NAMES = ["scaffold_det"] + [f"scaffold_s{s}" for s in SEEDS] + [f"random_s{s}" for s in SEEDS]
ABLATIONS = RESULTS / "ablations"
