"""Pytest process-level settings.

macOS / OpenMP: torch and xgboost both bring an OpenMP runtime, and they crash each other
when loaded in the wrong order into one process. Empirically (torch 2.8, xgboost 3.2):
importing torch first and using xgboost afterwards is safe, while importing xgboost first
segfaults the first heavier torch op. The default alphabetical test collection already
imports torch (the GNN modules) before xgboost (test_models_aml), which is why the plain
`pytest` run works; conftest keeps that order stable on purpose. tox.train_hybrid follows
the same torch-first rule (see its import-order guard).
"""
