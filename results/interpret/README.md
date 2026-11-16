# results/interpret

Per-atom attributions for the graph model (plan P3), written by `python -m tox.analysis.interpret_gnn`.

One CSV per (model, split, seed, endpoint) listing the molecules explained and their three highest-weighted atoms; the maps themselves are `results/figures/interpret_*.png`.

A weight is the GNNExplainer node mask: how much the model's logit for that endpoint relies on that atom, for that molecule. It describes the model, not the chemistry.
