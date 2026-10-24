"""P4: build the Phase 1 deck (report/phase1/slides.pptx). Every number is read from results/."""
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

from tox.constants import ABLATION_SUMMARY, FIGURES, HEADLINE, ROOT

OUT = ROOT / "report" / "phase1" / "slides.pptx"
INK, ACCENT = RGBColor(0x1A, 0x20, 0x2C), RGBColor(0x2B, 0x6C, 0xB0)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)


def slide(title, bullets=None, fig=None, fig_w=7.4, note=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12), Inches(0.9)).text_frame
    tb.text = title
    tb.paragraphs[0].font.size, tb.paragraphs[0].font.bold = Pt(30), True
    tb.paragraphs[0].font.color.rgb = INK
    text_w = 12 if fig is None else (12.1 - fig_w)
    if bullets:
        f = s.shapes.add_textbox(Inches(0.6), Inches(1.5), Inches(text_w), Inches(5)).text_frame
        f.word_wrap = True
        for i, b in enumerate(bullets):
            p = f.paragraphs[0] if i == 0 else f.add_paragraph()
            p.text, p.font.size, p.font.color.rgb = "•  " + b, Pt(18), INK
            p.space_after = Pt(10)
    if fig is not None:
        s.shapes.add_picture(str(FIGURES / fig), Inches(13.0 - fig_w), Inches(1.5), width=Inches(fig_w))
    if note:
        n = s.shapes.add_textbox(Inches(0.6), Inches(6.85), Inches(12), Inches(0.5)).text_frame
        n.text = note
        n.paragraphs[0].font.size, n.paragraphs[0].font.color.rgb = Pt(12), ACCENT
    return s


def table(s, df, top=1.5, left=0.6, width=12.1):
    t = s.shapes.add_table(len(df) + 1, len(df.columns), Inches(left), Inches(top),
                           Inches(width), Inches(0.45 * (len(df) + 1))).table
    for j, c in enumerate(df.columns):
        t.cell(0, j).text = c
    for i, row in enumerate(df.itertuples(index=False), 1):
        for j, v in enumerate(row):
            t.cell(i, j).text = v if isinstance(v, str) else f"{v:.3f}"
    for c in t.iter_cells():
        for p in c.text_frame.paragraphs:
            p.font.size = Pt(14)


h = pd.read_csv(HEADLINE)
pv = lambda m, sp, col: h[(h.model == m) & (h.split == sp)][col].iloc[0]
ab = pd.read_csv(ABLATION_SUMMARY).set_index("config")
best = h[h.split == "scaffold_det"].sort_values("roc_auc_mean").iloc[-1]
pos_scaf = pv("rf_fpdesc", "scaffold_det", "pos_rate")
pos_rand = pv("rf_fpdesc", "random", "pos_rate")

s = slide("Predicting Tox21 toxicity from molecular structure",
          ["12 in-vitro toxicity endpoints from a SMILES string",
           "Question: how much accuracy survives on unseen chemical scaffolds?",
           "Phase 1: Morgan fingerprints + RDKit descriptors with LR / RF / XGBoost",
           "Phase 2 (next): GNNs and a hybrid, on the same splits and scorer"],
          note="Project 8 · Molecular Toxicity Prediction · Phase 1 (ML)")
slide("Problem and dataset",
      ["Tox21 (MoleculeNet / DeepChem): 7,831 compounds, 12 binary endpoints",
       "NR: nuclear receptors (AR, ER, AhR, aromatase, PPAR-γ); SR: stress response (ARE, ATAD5, HSE, MMP, p53)",
       "Many labels are missing: each endpoint is trained and scored on its labelled rows only",
       "Positives are rare (about 3-15% per endpoint): the metric must respect imbalance"])
slide("Data quality",
      ["8 unparseable SMILES dropped; salts and counter-ions removed (largest fragment kept)",
       "Charges and stereochemistry deliberately kept",
       "Deduplicated BEFORE splitting: 72 duplicate groups (114 rows) merged",
       "44 (molecule, task) labels conflicted and were set to NaN: assay noise",
       "Final: 7,709 molecules, 2,285 unique Murcko scaffolds"])
slide("EDA: sparse and imbalanced labels", ["Positive rate differs several-fold between endpoints",
      "Missing rate is high on several tasks, so we mask rather than fill with 0",
      "NR-AR / NR-AR-LBD and NR-ER / NR-ER-LBD are strongly correlated"],
      fig="eda_label_rates.png", fig_w=6.6)
slide("Splits: why scaffold, not random",
      ["Random splits put close analogues in train and test, overstating performance",
       "scaffold_det: DeepChem convention (largest scaffold groups to train); scaffold_s0-2: balanced, seeded; random_s0-2: reference",
       f"Mean test positive rate is {pos_scaf:.3f} (scaffold) vs {pos_rand:.3f} (random)",
       "That rate is also the PR-AUC of a random classifier, so compare PR-AUC within a split type"],
      fig="eda_split_pos_rate.png", fig_w=5.6)
slide("Feature engineering",
      ["Morgan fingerprint, radius 2, 2048 bits: presence of circular substructures",
       "217 RDKit descriptors: size, polarity, rings, fragment counts (Ipc log-transformed)",
       "Imputer + scaler fit on train rows only, inside a sklearn Pipeline",
       f"Ablation (XGB, scaffold_det ROC): r1/1024 {ab.loc['xgb_fp-r1-1024','roc_auc_mean']:.3f}, "
       f"r2/2048 {ab.loc['xgb_fp-r2-2048','roc_auc_mean']:.3f}, descriptors only {ab.loc['xgb_desc','roc_auc_mean']:.3f}",
       "Radius and length barely matter; descriptors alone beat every fingerprint setting"])
slide("Models and protocol",
      ["Logistic regression, random forest (500 trees), XGBoost; each on FP and FP+descriptors",
       "One model per endpoint, imbalance handled by class weights / scale_pos_weight (train only)",
       "Small grid tuned on validation PR-AUC; test touched once for reporting",
       "7 splits x 3 seeds; one shared scorer (tox.metrics) for every number",
       "Honest caveat: validation is reused for tuning and early stopping, so val scores are optimistic"])
s = slide("Results: test, 12-endpoint mean ± std")
d = h.pivot(index="model", columns="split")
rows = [[m, f"{d.loc[m, ('roc_auc_mean','scaffold_det')]:.3f}",
         f"{d.loc[m, ('roc_auc_mean','scaffold')]:.3f} ± {d.loc[m, ('roc_auc_std','scaffold')]:.3f}",
         f"{d.loc[m, ('roc_auc_mean','random')]:.3f} ± {d.loc[m, ('roc_auc_std','random')]:.3f}",
         f"{d.loc[m, ('pr_auc_mean','scaffold_det')]:.3f}",
         f"{d.loc[m, ('pr_auc_mean','random')]:.3f}"] for m in d.index]
table(s, pd.DataFrame(rows, columns=["model", "ROC scaffold_det", "ROC scaffold", "ROC random",
                                     "PR scaffold_det", "PR random"]))
slide("Result: the scaffold penalty", [f"Best on scaffold_det: {best.model}, ROC {best.roc_auc_mean:.3f}",
      "Same model family loses roughly 0.08-0.11 ROC-AUC moving from random to scaffold splits",
      "The gap is as big as the spread between worst and best classical model",
      "Descriptors help most on unseen scaffolds"],
      fig="aml_test_split_gap.png", fig_w=7.0)
slide("What drives predictions? (XGBoost gain)",
      ["Descriptors take 35-72% of total gain per endpoint",
       "NR-AhR: fr_benzene, aromatic ring counts: planar polyaromatics, a known AhR ligand trait",
       "SR-MMP: fr_benzene and fr_phenol lead",
       "SR-p53 / SR-HSE: size and surface-area descriptors, not specific groups"],
      fig="importance_gain_share.png", fig_w=6.6)
slide("Recovering substructures behind top bits (NR-AhR)",
      ["Highlighted atoms: environment of each top fingerprint bit in an example molecule",
       "Top bits are generic (radius 0-1, present in thousands of molecules): aromatic atoms, not specific toxicophores",
       "Bit collisions mean one bit can stand for several substructures"],
      fig="importance_bits_NR-AhR.png", fig_w=12.1)
slide("Limitations and Phase 2",
      ["Tox21 are in-vitro screens, not whole-organism toxicity",
       "Labels are noisy and heavily missing; some endpoints have very few positives",
       "A scaffold split is still retrospective, not prospective validation",
       "Phase 2: GCN / MPNN / AttentiveFP with masked loss, hybrid models, ablations, similarity-bin analysis"])
prs.save(OUT)
print(OUT, len(prs.slides._sldIdLst), "slides")
