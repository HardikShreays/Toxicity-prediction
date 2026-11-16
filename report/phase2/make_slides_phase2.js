// Phase 2 deck: Tox21 toxicity prediction with GNNs and hybrids.
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/Users/arjunsingh/.claude/skills/synced/c99dba6e-3172-4286-94b7-94da8b4b6a5e_4afbc227-6ddd-4559-8cd4-6abafa787524/pptx/scripts/apply_theme.js");

const ROOT = "/Users/arjunsingh/Desktop/Toxicity-prediction-main1";
const FIG = ROOT + "/results/figures/";
const RFIG = ROOT + "/report/phase2/figures/";
const OUT = ROOT + "/report/phase2/report_phase2_slides.pptx";

const THEME = {
  name: "Tox21 Phase 2",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "10292B", // ink
    lt1: "FFFFFF",
    dk2: "014A52", // deep teal, headings
    lt2: "EAF3F3", // tint surface
    accent1: "028090", // teal (dominant)
    accent2: "00A896", // seafoam
    accent3: "02C39A", // mint
    accent4: "C75B39", // terracotta, for negative results
    accent5: "5F7E80", // muted slate
    accent6: "D9A404", // gold highlight
    hlink: "028090",
    folHlink: "5F7E80",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Tox21 Toxicity Prediction Project";
pres.title = "Report Phase 2 - Molecular Toxicity Prediction";
const C = pres.SchemeColor;

const FOOTER = "Tox21 toxicity prediction  |  Phase 2";

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: THEME.colors.dk2 },
  objects: [
    { placeholder: { options: { name: "eyebrow", type: "body", x: 0.9, y: 2.25, w: 11.5, h: 0.45,
        fontSize: 15, color: C.accent3, bold: true, charSpacing: 3, valign: "bottom" }, text: " " } },
    { placeholder: { options: { name: "title", type: "title", x: 0.9, y: 2.75, w: 11.5, h: 1.9,
        fontSize: 40, bold: true, color: C.background1, align: "left", valign: "top" }, text: " " } },
    { placeholder: { options: { name: "body", type: "body", x: 0.9, y: 4.75, w: 11.5, h: 1.5,
        fontSize: 16, color: C.accent3, align: "left", valign: "top" }, text: " " } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.42, w: 12.1, h: 0.8,
        fontSize: 30, bold: true, color: C.text2, align: "left", valign: "middle" }, text: " " } },
    { text: { text: FOOTER, options: { x: 0.6, y: 6.92, w: 8, h: 0.3, fontSize: 10,
        color: C.accent5, isTextBox: true, margin: 0 } } },
  ],
  slideNumber: { x: 12.5, y: 6.92, w: 0.4, h: 0.3, fontSize: 10, color: THEME.colors.accent5, align: "right" },
});

pres.defineSlideMaster({
  title: "CLOSING",
  background: { color: THEME.colors.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.9, y: 0.6, w: 11.5, h: 0.9,
        fontSize: 30, bold: true, color: C.background1, align: "left", valign: "middle" }, text: " " } },
  ],
});

// ---------------------------------------------------------------- helpers
const card = (s, { x, y, w, h, fill = C.background2, name }) =>
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: fill }, objectName: name,
  });

const circleNum = (s, { x, y, n, color = C.accent1, d = 0.46 }) => {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color }, line: { color } });
  s.addText(String(n), { x, y, w: d, h: d, fontSize: 14, bold: true, color: C.background1,
    align: "center", valign: "middle", margin: 0, isTextBox: true });
};

const stat = (s, { x, y, w, value, label, color = C.accent1, size = 40 }) => {
  s.addText(value, { x, y, w, h: 0.8, fontSize: size, bold: true, color, align: "left",
    valign: "bottom", margin: 0, isTextBox: true, fontFace: THEME.headFontFace });
  s.addText(label, { x, y: y + 0.82, w, h: 0.6, fontSize: 12, color: C.text1, align: "left",
    valign: "top", margin: 0, isTextBox: true });
};

const bullets = (s, items, o) =>
  s.addText(
    items.map((t, i) => ({
      text: t,
      options: { bullet: { indent: 14 }, breakLine: i < items.length - 1, paraSpaceAfter: 10 },
    })),
    Object.assign({ fontSize: 15, color: C.text1, isTextBox: true, valign: "top" }, o)
  );

// ================================================================ 1. Title
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Opening" });
s.addText("PROJECT REPORT  —  PHASE 2", { placeholder: "eyebrow" });
s.addText("Molecular toxicity prediction with graph neural networks and hybrid models",
  { placeholder: "title" });
s.addText("A controlled comparison against fingerprint baselines on the 12 Tox21 endpoints",
  { placeholder: "body" });
s.addText("Hardik Shreyas  ·  Saman  ·  Arjun  ·  Bhavya  ·  Priyansh",
  { x: 0.9, y: 6.1, w: 11.5, h: 0.4, fontSize: 14, color: C.accent3, isTextBox: true, margin: 0 });
s.addNotes("Phase 1 built the classical pipeline. Phase 2 asks whether learning from the molecular graph beats engineered features, and whether combining the two helps. Everything is measured on the same splits with the same scorer.");

// ================================================================ 2. Problem
pres.addSection({ title: "Setup" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Setup" });
s.addText("Twelve assays, sparse labels, and a split that decides the answer", { placeholder: "title" });
bullets(s, [
  "Predict 12 Tox21 in-vitro toxicity endpoints from a SMILES string alone",
  "Positives are rare (2.9%–15.9%) and 17% of the label matrix is missing — because a compound was not tested, not because it was inactive",
  "An unmeasured label must never be read as a negative: the loss and the scorer both mask it",
  "A random split puts close analogues of every test molecule in training; a scaffold split holds out whole molecular frameworks",
  "The Phase 1 baseline to beat: rf_fpdesc at 0.839 mean test ROC-AUC on the balanced scaffold splits",
], { x: 0.6, y: 1.5, w: 6.5, h: 3.9, fontSize: 15.5 });

[["7,709", "unique molecules\nafter dedup", 7.45], ["12", "binary toxicity\nendpoints", 9.35],
 ["17%", "of labels\nmissing", 11.1]].forEach(([v, l, x]) =>
  stat(s, { x, y: 1.6, w: 1.8, value: v, label: l, size: 36 }));
card(s, { x: 7.35, y: 3.6, w: 5.4, h: 1.75, name: "callout" });
s.addText([
  { text: "The honest question\n", options: { bold: true, fontSize: 15, color: C.text2, breakLine: true } },
  { text: "Not “can a GNN learn chemistry?” but “does it beat a well-tuned fingerprint model on chemistry it has never seen?”", options: { fontSize: 14, color: C.text1 } },
], { x: 7.65, y: 3.82, w: 4.8, h: 1.4, isTextBox: true, valign: "top", margin: 0 });
card(s, { x: 0.6, y: 5.6, w: 12.13, h: 1.0, name: "s2band" });
s.addText([
  { text: "Phase 1 gave the baseline.  ", options: { bold: true, fontSize: 15, color: C.text2 } },
  { text: "Phase 2 asks whether learning the representation from the molecular graph beats it, and whether combining the two representations beats either one.", options: { fontSize: 14, color: C.text1 } },
], { x: 0.9, y: 5.78, w: 11.5, h: 0.7, isTextBox: true, valign: "middle", margin: 0 });
s.addNotes("Two structural difficulties: extreme imbalance plus non-random missingness, and the fact that the evaluation split dominates apparent performance. Everything in the deck is reported on scaffold splits unless stated.");

// ================================================================ 3. Protocol
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Setup" });
s.addText("One pipeline, seven shared splits, a single scorer", { placeholder: "title" });
const steps = [
  ["Clean & dedup", "Largest fragment, canonical SMILES, conflicting duplicate labels → missing"],
  ["7 shared splits", "1 deterministic scaffold, 3 randomized scaffold, 3 random"],
  ["Two representations", "Morgan fingerprint + RDKit descriptors, and atom–bond graphs"],
  ["12 models", "6 classical, 3 graph networks, 3 hybrids — 9 runs each"],
  ["One masked scorer", "Endpoint-wise ROC-AUC and PR-AUC, missing labels dropped"],
];
steps.forEach(([h, d], i) => {
  const x = 0.6 + i * 2.48;
  card(s, { x, y: 1.75, w: 2.26, h: 2.3, name: "step" + i });
  circleNum(s, { x: x + 0.22, y: 1.97, n: i + 1 });
  s.addText(h, { x: x + 0.22, y: 2.55, w: 1.85, h: 0.5, fontSize: 14, bold: true, color: C.text2,
    isTextBox: true, margin: 0, valign: "top" });
  s.addText(d, { x: x + 0.22, y: 3.02, w: 1.85, h: 1.1, fontSize: 11, color: C.text1,
    isTextBox: true, margin: 0, valign: "top" });
  if (i < steps.length - 1)
    s.addShape(pres.ShapeType.rightArrow, { x: x + 2.3, y: 2.85, w: 0.14, h: 0.22,
      fill: { color: THEME.colors.accent5 }, line: { color: THEME.colors.accent5 } });
});
card(s, { x: 0.6, y: 4.6, w: 12.13, h: 1.5, name: "rules" });
s.addText([
  { text: "Two rules that keep the comparison honest\n", options: { bold: true, fontSize: 15, color: C.text2, breakLine: true } },
  { text: "Model selection — hyper-parameters, early stopping and the reported configuration — uses validation data only; test labels are read once per run, after training.  A graph encoder reused inside a hybrid always carries weights trained on that same split's training rows, so no hybrid sees data its parent did not.", options: { fontSize: 14, color: C.text1 } },
], { x: 0.9, y: 4.8, w: 11.5, h: 1.2, isTextBox: true, valign: "top", margin: 0 });
s.addNotes("The point of the shared pipeline is that any difference between model families is attributable to the model, not to preprocessing, splits or scoring.");

// ================================================================ 4. Graphs + loss
pres.addSection({ title: "Method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("A molecule is a graph; a missing label is a hole in the loss", { placeholder: "title" });
card(s, { x: 0.6, y: 1.45, w: 5.95, h: 2.75, name: "graphcard" });
s.addText("The graph encoding", { x: 0.9, y: 1.65, w: 5.3, h: 0.35, fontSize: 15, bold: true,
  color: C.text2, isTextBox: true, margin: 0 });
bullets(s, [
  "Nodes = heavy atoms (36 binary features), edges = bonds (6 features)",
  "Each bond stored as two directed edges, so messages flow both ways",
  "After L layers an atom sees every atom within L bonds",
  "Readout: mean ‖ max over atoms → one molecule vector → 12 logits",
], { x: 0.9, y: 2.08, w: 5.4, h: 2.0, fontSize: 12.5 });

card(s, { x: 6.78, y: 1.45, w: 5.95, h: 2.75, name: "losscard" });
s.addText("The masked multi-task loss", { x: 7.08, y: 1.65, w: 5.3, h: 0.35, fontSize: 15, bold: true,
  color: C.text2, isTextBox: true, margin: 0 });
s.addText("∂L / ∂zᵢⱼ  =  (mᵢⱼ / Σm) · [ (1 − yᵢⱼ) pᵢⱼ  −  wⱼ yᵢⱼ (1 − pᵢⱼ) ]",
  { x: 7.08, y: 2.08, w: 5.35, h: 0.5, fontSize: 15, color: C.accent1, bold: true,
    isTextBox: true, margin: 0, fontFace: "Cambria" });
bullets(s, [
  "The gradient is exactly zero wherever the mask mᵢⱼ = 0 — a missing label contributes nothing",
  "Filling missing entries with 0 would assert that “unmeasured” means “non-toxic”",
  "Positive weight wⱼ = min(n⁻/n⁺, 30), computed on training labels only",
], { x: 7.08, y: 2.64, w: 5.4, h: 1.5, fontSize: 12.5 });

[["GCN", "Degree-normalised neighbour average. No bond features at all — the control.", C.accent5],
 ["MPNN (GINE)", "Bonds enter each message; sums neighbours, so it separates neighbourhoods a mean cannot.", C.accent1],
 ["AttentiveFP", "Learned attention over atoms and a learned readout — the strongest published graph baseline.", C.accent2],
].forEach(([h, d, col], i) => {
  const x = 0.6 + i * 4.09;
  card(s, { x, y: 4.45, w: 3.85, h: 1.6, name: "arch" + i });
  s.addShape(pres.ShapeType.ellipse, { x: x + 0.3, y: 4.72, w: 0.22, h: 0.22, fill: { color: col }, line: { color: col } });
  s.addText(h, { x: x + 0.62, y: 4.67, w: 3, h: 0.35, fontSize: 15, bold: true, color: C.text2,
    isTextBox: true, margin: 0 });
  s.addText(d, { x: x + 0.3, y: 5.12, w: 3.3, h: 1.0, fontSize: 12.5, color: C.text1,
    isTextBox: true, margin: 0, valign: "top" });
});
s.addNotes("The three architectures isolate one design axis each: no bonds, bond-aware summation, learned attention. The loss derivation is the part the rubric asks for — the mask zeroes the gradient exactly, it is not an approximation.");

// ================================================================ 5. GNN results chart
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Graph networks beat fingerprints alone — but not rf_fpdesc", { placeholder: "title" });
s.addChart(pres.ChartType.bar, [{
  name: "scaffold ROC-AUC",
  labels: ["xgb_fp", "lr_fp", "rf_fp", "mpnn", "gcn", "afp", "rf_fpdesc"],
  values: [0.767, 0.775, 0.803, 0.813, 0.826, 0.828, 0.839],
}], {
  x: 0.6, y: 1.5, w: 7.5, h: 4.6,
  barDir: "bar", barGapWidthPct: 45,
  chartColors: [THEME.colors.accent5, THEME.colors.accent5, THEME.colors.accent5,
    THEME.colors.accent1, THEME.colors.accent1, THEME.colors.accent1, THEME.colors.accent6],
  showTitle: true, title: "Test ROC-AUC, 12-endpoint mean, balanced scaffold splits",
  titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: THEME.colors.dk2,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
  dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: THEME.colors.dk1,
  valAxisMinVal: 0.7, valAxisMaxVal: 0.88,
  catAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, catAxisLabelColor: THEME.colors.dk1,
  valAxisLabelFontFace: "+mn-lt", valAxisLabelFontSize: 10, valAxisLabelColor: THEME.colors.accent5,
  valGridLine: { color: "DCE6E6", size: 1 }, catGridLine: { style: "none" },
  showLegend: false, objectName: "gnn-bar",
});
s.addText("What the numbers say", { x: 8.45, y: 1.65, w: 4.3, h: 0.35, fontSize: 15, bold: true,
  color: C.text2, isTextBox: true, margin: 0 });
bullets(s, [
  "All three graph models beat every fingerprint-only model, on every split family",
  "None of them beats rf_fpdesc — fingerprints plus descriptors stay ahead (Wilcoxon p = 0.02–0.03)",
  "The ordering inside the graph family is not the one the design predicts: bond-aware MPNN is last",
  "Every model loses 0.02–0.03 ROC-AUC moving from random to scaffold splits",
], { x: 8.45, y: 2.1, w: 4.3, h: 3.4, fontSize: 13 });
s.addNotes("This reproduces Yang et al. (2019) on our data: well-tuned fingerprint baselines remain competitive. Note that the deterministic scaffold split is harder still — all numbers drop by about 0.06 there.");

// ================================================================ 6. Ablations
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("The MPNN ablations point at the tuning budget, not the graph", { placeholder: "title" });
s.addText("Five one-factor ablations on scaffold_det, 3 seeds each. Every arm but the 5-layer one scores at or above the frozen baseline (ROC-AUC 0.732).",
  { x: 0.6, y: 1.3, w: 12.1, h: 0.4, fontSize: 14, color: C.text1, isTextBox: true, margin: 0 });
const ab = [
  ["Depth", "2 layers 0.747  →  5 layers 0.731", "Monotone decline with depth — consistent with over-smoothing, not proof of it", C.accent4],
  ["Bond features", "zeroed: 0.743  vs  0.732", "The feature that distinguishes MPNN from GCN does not pay for itself here", C.accent4],
  ["Readout", "mean 0.743 · max 0.743", "Both beat the mean‖max concatenation the model was built on", C.accent4],
  ["Positive weights", "switched off: 0.746 / PR 0.345", "Class weighting was meant to help rare endpoints; measured, it hurts", C.accent4],
  ["Multi-task", "12 single-task models: 0.747", "Shared representation unsupported for mpnn — at 12× the compute", C.accent4],
];
ab.forEach(([h, v, d, col], i) => {
  const y = 1.85 + i * 0.92;
  card(s, { x: 0.6, y, w: 12.13, h: 0.8, name: "ab" + i });
  s.addShape(pres.ShapeType.ellipse, { x: 0.85, y: y + 0.29, w: 0.22, h: 0.22, fill: { color: col }, line: { color: col } });
  s.addText(h, { x: 1.2, y: y + 0.16, w: 2.1, h: 0.5, fontSize: 14, bold: true, color: C.text2,
    isTextBox: true, margin: 0, valign: "middle" });
  s.addText(v, { x: 3.3, y: y + 0.16, w: 3.2, h: 0.5, fontSize: 13.5, bold: true, color: C.accent1,
    isTextBox: true, margin: 0, valign: "middle" });
  s.addText(d, { x: 6.6, y: y + 0.16, w: 5.9, h: 0.5, fontSize: 12.5, color: C.text1,
    isTextBox: true, margin: 0, valign: "middle" });
});
s.addText("Reading: the frozen configuration — chosen on a single seed by validation PR-AUC — is demonstrably not the best MPNN within reach. Conclusions stated for mpnn only.",
  { x: 0.6, y: 6.5, w: 12.1, h: 0.35, fontSize: 12, italic: true, color: C.accent5, isTextBox: true, margin: 0 });
s.addNotes("This is the slide to be honest about. We are not claiming over-smoothing; the layer trend is consistent with it but equally consistent with overfitting at this data size. The useful conclusion is about tuning budget.");

// ================================================================ 7. Hybrids
pres.addSection({ title: "Hybrids" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Hybrids" });
s.addText("Three ways to combine the parents, in increasing order of coupling", { placeholder: "title" });
const hy = [
  ["hyb_stack", "Loose", "A per-endpoint logistic regression on the validation probabilities of xgb_fpdesc and mpnn. Fitted on validation, never on test.", "0.836", C.accent3],
  ["hyb_embxgb", "Medium", "The frozen MPNN readout vector concatenated with the 2048-bit Morgan fingerprint; one XGBoost per endpoint on top.", "0.830", C.accent2],
  ["hyb_fusion", "Tight", "Fingerprint MLP branch and MPNN encoder branch concatenated under one head, trained end to end from a warm start.", "0.802", C.accent4],
];
hy.forEach(([n, coupling, d, score, col], i) => {
  const x = 0.6 + i * 4.09;
  card(s, { x, y: 1.5, w: 3.85, h: 3.55, name: "hyb" + i });
  s.addText(coupling.toUpperCase(), { x: x + 0.3, y: 1.72, w: 3.2, h: 0.3, fontSize: 11, bold: true,
    color: col, charSpacing: 2, isTextBox: true, margin: 0 });
  s.addText(n, { x: x + 0.3, y: 2.02, w: 3.2, h: 0.4, fontSize: 19, bold: true, color: C.text2,
    isTextBox: true, margin: 0, fontFace: THEME.headFontFace });
  s.addText(d, { x: x + 0.3, y: 2.52, w: 3.25, h: 1.6, fontSize: 12.5, color: C.text1,
    isTextBox: true, margin: 0, valign: "top" });
  s.addText(score, { x: x + 0.3, y: 3.85, w: 1.5, h: 0.6, fontSize: 30, bold: true, color: col,
    isTextBox: true, margin: 0, fontFace: THEME.headFontFace, valign: "bottom" });
  s.addText("scaffold ROC-AUC", { x: x + 0.3, y: 4.5, w: 3.2, h: 0.3, fontSize: 11, color: C.accent5,
    isTextBox: true, margin: 0 });
});
card(s, { x: 0.6, y: 5.35, w: 12.13, h: 1.05, name: "hybtake" });
s.addText([
  { text: "They order inversely to coupling.  ", options: { bold: true, fontSize: 15, color: C.text2 } },
  { text: "Tight coupling exposes the joint model to the weaker parent's failure modes; loose coupling lets each parent contribute only what it is good at.", options: { fontSize: 14, color: C.text1 } },
], { x: 0.9, y: 5.5, w: 11.5, h: 0.75, isTextBox: true, valign: "middle", margin: 0 });
s.addNotes("The three variants differ only in how tightly the parents are coupled — same parents, same splits, same scorer. That is what makes the ordering interpretable.");

// ================================================================ 8. Final comparison
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Hybrids" });
s.addText("Final comparison: stacking ties the baseline, fusion falls back", { placeholder: "title" });
s.addImage({ path: RFIG + "phase2_model_comparison.png", x: 0.6, y: 1.45, w: 7.9, h: 4.74 });
s.addText("Three findings", { x: 8.75, y: 1.6, w: 4.0, h: 0.35, fontSize: 15, bold: true,
  color: C.text2, isTextBox: true, margin: 0 });
bullets(s, [
  "hyb_stack beats both of its own parents (0.813, 0.813 → 0.836)",
  "It is the only model the paired Wilcoxon test cannot separate from rf_fpdesc (p = 0.47)",
  "hyb_embxgb is close behind and also not significantly behind the baseline (p = 0.15)",
  "hyb_fusion is below both parents — a negative result with a traceable cause",
], { x: 8.75, y: 2.05, w: 4.0, h: 3.6, fontSize: 13 });
s.addText("12 paired endpoint means · exploratory, no multiplicity correction",
  { x: 8.75, y: 5.75, w: 4.0, h: 0.35, fontSize: 11, italic: true, color: C.accent5, isTextBox: true, margin: 0 });
s.addNotes("Error bars are the standard deviation over the three runs of the split family. The stack's tie with rf_fpdesc is a tie, not a win — and it comes with a calibration cost we show in two slides.");

// ================================================================ 9. Why fusion failed
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Hybrids" });
s.addText("Why end-to-end fusion failed, and how the ablations prove it", { placeholder: "title" });
card(s, { x: 0.6, y: 1.45, w: 12.13, h: 1.45, name: "cause" });
s.addText([
  { text: "The cause\n", options: { bold: true, fontSize: 15, color: C.text2, breakLine: true } },
  { text: "Fine-tuning the whole 256-unit graph encoder at the GNN recipe's learning rate of 10⁻³ degrades the model within a few epochs. Early stopping then restores the epoch-0 or epoch-1 warm-start weights in 7 of 9 runs — so the “jointly trained” model is, in most runs, barely trained at all.", options: { fontSize: 14, color: C.text1 } },
], { x: 0.9, y: 1.62, w: 11.5, h: 1.2, isTextBox: true, valign: "top", margin: 0 });
s.addText("The ablations locate it", { x: 0.6, y: 3.2, w: 6, h: 0.35, fontSize: 15, bold: true,
  color: C.text2, isTextBox: true, margin: 0 });
[["0.748", "fusion\n(both branches)", C.accent5], ["0.744", "without the\nfingerprint branch", C.accent5],
 ["0.671", "without the\ngraph branch", C.accent4]].forEach(([v, l, col], i) =>
  stat(s, { x: 0.75 + i * 2.6, y: 3.75, w: 2.4, value: v, label: l, color: col, size: 42 }));
card(s, { x: 8.6, y: 3.6, w: 4.13, h: 2.1, name: "fusiontake" });
s.addText([
  { text: "Removing the fingerprint branch changes almost nothing; removing the graph branch collapses the model.\n\n", options: { fontSize: 13.5, color: C.text1, breakLine: true } },
  { text: "The joint head never learned what the fingerprints know.", options: { fontSize: 13.5, bold: true, color: C.text2 } },
], { x: 8.9, y: 3.8, w: 3.6, h: 1.7, isTextBox: true, valign: "top", margin: 0 });
s.addText("The fix is known and cheap: freeze the encoder for a warm-up, then fine-tune the joint model at 10⁻⁴. We report the negative result rather than tune until it wins, because the budget was spent on the comparison itself.",
  { x: 0.75, y: 5.95, w: 11.9, h: 0.6, fontSize: 13, italic: true, color: C.accent5, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("The fix is known and cheap: a lower learning rate with a frozen warm-up before joint fine-tuning. We report it as a negative result rather than tuning until it wins, because the budget was spent elsewhere.");

// ================================================================ 10. Where gains come from
pres.addSection({ title: "Validation" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Validation" });
s.addText("Where the stacking gain actually comes from", { placeholder: "title" });
s.addImage({ path: FIG + "hybrid_sim_bins.png", x: 0.6, y: 1.5, w: 7.6, h: 3.69 });
s.addText("We predicted hybrids would win on novel chemistry", { x: 8.45, y: 1.6, w: 4.3, h: 0.6,
  fontSize: 15, bold: true, color: C.text2, isTextBox: true, margin: 0, valign: "top" });
bullets(s, [
  "Test molecules are binned by their maximum Tanimoto similarity to the training set",
  "In the hardest bin [0, 0.3) — 172 genuinely novel molecules — hyb_stack (0.724) clearly beats its graph parent mpnn (0.645)",
  "But rf_fpdesc is stronger there still (0.751), and it also leads on the 4 sparsest endpoints",
], { x: 8.45, y: 2.3, w: 4.3, h: 2.9, fontSize: 13 });
card(s, { x: 0.6, y: 5.45, w: 12.13, h: 1.05, name: "simtake" });
s.addText([
  { text: "So the stated rationale is not confirmed.  ", options: { bold: true, fontSize: 15, color: C.text2 } },
  { text: "The stack's advantage over its parents is real, but it comes from averaging two different error profiles — not from a special affinity of hybrids for unseen scaffolds. That is a weaker claim, and it is the one the evidence supports.", options: { fontSize: 14, color: C.text1 } },
], { x: 0.9, y: 5.62, w: 11.5, h: 0.75, isTextBox: true, valign: "middle", margin: 0 });
s.addNotes("This is the ablation that tests our own hypothesis rather than the model. We stated the rationale in the plan, measured it, and it did not hold in the form we stated it.");

// ================================================================ 11. Calibration
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Validation" });
s.addText("Ranking well is not the same as predicting well", { placeholder: "title" });
s.addImage({ path: FIG + "calibration_rf_fpdesc.png", x: 1.25, y: 1.55, w: 3.5, h: 3.5 });
s.addImage({ path: FIG + "calibration_hyb_stack.png", x: 5.1, y: 1.55, w: 3.5, h: 3.5 });
s.addText("rf_fpdesc  —  ECE 0.025", { x: 1.25, y: 5.1, w: 3.5, h: 0.35, fontSize: 13, bold: true,
  color: C.accent1, align: "center", isTextBox: true, margin: 0 });
s.addText("hyb_stack  —  ECE 0.278", { x: 5.1, y: 5.1, w: 3.5, h: 0.35, fontSize: 13, bold: true,
  color: C.accent4, align: "center", isTextBox: true, margin: 0 });
card(s, { x: 9.0, y: 1.55, w: 3.73, h: 3.9, name: "calcard" });
s.addText("The caveat on our best hybrid", { x: 9.3, y: 1.78, w: 3.15, h: 0.6, fontSize: 15,
  bold: true, color: C.text2, isTextBox: true, margin: 0, valign: "top" });
bullets(s, [
  "Expected calibration error over 65,815 labelled (task, molecule) pairs",
  "rf_fpdesc is both the best ranker and by far the best calibrated",
  "hyb_stack ties it on ranking but has the worst ECE of all 12 models",
  "Its curve sits below the diagonal: it over-states toxicity at every level",
], { x: 9.3, y: 2.42, w: 3.15, h: 2.9, fontSize: 12.5 });
s.addText("Any deployment that thresholds the stack's output needs a calibration step — Platt scaling or isotonic regression on validation — which this phase does not perform.",
  { x: 0.6, y: 5.7, w: 8.0, h: 0.8, fontSize: 13, italic: true, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("A screening decision consumes a probability, not a ranking. This is the single most actionable caveat in the report.");

// ================================================================ 12. Interpretability
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Validation" });
s.addText("What the model looks at — and what it mistakes", { placeholder: "title" });
s.addImage({ path: FIG + "interpret_mpnn_SR-MMP_tp1.png", x: 1.25, y: 1.55, w: 3.5, h: 3.5 });
s.addImage({ path: FIG + "interpret_mpnn_SR-MMP_fp2.png", x: 5.1, y: 1.55, w: 3.5, h: 3.5 });
s.addText("true positive  —  p = 0.997", { x: 1.25, y: 5.1, w: 3.5, h: 0.35, fontSize: 13, bold: true,
  color: C.accent1, align: "center", isTextBox: true, margin: 0 });
s.addText("false positive  —  p = 0.989", { x: 5.1, y: 5.1, w: 3.5, h: 0.35, fontSize: 13, bold: true,
  color: C.accent4, align: "center", isTextBox: true, margin: 0 });
card(s, { x: 9.0, y: 1.55, w: 3.73, h: 3.9, name: "interpcard" });
s.addText("GNNExplainer on SR-MMP", { x: 9.3, y: 1.78, w: 3.15, h: 0.6, fontSize: 15,
  bold: true, color: C.text2, isTextBox: true, margin: 0, valign: "top" });
bullets(s, [
  "Atom masks for the 5 most confident true and false positives, mpnn / scaffold_det",
  "Right: it is right for a reason — attribution lands on the halogenated phenol of an uncoupler",
  "Wrong 1: polyphenols (galloyl, catechol) — the classic assay-interference chemotype",
  "Wrong 2: sterols and bare polyaromatics — the fused-ring rule generalised too far",
], { x: 9.3, y: 2.42, w: 3.15, h: 2.9, fontSize: 12.5 });
s.addText("A mask says the logit drops when that atom is hidden. It is evidence about this model on this molecule — a debugging instrument, not a toxicophore claim.",
  { x: 0.6, y: 5.7, w: 8.0, h: 0.8, fontSize: 13, italic: true, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("This is how the two failure groups were found, not a post-hoc story. Both groups say the same thing: the strongest feature the model learned for SR-MMP is coarse - extended fused rings and electron-rich phenols - good enough to rank, blunt enough to misfire on unseen chemistry.");

// ================================================================ 13. Conclusions
pres.addSection({ title: "Closing" });
s = pres.addSlide({ masterName: "CLOSING", sectionTitle: "Closing" });
s.addText("What Phase 2 establishes", { placeholder: "title" });
const concl = [
  ["Graph networks are competitive, not superior", "They beat every fingerprint-only model, and lose to fingerprints plus descriptors on every split family — at this data scale and tuning budget."],
  ["Coupling decides whether hybrids help", "Loose stacking ties the best baseline; tight end-to-end fusion underperforms both of its parents. The cheapest combination is the most effective one."],
  ["Discrimination and calibration come apart", "The model that ties on ROC-AUC is the worst calibrated of the twelve. Both have to be reported."],
];
concl.forEach(([h, d], i) => {
  const y = 1.75 + i * 1.32;
  circleNum(s, { x: 0.9, y: y + 0.06, n: i + 1, color: THEME.colors.accent3, d: 0.42 });
  s.addText(h, { x: 1.5, y, w: 10.9, h: 0.42, fontSize: 18, bold: true, color: C.background1,
    isTextBox: true, margin: 0, valign: "middle", fontFace: THEME.headFontFace });
  s.addText(d, { x: 1.5, y: y + 0.46, w: 10.6, h: 0.72, fontSize: 14, color: THEME.colors.lt2,
    isTextBox: true, margin: 0, valign: "top" });
});
s.addText("Next", { x: 0.9, y: 5.75, w: 2, h: 0.35, fontSize: 13, bold: true, color: THEME.colors.accent3,
  charSpacing: 2, isTextBox: true, margin: 0 });
s.addText("Multi-seed tuning for the graph models  ·  post-hoc calibration of the stack  ·  frozen warm-up plus a lower learning rate for fusion  ·  external or prospective validation",
  { x: 0.9, y: 6.12, w: 11.5, h: 0.7, fontSize: 14, color: C.background1, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("Close on the measured claim: graph-learned representations are useful complements to engineered chemical features, not automatic replacements for them.");

// ---------------------------------------------------------------- write
(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();
