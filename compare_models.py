"""Compare all models: logistic regression and XGBoost, each as
unaware / surrogate-fair (Exponentiated Gradient) / graph-informed.

    python compare_models.py            # ~6 min -> results/model_comparison.csv, figures/fig11_model_comparison.png
    python compare_models.py --seeds 1  # quicker
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fairsim.evaluation import evaluate

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=3)
args = p.parse_args()

GRAPHS = ["G0", "G1", "G2", "G3", "G4"]
rows = [r for m in ("stylised", "realistic") for g in GRAPHS for s in range(args.seeds)
        for r in evaluate(m, g, kappa=1.0, seed=s, learners=("logreg", "xgboost"))]
res = pd.DataFrame(rows)
(ROOT / "results").mkdir(exist_ok=True)
res.to_csv(ROOT / "results" / "model_comparison.csv", index=False)

pd.set_option("display.width", 200)
for metric in ["AUC", "model_TE"]:
    print(f"\n{metric} (mean over seeds)")
    print(res.pivot_table(index=["model", "graph"], columns=["learner", "variant"], values=metric).round(3).to_string())
real = res[res.model == "realistic"]
print("\nRealistic: premium gap male - female in EUR/year (true vs model)")
t = real.pivot_table(index="graph", columns=["learner", "variant"], values="model_TE_premium_eur").round(0)
t.insert(0, ("true", ""), real.groupby("graph")["true_TE_premium_eur"].mean().round(0))
print(t.to_string())

# ---- summary figure ----------------------------------------------------------------
BLUE, ORANGE, AQUA, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
plt.rcParams.update({"figure.dpi": 200, "savefig.bbox": "tight", "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "axes.grid.axis": "y", "grid.color": "#e8e7e3", "legend.frameon": False,
                     "axes.titleweight": "bold"})
VARS = [("unaware", "Unaware", BLUE), ("surrogate-fair", "Surrogate-fair (EG)", AQUA),
        ("graph-informed", "Graph-informed", ORANGE)]
LEARN = [("logreg", "Logistic regression"), ("xgboost", "XGBoost")]
unfair = res[res.graph != "G0"].copy()
unfair["share"] = 100 * unfair["model_TE"] / unfair["true_TE"]
summ = unfair.groupby(["model", "learner", "variant"])[["AUC", "share"]].mean()

fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
x = np.arange(len(LEARN)); w = 0.26
for r_, model in enumerate(["stylised", "realistic"]):
    for c_, (metric, ylabel) in enumerate([("AUC", "AUC (G1-G4 average)"),
                                           ("share", "Unfairness inherited\n(% of true gender effect, G1-G4)")]):
        ax = axes[r_, c_]
        for i, (v, lab, col) in enumerate(VARS):
            vals = [summ.loc[(model, l, v), metric] for l, _ in LEARN]
            ax.bar(x + (i - 1) * w, vals, w - 0.02, color=col, label=lab)
            for xi, val in zip(x + (i - 1) * w, vals):
                ax.text(xi, val + (0.002 if metric == "AUC" else 1), f"{val:.3f}" if metric == "AUC" else f"{val:.0f}%",
                        ha="center", fontsize=9, color=INK2)
        ax.set_xticks(x, [l for _, l in LEARN])
        if metric == "AUC":
            lo = summ.loc[model]["AUC"].min()
            ax.set_ylim(lo - 0.03, summ.loc[model]["AUC"].max() + 0.015)
        else:
            ax.set_ylim(0, 75)
        ax.set_ylabel(ylabel, color=INK2)
        ax.set_title(f"{model.capitalize()} model: {'accuracy' if metric == 'AUC' else 'hidden unfairness'}")
        if r_ == 0 and c_ == 1:
            ax.legend(loc="upper right", fontsize=10)
fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "fig11_model_comparison.png")
print("\nsaved figures/fig11_model_comparison.png and results/model_comparison.csv")
