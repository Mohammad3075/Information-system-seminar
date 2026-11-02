"""Create all presentation figures (both simulators).

    python make_figures.py              # ~2 minutes -> figures/*.png, results/*.csv
    python make_figures.py --quick      # fewer seeds, for testing
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
from matplotlib.patches import FancyArrowPatch
from sklearn.linear_model import LogisticRegression

from fairsim import realistic, stylised
from fairsim.evaluation import evaluate

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true")
args = p.parse_args()
SEEDS_STYL = range(3 if args.quick else 10)
SEEDS_REAL = range(2 if args.quick else 5)

FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
RES = ROOT / "results"; RES.mkdir(exist_ok=True)

# ---- style ------------------------------------------------------------------------
BLUE, ORANGE, AQUA, VIOLET, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"
GREY, INK, INK2 = "#9a9893", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 200, "savefig.bbox": "tight", "font.size": 13,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelcolor": INK2,
    "axes.edgecolor": "#c9c8c3", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e8e7e3", "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
})
G_ORDER = ["G0", "G1", "G2", "G3", "G4"]
G_DESC = ["fair", "direct", "via mileage", "via fault bias", "direct + mileage"]
V_LABEL = {"unaware": "Unaware model", "surrogate-fair": "Surrogate-fair model (EG)",
           "graph-informed": "Graph-informed model"}
V_COLOR = {"unaware": BLUE, "surrogate-fair": AQUA, "graph-informed": ORANGE}


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("saved", FIG / name)


def run(model, seeds):
    rows = [r for g in G_ORDER for s in seeds for r in evaluate(model, g, kappa=1.0, seed=s)]
    df = pd.DataFrame(rows)
    df.to_csv(RES / f"results_{model}.csv", index=False)
    return df


def bars_true_vs_model(res, col, ylabel, title, name, fmt="{:.2f}", pad=0.006):
    mean = res.groupby(["graph", "variant"]).mean(numeric_only=True)
    sd = res.groupby(["graph", "variant"]).std(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(G_ORDER)); w = 0.2; off = 1.5
    series = [("True effect of gender", GREY, mean.xs("unaware", level=1)[f"true_{col}"], None)]
    series += [(V_LABEL[v], V_COLOR[v], mean.xs(v, level=1)[f"model_{col}"], sd.xs(v, level=1)[f"model_{col}"])
               for v in V_LABEL]
    for i, (lab, c, m, s) in enumerate(series):
        ax.bar(x + (i - off) * w, m.values, w - 0.02, color=c, label=lab,
               yerr=None if s is None else s.values, error_kw=dict(ecolor=INK2, lw=1, capsize=3))
        top = m.values + (0 if s is None else s.values)
        for xi, val, t in zip(x + (i - off) * w, m.values, top):
            val = 0.0 if abs(val) < 0.5 * 10 ** -(fmt.count("0") or 2) else val
            ax.text(xi, max(t, 0) + pad, fmt.format(val), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"{g}\n{d}" for g, d in zip(G_ORDER, G_DESC)])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(loc="upper left")
    save(fig, name)


# =====================================================================================
# PART A: stylised model
# =====================================================================================
res_s = run("stylised", SEEDS_STYL)
mean_s = res_s.groupby(["graph", "variant"]).mean(numeric_only=True)

# Fig 0: the five causal graphs
POS = {"gender": (-0.2, 1), "engine_kw": (1.1, 2), "mileage": (1.1, 1), "fault_bias": (1.1, 0),
       "skill": (2, 2.6), "experience": (2.9, 2.6), "claim": (2.4, 1)}
LBL = {"gender": "Gender", "engine_kw": "Engine\npower", "mileage": "Mileage",
       "fault_bias": "Fault\nbias", "skill": "Skill", "experience": "Experi-\nence", "claim": "Claim"}
HIDDEN = {"gender", "skill", "fault_bias"}
FIXED = [("gender", "engine_kw"), ("skill", "experience"), ("skill", "claim"),
         ("experience", "claim"), ("mileage", "claim"), ("fault_bias", "claim")]
UNFAIR = {"mileage": ("gender", "mileage"), "fault_bias": ("gender", "fault_bias"),
          "claim": ("gender", "claim")}
fig, axes = plt.subplots(1, 5, figsize=(18, 4.4))
for ax, g in zip(axes, G_ORDER):
    ax.set_xlim(-0.7, 3.4); ax.set_ylim(-0.6, 3.1); ax.axis("off")
    ax.set_title(g, color=INK)
    edges = [(e, GREY, 1.4) for e in FIXED]
    edges += [(UNFAIR[k], ORANGE, 3.0) for k, on in stylised.GRAPHS[g].items() if on]
    for (a, b), c, lw in edges:
        rad = 0.6 if (a, b) == ("gender", "claim") else 0.0
        ax.add_patch(FancyArrowPatch(POS[a], POS[b], arrowstyle="-|>", mutation_scale=14,
                                     color=c, lw=lw, shrinkA=20, shrinkB=20,
                                     connectionstyle=f"arc3,rad={rad}"))
    for node, (x, y) in POS.items():
        ax.scatter(x, y, s=1500, facecolor="white", edgecolor=GREY if node in HIDDEN else INK2,
                   linewidth=1.6, linestyle="--" if node in HIDDEN else "-", zorder=3)
        ax.text(x, y, LBL[node], ha="center", va="center", fontsize=8.5, color=INK, zorder=4)
fig.suptitle("Five causal structures: orange arrows are unfair paths, dashed nodes are hidden from the insurer",
             fontsize=14, color=INK2, y=0.02)
save(fig, "fig0_causal_graphs.png")

# Fig 1: key result
bars_true_vs_model(res_s, "TE", "Change in claim probability\nfemale → male (same driver)",
                   "A model without gender still inherits part of the gender effect",
                   "fig1_true_vs_model.png")

# Fig 2: through which variable
fig, ax = plt.subplots(figsize=(10, 5.2))
un = mean_s.xs("unaware", level=1)
left = np.zeros(len(G_ORDER))
for col, lab, c in [("model_PSE_engine_kw", "via engine power (proxy)", VIOLET),
                    ("model_PSE_mileage", "via mileage (mediator)", AQUA)]:
    vals = un[col].clip(lower=0).values
    ax.barh(G_ORDER, vals, left=left, color=c, label=lab, edgecolor="white", linewidth=2, height=0.55)
    left += vals
ax.scatter(un["true_TE"], G_ORDER, marker="|", s=500, color=INK, linewidth=2.5, label="true effect", zorder=4)
ax.invert_yaxis(); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_xlim(0, 0.26)
ax.set_xlabel("Change in predicted claim probability, female → male")
ax.set_title("Unaware model: where the gender effect comes from")
ax.legend(loc="upper right")
save(fig, "fig2_path_decomposition.png")

# Fig 3: accuracy vs fairness
fig, ax = plt.subplots(figsize=(9, 5.6))
ax.grid(axis="x")
for v in V_LABEL:
    m = mean_s.xs(v, level=1)
    ax.scatter(m["model_TE"].abs(), m["AUC"], s=110, color=V_COLOR[v], label=V_LABEL[v],
               edgecolor="white", linewidth=2, zorder=3)
    pts = {}
    for g, r in m.iterrows():
        pts.setdefault((round(abs(r["model_TE"]), 3), round(r["AUC"], 3)), []).append(g)
    for (xv, yv), gs in pts.items():
        dy = -14 if v == "surrogate-fair" else 4
        ax.annotate(", ".join(gs), (xv, yv), xytext=(7, dy), textcoords="offset points", fontsize=10,
                    color=V_COLOR[v] if v == "surrogate-fair" else INK2)
ax.set_xlabel("Causal unfairness of the model |effect of gender|")
ax.set_ylabel("AUC")
ax.set_title("Fairness costs accuracy")
ax.legend(loc="lower right")
save(fig, "fig3_accuracy_vs_fairness.png")

# Fig 4: bias strength
kappas = [0, 0.25, 0.5, 0.75, 1.0, 1.5]
sw = pd.DataFrame([r for g in G_ORDER[1:] for k in kappas for s in SEEDS_STYL[:5]
                   for r in evaluate("stylised", g, kappa=k, seed=s, variants=("unaware",))])
sw = sw.groupby(["graph", "kappa"]).mean(numeric_only=True).reset_index()
fig, ax = plt.subplots(figsize=(10, 5.6))
for g, c in zip(G_ORDER[1:], [BLUE, ORANGE, AQUA, VIOLET]):
    d = sw[sw.graph == g]
    ax.plot(d.kappa, d.model_TE, color=c, lw=2, marker="o", ms=8, markeredgecolor="white", mew=2,
            ls="--" if g == "G3" else "-")
    if g != "G1":
        ax.annotate("G1 = G3" if g == "G3" else g, (d.kappa.iloc[-1], d.model_TE.iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=12, color=INK)
ax.set_xlabel("Strength of hidden bias κ")
ax.set_ylabel("Model's effect of gender\non predicted claim probability")
ax.set_title("Hidden unfairness grows with bias strength (unaware model)")
save(fig, "fig4_bias_strength.png")

# Fig 5: proxy
ds = stylised.generate(stylised.Config(graph="G0", seed=0, n=20_000))
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, 220, 45)
for gval, lab, c in [(0, "Women", ORANGE), (1, "Men", BLUE)]:
    ax.hist(ds.df.loc[ds.gender == gval, "engine_kw"], bins=bins, color=c, alpha=0.55, label=lab,
            edgecolor="white", linewidth=0.8)
ax.set_xlabel("Engine power (kW), observed by the insurer")
ax.set_ylabel("Number of drivers")
ax.set_title("Gender is removed, but engine power still carries it")
ax.legend()
save(fig, "fig5_proxy_engine_power.png")

# Fig 6: observational equivalence
u, f = stylised.equivalent_pair(n=20_000)
clf = LogisticRegression(max_iter=1000).fit(u[stylised.OBSERVED], u[stylised.LABEL])
p_model = clf.predict_proba(u[stylised.OBSERVED])[:, 1]
true_u = stylised.true_effects(stylised.generate(stylised.Config(graph="G2", n=20_000)))["TE"]
gap = lambda df: p_model[df.gender == 1].mean() - p_model[df.gender == 0].mean()
fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(2); w = 0.36
ax.bar(x - w / 2, [true_u, 0.0], w - 0.02, color=GREY, label="True effect of gender")
ax.bar(x + w / 2, [gap(u), gap(f)], w - 0.02, color=BLUE, label="Model's gender gap")
for xi, val in zip(np.r_[x - w / 2, x + w / 2], [true_u, 0.0, gap(u), gap(f)]):
    ax.text(xi, val + 0.004, f"{abs(val) if abs(val) < 0.005 else val:.2f}", ha="center", fontsize=11, color=INK2)
ax.set_xticks(x, ["World 1\nfactor = gender", "World 2\nfactor = rural residence"])
ax.set_ylim(0, max(true_u, gap(u)) * 1.3)
ax.set_ylabel("Effect on claim probability")
ax.set_title("Identical data, opposite fairness verdicts")
ax.text(0.5, -0.17, "Both worlds produce exactly the same observed data and the same model.",
        transform=ax.transAxes, ha="center", fontsize=11, color=INK2)
ax.legend(loc="upper right")
save(fig, "fig6_equivalence.png")

# =====================================================================================
# PART B: realistic model
# =====================================================================================
res_r = run("realistic", SEEDS_REAL)
mean_r = res_r.groupby(["graph", "variant"]).mean(numeric_only=True)

# Fig 7: premium gap in EUR
bars_true_vs_model(res_r, "TE_premium_eur", "Premium difference, male − female\n(same driver, EUR per year)",
                   "Realistic tariff: how much more men pay because of gender",
                   "fig7_realistic_premium_gap.png", fmt="€{:.0f}", pad=8)

# Fig 8: which rating factors carry gender
fig, ax = plt.subplots(figsize=(10, 5.2))
un = mean_r.xs("unaware", level=1)
left = np.zeros(len(G_ORDER))
for col, lab, c in [("model_PSE_engine_kw", "via engine power / car value (proxy)", VIOLET),
                    ("model_PSE_occupation", "via occupation (proxy)", MAGENTA),
                    ("model_PSE_mileage", "via mileage (mediator)", AQUA)]:
    vals = un[col].clip(lower=0).values
    ax.barh(G_ORDER, vals, left=left, color=c, label=lab, edgecolor="white", linewidth=2, height=0.55)
    left += vals
ax.scatter(un["true_TE"], G_ORDER, marker="|", s=500, color=INK, linewidth=2.5, label="true effect", zorder=4)
ax.invert_yaxis(); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_xlim(0, un["true_TE"].max() * 1.15)
ax.set_xlabel("Change in predicted claim probability, female → male")
ax.set_title("Realistic tariff: gender leaks through several rating factors")
ax.legend(loc="upper right", fontsize=11)
save(fig, "fig8_realistic_path_decomposition.png")

# Fig 9: what the realistic data look like
d = realistic.generate(realistic.Config(graph="G0", n=50_000, seed=0)).df
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 4.8))
by_age = d.groupby(d.age // 2 * 2)["p_claim"].mean()
a1.plot(by_age.index, by_age.values, color=BLUE, lw=2)
a1.set_xlabel("Driver age"); a1.set_ylabel("True claim probability"); a1.set_title("U-shaped age risk")
occ = d.groupby("gender", observed=True)["occupation"].value_counts(normalize=True).unstack()[realistic.CATEGORICAL["occupation"]]
xo = np.arange(occ.shape[1]); w = 0.38
for i, (gval, lab, c) in enumerate([(0, "Women", ORANGE), (1, "Men", BLUE)]):
    a2.bar(xo + (i - 0.5) * w, occ.loc[gval].values, w - 0.02, color=c, label=lab)
a2.set_xticks(xo, occ.columns); a2.set_ylabel("Share of drivers"); a2.set_title("Occupation: a weak proxy")
a2.legend()
bins = np.linspace(40, 200, 40)
for gval, lab, c in [(0, "Women", ORANGE), (1, "Men", BLUE)]:
    a3.hist(d.loc[d.gender == gval, "engine_kw"], bins=bins, color=c, alpha=0.55, label=lab,
            edgecolor="white", linewidth=0.6)
a3.set_xlabel("Engine power (kW)"); a3.set_ylabel("Number of drivers"); a3.set_title("Engine power: a proxy")
fig.tight_layout()
save(fig, "fig9_realistic_data.png")

# Fig 10: stylised vs realistic, share of the true effect the unaware model inherits
share = lambda m: (m.xs("unaware", level=1)["model_TE"] / m.xs("unaware", level=1)["true_TE"]).loc[G_ORDER[1:]]
fig, ax = plt.subplots(figsize=(10, 5.4))
x = np.arange(4); w = 0.36
for i, (lab, m, c) in enumerate([("Stylised model", mean_s, BLUE), ("Realistic model", mean_r, AQUA)]):
    vals = share(m).values * 100
    ax.bar(x + (i - 0.5) * w, vals, w - 0.02, color=c, label=lab)
    for xi, v in zip(x + (i - 0.5) * w, vals):
        ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10, color=INK2)
ax.set_xticks(x, [f"{g}\n{d}" for g, d in zip(G_ORDER[1:], G_DESC[1:])])
ax.set_ylabel("Share of the true gender effect\npicked up by the unaware model (%)")
ax.set_title("The finding holds in both models")
ax.set_ylim(0, 115); ax.legend(loc="upper left")
save(fig, "fig10_stylised_vs_realistic.png")

# Fig 12: how good must the inferred gender be? (surrogate noise rate)
rates = [0.0, 0.1, 0.2, 0.3, 0.4]
nz = pd.DataFrame([dict(r, rate=q) for m, sd_ in [("stylised", SEEDS_STYL[:3]), ("realistic", SEEDS_REAL[:2])]
                   for q in rates for s in sd_
                   for r in evaluate(m, "G2", seed=s, variants=("surrogate-fair",), noise_rate=q)])
nz = nz.groupby(["model", "rate"]).mean(numeric_only=True).reset_index()
nz["share"] = 100 * nz["model_TE"] / nz["true_TE"]
fig, ax = plt.subplots(figsize=(9.5, 5.4))
for m, c in [("stylised", BLUE), ("realistic", AQUA)]:
    d_ = nz[nz.model == m]
    ax.plot(d_.rate * 100, d_.share, color=c, lw=2, marker="o", ms=8, markeredgecolor="white", mew=2)
    ax.annotate(f"{m.capitalize()} model", (d_.rate.iloc[-1] * 100, d_.share.iloc[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=12, color=INK)
ax.set_xlabel("Share of wrongly inferred gender labels in the surrogate (%)")
ax.set_ylabel("Unfairness left in the model\n(% of true gender effect, G2)")
ax.set_title("The fair model is only as good as the inferred gender")
ax.set_ylim(bottom=0); ax.set_xlim(-2, 50)
save(fig, "fig12_surrogate_noise.png")
nz.to_csv(RES / "surrogate_noise.csv", index=False)

print("\nStylised (mean over seeds):")
print(mean_s[["AUC", "model_TE", "true_TE"]].round(3).to_string())
print("\nRealistic (mean over seeds):")
print(mean_r[["AUC", "model_TE", "model_TE_premium_eur", "true_TE", "true_TE_premium_eur"]].round(3).to_string())
