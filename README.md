# Seminar Information Systems: Fairness under a Hidden Unfair Attribute

**Humboldt-Universität zu Berlin · Master Seminar Information Systems · WiSe 26/27 · Topic 2**
*Fairness evaluation in the presence of unobserved unfair attributes*

> ⚠️ Keep this repository **private**. Seminar rules: no publishing of results without supervisor consent.

---

## 1. The question

Since December 2012, EU insurers may not use **gender** to price car insurance (Test-Achats ruling). They simply drop the column. But dropping a column does not remove its influence: gender can still reach the price through variables that correlate with it, such as engine power or mileage, or through biased claim records.

Because gender is not in the data, **standard fairness metrics cannot even be computed**. A model can be unfair without anyone being able to see it.

This project builds **simulators where the hidden truth is known**: we generate drivers with a gender, hide it from the model, and then measure exactly how much gender still affects the predicted claim risk and the premium.

**Research questions**
- **RQ1**: For which causal structures can unfairness caused by a hidden attribute be detected or removed using only observed data?
- **RQ2**: How do direct and indirect unfair paths differ in how much unfairness a model inherits?

**Hypotheses**
- **H1**: A model that never sees gender is still unfair whenever an unfair path exists, because proxies let it partly rebuild gender.
- **H2**: Removing the descendants of gender removes unfairness from the model, but not from biased labels (direct or hidden-mediator paths).
- **H3**: Different causal structures can produce identical data but different fairness verdicts, so fairness w.r.t. a hidden attribute is not identifiable from data alone.

---

## 2. Repository structure

```
seminar-information-system/
├── fairsim/                  # the Python package
│   ├── stylised.py           # simulator 1: small, clean, 7 variables
│   ├── realistic.py          # simulator 2: realistic tariff, 13 variables, premium in EUR
│   └── evaluation.py         # learners (logistic regression, XGBoost), fairness method (EG), evaluation
├── run_all.py                # runs everything below in the right order
├── run_checks.py             # step 1: tests + quick results table
├── make_datasets.py          # step 2: exports the experiment grid as CSV
├── make_figures.py           # step 3: presentation figures 0-10, 12 + results tables
├── compare_models.py         # step 4: all models side by side, figure 11
├── figures/                  # created by the scripts (PNG)
├── results/                  # created by the scripts (CSV)
├── data/                     # created by make_datasets.py (git-ignored)
└── requirements.txt
```

---

## 3. Quick start

```bash
# 1. install (once)
pip install -r requirements.txt

# 2. run everything (from inside this folder)
python run_all.py           # ~15 min, or: python run_all.py --quick  (~5 min)
```

Or step by step, in this order:

| Step | Command | Time | Output |
|---|---|---|---|
| 1 | `python run_checks.py` | ~1 min | tests both simulators, prints a results table (all models) |
| 2 | `python make_datasets.py` | ~2 min | `data/stylised/` and `data/realistic/`: 200 CSVs each + `manifest.csv` |
| 3 | `python make_figures.py` | ~5 min | `figures/fig0`–`fig10`, `fig12`, `results/results_*.csv`, `results/surrogate_noise.csv` |
| 4 | `python compare_models.py` | ~5 min | `figures/fig11_model_comparison.png`, `results/model_comparison.csv` |

On Windows, if `python` is not found, use the full path, e.g. `& "D:\New folder\python\python.exe" run_all.py`.
Always run from inside this folder (open it in VS Code with **File → Open Folder**).

**What's in `data/`**: one CSV per dataset; the name says the settings (`real_G2_k1_gaussian_s0.csv` = realistic, graph G2, κ = 1, Gaussian noise, seed 0). `manifest.csv` lists every dataset with its settings and its **true** gender effect (`TE`, `PSE_*`, `TE_premium_eur`), the correct answer that models are checked against.
`python make_datasets.py --model realistic --noise gaussian` writes only part of the grid.

**Use the simulators in your own code**
```python
from fairsim import realistic as sim          # or: from fairsim import stylised as sim

ds = sim.generate(sim.Config(graph="G4", kappa=1.0, seed=0))
ds.df            # full table: observed + hidden columns
ds.X, ds.y       # model inputs and claim labels  (realistic: ds.X is already one-hot encoded)
ds.gender        # hidden attribute, ONLY for evaluation

sim.true_effects(ds)                      # true effect of gender, total and per path
ds.counterfactual(paths=["mileage"])      # the same drivers, gender flipped on one path only
sim.model_effects(ds, lambda X: clf.predict_proba(X)[:, 1])   # how unfair is a fitted model?
```

---

## 4. The causal structures (G0–G4)

Both simulators share the same five graphs. Each graph switches **unfair arrows out of gender** on or off; everything else is identical.

![causal graphs](figures/fig0_causal_graphs.png)

| Graph | Unfair path | Real-world story |
|---|---|---|
| **G0** | none | Fair reference: gender has no effect on claims |
| **G1** | gender → claim | Direct: claim records are biased against men |
| **G2** | gender → mileage → claim | Observed mediator: men drive more, more km = more claims |
| **G3** | gender → fault bias → claim | Hidden mediator: claims handlers assign fault to men more often |
| **G4** | G1 + G2 | Direct and indirect together |

The arrow **gender → engine power** (a proxy) exists in every graph. Proxies do not cause claims, but they let a model guess gender.
κ (kappa) sets the strength of every unfair arrow (default 0.5 and 1.0).

---

## 5. Simulator 1: stylised (`fairsim/stylised.py`)

Small and fully transparent: one line per variable, made for explaining the theory.

| Column | Role | Equation (z = standardised) | Insurer sees it? |
|---|---|---|---|
| `gender` | unfair attribute A | Bernoulli(0.5), 1 = male | ❌ |
| `skill` | latent merit U | N(0, 1) | ❌ |
| `experience` | legitimate | 15 + 4·(skill + ε) years | ✅ |
| `engine_kw` | proxy | 100 + 25·(α·gender + ε) | ✅ |
| `mileage` | mediator | 13000 + 4000·(κ·gender + ε) km | ✅ |
| `fault_bias` | hidden mediator | κ·gender + ε | ❌ |
| `claim` | label | Bernoulli(σ(−2 − exp − skill + mileage + fault_bias + κ·gender)) | ✅ (label) |

Extras: `Config(car_class=True)` replaces engine power with a 4-level categorical car class; `equivalent_pair()` builds the H3 example.

## 6. Simulator 2: realistic (`fairsim/realistic.py`)

Closer to a real motor tariff: more rating factors, mixed types, non-linear effects, claim counts and premiums.

| Column | Type | Role | Insurer sees it? |
|---|---|---|---|
| `gender` | binary | unfair attribute | ❌ |
| `skill` | continuous | latent merit | ❌ |
| `fault_bias` | continuous | hidden mediator (biased claims handling) | ❌ |
| `age` | 18–80 | legitimate, **U-shaped** risk | ✅ |
| `years_licensed` | years | legitimate | ✅ |
| `region` | urban / suburban / rural | legitimate | ✅ |
| `parking` | garage / street | legitimate | ✅ |
| `vehicle_age` | years | legitimate | ✅ |
| `bonus_class` | 0–15 | legitimate (no-claims bonus) | ✅ |
| `occupation` | office / manual / student / retired | **proxy** | ✅ |
| `engine_kw` | kW | **proxy** | ✅ |
| `car_value` | EUR | **proxy** (via engine power) | ✅ |
| `mileage` | km/year | legitimate risk, **mediator** in G2/G4 | ✅ |
| `claim_count` | Poisson | number of claims | ✅ |
| `claim` | 0/1 | label (≥ 1 claim) | ✅ |
| `premium` | EUR | true risk premium = λ · €3,500 · 1.25 | ❌ |

Calibration: ~9% of drivers claim per year, average premium ~€430, AUC of a standard model ~0.68. These are typical values for motor insurance.

## 7. How the ground truth works

Every random draw (noise) is **stored**. To ask "what if this driver were a woman instead of a man?", we re-run the same equations with the **same noise** and only gender changed. This gives exact counterfactuals:

- **Total effect (TE)**: switch gender everywhere.
- **Path-specific effect (PSE)**: switch gender on one arrow only, e.g. only gender → mileage. This separates direct from indirect unfairness.
- **Model effect**: feed the counterfactual drivers into a trained model and measure how its prediction changes. The model never sees gender, but its inputs (engine power, mileage, …) change with it.

## 8. Models and metrics (`fairsim/evaluation.py`)

**Learners**

| Learner | What it is | Why |
|---|---|---|
| **Logistic regression** | a GLM (generalised linear model): weighted sum of features → probability | the insurer's standard, transparent baseline |
| **XGBoost** | gradient-boosted decision trees | the strongest standard method for tabular data; learns interactions and non-linear patterns |

XGBoost settings: `n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_lambda=5` (best of 5 tested). Change them in `make_learner()`.

**Variants** (each learner is trained in all three)

| Variant | What it does | What it needs |
|---|---|---|
| **Unaware** | all observed features, gender never included ("fairness through unawareness", what EU insurers do today) | nothing |
| **Surrogate-fair (EG)** | all observed features + **Exponentiated Gradient** (Agarwal et al., 2018; `fairlearn`) with a **demographic-parity** constraint, enforced on a **surrogate gender**: the true gender with **20% of values flipped**, as if the insurer had inferred it (e.g. from first names) | an inferred gender, only during training |
| **Graph-informed** | drops every descendant of gender (engine power, car value, occupation, mileage) | the true causal graph |

How the surrogate-fair model is built (`SurrogateFair` in `evaluation.py`):
1. Surrogate gender = true gender with 20% flipped (`noise_rate`).
2. Inputs standardised.
3. `ExponentiatedGradient(base learner, DemographicParity(difference_bound=0.01), max_iter=30)`. Base learners are **class-balanced**: claims are rare, and without balancing EG collapses to "nobody claims".
4. Score = weighted mix of the learned predictors' probabilities.
5. One global Platt recalibration (no gender involved), so scores are real probabilities again and premiums stay correct.

At prediction time the model needs no gender at all.

70/30 stratified train/test split. For the realistic simulator the inputs are prepared like an insurer would: one-hot categories, age bands, log car value.

| Metric | Meaning |
|---|---|
| `AUC` | prediction accuracy |
| `score_gap` | average predicted risk, men − women (needs the hidden gender) |
| `model_TE` | causal effect of gender on the model's prediction |
| `model_PSE_*` | part of it flowing through each proxy/mediator |
| `true_TE`, `true_PSE_*` | effect of gender on the true claim probability |
| `*_premium_eur` | the same in EUR per year (realistic simulator) |

---

## 9. Results

| Figure | Message |
|---|---|
| ![](figures/fig1_true_vs_model.png) | **H1**: a model without gender still inherits part of the gender effect. G2: almost all of it. G1/G3: ~20% through the proxy. |
| ![](figures/fig2_path_decomposition.png) | Where it comes from: engine power (proxy) vs mileage (mediator). |
| ![](figures/fig3_accuracy_vs_fairness.png) | **H2**: dropping descendants of gender makes the model fair but costs AUC. Labels stay biased in G1/G3/G4. |
| ![](figures/fig4_bias_strength.png) | Hidden unfairness grows with bias strength κ. |
| ![](figures/fig5_proxy_engine_power.png) | Engine power differs by gender: the proxy problem. |
| ![](figures/fig6_equivalence.png) | **H3**: identical data, opposite verdicts. No method using only observed data can tell the worlds apart. |
| ![](figures/fig7_realistic_premium_gap.png) | Realistic tariff in EUR: men pay €43–265 more per year because of gender, without gender being used. |
| ![](figures/fig8_realistic_path_decomposition.png) | In the realistic tariff gender leaks through several weak proxies at once. |
| ![](figures/fig9_realistic_data.png) | What the realistic data look like. |
| ![](figures/fig10_stylised_vs_realistic.png) | The finding holds in both simulators. |
| ![](figures/fig11_model_comparison.png) | **All six models**: logistic regression and XGBoost × unaware / surrogate-fair / graph-informed. Accuracy vs hidden unfairness. |
| ![](figures/fig12_surrogate_noise.png) | The fair model is only as good as the inferred gender: the more wrong labels in the surrogate, the more unfairness remains. |

*Figures appear after running the scripts.*

**Key numbers (κ = 1)**

| Graph | True gender effect | Unaware model (stylised) | Unaware model (realistic, EUR/year) | Graph-informed |
|---|---|---|---|---|
| G0 | 0 | 0 | €0 | 0 |
| G1 | +0.11 | +0.02 | +€43 of €208 | 0 |
| G2 | +0.11 | +0.11 | +€128 of €146 | 0 |
| G3 | +0.11 | +0.02 | +€43 of €208 | 0 |
| G4 | +0.24 | +0.16 | +€265 of €426 | 0 |

**All models (κ = 1, average over G1–G4)**

| Model | AUC stylised | Unfairness inherited, stylised | AUC realistic | Unfairness inherited, realistic |
|---|---|---|---|---|
| Logistic regression, unaware | 0.861 | 51% | 0.683 | 54% |
| Logistic regression, **surrogate-fair** | 0.847 | **17%** | 0.672 | **16%** |
| Logistic regression, graph-informed | 0.812 | 0% | 0.652 | 0% |
| XGBoost, unaware | 0.855 | 50% | 0.680 | 52% |
| XGBoost, **surrogate-fair** | 0.842 | **15%** | 0.661 | **12%** |
| XGBoost, graph-informed | 0.806 | 0% | 0.648 | 0% |

Realistic tariff, premium gap male − female (EUR/year, logistic regression): G2 true €146 → unaware €130 → **surrogate-fair €35** → graph-informed €0; G4 true €426 → unaware €290 → **surrogate-fair €41** → graph-informed €0.

**Surrogate-fair is the practical middle ground.** It removes about two-thirds of the hidden unfairness without knowing the causal graph, and keeps more accuracy than the graph-informed model. But (fig12) it depends on the quality of the inferred gender: with 40% wrong labels most of the unfairness comes back. And it only equalises **predictions**: the biased labels in G1/G3/G4 stay biased.

**XGBoost vs logistic regression.** XGBoost is slightly less accurate (AUC about −0.005) and inherits the same share of the gender effect. The reason is that both simulators generate claims with a log-linear formula, so logistic regression is the correctly specified model and there are no interactions for XGBoost to exploit. Takeaway: in this setting, the **causal path** decides how much unfairness a model inherits, not the model's power.

Note that G1 and G3 look identical to every model: a direct bias and a hidden-mediator bias cannot be told apart from observed data.

---

## 10. Assumptions and limitations

- The causal graph is assumed known (needed for the graph-informed model and for path-specific effects).
- Gender is binary; relationships inside the claim model are log-linear.
- Proxies (engine power, car value, occupation) are assumed to have **no** causal effect on claims. In reality engine power may matter; changing this turns a proxy into a mediator.
- The premium is the true risk premium (frequency × fixed mean claim size × loading); claim size does not vary.
- All results are simulations: they show what *can* happen, not how large the effect is in any real portfolio.

## 11. Next steps

- [ ] Add realistic interactions to the realistic simulator (young × urban × street parking, young × mileage) and test whether XGBoost then rebuilds more of the hidden gender.
- [x] Surrogate fairness model: Exponentiated Gradient (fairlearn) with a noisy, inferred gender (20% flipped).
- [ ] Other fairness methods from fairlearn (Correlation Remover before training, Threshold Optimizer after training).
- [ ] Full grid over κ and noise type, mean ± sd over 10 seeds.

## 12. References

Starting points from the seminar: arXiv:2407.05732 · Information Sciences S0020025525007741 · WHY-21 paper 16.
Agarwal et al. (2018) A reductions approach to fair classification (Exponentiated Gradient) · Chen & Guestrin (2016) XGBoost · Kusner et al. (2017) Counterfactual fairness · Chiappa (2019) Path-specific counterfactual fairness · Kilbertus et al. (2017) Avoiding discrimination through causal reasoning · Zhang & Bareinboim (2018) Fairness in decision-making · Hardt et al. (2016) Equality of opportunity · Lahoti et al. (2020) Fairness without demographics · CJEU C-236/09 Test-Achats (2011).

## 13. Use of generative AI

Parts of the code were written with the help of Claude (Anthropic). As required by the seminar, a summary of the AI use and the chat protocol are submitted with the paper. The team is responsible for the correctness of all code and results.
