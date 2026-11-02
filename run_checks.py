"""Sanity checks for both simulators + a quick baseline table. Run this first.

    python run_checks.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from fairsim import realistic, stylised
from fairsim.evaluation import evaluate

for name, sim, n_big in [("stylised", stylised, 20_000), ("realistic", realistic, 20_000)]:
    # 1. same seed -> same data
    a, b = sim.generate(sim.Config(seed=3)), sim.generate(sim.Config(seed=3))
    assert a.df.equals(b.df), name
    # 2. do(gender = factual gender) reproduces the factual world
    col = "p_claim"
    assert np.allclose(a.intervene(a.exo["gender"])[col], a.df[col]), name
    # 3. each graph's effect sits exactly on its own edges; G0 is fair
    for g in sim.GRAPHS:
        te = sim.true_effects(sim.generate(sim.Config(graph=g, kappa=1.0, n=n_big)))
        for e, on in sim.GRAPHS[g].items():
            ok = te[f"PSE_{e}"] > 0.005 if on else abs(te[f"PSE_{e}"]) < 1e-12
            assert ok, (name, g, e, te)
    # 4. Laplace noise works
    sim.generate(sim.Config(noise="laplace"))
    print(f"[ok] {name} simulator")

# stylised extras
u, f = stylised.equivalent_pair()
assert u[stylised.OBSERVED + [stylised.LABEL]].equals(f[stylised.OBSERVED + [stylised.LABEL]])
assert set(stylised.generate(stylised.Config(car_class=True)).X["car_class"].unique()) <= {0, 1, 2, 3}
# realistic extras: plausible portfolio
d = realistic.generate(realistic.Config()).df
assert 0.05 < d.claim.mean() < 0.15 and 300 < d.premium.mean() < 700
print("[ok] extras (equivalence pair, categorical car class, realistic calibration)")
print("\nall sanity checks passed\n")

rows = [r for m in ("stylised", "realistic") for g in stylised.GRAPHS for r in evaluate(m, g, seed=0, learners=("logreg", "xgboost"))]
res = pd.DataFrame(rows)
cols = ["AUC", "model_TE", "true_TE"]
print(res.set_index(["model", "graph", "learner", "variant"])[cols].round(3).to_string())
