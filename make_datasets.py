"""Write the experiment grid as CSV files + a manifest with the true effects.

    python make_datasets.py                          # both simulators
    python make_datasets.py --model realistic        # only one
    python make_datasets.py --noise gaussian --seeds 5
"""
import argparse
import itertools
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from fairsim import realistic, stylised

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["stylised", "realistic", "both"], default="both")
p.add_argument("--kappas", type=float, nargs="+", default=[0.5, 1.0])
p.add_argument("--noise", nargs="+", default=["gaussian", "laplace"])
p.add_argument("--seeds", type=int, default=10)
args = p.parse_args()

models = {"stylised": stylised, "realistic": realistic}
if args.model != "both":
    models = {args.model: models[args.model]}

for name, sim in models.items():
    out = ROOT / "data" / name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for g, k, nz, s in itertools.product(sim.GRAPHS, args.kappas, args.noise, range(args.seeds)):
        cfg = sim.Config(graph=g, kappa=k, noise=nz, seed=s)
        ds = sim.generate(cfg)
        ds.df.to_csv(out / f"{cfg.name}.csv", index=False, float_format="%.4f")
        rows.append({"file": f"{cfg.name}.csv", **asdict(cfg), "description": sim.GRAPH_DESC[g],
                     "base_rate": ds.y.mean(), **sim.true_effects(ds)})
    man = pd.DataFrame(rows)
    man.to_csv(out / "manifest.csv", index=False)
    print(f"{name}: {len(man)} datasets -> {out}")
    show = [c for c in ["base_rate", "TE", "TE_premium_eur"] if c in man]
    print(man.groupby(["graph", "kappa"])[show].mean().round(3).to_string(), "\n")
