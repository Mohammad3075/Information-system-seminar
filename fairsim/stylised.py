"""
Car insurance SCM simulator for
"Fairness under an Unobserved Unfair Attribute" (Seminar IS, WiSe 26/27, Topic 2).

The insurer predicts whether a driver files a claim next year. Gender is NOT recorded
(EU Test-Achats ruling), but it can still reach the claim label through other variables.

Variables (theory symbol in brackets)
    gender       [A]   1 = male, 0 = female            hidden   (unfair attribute)
    skill        [U]   driving skill, N(0,1)           hidden   (latent merit)
    experience   [XL]  years of driving experience     observed (legitimate)
    engine_kw    [XP]  engine power in kW              observed (proxy: no effect on claims)
    mileage      [XM]  km driven per year              observed (mediator: raises claim risk)
    fault_bias   [M]   biased fault assignment         hidden   (hidden mediator)
    claim        [Y]   claim next year, 0/1            observed as label

Structural equations (z_* are the standardised versions used inside the claim model)
    gender     ~ Bernoulli(0.5)
    skill      ~ N(0, 1)
    z_exp      = skill + e_L                 experience = 15 + 4 * z_exp   (years, >= 0)
    z_engine   = alpha * gender + e_P        engine_kw  = 100 + 25 * z_engine
    z_mileage  = k_M * gender + e_M          mileage    = 13000 + 4000 * z_mileage
    fault_bias = k_H * gender + e_H
    p_claim    = sigmoid(b0 - b_L*z_exp - b_U*skill + b_M*z_mileage + b_H*fault_bias + k_D*gender)
    claim      = 1[u_Y < p_claim]

Unfair edges out of gender, switched on per graph:
    G0  none                                 (fair reference)
    G1  gender -> claim                      (direct: biased claim records)
    G2  gender -> mileage -> claim           (observed mediator)
    G3  gender -> fault_bias -> claim        (hidden mediator)
    G4  gender -> claim and gender -> mileage
gender -> engine_kw exists in every graph (the proxy).

All unfair edges raise the claim probability of men, so a positive effect = men pay more.
All noise is stored, so counterfactuals are exact: re-run the equations with the same
noise and gender switched, on all edges (total effect) or on one edge (path-specific).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Union

import numpy as np
import pandas as pd

# Edges out of gender, named by the child that receives it.
EDGES = ("engine_kw", "mileage", "fault_bias", "claim")

GRAPHS: Dict[str, Dict[str, bool]] = {
    "G0": dict(mileage=False, fault_bias=False, claim=False),
    "G1": dict(mileage=False, fault_bias=False, claim=True),
    "G2": dict(mileage=True,  fault_bias=False, claim=False),
    "G3": dict(mileage=False, fault_bias=True,  claim=False),
    "G4": dict(mileage=True,  fault_bias=False, claim=True),
}
GRAPH_DESC = {
    "G0": "no unfair path (fair reference)",
    "G1": "direct: gender -> claim",
    "G2": "observed mediator: gender -> mileage -> claim",
    "G3": "hidden mediator: gender -> fault_bias -> claim",
    "G4": "direct + observed mediator",
}

OBSERVED = ["engine_kw", "mileage", "experience"]   # model inputs
LABEL = "claim"
HIDDEN = ["gender", "skill", "fault_bias", "p_claim"]  # never give to a model

# fixed kW cut points for the categorical car class (small, compact, mid, large)
CAR_CLASS_CUTS = np.array([80.0, 110.0, 140.0])


@dataclass(frozen=True)
class Config:
    graph: str = "G0"
    kappa: float = 1.0           # strength of every active unfair edge
    noise: str = "gaussian"      # "gaussian" | "laplace" (both unit variance)
    n: int = 5000
    seed: int = 0
    alpha: float = 1.0           # proxy strength gender -> engine_kw
    p_male: float = 0.5
    b0: float = -2.0             # intercept: base claim rate ~25-35%
    b_L: float = 1.0             # experience lowers claims
    b_U: float = 1.0             # skill lowers claims
    b_M: float = 1.0             # mileage raises claims
    b_H: float = 1.0             # fault assignment raises recorded claims
    car_class: bool = False      # True: add categorical car_class (0..3) and use it instead of engine_kw

    def __post_init__(self):
        if self.graph not in GRAPHS:
            raise ValueError(f"graph must be one of {list(GRAPHS)}")
        if self.noise not in ("gaussian", "laplace"):
            raise ValueError("noise must be 'gaussian' or 'laplace'")

    def kappas(self) -> Dict[str, float]:
        on = GRAPHS[self.graph]
        return {e: (self.kappa if on[e] else 0.0) for e in on}

    @property
    def observed(self):
        return ["car_class", "mileage", "experience"] if self.car_class else OBSERVED

    @property
    def name(self) -> str:
        return f"{self.graph}_k{self.kappa:g}_{self.noise}_s{self.seed}"


def _noise(rng: np.random.Generator, kind: str, n: int) -> np.ndarray:
    if kind == "gaussian":
        return rng.standard_normal(n)
    return rng.laplace(0.0, 1.0 / np.sqrt(2.0), n)   # Var = 2 b^2 = 1


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _structural(cfg: Config, exo: Dict[str, np.ndarray], g=None) -> pd.DataFrame:
    """Compute every variable from the stored noise.
    g = None -> factual; 0/1/array -> gender set on all edges;
    dict {edge: value} -> gender changed only on those edges (others keep factual)."""
    G = exo["gender"]
    if g is None:
        ge = {e: G for e in EDGES}
    elif isinstance(g, dict):
        ge = {e: np.asarray(g.get(e, G)) for e in EDGES}
    else:
        ge = {e: np.broadcast_to(np.asarray(g), G.shape) for e in EDGES}

    k = cfg.kappas()
    skill = exo["skill"]
    z_exp = skill + exo["e_L"]
    z_engine = cfg.alpha * ge["engine_kw"] + exo["e_P"]
    z_mileage = k["mileage"] * ge["mileage"] + exo["e_M"]
    fault_bias = k["fault_bias"] * ge["fault_bias"] + exo["e_H"]
    logit = (cfg.b0 - cfg.b_L * z_exp - cfg.b_U * skill + cfg.b_M * z_mileage
             + cfg.b_H * fault_bias + k["claim"] * ge["claim"])
    p_claim = _sigmoid(logit)

    df = pd.DataFrame(dict(
        engine_kw=100 + 25 * z_engine,
        mileage=13000 + 4000 * z_mileage,
        experience=np.clip(15 + 4 * z_exp, 0, None),
        claim=(exo["u_Y"] < p_claim).astype(int),
        gender=G, skill=skill, fault_bias=fault_bias, p_claim=p_claim,
    ))
    if cfg.car_class:
        df.insert(0, "car_class", np.searchsorted(CAR_CLASS_CUTS, df["engine_kw"]))
    return df


@dataclass
class Dataset:
    cfg: Config
    exo: Dict[str, np.ndarray]
    df: pd.DataFrame = field(default=None)

    @property
    def X(self) -> pd.DataFrame:
        return self.df[self.cfg.observed]

    @property
    def y(self) -> np.ndarray:
        return self.df[LABEL].to_numpy()

    @property
    def gender(self) -> np.ndarray:
        return self.df["gender"].to_numpy()

    def intervene(self, g: Union[int, np.ndarray, Dict[str, Union[int, np.ndarray]]]) -> pd.DataFrame:
        """Same drivers (same noise), gender set to g. See _structural for the forms of g."""
        return _structural(self.cfg, self.exo, g)

    def counterfactual(self, paths=EDGES) -> pd.DataFrame:
        """Each driver with gender flipped, only along the given edges."""
        flipped = 1 - self.exo["gender"]
        return self.intervene({e: flipped for e in paths})


def generate(cfg: Config) -> Dataset:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n
    exo = dict(
        gender=(rng.random(n) < cfg.p_male).astype(int),
        skill=rng.standard_normal(n),
        e_L=_noise(rng, cfg.noise, n),
        e_P=_noise(rng, cfg.noise, n),
        e_M=_noise(rng, cfg.noise, n),
        e_H=_noise(rng, cfg.noise, n),
        u_Y=rng.random(n),
    )
    ds = Dataset(cfg=cfg, exo=exo)
    ds.df = _structural(cfg, exo)
    return ds


def _one_edge(e):
    return {e: 1, **{o: 0 for o in EDGES if o != e}}


def true_effects(ds: Dataset) -> Dict[str, float]:
    """Effect of switching gender female -> male on the TRUE claim probability:
    total (TE) and per unfair edge (PSE_*). Positive = men have more claims."""
    p = lambda g: ds.intervene(g)["p_claim"].mean()
    base = p(0)
    out = {"TE": p(1) - base}
    for e in ("mileage", "fault_bias", "claim"):
        out[f"PSE_{e}"] = p(_one_edge(e)) - base
    return out


def model_effects(ds: Dataset, predict) -> Dict[str, float]:
    """Causal unfairness of a fitted model: mean change in its prediction when gender
    is switched female -> male, total and per edge reaching the inputs.
    predict: DataFrame of observed features -> scores, e.g. lambda X: clf.predict_proba(X)[:, 1]"""
    cols = ds.cfg.observed
    f = lambda g: np.mean(predict(ds.intervene(g)[cols]))
    base = f(0)
    out = {"TE": f(1) - base}
    for e in ("engine_kw", "mileage"):
        out[f"PSE_{e}"] = f(_one_edge(e)) - base
    return out


def equivalent_pair(kappa: float = 1.0, n: int = 5000, seed: int = 0, noise: str = "gaussian"):
    """H3: two worlds with IDENTICAL observed data but opposite fairness verdicts.

      unfair world: the binary factor behind engine power and mileage is gender
                    -> gender -> mileage -> claim is an unfair path (graph G2).
      fair world:   the same factor is a legitimate one, e.g. rural residence
                    (column 'rural'); gender is an independent coin flip with no effect.

    Returns (unfair_df, fair_df). Observed columns and labels are equal row by row, so no
    method using only observed data can tell the worlds apart.
    """
    unfair = generate(Config(graph="G2", kappa=kappa, n=n, seed=seed, noise=noise))
    fair = unfair.df.copy()
    fair["rural"] = unfair.df["gender"]
    fair["gender"] = (np.random.default_rng(seed + 10_000).random(n) < 0.5).astype(int)
    return unfair.df, fair
