"""
Realistic car insurance simulator with a hidden unfair attribute (gender).

Same idea and same interface as `fairsim.stylised`, but closer to a real motor tariff:
  * 10 observed rating factors, mixed continuous and categorical
  * non-linear (U-shaped) age risk
  * several weak gender proxies instead of one strong one
  * claim COUNTS (Poisson) and a premium in EUR

Hidden variables
    gender        1 = male, 0 = female                      (unfair attribute A)
    skill         driving skill, N(0, 1)                     (latent merit U)
    fault_bias    biased fault assignment in claims handling (hidden mediator M)

Observed rating factors (what the insurer sees)
    age             18-80 years                    legitimate, U-shaped risk
    years_licensed  years since licence            legitimate (from age)
    region          urban / suburban / rural       legitimate
    parking         garage / street                legitimate (from region)
    vehicle_age     years                          legitimate
    bonus_class     no-claims bonus 0-15           legitimate (from skill, years_licensed)
    occupation      office / manual / student / retired   PROXY (gender, age); no effect on claims
    engine_kw       engine power                   PROXY (gender); no effect on claims
    car_value       EUR                            PROXY (from engine_kw); no effect on claims
    mileage         km per year                    legitimate risk; MEDIATOR when gender -> mileage is on

Outcome
    lambda      = exp(b0 + age_risk + region + parking + vehicle_age + bonus + skill
                      + log(mileage) + fault_bias + k_D * gender)       expected claims per year
    claim_count = Poisson(lambda)   (inverse-CDF of a stored uniform -> exact counterfactuals)
    claim       = 1[claim_count > 0]
    p_claim     = 1 - exp(-lambda)
    premium     = lambda * mean claim size * loading                (EUR per year, true risk premium)

Unfair edges out of gender, switched on per graph (same as the stylised model):
    G0 none | G1 gender -> claim | G2 gender -> mileage | G3 gender -> fault_bias | G4 G1 + G2
gender -> occupation and gender -> engine_kw (proxies) exist in every graph.
Positive effects = men have more claims / pay more.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Union

import numpy as np
import pandas as pd
from scipy.stats import poisson

EDGES = ("engine_kw", "occupation", "mileage", "fault_bias", "claim")
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

OBSERVED = ["age", "years_licensed", "region", "parking", "vehicle_age", "bonus_class",
            "occupation", "engine_kw", "car_value", "mileage"]
CATEGORICAL = {"region": ["urban", "suburban", "rural"],
               "parking": ["garage", "street"],
               "occupation": ["office", "manual", "student", "retired"]}
AGE_BANDS = [18, 25, 35, 50, 65, 81]
LABEL = "claim"
HIDDEN = ["gender", "skill", "fault_bias", "lam", "p_claim", "premium"]
DESCENDANTS_OF_GENDER = ["occupation", "engine_kw", "car_value", "mileage"]

MEAN_CLAIM_SIZE = 3500.0     # EUR
LOADING = 1.25               # costs + profit


@dataclass(frozen=True)
class Config:
    graph: str = "G0"
    kappa: float = 1.0           # strength of every active unfair edge
    noise: str = "gaussian"      # "gaussian" | "laplace"
    n: int = 20000
    seed: int = 0
    alpha: float = 1.0           # strength of the proxies (gender -> engine_kw, occupation)
    p_male: float = 0.5
    b0: float = -2.55            # intercept: ~10% of drivers claim per year

    def __post_init__(self):
        if self.graph not in GRAPHS:
            raise ValueError(f"graph must be one of {list(GRAPHS)}")
        if self.noise not in ("gaussian", "laplace"):
            raise ValueError("noise must be 'gaussian' or 'laplace'")

    def kappas(self) -> Dict[str, float]:
        on = GRAPHS[self.graph]
        return {e: (self.kappa if on[e] else 0.0) for e in on}

    @property
    def name(self) -> str:
        return f"real_{self.graph}_k{self.kappa:g}_{self.noise}_s{self.seed}"


def _noise(rng, kind, n):
    if kind == "gaussian":
        return rng.standard_normal(n)
    return rng.laplace(0.0, 1.0 / np.sqrt(2.0), n)


def age_risk(age):
    """U-shaped age effect on log claim frequency: young drivers high, middle low, 70+ rising."""
    return 1.3 * np.exp(-(age - 18) / 6.0) + 0.4 * np.clip((age - 65) / 10.0, 0, None) ** 2


def _structural(cfg: Config, exo: Dict[str, np.ndarray], g=None) -> pd.DataFrame:
    G = exo["gender"]
    if g is None:
        ge = {e: G for e in EDGES}
    elif isinstance(g, dict):
        ge = {e: np.asarray(g.get(e, G)) for e in EDGES}
    else:
        ge = {e: np.broadcast_to(np.asarray(g), G.shape) for e in EDGES}
    k = cfg.kappas()
    n = len(G)
    skill = exo["skill"]

    # --- legitimate background factors (not caused by gender) ---
    age = exo["age"]
    years_licensed = np.clip(age - 18 - exo["licence_delay"], 0, None)
    region = np.array(CATEGORICAL["region"])[np.searchsorted([0.40, 0.75], exo["u_region"])]
    rural = (region == "rural").astype(float)
    urban = (region == "urban").astype(float)
    p_garage = 0.35 + 0.35 * rural - 0.15 * urban
    parking = np.where(exo["u_parking"] < p_garage, "garage", "street")
    vehicle_age = exo["vehicle_age"]
    bonus_class = np.clip(np.round(0.6 * np.minimum(years_licensed, 20) + 1.5 * skill
                                   + 1.5 * exo["e_bonus"]), 0, 15)

    # --- proxies: caused by gender, no effect on claims ---
    a_occ = cfg.alpha * ge["occupation"]
    is_student = exo["u_occ1"] < np.where(age < 26, 0.6, 0.02)
    is_retired = (age >= 63) & (exo["u_occ1"] < 0.85)
    is_manual = exo["u_occ2"] < 0.15 + 0.30 * a_occ          # men more often manual jobs
    occupation = np.select([is_student, is_retired, is_manual], ["student", "retired", "manual"], "office")
    z_engine = 0.9 * cfg.alpha * ge["engine_kw"] + exo["e_engine"]
    engine_kw = np.clip(95 + 22 * z_engine - 6 * (age < 25), 40, None)   # young drivers: smaller cars
    car_value = np.exp(9.2 + 0.009 * engine_kw + 0.25 * exo["e_value"] - 0.06 * vehicle_age)

    # --- mediator: mileage (legit risk, unfair when gender -> mileage is on) ---
    z_mileage = 1.2 * k["mileage"] * ge["mileage"] + 0.5 * rural + exo["e_mileage"]
    mileage = np.clip(12000 + 4000 * z_mileage, 1000, None)

    # --- hidden mediator: biased fault assignment ---
    fault_bias = 0.4 * k["fault_bias"] * ge["fault_bias"] + 0.3 * exo["e_fault"]

    # --- claim frequency ---
    log_lam = (cfg.b0 + age_risk(age) + 0.30 * urban + 0.10 * (region == "suburban")
               + 0.15 * (parking == "street") + 0.02 * vehicle_age - 0.06 * bonus_class
               - 0.35 * skill + 0.9 * np.log(mileage / 12000) + fault_bias
               + 0.40 * k["claim"] * ge["claim"])
    lam = np.exp(log_lam)
    claim_count = poisson.ppf(exo["u_claim"], lam).astype(int)

    df = pd.DataFrame(dict(
        age=np.round(age).astype(int), years_licensed=np.round(years_licensed).astype(int),
        region=region, parking=parking, vehicle_age=vehicle_age.astype(int),
        bonus_class=bonus_class.astype(int), occupation=occupation,
        engine_kw=np.round(engine_kw, 1), car_value=np.round(car_value, -1),
        mileage=np.round(mileage, -2),
        claim_count=claim_count, claim=(claim_count > 0).astype(int),
        gender=G, skill=skill, fault_bias=fault_bias, lam=lam, p_claim=1 - np.exp(-lam),
        premium=lam * MEAN_CLAIM_SIZE * LOADING,
    ))
    for c, cats in CATEGORICAL.items():
        df[c] = pd.Categorical(df[c], categories=cats)
    return df


def design_matrix(df: pd.DataFrame, drop=()) -> pd.DataFrame:
    """Turn observed rating factors into numeric model inputs, as an insurer would:
    one-hot categories, age bands (captures the U-shape), log of car value."""
    X = df[[c for c in OBSERVED if c not in drop]].copy()
    if "age" in X:
        X["age_band"] = pd.cut(X["age"], AGE_BANDS, right=False,
                               labels=[f"{a}-{b - 1}" for a, b in zip(AGE_BANDS[:-1], AGE_BANDS[1:])])
    if "car_value" in X:
        X["car_value"] = np.log(X["car_value"])
    if "mileage" in X:
        X["mileage"] = X["mileage"] / 1000
    cats = [c for c in list(CATEGORICAL) + ["age_band"] if c in X]
    return pd.get_dummies(X, columns=cats, drop_first=True, dtype=float)


@dataclass
class Dataset:
    cfg: Config
    exo: Dict[str, np.ndarray]
    df: pd.DataFrame = field(default=None)

    def features(self, drop=()) -> pd.DataFrame:
        return design_matrix(self.df, drop)

    @property
    def X(self) -> pd.DataFrame:
        return self.features()

    @property
    def y(self) -> np.ndarray:
        return self.df[LABEL].to_numpy()

    @property
    def gender(self) -> np.ndarray:
        return self.df["gender"].to_numpy()

    def intervene(self, g: Union[int, np.ndarray, Dict[str, Union[int, np.ndarray]]]) -> pd.DataFrame:
        return _structural(self.cfg, self.exo, g)

    def counterfactual(self, paths=EDGES) -> pd.DataFrame:
        flipped = 1 - self.exo["gender"]
        return self.intervene({e: flipped for e in paths})


def generate(cfg: Config) -> Dataset:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n
    exo = dict(
        gender=(rng.random(n) < cfg.p_male).astype(int),
        skill=rng.standard_normal(n),
        age=18 + 62 * rng.beta(1.6, 2.2, n),
        licence_delay=rng.exponential(2.5, n),
        u_region=rng.random(n), u_parking=rng.random(n),
        vehicle_age=rng.poisson(7, n).astype(float),
        e_bonus=_noise(rng, cfg.noise, n),
        u_occ1=rng.random(n), u_occ2=rng.random(n),
        e_engine=_noise(rng, cfg.noise, n), e_value=_noise(rng, cfg.noise, n),
        e_mileage=_noise(rng, cfg.noise, n), e_fault=_noise(rng, cfg.noise, n),
        u_claim=rng.random(n),
    )
    ds = Dataset(cfg=cfg, exo=exo)
    ds.df = _structural(cfg, exo)
    return ds


def _one_edge(e):
    return {e: 1, **{o: 0 for o in EDGES if o != e}}


def true_effects(ds: Dataset) -> Dict[str, float]:
    """Effect of switching gender female -> male on the TRUE claim probability, total and per
    unfair edge, plus the total effect on the true risk premium in EUR."""
    w0 = ds.intervene(0)
    w1 = ds.intervene(1)
    out = {"TE": w1["p_claim"].mean() - w0["p_claim"].mean(),
           "TE_premium_eur": w1["premium"].mean() - w0["premium"].mean()}
    for e in ("mileage", "fault_bias", "claim"):
        out[f"PSE_{e}"] = ds.intervene(_one_edge(e))["p_claim"].mean() - w0["p_claim"].mean()
    return out


def premium_from_prob(p):
    """Price a predicted claim probability: lambda = -log(1 - p), premium = lambda * size * loading."""
    return -np.log1p(-np.clip(p, 0, 0.999)) * MEAN_CLAIM_SIZE * LOADING


def model_effects(ds: Dataset, predict, drop=()) -> Dict[str, float]:
    """Causal unfairness of a fitted model: mean change in its predicted claim probability
    (and premium) when gender is switched female -> male, total and per proxy/mediator edge.
    predict: numeric design matrix -> probabilities. `drop` must match the training columns."""
    cols = design_matrix(ds.df, drop).columns
    f = lambda g: predict(design_matrix(ds.intervene(g), drop).reindex(columns=cols, fill_value=0.0))
    p0, p1 = f(0), f(1)
    out = {"TE": p1.mean() - p0.mean(),
           "TE_premium_eur": premium_from_prob(p1).mean() - premium_from_prob(p0).mean()}
    for e in ("engine_kw", "occupation", "mileage"):
        out[f"PSE_{e}"] = f(_one_edge(e)).mean() - p0.mean()
    return out
