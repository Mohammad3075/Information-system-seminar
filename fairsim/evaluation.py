"""Shared evaluation: train the baseline models on one simulated dataset and score them
against the hidden ground truth. Works for both simulators.

Learners
    logreg    logistic regression (a GLM): transparent baseline, the insurer's standard
    xgboost   gradient-boosted trees: powerful black box, learns interactions

Model variants
    unaware         all observed features (gender is never available)
    graph-informed  drop every descendant of gender (assumes the causal graph is known)
    surrogate-fair  all observed features + Exponentiated Gradient (fairlearn) with a
                    demographic-parity constraint on a SURROGATE gender: the true gender with
                    20% of values flipped, as if the insurer had inferred it (e.g. from names)
"""
from __future__ import annotations

import warnings

import numpy as np
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from . import realistic, stylised

MODELS = {"stylised": stylised, "realistic": realistic}
VARIANTS = ("unaware", "graph-informed", "surrogate-fair")
LEARNERS = ("logreg", "xgboost")


def make_learner(name: str, seed: int = 0, balanced: bool = False, pos_weight: float = 1.0):
    """Return an unfitted classifier. Hyper-parameters are moderate defaults for tabular data.
    balanced=True up-weights claims (needed inside Exponentiated Gradient, see SurrogateFair)."""
    if name == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced" if balanced else None)
    if name == "xgboost":
        return XGBClassifier(n_estimators=150 if balanced else 300, max_depth=2,
                             learning_rate=0.1 if balanced else 0.05,
                             scale_pos_weight=pos_weight if balanced else 1.0,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_lambda=5,
                             random_state=seed, n_jobs=4, eval_metric="logloss")
    raise ValueError(f"learner must be one of {LEARNERS}")


class SurrogateFair:
    """Exponentiated Gradient (Agarwal et al., 2018) with a demographic-parity constraint,
    trained on a noisy surrogate of the hidden gender.

    Steps
      1. surrogate gender = true gender with `noise_rate` of the values flipped
      2. inputs standardised (fast, stable training)
      3. ExponentiatedGradient(base learner, DemographicParity(eps)) with class-balanced base
         learners (claims are rare; without balancing EG returns "nobody claims")
      4. score = weighted mix of the learned predictors' probabilities
      5. one global Platt recalibration (no gender involved) so scores are real
         probabilities again and premiums stay on the right scale
    At prediction time no gender (true or surrogate) is needed.
    """

    def __init__(self, learner="logreg", seed=0, noise_rate=0.2, eps=0.01, max_iter=30):
        self.learner, self.seed, self.noise_rate = learner, seed, noise_rate
        self.eps, self.max_iter = eps, max_iter

    def fit(self, X, y, gender):
        rng = np.random.default_rng(self.seed + 777)
        self.surrogate_ = np.where(rng.random(len(gender)) < self.noise_rate, 1 - gender, gender)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        pos_weight = (1 - y.mean()) / y.mean()
        base = make_learner(self.learner, self.seed, balanced=True, pos_weight=pos_weight)
        self.eg_ = ExponentiatedGradient(base, DemographicParity(difference_bound=self.eps),
                                         max_iter=self.max_iter)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.eg_.fit(Xs, y, sensitive_features=self.surrogate_)
        z = self._logit(self._mix(Xs))
        self.calib_ = LogisticRegression().fit(z.reshape(-1, 1), y)
        return self

    def _mix(self, Xs):
        p = np.zeros(len(Xs))
        for w, m in zip(self.eg_.weights_, self.eg_.predictors_):
            if w > 0:
                p += w * m.predict_proba(Xs)[:, 1]
        return p

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def predict_proba(self, X):
        z = self._logit(self._mix(self.scaler_.transform(X)))
        p = self.calib_.predict_proba(z.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - p, p])


def _features(model, ds, variant):
    """Returns (X, predict_wrapper_builder) for the chosen simulator and variant."""
    if model == "stylised":
        cols = ["experience"] if variant == "graph-informed" else stylised.OBSERVED
        X = ds.X[cols]
        effects = lambda clf: stylised.model_effects(ds, lambda Z: clf.predict_proba(Z[cols])[:, 1])
    else:
        drop = realistic.DESCENDANTS_OF_GENDER if variant == "graph-informed" else ()
        X = ds.features(drop)
        effects = lambda clf: realistic.model_effects(ds, lambda Z: clf.predict_proba(Z)[:, 1], drop=drop)
    return X, effects


def evaluate(model: str, graph: str, kappa: float = 1.0, seed: int = 0, noise: str = "gaussian",
             learners=("logreg",), variants=VARIANTS, noise_rate=0.2):
    """Generate one dataset, fit every learner x variant (70/30 split), return one row each."""
    sim = MODELS[model]
    ds = sim.generate(sim.Config(graph=graph, kappa=kappa, seed=seed, noise=noise))
    truth = sim.true_effects(ds)
    tr, te = train_test_split(np.arange(ds.cfg.n), test_size=0.3, random_state=seed,
                              stratify=ds.y)
    rows = []
    for learner in learners:
        for v in variants:
            X, effects = _features(model, ds, v)
            if v == "surrogate-fair":
                clf = SurrogateFair(learner, seed, noise_rate=noise_rate).fit(
                    X.iloc[tr], ds.y[tr], ds.gender[tr])
            else:
                clf = make_learner(learner, seed).fit(X.iloc[tr], ds.y[tr])
            s = clf.predict_proba(X.iloc[te])[:, 1]
            G = ds.gender[te]
            me = effects(clf)
            rows.append(dict(
                model=model, learner=learner, graph=graph, kappa=kappa, seed=seed, noise=noise, variant=v,
                base_rate=ds.y.mean(),
                AUC=roc_auc_score(ds.y[te], s),
                score_gap=s[G == 1].mean() - s[G == 0].mean(),   # observed gap, men - women
                **{f"model_{k}": val for k, val in me.items()},  # causal effect on the model
                **{f"true_{k}": val for k, val in truth.items()},
            ))
    return rows
