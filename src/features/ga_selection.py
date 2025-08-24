"""Genetic Algorithm based feature subset selection for XGBoost.

This implements a lightweight GA that selects a binary mask over features
and evaluates fitness using time-aware CV (PurgedKFold) with an XGBoost
classifier. Designed for quick exploratory runs; tune pop/gen for depth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import numpy as np
import pandas as pd
import random
import xgboost as xgb

from src.data.splits import PurgedKFold


@dataclass
class GAConfig:
    population_size: int = 20
    n_generations: int = 5
    crossover_prob: float = 0.75
    mutation_prob: float = 0.1
    elitism: int = 2
    n_splits: int = 3
    embargo: int = 10
    random_state: int = 42
    max_features: Optional[int] = None  # optional cap


@dataclass
class GAResult:
    best_mask: np.ndarray
    best_score: float
    history: List[float]
    selected_features: List[str]


def _evaluate_subset(
    X: pd.DataFrame,
    y: pd.Series,
    mask: np.ndarray,
    n_splits: int,
    embargo: int,
    random_state: int,
    metric: str = 'prauc',
) -> float:
    if not mask.any():
        return -1.0
    Xs = X.loc[:, X.columns[mask]]
    cv = PurgedKFold(n_splits=n_splits, embargo=embargo)
    scores = []
    for tr, va in cv.split(Xs, y):
        X_tr, X_va = Xs.iloc[tr], Xs.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        model = xgb.XGBClassifier(
            tree_method='hist',
            objective='binary:logistic',
            eval_metric='aucpr',
            learning_rate=0.1,
            max_depth=5,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        proba = model.predict_proba(X_va)[:, 1]
        # Primary metric: PR-AUC
        from sklearn.metrics import precision_recall_curve, auc, f1_score
        precision, recall, _ = precision_recall_curve(y_va, proba)
        pr_auc = auc(recall, precision)
        if metric == 'f1':
            thresh_idx = np.argmax(2 * (precision * recall) / (precision + recall + 1e-10)[:-1])
            th = _[thresh_idx] if len(_) > 0 else 0.5
            f1 = f1_score(y_va, (proba >= th).astype(int), zero_division=0)
            scores.append(0.7 * pr_auc + 0.3 * f1)
        else:
            scores.append(pr_auc)
    return float(np.mean(scores))


def run_ga_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: GAConfig,
    metric: str = 'prauc',
    logger: Optional[Callable[[str, dict], None]] = None,
) -> GAResult:
    rng = np.random.RandomState(cfg.random_state)
    random.seed(cfg.random_state)
    n_features = X.shape[1]
    max_feats = cfg.max_features or n_features

    def log(event: str, **kw):
        if logger:
            logger(event, kw)

    # Initialize population with diverse sparsity
    pop = []
    for _ in range(cfg.population_size):
        k = rng.randint(5, max(6, min(max_feats, n_features // 2)))
        mask = np.zeros(n_features, dtype=bool)
        mask[rng.choice(n_features, size=k, replace=False)] = True
        pop.append(mask)

    history: List[float] = []
    best_mask = None
    best_score = -np.inf

    for gen in range(cfg.n_generations):
        fitness = []
        for mask in pop:
            score = _evaluate_subset(
                X, y, mask, cfg.n_splits, cfg.embargo, cfg.random_state, metric
            )
            fitness.append(score)
        fitness = np.array(fitness)
        history.append(float(fitness.max()))

        # Track best
        gen_best_idx = int(fitness.argmax())
        if fitness[gen_best_idx] > best_score:
            best_score = float(fitness[gen_best_idx])
            best_mask = pop[gen_best_idx].copy()
        log("ga_generation_completed", gen=gen, best=best_score)

        # Selection (tournament)
        def tournament(k=3):
            idx = rng.choice(len(pop), size=k, replace=False)
            return pop[idx[np.argmax(fitness[idx])]].copy()

        # Elitism
        elite_idx = fitness.argsort()[-cfg.elitism:][::-1]
        new_pop = [pop[i].copy() for i in elite_idx]

        # Crossover and mutation
        while len(new_pop) < cfg.population_size:
            p1, p2 = tournament(), tournament()
            if rng.rand() < cfg.crossover_prob:
                cx_point = rng.randint(1, n_features - 1)
                c1 = np.r_[p1[:cx_point], p2[cx_point:]]
                c2 = np.r_[p2[:cx_point], p1[cx_point:]]
            else:
                c1, c2 = p1, p2
            # Mutation
            for c in (c1, c2):
                mut_mask = rng.rand(n_features) < cfg.mutation_prob
                c[mut_mask] = ~c[mut_mask]
                # Ensure not empty
                if not c.any():
                    c[rng.randint(0, n_features)] = True
            new_pop.extend([c1, c2])

        pop = new_pop[: cfg.population_size]

    selected = [c for c, m in zip(X.columns, best_mask) if m]
    return GAResult(best_mask=best_mask, best_score=best_score, history=history, selected_features=selected)

