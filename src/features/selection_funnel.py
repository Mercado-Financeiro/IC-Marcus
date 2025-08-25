from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, matthews_corrcoef
from xgboost import XGBClassifier  # type: ignore

from src.data.splits import PurgedKFold


@dataclass
class FunnelConfig:
    corr_threshold: float = 0.92
    mrmr_k: int = 50
    final_k: int = 40
    lambda_penalty: float = 0.02
    n_splits: int = 5
    embargo: int = 10
    seed: int = 42


def _spearman_corr_abs(X: pd.DataFrame) -> pd.DataFrame:
    return X.rank(axis=0).corr(method="pearson").abs()


def _auprc_norm_cv(x: np.ndarray, y: np.ndarray, n_splits: int, embargo: int) -> float:
    cv = PurgedKFold(n_splits=n_splits, embargo=embargo)
    scores = []
    for tr, va in cv.split(x, y):
        yv = y[va]
        xv = x[va]
        # simple monotonic transform if constant
        if np.all(xv == xv[0]):
            p = np.full_like(yv, yv.mean(), dtype=float)
        else:
            # normalize feature then map to prob via min-max
            xv_std = (xv - np.nanmin(xv)) / (np.nanmax(xv) - np.nanmin(xv) + 1e-12)
            p = np.clip(xv_std, 0, 1)
        pr = average_precision_score(yv, p)
        p_base = float(yv.mean())
        scores.append((pr - p_base) / (1 - p_base + 1e-12))
    return float(np.mean(scores)) if scores else 0.0


def prune_redundancy_spearman(
    X: pd.DataFrame, y: pd.Series, cfg: FunnelConfig
) -> List[str]:
    """Greedy cluster by correlation and pick medoids via CV AUPRC_norm.

    Returns list of representative feature names.
    """
    corr = _spearman_corr_abs(X)
    reps: List[str] = []
    assigned: Dict[str, str] = {}
    # Sort by variance (higher first) to seed clusters with informative cols
    order = X.var().sort_values(ascending=False).index.tolist()
    for f in order:
        if f in assigned:
            continue
        reps.append(f)
        # assign neighbors highly correlated to this rep
        high = corr.index[(corr[f] >= cfg.corr_threshold)].tolist()
        for h in high:
            if h not in assigned:
                assigned[h] = f
    # For each rep cluster, choose medoid by best univariate normalized AUPRC
    medoids: List[str] = []
    y_arr = y.values.astype(int)
    for rep in reps:
        cluster_members = [k for k, v in assigned.items() if v == rep] or [rep]
        best_f = None
        best_s = -1.0
        for f in cluster_members:
            s = _auprc_norm_cv(X[f].values, y_arr, cfg.n_splits, cfg.embargo)
            if s > best_s:
                best_s, best_f = s, f
        if best_f is not None:
            medoids.append(best_f)
    # Deduplicate and keep order
    seen = set()
    out = []
    for f in order:
        if f in medoids and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def mrmr_time_aware(
    X: pd.DataFrame, y: pd.Series, candidates: List[str], k: int, cfg: FunnelConfig
) -> List[str]:
    y_arr = y.values.astype(int)
    rel: Dict[str, float] = {
        f: _auprc_norm_cv(X[f].values, y_arr, cfg.n_splits, cfg.embargo) for f in candidates
    }
    selected: List[str] = []
    remaining = set(candidates)
    corr = _spearman_corr_abs(X[candidates])
    while len(selected) < min(k, len(remaining)) and remaining:
        best_f, best_s = None, -1e9
        for f in list(remaining):
            red = 0.0
            if selected:
                red = float(np.mean([corr.loc[f, s] for s in selected]))
            score = rel.get(f, 0.0) - red
            if score > best_s:
                best_s, best_f = score, f
        if best_f is None:
            break
        selected.append(best_f)
        remaining.remove(best_f)
    return selected


def _cv_mcc(model: XGBClassifier, X: pd.DataFrame, y: pd.Series, cfg: FunnelConfig) -> float:
    cv = PurgedKFold(n_splits=cfg.n_splits, embargo=cfg.embargo)
    scores = []
    Xv = X.values
    yv = y.values.astype(int)
    for tr, va in cv.split(Xv, yv):
        model.fit(Xv[tr], yv[tr])
        p = model.predict_proba(Xv[va])[:, 1]
        yhat = (p >= 0.5).astype(int)
        scores.append(matthews_corrcoef(yv[va], yhat))
    return float(np.mean(scores)) if scores else 0.0


def sffs_wrapper(
    X: pd.DataFrame,
    y: pd.Series,
    pool: List[str],
    cfg: FunnelConfig,
) -> List[str]:
    rng = np.random.default_rng(cfg.seed)
    selected: List[str] = []
    remaining = list(pool)
    best_J = -1e9
    patience = 2
    no_improve = 0
    while remaining and len(selected) < cfg.final_k:
        cand_best, cand_feat = -1e9, None
        for f in remaining:
            feats = selected + [f]
            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=2.0,
                tree_method="hist",
                random_state=cfg.seed,
                eval_metric="aucpr",
            )
            mcc = _cv_mcc(model, X[feats], y, cfg)
            J = mcc - cfg.lambda_penalty * np.sqrt(len(feats))
            if J > cand_best:
                cand_best, cand_feat = J, f
        if cand_feat is None:
            break
        selected.append(cand_feat)
        remaining.remove(cand_feat)
        if cand_best > best_J + 1e-4:
            best_J = cand_best
            no_improve = 0
        else:
            no_improve += 1
        # Backward step
        improved = True
        while improved and len(selected) > 2:
            improved = False
            for f in list(selected):
                trial = [s for s in selected if s != f]
                model = XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    tree_method="hist",
                    random_state=cfg.seed,
                    eval_metric="aucpr",
                )
                mcc = _cv_mcc(model, X[trial], y, cfg)
                J = mcc - cfg.lambda_penalty * np.sqrt(len(trial))
                if J > best_J + 1e-4:
                    best_J = J
                    selected = trial
                    improved = True
                    break
        if no_improve >= patience:
            break
    return selected


def run_funnel(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    embargo: int = 10,
    corr_threshold: float = 0.92,
    mrmr_k: int = 50,
    final_k: int = 40,
    lambda_penalty: float = 0.02,
    seed: int = 42,
) -> List[str]:
    cfg = FunnelConfig(
        corr_threshold=corr_threshold,
        mrmr_k=mrmr_k,
        final_k=final_k,
        lambda_penalty=lambda_penalty,
        n_splits=n_splits,
        embargo=embargo,
        seed=seed,
    )
    # 1) Redundancy prune
    reps = prune_redundancy_spearman(X, y, cfg)
    # 2) mRMR
    mrmr_sel = mrmr_time_aware(X, y, reps, min(cfg.mrmr_k, len(reps)), cfg)
    # 3) SFFS wrapper
    final_sel = sffs_wrapper(X, y, mrmr_sel, cfg)
    return final_sel

