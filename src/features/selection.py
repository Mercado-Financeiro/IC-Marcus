"""Feature selection and dimensionality reduction utilities.

This module provides small, leak-safe helpers to reduce feature space
for faster experiments: PCA and SelectKBest. It follows the contract
that any fitting occurs on training data only, and the same transform
is applied to validation/test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


@dataclass
class SelectionResult:
    X_train: pd.DataFrame
    X_val: Optional[pd.DataFrame]
    X_test: Optional[pd.DataFrame]
    feature_names: List[str]
    method: str
    details: dict


def apply_pca(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    n_components: int = 50,
    random_state: int = 42,
) -> SelectionResult:
    """Fit PCA on train and transform val/test.

    Returns DataFrames with columns PC1..PCk to keep names explicit.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    Xt = pca.fit_transform(X_train.values)
    cols = [f"PC{i+1}" for i in range(Xt.shape[1])]

    X_train_t = pd.DataFrame(Xt, index=X_train.index, columns=cols)

    def _tx(X: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if X is None:
            return None
        Xv = pca.transform(X.values)
        return pd.DataFrame(Xv, index=X.index, columns=cols)

    return SelectionResult(
        X_train=X_train_t,
        X_val=_tx(X_val),
        X_test=_tx(X_test),
        feature_names=cols,
        method="pca",
        details={
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "n_components": len(cols),
        },
    )


def apply_select_kbest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    k: int = 100,
    score_func: str = "f_classif",
) -> SelectionResult:
    """Fit SelectKBest on train and transform val/test.

    score_func: one of {"f_classif", "mutual_info"}
    """
    if score_func == "mutual_info":
        scorer = mutual_info_classif
    else:
        scorer = f_classif

    skb = SelectKBest(score_func=scorer, k=min(k, X_train.shape[1]))
    skb.fit(X_train, y_train)

    selected_mask = skb.get_support()
    selected_cols = [c for c, m in zip(X_train.columns, selected_mask) if m]

    def _tx(X: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if X is None:
            return None
        return X.loc[:, selected_cols]

    return SelectionResult(
        X_train=_tx(X_train),
        X_val=_tx(X_val),
        X_test=_tx(X_test),
        feature_names=selected_cols,
        method="select_kbest",
        details={
            "k": len(selected_cols),
            "score_func": score_func,
        },
    )

