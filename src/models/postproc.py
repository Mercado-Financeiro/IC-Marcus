"""
Post-processing module for model calibration and threshold optimization.
Enhanced version with proper calibration methods.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional, Any
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, f1_score, average_precision_score, matthews_corrcoef, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from .utils_common import try_import


def _brier_score(y_true, p_pred) -> float:
    """Calculate Brier score manually if sklearn not available."""
    n = len(y_true)
    return sum((float(p_pred[i]) - float(y_true[i])) ** 2 for i in range(n)) / max(n, 1)


def choose_threshold_by_ev_tplus1(
    p_val,
    returns_next,
    fee_bps: float,
    slippage_bps: float,
    mode: str = "long_only",
) -> float:
    """Threshold that maximizes EV using next-bar returns (t+1) and costs.

    EV(th) = mean(side_t(th) * ret_{t+1}) - turnover_t(th) * total_cost

    - mode="long_only": side_t in {0, 1}
    - mode="long_short": side_t in {-1, 1}
    """
    import numpy as np

    costs = (fee_bps + slippage_bps) / 1e4
    best_th, best_ev = 0.5, float("-inf")
    p_val = np.asarray(p_val).astype(float)
    returns_next = np.asarray(returns_next).astype(float)

    for k in range(5, 96):
        th = k / 100.0
        if mode == "long_short":
            side = np.where(p_val >= th, 1, -1)
        else:
            side = (p_val >= th).astype(int)
        # EV bruto com execução t+1
        gross = float(np.mean(side * returns_next))
        # Turnover por barra
        turnover = float(np.mean(np.abs(np.diff(side, prepend=0))))
        ev = gross - turnover * costs
        if ev > best_ev:
            best_ev, best_th = ev, th
    return best_th


def choose_threshold_by_f1(p_val, y_val) -> float:
    """Find threshold that maximizes F1 score.
    
    Args:
        p_val: Validation probabilities
        y_val: Validation labels
        
    Returns:
        Optimal threshold for F1
    """
    best_th, best_f1 = 0.5, 0.0
    
    for k in range(5, 96):
        th = k / 100.0
        y_pred = [1 if p >= th else 0 for p in p_val]
        
        # Calculate F1
        tp = sum(1 for i in range(len(y_val)) if y_val[i] == 1 and y_pred[i] == 1)
        fp = sum(1 for i in range(len(y_val)) if y_val[i] == 0 and y_pred[i] == 1)
        fn = sum(1 for i in range(len(y_val)) if y_val[i] == 1 and y_pred[i] == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1, best_th = f1, th
    
    return best_th


def calibrate_and_threshold(
    model,
    X_tr,
    y_tr,
    X_val,
    y_val,
    costs: Dict[str, float] | None = None,
    calibration_method: str = 'isotonic',
    returns_val_next: Optional[Any] = None,
) -> Tuple[Any, Dict[str, float]]:
    """Calibrate probs if sklearn is available; else no-op. Compute thresholds.

    Returns (calibrator_or_model, metrics_dict)
    """
    """Enhanced calibration with multiple methods and threshold optimization.
    
    Args:
        model: Base model with predict_proba method
        X_tr: Training features
        y_tr: Training labels
        X_val: Validation features
        y_val: Validation labels
        costs: Cost dictionary with 'fee_bps' and 'slippage_bps'
        calibration_method: 'isotonic', 'sigmoid', or 'both'
        
    Returns:
        Tuple of (calibrated_model, metrics_dict)
    """
    sk = try_import("sklearn")
    calibrated_model = model
    p_val = None
    metrics: Dict[str, float] = {}
    
    if sk is not None:
        from sklearn.calibration import CalibratedClassifierCV  # type: ignore
        from sklearn.metrics import (  # type: ignore
            f1_score, average_precision_score, roc_auc_score,
            matthews_corrcoef, brier_score_loss
        )
        
        # Try calibration with specified method
        best_calibrator = None
        best_brier = float('inf')
        
        methods_to_try = []
        if calibration_method == 'both':
            methods_to_try = ['isotonic', 'sigmoid']
        else:
            methods_to_try = [calibration_method]
        
        for method in methods_to_try:
            try:
                cal_model = CalibratedClassifierCV(model, method=method, cv='prefit')
                cal_model.fit(X_tr, y_tr)
                p_cal = cal_model.predict_proba(X_val)[:, 1]
                brier = brier_score_loss(y_val, p_cal)
                
                if brier < best_brier:
                    best_brier = brier
                    best_calibrator = cal_model
                    p_val = p_cal
                    metrics[f"brier_{method}"] = float(brier)
                    
            except Exception as e:
                print(f"   Warning: {method} calibration failed: {e}")
        
        if best_calibrator is not None:
            calibrated_model = best_calibrator
            metrics["brier"] = float(best_brier)
        else:
            # Fallback to uncalibrated
            p_val = model.predict_proba(X_val)[:, 1]
            metrics["brier"] = float(brier_score_loss(y_val, p_val))
        
        # Calculate metrics with optimal thresholds
        y_pred_default = (p_val >= 0.5).astype(int)

        # F1 threshold
        th_f1 = choose_threshold_by_f1(p_val, y_val)
        y_pred_f1 = (p_val >= th_f1).astype(int)

        metrics["f1_default"] = float(f1_score(y_val, y_pred_default))
        metrics["f1_optimal"] = float(f1_score(y_val, y_pred_f1))
        metrics["threshold_f1"] = float(th_f1)

        # Other metrics
        metrics["prauc"] = float(average_precision_score(y_val, p_val))
        metrics["roc_auc"] = float(roc_auc_score(y_val, p_val))
        metrics["mcc_default"] = float(matthews_corrcoef(y_val, y_pred_default))
        metrics["mcc_optimal"] = float(matthews_corrcoef(y_val, y_pred_f1))

        # Calibration curve -> ECE
        frac_pos, mean_pred = calibration_curve(y_val, p_val, n_bins=10)
        ece = float(np.mean(np.abs(frac_pos - mean_pred)))
        metrics["ece"] = ece

        # Baselines
        prevalence = np.mean(y_val)
        metrics["prevalence"] = float(prevalence)
        metrics["baseline_acc"] = float(max(prevalence, 1 - prevalence))
        metrics["baseline_prauc"] = float(prevalence)
        metrics["prauc_normalized"] = float((metrics["prauc"] - prevalence) / (1 - prevalence)) if prevalence < 1 else 0.0
        
    else:
        # Degraded mode: limited functionality
        try:
            if hasattr(model, 'predict_proba'):
                p_val = model.predict_proba(X_val)[:, 1]
            else:
                p_val = [0.5 for _ in y_val]
        except Exception:
            p_val = [0.5] * len(list(y_val))
        
        metrics["f1_default"] = 0.0
        metrics["prauc"] = 0.0
        metrics["roc_auc"] = 0.5
        metrics["brier"] = float(_brier_score(y_val, p_val))
        metrics["degraded_mode"] = 1.0
    
    # Calculate EV-based threshold (prefer t+1 if returns provided)
    fee = float((costs or {}).get("fee_bps", 10.0))
    slp = float((costs or {}).get("slippage_bps", 5.0))
    if returns_val_next is not None:
        th_ev = choose_threshold_by_ev_tplus1(p_val, returns_val_next, fee, slp)
    else:
        # Fallback: proxy using labels (+1/-1)
        y_signed = np.array(y_val) * 2 - 1
        th_ev = choose_threshold_by_ev_tplus1(p_val, y_signed, fee, slp)
    metrics["threshold_ev"] = float(th_ev)
    metrics["total_cost_bps"] = float(fee + slp)

    return calibrated_model, metrics
