"""Metrics calculation for XGBoost models."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix, classification_report
)


class TradingMetrics:
    """Calculate trading-specific metrics."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            periods_per_year: Number of trading periods per year
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return np.sqrt(periods_per_year) * mean_return / std_return
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, periods_per_year: int = 252,
                               target_return: float = 0) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: Array of returns
            periods_per_year: Number of trading periods per year
            target_return: Target return for downside deviation
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0.0
        
        mean_excess_return = np.mean(excess_returns)
        return np.sqrt(periods_per_year) * mean_excess_return / downside_std
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Cumulative returns or equity curve
            
        Returns:
            Tuple of (max_drawdown, peak_idx, trough_idx)
        """
        if len(equity_curve) == 0:
            return 0.0, 0, 0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = np.min(drawdown)
        trough_idx = np.argmin(drawdown)
        
        # Find peak before trough
        peak_idx = np.argmax(equity_curve[:trough_idx+1])
        
        return abs(max_dd), peak_idx, trough_idx
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio.
        
        Args:
            returns: Array of returns
            periods_per_year: Number of trading periods per year
            
        Returns:
            Calmar ratio
        """
        if len(returns) == 0:
            return 0.0
        
        # Annualized return
        mean_return = np.mean(returns) * periods_per_year
        
        # Maximum drawdown
        equity_curve = (1 + returns).cumprod()
        max_dd, _, _ = TradingMetrics.calculate_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return 0.0
        
        return mean_return / max_dd
    
    @staticmethod
    def calculate_win_rate(returns: np.ndarray) -> float:
        """Calculate win rate (percentage of positive returns)."""
        if len(returns) == 0:
            return 0.0
        return np.mean(returns > 0)
    
    @staticmethod
    def calculate_profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(returns) == 0:
            return 0.0
        
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return np.inf if profits > 0 else 0.0
        
        return profits / losses
    
    @staticmethod
    def calculate_expected_value(signals: np.ndarray, returns: np.ndarray,
                                costs: Dict[str, float]) -> float:
        """
        Calculate expected value considering costs.
        
        Args:
            signals: Trading signals (1 for buy, 0 for no position)
            returns: Actual returns
            costs: Dictionary with 'fee_bps' and 'slippage_bps'
            
        Returns:
            Expected value
        """
        if len(signals) == 0 or len(returns) == 0:
            return 0.0
        
        # Calculate gross returns
        gross_returns = signals * returns
        
        # Calculate costs
        total_cost_bps = costs.get('fee_bps', 5) + costs.get('slippage_bps', 5)
        trade_mask = signals != 0
        costs_pct = (total_cost_bps / 10000) * trade_mask
        
        # Net returns
        net_returns = gross_returns - costs_pct
        
        return net_returns.mean()


def calculate_ml_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate machine learning metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    # Add probability-based metrics if available
    if y_pred_proba is not None:
        metrics.update({
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba),
            'brier': brier_score_loss(y_true, y_pred_proba)
        })
    
    return metrics


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         y_pred_proba: np.ndarray, returns: np.ndarray,
                         costs: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate all metrics (ML and trading).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        returns: Actual returns
        costs: Trading costs
        
    Returns:
        Dictionary of all metrics
    """
    # ML metrics
    ml_metrics = calculate_ml_metrics(y_true, y_pred, y_pred_proba)
    
    # Trading metrics
    trading_metrics_calc = TradingMetrics()
    
    # Calculate returns based on predictions
    strategy_returns = y_pred * returns
    
    trading_metrics = {
        'sharpe': trading_metrics_calc.calculate_sharpe_ratio(strategy_returns),
        'sortino': trading_metrics_calc.calculate_sortino_ratio(strategy_returns),
        'calmar': trading_metrics_calc.calculate_calmar_ratio(strategy_returns),
        'max_drawdown': trading_metrics_calc.calculate_max_drawdown(
            (1 + strategy_returns).cumprod())[0],
        'win_rate': trading_metrics_calc.calculate_win_rate(strategy_returns),
        'profit_factor': trading_metrics_calc.calculate_profit_factor(strategy_returns),
        'expected_value': trading_metrics_calc.calculate_expected_value(y_pred, returns, costs)
    }
    
    # Combine all metrics
    all_metrics = {**ml_metrics, **trading_metrics}
    
    return all_metrics