"""Threshold optimization for XGBoost models."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import f1_score, precision_recall_curve


class ThresholdOptimizer:
    """Optimize classification thresholds using different strategies."""
    
    def __init__(self, strategies: list = None):
        """Initialize threshold optimizer.
        
        Args:
            strategies: List of optimization strategies to use
        """
        self.strategies = strategies or ['f1', 'ev', 'profit']
        self.thresholds = {}
        
    def optimize_f1(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Find threshold that maximizes F1 score.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Optimal threshold
        """
        # Try different thresholds
        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold
        
        self.thresholds['f1'] = best_threshold
        return best_threshold
    
    def optimize_pr_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Find threshold using precision-recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Optimal threshold
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # F1 scores for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Find threshold with max F1
        best_idx = np.argmax(f1_scores[:-1])  # Last element is 1.0
        best_threshold = thresholds[best_idx]
        
        self.thresholds['pr_curve'] = best_threshold
        return best_threshold
    
    def optimize_ev(
        self, 
        y_true: pd.Series, 
        y_pred_proba: np.ndarray,
        costs: Dict[str, float]
    ) -> float:
        """Optimize threshold for expected value.
        
        Args:
            y_true: True labels with returns
            y_pred_proba: Predicted probabilities
            costs: Trading costs (fee_bps, slippage_bps)
            
        Returns:
            Optimal threshold
        """
        # Calculate returns for different thresholds
        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_ev = -np.inf
        
        total_cost_bps = costs.get('fee_bps', 5) + costs.get('slippage_bps', 5)
        
        for threshold in thresholds:
            # Generate signals
            signals = (y_pred_proba >= threshold).astype(int)
            
            # Calculate returns (assuming y_true contains actual returns)
            if hasattr(y_true, 'values'):
                returns = y_true.values * signals
            else:
                returns = y_true * signals
            
            # Apply costs on trades
            trade_mask = signals != 0
            costs_pct = (total_cost_bps / 10000) * trade_mask
            
            # Net returns
            net_returns = returns - costs_pct
            ev = net_returns.mean()
            
            if ev > best_ev:
                best_ev = ev
                best_threshold = threshold
        
        self.thresholds['ev'] = best_threshold
        return best_threshold
    
    def optimize_profit(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        initial_capital: float = 100000,
        position_size: float = 0.1,
        costs: Dict[str, float] = None
    ) -> Tuple[float, Dict]:
        """Optimize threshold for profit maximization.
        
        Args:
            y_true: True labels with returns
            y_pred_proba: Predicted probabilities
            initial_capital: Initial capital
            position_size: Position size as fraction of capital
            costs: Trading costs
            
        Returns:
            Tuple of (optimal_threshold, metrics_dict)
        """
        if costs is None:
            costs = {'fee_bps': 5, 'slippage_bps': 5}
        
        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_profit = -np.inf
        best_metrics = {}
        
        for threshold in thresholds:
            # Generate signals
            signals = (y_pred_proba >= threshold).astype(int)
            
            # Simple profit calculation
            capital = initial_capital
            trades = []
            
            for i, signal in enumerate(signals):
                if signal == 1:
                    # Calculate position value
                    position_value = capital * position_size
                    
                    # Apply costs
                    total_cost_bps = costs['fee_bps'] + costs['slippage_bps']
                    cost = position_value * (total_cost_bps / 10000)
                    
                    # Calculate return (simplified)
                    if i < len(y_true):
                        ret = float(y_true.iloc[i] if hasattr(y_true, 'iloc') else y_true[i])
                        profit = position_value * ret - cost
                        capital += profit
                        trades.append(profit)
            
            total_profit = capital - initial_capital
            
            if total_profit > best_profit:
                best_profit = total_profit
                best_threshold = threshold
                best_metrics = {
                    'total_profit': total_profit,
                    'return_pct': (total_profit / initial_capital) * 100,
                    'n_trades': len(trades),
                    'avg_trade': np.mean(trades) if trades else 0,
                    'win_rate': np.mean([t > 0 for t in trades]) if trades else 0
                }
        
        self.thresholds['profit'] = best_threshold
        return best_threshold, best_metrics
    
    def optimize_all(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        costs: Dict[str, float] = None
    ) -> Dict[str, float]:
        """Optimize thresholds using all strategies.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            costs: Trading costs
            
        Returns:
            Dictionary of thresholds by strategy
        """
        results = {}
        
        if 'f1' in self.strategies:
            results['f1'] = self.optimize_f1(y_true, y_pred_proba)
        
        if 'pr_curve' in self.strategies:
            results['pr_curve'] = self.optimize_pr_curve(y_true, y_pred_proba)
        
        if 'ev' in self.strategies and costs is not None:
            results['ev'] = self.optimize_ev(y_true, y_pred_proba, costs)
        
        if 'profit' in self.strategies and costs is not None:
            threshold, metrics = self.optimize_profit(y_true, y_pred_proba, costs=costs)
            results['profit'] = threshold
            results['profit_metrics'] = metrics
        
        self.thresholds.update(results)
        return results