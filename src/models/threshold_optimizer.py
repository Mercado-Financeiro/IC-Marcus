"""
Threshold Optimizer by Expected Value (EV)

Optimizes classification threshold based on net expected value after costs,
not on F1 or accuracy. This is where probability becomes profit.

References:
- Cost-sensitive learning: Elkan (2001)
- Threshold tuning: Hernandez-Orallo et al. (2012)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import warnings
from scipy.optimize import golden
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from src.utils.logging import log as logger


@dataclass
class TradingCosts:
    """Real trading costs in basis points."""
    fee_bps: float = 5.0          # 0.05% exchange fee
    slippage_bps: float = 5.0      # 0.05% typical slippage
    impact_bps: float = 2.0        # 0.02% market impact (simplified)
    
    @property
    def total_bps(self) -> float:
        """Total roundtrip cost in bps."""
        return (self.fee_bps + self.slippage_bps + self.impact_bps) * 2  # roundtrip
    
    @property
    def total_pct(self) -> float:
        """Total roundtrip cost as percentage."""
        return self.total_bps / 10000


@dataclass 
class EVResults:
    """Results from EV optimization."""
    optimal_threshold: float
    max_ev: float
    ev_per_trade: float
    expected_trades_per_period: int
    precision_at_threshold: float
    recall_at_threshold: float
    ev_curve: np.ndarray
    threshold_range: np.ndarray
    metadata: Dict


class ThresholdOptimizer:
    """
    Optimizes classification threshold to maximize expected value (EV).
    
    This is the bridge between ML probabilities and trading decisions.
    We don't care about F1 score - we care about money after costs.
    """
    
    def __init__(self, costs: Optional[TradingCosts] = None):
        """
        Initialize optimizer with trading costs.
        
        Args:
            costs: Trading cost structure. If None, uses defaults.
        """
        self.costs = costs or TradingCosts()
        self.last_results = None
        
    def compute_ev(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float,
        avg_win_pct: float = 0.015,  # 1.5% average win
        avg_loss_pct: float = 0.005,  # 0.5% average loss
    ) -> Tuple[float, Dict]:
        """
        Compute expected value at a specific threshold.
        
        EV = P(win|signal) * avg_win - P(loss|signal) * avg_loss - cost_per_trade
        
        Args:
            y_true: True labels (0 or 1)
            y_proba: Calibrated probabilities [0, 1]
            threshold: Decision threshold
            avg_win_pct: Average return on correct prediction
            avg_loss_pct: Average loss on incorrect prediction (positive value)
            
        Returns:
            (ev_per_opportunity, metrics_dict)
        """
        # Generate signals
        signals = (y_proba >= threshold).astype(int)
        n_signals = signals.sum()
        n_total = len(y_true)
        
        if n_signals == 0:
            return 0.0, {
                'n_trades': 0, 
                'precision': 0, 
                'recall': 0,
                'gross_return_per_trade': 0,
                'net_return_per_trade': 0,
                'ev_per_opportunity': 0,
                'trade_frequency': 0
            }
        
        # Calculate confusion matrix components
        tp = np.sum((signals == 1) & (y_true == 1))
        fp = np.sum((signals == 1) & (y_true == 0))
        fn = np.sum((signals == 0) & (y_true == 1))
        tn = np.sum((signals == 0) & (y_true == 0))
        
        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Expected return per trade (before costs)
        gross_return_per_trade = precision * avg_win_pct - (1 - precision) * avg_loss_pct
        
        # Net return per trade (after costs)
        net_return_per_trade = gross_return_per_trade - self.costs.total_pct
        
        # Expected value per opportunity (normalized by total samples)
        ev_per_opportunity = net_return_per_trade * (n_signals / n_total)
        
        metrics = {
            'n_trades': n_signals,
            'precision': precision,
            'recall': recall,
            'gross_return_per_trade': gross_return_per_trade,
            'net_return_per_trade': net_return_per_trade,
            'ev_per_opportunity': ev_per_opportunity,
            'trade_frequency': n_signals / n_total
        }
        
        return ev_per_opportunity, metrics
    
    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        avg_win_pct: float = 0.015,
        avg_loss_pct: float = 0.005,
        method: str = 'adaptive',
        n_points: int = 100
    ) -> EVResults:
        """
        Find threshold that maximizes expected value.
        
        Args:
            y_true: True labels
            y_proba: Calibrated probabilities
            avg_win_pct: Average win return
            avg_loss_pct: Average loss (positive value)
            method: 'grid', 'adaptive', or 'golden'
            n_points: Number of points for grid search
            
        Returns:
            EVResults with optimal threshold and metrics
        """
        logger.info(
            "Starting threshold optimization",
            method=method,
            costs_bps=self.costs.total_bps,
            avg_win=f"{avg_win_pct:.2%}",
            avg_loss=f"{avg_loss_pct:.2%}"
        )
        
        if method == 'golden':
            # Golden section search for smooth EV curve
            def neg_ev(tau):
                ev, _ = self.compute_ev(y_true, y_proba, tau, avg_win_pct, avg_loss_pct)
                return -ev
            
            # Search in [0.05, 0.95] to avoid edge cases
            result = golden(neg_ev, brack=(0.05, 0.95), tol=1e-4)
            optimal_threshold = result
            
            # Compute full curve for visualization
            threshold_range = np.linspace(0.05, 0.95, n_points)
            
        elif method == 'adaptive':
            # Adaptive grid: denser around high-probability regions
            # Start with coarse grid
            coarse_range = np.linspace(0.05, 0.95, 20)
            coarse_evs = []
            
            for tau in coarse_range:
                ev, _ = self.compute_ev(y_true, y_proba, tau, avg_win_pct, avg_loss_pct)
                coarse_evs.append(ev)
            
            # Find promising region (around max)
            max_idx = np.argmax(coarse_evs)
            center = coarse_range[max_idx]
            
            # Fine grid around promising region
            fine_range = np.linspace(
                max(0.05, center - 0.15),
                min(0.95, center + 0.15),
                n_points
            )
            threshold_range = fine_range
            
        else:  # grid
            threshold_range = np.linspace(0.05, 0.95, n_points)
        
        # Compute EV for all thresholds
        ev_curve = []
        all_metrics = []
        
        for tau in threshold_range:
            ev, metrics = self.compute_ev(y_true, y_proba, tau, avg_win_pct, avg_loss_pct)
            ev_curve.append(ev)
            all_metrics.append(metrics)
        
        ev_curve = np.array(ev_curve)
        
        # Find optimal threshold
        if method != 'golden':
            optimal_idx = np.argmax(ev_curve)
            optimal_threshold = threshold_range[optimal_idx]
        else:
            optimal_idx = np.argmin(np.abs(threshold_range - optimal_threshold))
        
        max_ev = ev_curve[optimal_idx]
        optimal_metrics = all_metrics[optimal_idx] if all_metrics else None
        
        # If metrics not available (golden method), compute them
        if optimal_metrics is None:
            _, optimal_metrics = self.compute_ev(
                y_true, y_proba, optimal_threshold, avg_win_pct, avg_loss_pct
            )
        
        # Create results
        results = EVResults(
            optimal_threshold=optimal_threshold,
            max_ev=max_ev,
            ev_per_trade=optimal_metrics['net_return_per_trade'],
            expected_trades_per_period=optimal_metrics['n_trades'],
            precision_at_threshold=optimal_metrics['precision'],
            recall_at_threshold=optimal_metrics['recall'],
            ev_curve=ev_curve,
            threshold_range=threshold_range,
            metadata={
                'avg_win_pct': avg_win_pct,
                'avg_loss_pct': avg_loss_pct,
                'total_costs_bps': self.costs.total_bps,
                'trade_frequency': optimal_metrics['trade_frequency'],
                'gross_return_per_trade': optimal_metrics['gross_return_per_trade']
            }
        )
        
        self.last_results = results
        
        logger.info(
            "Threshold optimization complete",
            optimal_threshold=f"{optimal_threshold:.3f}",
            max_ev_per_opportunity=f"{max_ev:.5f}",
            ev_per_trade=f"{optimal_metrics['net_return_per_trade']:.4%}",
            precision=f"{optimal_metrics['precision']:.2%}",
            expected_trades=int(optimal_metrics['n_trades']),  # Convert to int for JSON serialization
            note="This is NET after all costs"
        )
        
        return results
    
    def plot_ev_curve(
        self,
        results: Optional[EVResults] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the EV curve showing optimal threshold.
        
        Args:
            results: EVResults to plot. If None, uses last results.
            save_path: Path to save figure
            
        Returns:
            matplotlib figure
        """
        if results is None:
            results = self.last_results
            if results is None:
                raise ValueError("No results to plot. Run optimize_threshold first.")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: EV curve
        ax1.plot(results.threshold_range, results.ev_curve, 'b-', linewidth=2)
        ax1.axvline(results.optimal_threshold, color='r', linestyle='--', 
                   label=f'Optimal τ={results.optimal_threshold:.3f}')
        ax1.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax1.fill_between(results.threshold_range, 0, results.ev_curve,
                         where=(results.ev_curve > 0), alpha=0.3, color='green',
                         label='Profitable region')
        ax1.fill_between(results.threshold_range, results.ev_curve, 0,
                         where=(results.ev_curve <= 0), alpha=0.3, color='red',
                         label='Loss region')
        
        ax1.set_xlabel('Threshold (τ)')
        ax1.set_ylabel('Expected Value per Opportunity')
        ax1.set_title(f'EV Optimization (Costs: {results.metadata["total_costs_bps"]:.1f} bps)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Components
        ax2.plot(results.threshold_range, 
                [m['precision'] for m in all_metrics], 
                'g-', label='Precision')
        ax2.plot(results.threshold_range,
                [m['recall'] for m in all_metrics],
                'b-', label='Recall')
        ax2.plot(results.threshold_range,
                [m['trade_frequency'] for m in all_metrics],
                'orange', label='Trade Frequency')
        
        ax2.axvline(results.optimal_threshold, color='r', linestyle='--')
        ax2.set_xlabel('Threshold (τ)')
        ax2.set_ylabel('Metric Value')
        ax2.set_title('Precision, Recall, and Trade Frequency vs Threshold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"EV curve saved to {save_path}")
        
        return fig
    
    def compare_with_f1_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        avg_win_pct: float = 0.015,
        avg_loss_pct: float = 0.005
    ) -> Dict:
        """
        Compare EV-optimal threshold with F1-optimal threshold.
        
        Shows why optimizing for F1 loses money.
        
        Returns:
            Comparison metrics
        """
        # Get EV-optimal threshold
        ev_results = self.optimize_threshold(
            y_true, y_proba, avg_win_pct, avg_loss_pct
        )
        
        # Get F1-optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        f1_optimal_idx = np.argmax(f1_scores[:-1])
        f1_threshold = thresholds[f1_optimal_idx]
        
        # Compute EV at F1 threshold
        ev_at_f1, metrics_at_f1 = self.compute_ev(
            y_true, y_proba, f1_threshold, avg_win_pct, avg_loss_pct
        )
        
        # Compute EV at fixed 0.5
        ev_at_half, metrics_at_half = self.compute_ev(
            y_true, y_proba, 0.5, avg_win_pct, avg_loss_pct
        )
        
        comparison = {
            'ev_optimal': {
                'threshold': ev_results.optimal_threshold,
                'ev': ev_results.max_ev,
                'ev_per_trade': ev_results.ev_per_trade,
                'precision': ev_results.precision_at_threshold,
                'trades': ev_results.expected_trades_per_period
            },
            'f1_optimal': {
                'threshold': f1_threshold,
                'ev': ev_at_f1,
                'ev_per_trade': metrics_at_f1['net_return_per_trade'],
                'precision': metrics_at_f1['precision'],
                'trades': metrics_at_f1['n_trades'],
                'f1_score': f1_scores[f1_optimal_idx]
            },
            'fixed_0.5': {
                'threshold': 0.5,
                'ev': ev_at_half,
                'ev_per_trade': metrics_at_half['net_return_per_trade'],
                'precision': metrics_at_half['precision'],
                'trades': metrics_at_half['n_trades']
            },
            'improvement_over_f1': (ev_results.max_ev - ev_at_f1) / abs(ev_at_f1) * 100 if ev_at_f1 != 0 else np.inf,
            'improvement_over_0.5': (ev_results.max_ev - ev_at_half) / abs(ev_at_half) * 100 if ev_at_half != 0 else np.inf
        }
        
        logger.info(
            "Threshold comparison",
            ev_optimal=f"τ={comparison['ev_optimal']['threshold']:.3f}, EV={comparison['ev_optimal']['ev']:.5f}",
            f1_optimal=f"τ={comparison['f1_optimal']['threshold']:.3f}, EV={comparison['f1_optimal']['ev']:.5f}",
            fixed_half=f"τ=0.500, EV={comparison['fixed_0.5']['ev']:.5f}",
            improvement_vs_f1=f"{comparison['improvement_over_f1']:.1f}%",
            improvement_vs_half=f"{comparison['improvement_over_0.5']:.1f}%"
        )
        
        return comparison


def estimate_returns_from_data(
    y_true: np.ndarray,
    returns: np.ndarray,
    percentile_win: float = 75,
    percentile_loss: float = 25
) -> Tuple[float, float]:
    """
    Estimate average win/loss from actual return data.
    
    Don't use fixed values - estimate from your backtest!
    
    Args:
        y_true: True labels
        returns: Actual returns for each sample
        percentile_win: Percentile for win estimation
        percentile_loss: Percentile for loss estimation
        
    Returns:
        (avg_win_pct, avg_loss_pct)
    """
    wins = returns[y_true == 1]
    losses = returns[y_true == 0]
    
    if len(wins) > 0:
        avg_win = np.percentile(wins[wins > 0], percentile_win) if np.any(wins > 0) else 0.015
    else:
        avg_win = 0.015  # fallback
        
    if len(losses) > 0:
        avg_loss = abs(np.percentile(losses[losses < 0], 100 - percentile_loss)) if np.any(losses < 0) else 0.005
    else:
        avg_loss = 0.005  # fallback
    
    logger.info(
        "Estimated returns from data",
        avg_win=f"{avg_win:.2%}",
        avg_loss=f"{avg_loss:.2%}",
        n_wins=len(wins),
        n_losses=len(losses)
    )
    
    return avg_win, avg_loss