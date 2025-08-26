"""Threshold optimization for XGBoost models."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Union
from sklearn.metrics import (
    f1_score, precision_recall_curve, precision_score,
    recall_score, matthews_corrcoef
)
import structlog

from .config import ThresholdConfig

log = structlog.get_logger()


class ThresholdOptimizer:
    """Optimize classification threshold for different objectives."""
    
    def __init__(self, config: Optional[ThresholdConfig] = None):
        """Initialize threshold optimizer.
        
        Args:
            config: Threshold optimization configuration
        """
        self.config = config or ThresholdConfig()
        self.threshold_f1 = 0.5
        self.threshold_ev = 0.5
        self.threshold_profit = 0.5
        self.optimization_results = {}
        
    def optimize_all(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Optimize threshold using all configured methods.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            returns: Optional returns for EV/profit optimization
            
        Returns:
            Dictionary with optimized thresholds
        """
        results = {}
        
        if self.config.optimize_f1:
            self.threshold_f1 = self.optimize_f1_threshold(y_true, y_pred_proba)
            results['threshold_f1'] = self.threshold_f1
        
        if self.config.optimize_ev and returns is not None:
            self.threshold_ev = self.optimize_ev_threshold(
                y_true, y_pred_proba, returns
            )
            results['threshold_ev'] = self.threshold_ev
        
        if self.config.optimize_profit and returns is not None:
            self.threshold_profit = self.optimize_profit_threshold(
                y_true, y_pred_proba, returns
            )
            results['threshold_profit'] = self.threshold_profit
        
        self.optimization_results = results
        
        log.info(
            "thresholds_optimized",
            **results
        )
        
        return results
    
    def optimize_f1_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        """Optimize threshold for F1 score.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Optimal threshold
        """
        # Get precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(
            y_true, y_pred_proba
        )
        
        # Calculate F1 scores
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        # Find best threshold
        best_idx = np.argmax(f1_scores[:-1])  # Exclude last point
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])
        
        log.info(
            "f1_threshold_optimized",
            threshold=best_threshold,
            f1_score=best_f1
        )
        
        return best_threshold
    
    def optimize_ev_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        returns: np.ndarray
    ) -> float:
        """Optimize threshold for expected value.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            returns: Returns for each sample
            
        Returns:
            Optimal threshold
        """
        # Create threshold candidates
        thresholds = np.linspace(
            self.config.threshold_min,
            self.config.threshold_max,
            self.config.threshold_steps
        )
        
        best_ev = -np.inf
        best_threshold = 0.5
        
        # Total cost in basis points
        total_cost_bps = (
            self.config.transaction_cost_bps +
            self.config.slippage_bps
        )
        
        for threshold in thresholds:
            # Get predictions for this threshold
            predictions = (y_pred_proba >= threshold).astype(int)
            
            # Calculate positions (1 for long, 0 for no position)
            positions = predictions
            
            # Calculate returns when in position
            strategy_returns = positions * returns
            
            # Calculate turnover
            position_changes = np.abs(np.diff(positions, prepend=0))
            turnover = position_changes.mean()
            
            # Apply transaction costs
            cost = turnover * total_cost_bps / 10000
            net_returns = strategy_returns - cost
            
            # Calculate expected value
            ev = net_returns.mean()
            
            if ev > best_ev:
                best_ev = ev
                best_threshold = float(threshold)
        
        log.info(
            "ev_threshold_optimized",
            threshold=best_threshold,
            expected_value=best_ev
        )
        
        return best_threshold
    
    def optimize_profit_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        returns: np.ndarray
    ) -> float:
        """Optimize threshold for profit considering Kelly criterion.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            returns: Returns for each sample
            
        Returns:
            Optimal threshold
        """
        thresholds = np.linspace(
            self.config.threshold_min,
            self.config.threshold_max,
            self.config.threshold_steps
        )
        
        best_profit = -np.inf
        best_threshold = 0.5
        
        for threshold in thresholds:
            predictions = (y_pred_proba >= threshold).astype(int)
            
            # Calculate win rate and average win/loss
            correct_predictions = predictions == y_true
            win_rate = correct_predictions[predictions == 1].mean() if (predictions == 1).any() else 0
            
            if win_rate > 0 and win_rate < 1:
                # Calculate average win and loss
                wins = returns[(predictions == 1) & (y_true == 1)]
                losses = returns[(predictions == 1) & (y_true == 0)]
                
                avg_win = wins.mean() if len(wins) > 0 else 0
                avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
                
                # Kelly criterion
                if avg_loss > 0:
                    kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                    kelly_f = np.clip(kelly_f, 0, 1)
                    
                    # Apply Kelly fraction
                    position_size = kelly_f * self.config.kelly_fraction
                    position_size = min(position_size, self.config.max_position_size)
                    
                    # Calculate profit
                    strategy_returns = predictions * returns * position_size
                    
                    # Apply costs
                    turnover = np.abs(np.diff(predictions, prepend=0)).mean()
                    cost = turnover * (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000
                    
                    profit = strategy_returns.mean() - cost
                    
                    if profit > best_profit:
                        best_profit = profit
                        best_threshold = float(threshold)
        
        log.info(
            "profit_threshold_optimized",
            threshold=best_threshold,
            expected_profit=best_profit
        )
        
        return best_threshold
    
    def evaluate_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate metrics at given threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Threshold to evaluate
            returns: Optional returns for financial metrics
            
        Returns:
            Dictionary of metrics
        """
        predictions = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'threshold': threshold,
            'precision': precision_score(y_true, predictions, zero_division=0),
            'recall': recall_score(y_true, predictions, zero_division=0),
            'f1_score': f1_score(y_true, predictions, zero_division=0),
            'mcc': matthews_corrcoef(y_true, predictions),
            'n_positive_predictions': predictions.sum(),
            'positive_rate': predictions.mean()
        }
        
        if returns is not None:
            # Financial metrics
            strategy_returns = predictions * returns
            metrics['mean_return'] = strategy_returns.mean()
            metrics['total_return'] = strategy_returns.sum()
            metrics['win_rate'] = (strategy_returns > 0).mean()
            
            # Sharpe ratio
            if strategy_returns.std() > 0:
                metrics['sharpe_ratio'] = (
                    strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                )
            else:
                metrics['sharpe_ratio'] = 0
        
        return metrics
    
    def choose_threshold_by_ev(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        cost_per_trade_bps: float = 10.0,
        win_return: float = 0.01,
        loss_return: float = 0.01,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
        n_thresholds: int = 100
    ) -> Tuple[float, Dict]:
        """
        Encontra threshold que maximiza Expected Value líquido.
        
        Implementação conforme especificação do CLAUDE.md:
        EV = P(correct) * win_return - P(incorrect) * loss_return - cost_per_trade
        
        Args:
            y_true: Labels verdadeiros
            y_pred_proba: Probabilidades preditas
            cost_per_trade_bps: Custo total por trade em basis points
            win_return: Retorno esperado em trades corretos (decimal)
            loss_return: Perda esperada em trades incorretos (decimal)
            min_threshold: Threshold mínimo a testar
            max_threshold: Threshold máximo a testar
            n_thresholds: Número de thresholds a testar
            
        Returns:
            Tuple com (threshold_ótimo, dict_com_métricas)
        """
        thresholds = np.linspace(min_threshold, max_threshold, n_thresholds)
        
        results = {
            'thresholds': [],
            'ev_net': [],
            'ev_gross': [],
            'win_rate': [],
            'n_trades': [],
            'cost_total': []
        }
        
        best_ev = -np.inf
        best_threshold = 0.5
        best_metrics = {}
        
        cost_per_trade = cost_per_trade_bps / 10000.0  # Convert bps to decimal
        
        for threshold in thresholds:
            # Gerar sinais
            signals = (y_pred_proba >= threshold).astype(int)
            n_trades = signals.sum()
            
            if n_trades == 0:
                ev_net = 0.0
                ev_gross = 0.0
                win_rate = 0.0
                cost_total = 0.0
            else:
                # Calcular win rate nos trades sinalizados
                correct_trades = np.sum((signals == 1) & (y_true == 1))
                incorrect_trades = np.sum((signals == 1) & (y_true == 0))
                
                win_rate = correct_trades / n_trades if n_trades > 0 else 0.0
                loss_rate = incorrect_trades / n_trades if n_trades > 0 else 0.0
                
                # Expected Value bruto
                ev_gross = (win_rate * win_return) - (loss_rate * loss_return)
                
                # Custo total
                cost_total = n_trades * cost_per_trade
                
                # Expected Value líquido
                ev_net = ev_gross - cost_total
            
            # Armazenar resultados
            results['thresholds'].append(threshold)
            results['ev_net'].append(ev_net)
            results['ev_gross'].append(ev_gross)
            results['win_rate'].append(win_rate)
            results['n_trades'].append(n_trades)
            results['cost_total'].append(cost_total)
            
            # Verificar se é o melhor
            if ev_net > best_ev:
                best_ev = ev_net
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'ev_net': ev_net,
                    'ev_gross': ev_gross,
                    'win_rate': win_rate,
                    'n_trades': n_trades,
                    'cost_total': cost_total,
                    'cost_per_trade_bps': cost_per_trade_bps
                }
        
        log.info(
            "ev_threshold_optimized",
            threshold=best_threshold,
            ev_net=best_ev,
            win_rate=best_metrics['win_rate'],
            n_trades=best_metrics['n_trades']
        )
        
        # Adicionar curva completa aos resultados
        results['best'] = best_metrics
        
        return best_threshold, results
        
    def get_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Analyze performance across different thresholds.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            returns: Optional returns
            
        Returns:
            DataFrame with threshold analysis
        """
        thresholds = np.linspace(0.1, 0.9, 17)  # Every 0.05
        
        results = []
        for threshold in thresholds:
            metrics = self.evaluate_threshold(
                y_true, y_pred_proba, threshold, returns
            )
            results.append(metrics)
        
        return pd.DataFrame(results)