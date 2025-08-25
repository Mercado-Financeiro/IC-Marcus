"""Métricas avançadas de trading para avaliação de estratégias."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from scipy import stats
from dataclasses import dataclass
import structlog

log = structlog.get_logger()


@dataclass
class MetricsConfig:
    """Configuração para cálculo de métricas."""
    
    risk_free_rate: float = 0.02  # 2% anual
    trading_days: int = 365  # Crypto trades 24/7
    confidence_level: float = 0.95
    min_samples: int = 30
    transaction_cost_bps: float = 10  # basis points


class AdvancedTradingMetrics:
    """Calculador de métricas avançadas de trading."""
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """Inicializa o calculador de métricas.
        
        Args:
            config: Configuração para cálculo das métricas
        """
        self.config = config or MetricsConfig()
        
    def calculate_psr(self,
                     returns: pd.Series,
                     benchmark_sr: float = 0.0,
                     min_track_record: int = None) -> Dict[str, float]:
        """Calcula Probabilistic Sharpe Ratio (PSR).
        
        PSR considera o tamanho da amostra e ajusta para múltiplos testes.
        
        Args:
            returns: Series com retornos
            benchmark_sr: Sharpe ratio de referência
            min_track_record: Comprimento mínimo do track record
            
        Returns:
            Dict com PSR e métricas relacionadas
        """
        n = len(returns)
        if n < self.config.min_samples:
            log.warning("insufficient_samples_for_psr", n=n)
            return {'psr': 0.0, 'sr': 0.0, 'sr_std': np.inf}
        
        # Sharpe Ratio observado
        sr = self.calculate_sharpe_ratio(returns)
        
        # Skewness e Kurtosis
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns, fisher=True)
        
        # Erro padrão do Sharpe Ratio (ajustado para higher moments)
        sr_std = np.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / n)
        
        # PSR: probabilidade de que SR verdadeiro > benchmark
        psr = stats.norm.cdf((sr - benchmark_sr) / sr_std)
        
        # Minimum Track Record Length (Tempo mínimo para confiança)
        if min_track_record is None:
            # Bailey & López de Prado formula
            min_track_record = 1 + (1 - skew * sr + (kurt / 4) * sr**2) * \
                               (stats.norm.ppf(self.config.confidence_level) / \
                                (sr - benchmark_sr))**2
        
        log.info("psr_calculated",
                sr=float(sr),
                psr=float(psr),
                min_track_record=int(min_track_record))
        
        return {
            'psr': float(psr),
            'sharpe_ratio': float(sr),
            'sr_std_error': float(sr_std),
            'min_track_record': int(min_track_record),
            'skewness': float(skew),
            'kurtosis': float(kurt)
        }
    
    def calculate_dsr(self,
                     returns: pd.Series,
                     num_trials: int = 1,
                     independent_trials: bool = True) -> Dict[str, float]:
        """Calcula Deflated Sharpe Ratio (DSR).
        
        DSR ajusta para múltiplos testes e data mining.
        
        Args:
            returns: Series com retornos
            num_trials: Número de estratégias testadas
            independent_trials: Se os testes são independentes
            
        Returns:
            Dict com DSR e métricas relacionadas
        """
        # PSR primeiro
        psr_result = self.calculate_psr(returns)
        sr = psr_result['sharpe_ratio']
        
        n = len(returns)
        
        # Número efetivo de trials (considerando correlação)
        if not independent_trials:
            # Assumir correlação média de 0.4 entre estratégias
            avg_correlation = 0.4
            num_trials_eff = num_trials * (1 - avg_correlation) + avg_correlation
        else:
            num_trials_eff = num_trials
        
        # Ajuste para múltiplos testes (Bonferroni-like)
        # Probabilidade de pelo menos 1 falso positivo
        fdr = 1 - (1 - 0.05)**(1/num_trials_eff)  # False Discovery Rate
        
        # SR esperado sob H0 (null hypothesis)
        e_max_sr_h0 = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1/(num_trials_eff * np.e)) + \
                      np.euler_gamma * stats.norm.ppf(1 - 1/num_trials_eff)
        
        # Variance do max SR sob H0
        var_max_sr_h0 = ((np.pi**2 / 6) - 1) / n
        
        # DSR: SR deflacionado
        dsr = (sr - e_max_sr_h0) / np.sqrt(var_max_sr_h0)
        
        # Probabilidade de que a estratégia seja genuína
        p_genuine = stats.norm.cdf(dsr)
        
        log.info("dsr_calculated",
                sr=float(sr),
                dsr=float(dsr),
                p_genuine=float(p_genuine),
                num_trials=num_trials)
        
        return {
            'dsr': float(dsr),
            'sharpe_ratio': float(sr),
            'expected_max_sr': float(e_max_sr_h0),
            'p_genuine': float(p_genuine),
            'false_discovery_rate': float(fdr),
            'num_trials_effective': float(num_trials_eff)
        }
    
    def calculate_capacity_analysis(self,
                                   returns: pd.Series,
                                   volumes: pd.Series,
                                   market_impact_model: str = 'sqrt') -> Dict[str, float]:
        """Analisa capacidade da estratégia (AUM máximo).
        
        Args:
            returns: Series com retornos
            volumes: Series com volumes negociados
            market_impact_model: Modelo de impacto ('linear', 'sqrt', 'power')
            
        Returns:
            Dict com análise de capacidade
        """
        # Turnover médio
        position_changes = returns.abs()
        avg_turnover = position_changes.mean()
        
        # Volume médio disponível
        avg_volume = volumes.mean()
        participation_rate = 0.1  # Máximo 10% do volume
        
        # Capacidade base (sem considerar slippage)
        base_capacity = avg_volume * participation_rate
        
        # Modelo de impacto de mercado
        if market_impact_model == 'linear':
            # Impacto linear: cost = k * size
            impact_coef = 0.1  # 10 bps por 1% do volume
            max_acceptable_cost = 50  # 50 bps máximo
            capacity = base_capacity * (max_acceptable_cost / impact_coef)
            
        elif market_impact_model == 'sqrt':
            # Modelo square-root (mais realista)
            # cost = k * sqrt(size / avg_volume)
            impact_coef = 10  # bps
            max_acceptable_cost = 50  # bps
            capacity = base_capacity * (max_acceptable_cost / impact_coef)**2
            
        else:  # power
            # Modelo power-law: cost = k * (size / volume)^alpha
            alpha = 1.5
            impact_coef = 5
            max_acceptable_cost = 50
            capacity = base_capacity * (max_acceptable_cost / impact_coef)**(1/alpha)
        
        # Estimar degradação do Sharpe com AUM
        sr_current = self.calculate_sharpe_ratio(returns)
        
        # Assumir degradação linear do Sharpe com AUM
        sr_at_capacity = sr_current * 0.5  # 50% de degradação na capacidade máxima
        
        # Break-even capacity (onde SR = 0)
        breakeven_capacity = capacity * (sr_current / (sr_current - 0))
        
        log.info("capacity_analysis",
                base_capacity=float(base_capacity),
                effective_capacity=float(capacity),
                current_sharpe=float(sr_current),
                expected_sharpe_at_capacity=float(sr_at_capacity))
        
        return {
            'base_capacity_usd': float(base_capacity),
            'effective_capacity_usd': float(capacity),
            'breakeven_capacity_usd': float(breakeven_capacity),
            'current_sharpe': float(sr_current),
            'expected_sharpe_at_capacity': float(sr_at_capacity),
            'avg_turnover': float(avg_turnover),
            'participation_rate': float(participation_rate),
            'market_impact_model': market_impact_model
        }
    
    def calculate_turnover_analysis(self,
                                   positions: pd.Series,
                                   prices: pd.Series) -> Dict[str, float]:
        """Analisa turnover e custos de transação.
        
        Args:
            positions: Series com posições
            prices: Series com preços
            
        Returns:
            Dict com análise de turnover
        """
        # Mudanças de posição
        position_changes = positions.diff().abs()
        
        # Turnover (fração do portfolio negociada)
        avg_position = positions.abs().mean()
        if avg_position > 0:
            turnover_rate = position_changes.mean() / avg_position
        else:
            turnover_rate = 0
        
        # Turnover anualizado
        periods_per_year = self.config.trading_days * 24  # Hourly data
        annual_turnover = turnover_rate * periods_per_year
        
        # Custo estimado de transação
        cost_per_trade = self.config.transaction_cost_bps / 10000
        annual_cost = annual_turnover * cost_per_trade
        
        # Round-trip trades
        # Contar mudanças de sinal como round-trips
        sign_changes = np.diff(np.sign(positions))
        round_trips = np.sum(np.abs(sign_changes) == 2)
        
        # Holding period médio
        if round_trips > 0:
            avg_holding_period = len(positions) / (round_trips * 2)
        else:
            avg_holding_period = len(positions)
        
        log.info("turnover_analysis",
                annual_turnover=float(annual_turnover),
                annual_cost_pct=float(annual_cost * 100),
                round_trips=int(round_trips))
        
        return {
            'turnover_rate': float(turnover_rate),
            'annual_turnover': float(annual_turnover),
            'annual_cost_percent': float(annual_cost * 100),
            'round_trips': int(round_trips),
            'avg_holding_period': float(avg_holding_period),
            'total_trades': int(position_changes.sum())
        }
    
    def calculate_sharpe_ratio(self,
                              returns: pd.Series,
                              risk_free: Optional[float] = None) -> float:
        """Calcula Sharpe Ratio.
        
        Args:
            returns: Series com retornos
            risk_free: Taxa livre de risco
            
        Returns:
            Sharpe Ratio anualizado
        """
        if risk_free is None:
            risk_free = self.config.risk_free_rate / self.config.trading_days
        
        excess_returns = returns - risk_free
        
        if len(returns) < 2:
            return 0.0
        
        mean_return = excess_returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        # Anualizar
        periods_per_year = self.config.trading_days
        sharpe = mean_return / std_return * np.sqrt(periods_per_year)
        
        return float(sharpe)
    
    def calculate_sortino_ratio(self,
                               returns: pd.Series,
                               target_return: float = 0) -> float:
        """Calcula Sortino Ratio.
        
        Args:
            returns: Series com retornos
            target_return: Retorno alvo
            
        Returns:
            Sortino Ratio
        """
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) < 2:
            return 0.0
        
        mean_return = excess_returns.mean()
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0.0
        
        # Anualizar
        periods_per_year = self.config.trading_days
        sortino = mean_return / downside_std * np.sqrt(periods_per_year)
        
        return float(sortino)
    
    def calculate_calmar_ratio(self,
                              returns: pd.Series,
                              max_drawdown: Optional[float] = None) -> float:
        """Calcula Calmar Ratio.
        
        Args:
            returns: Series com retornos
            max_drawdown: Drawdown máximo (se None, calcula)
            
        Returns:
            Calmar Ratio
        """
        # Retorno anualizado
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        annual_return = (1 + total_return)**(self.config.trading_days / n_periods) - 1
        
        # Max Drawdown
        if max_drawdown is None:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
        
        if max_drawdown == 0:
            return 0.0
        
        calmar = annual_return / abs(max_drawdown)
        
        return float(calmar)
    
    def calculate_information_ratio(self,
                                   returns: pd.Series,
                                   benchmark_returns: pd.Series) -> float:
        """Calcula Information Ratio.
        
        Args:
            returns: Series com retornos da estratégia
            benchmark_returns: Series com retornos do benchmark
            
        Returns:
            Information Ratio
        """
        active_returns = returns - benchmark_returns
        
        if len(active_returns) < 2:
            return 0.0
        
        mean_active = active_returns.mean()
        std_active = active_returns.std()
        
        if std_active == 0:
            return 0.0
        
        # Anualizar
        periods_per_year = self.config.trading_days
        ir = mean_active / std_active * np.sqrt(periods_per_year)
        
        return float(ir)
    
    def calculate_omega_ratio(self,
                            returns: pd.Series,
                            threshold: float = 0) -> float:
        """Calcula Omega Ratio.
        
        Args:
            returns: Series com retornos
            threshold: Limite de retorno
            
        Returns:
            Omega Ratio
        """
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = -excess[excess < 0].sum()
        
        if losses == 0:
            return np.inf if gains > 0 else 0.0
        
        omega = gains / losses
        
        return float(omega)
    
    def calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calcula Tail Ratio.
        
        Razão entre o percentil 95 e o percentil 5 dos retornos.
        
        Args:
            returns: Series com retornos
            
        Returns:
            Tail Ratio
        """
        if len(returns) < 20:
            return 0.0
        
        right_tail = np.percentile(returns, 95)
        left_tail = abs(np.percentile(returns, 5))
        
        if left_tail == 0:
            return np.inf if right_tail > 0 else 0.0
        
        tail_ratio = right_tail / left_tail
        
        return float(tail_ratio)
    
    def calculate_all_metrics(self,
                             returns: pd.Series,
                             positions: Optional[pd.Series] = None,
                             prices: Optional[pd.Series] = None,
                             volumes: Optional[pd.Series] = None,
                             benchmark_returns: Optional[pd.Series] = None,
                             num_trials: int = 1) -> Dict[str, any]:
        """Calcula todas as métricas avançadas.
        
        Args:
            returns: Series com retornos
            positions: Series com posições (opcional)
            prices: Series com preços (opcional)
            volumes: Series com volumes (opcional)
            benchmark_returns: Series com retornos do benchmark (opcional)
            num_trials: Número de estratégias testadas
            
        Returns:
            Dict com todas as métricas
        """
        log.info("calculating_all_advanced_metrics", n_returns=len(returns))
        
        metrics = {}
        
        # Métricas básicas
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns)
        metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns)
        metrics['omega_ratio'] = self.calculate_omega_ratio(returns)
        metrics['tail_ratio'] = self.calculate_tail_ratio(returns)
        
        # PSR e DSR
        psr_result = self.calculate_psr(returns)
        metrics.update({f'psr_{k}': v for k, v in psr_result.items()})
        
        dsr_result = self.calculate_dsr(returns, num_trials)
        metrics.update({f'dsr_{k}': v for k, v in dsr_result.items()})
        
        # Information Ratio (se benchmark disponível)
        if benchmark_returns is not None:
            metrics['information_ratio'] = self.calculate_information_ratio(
                returns, benchmark_returns
            )
        
        # Turnover Analysis (se posições disponíveis)
        if positions is not None and prices is not None:
            turnover_result = self.calculate_turnover_analysis(positions, prices)
            metrics.update({f'turnover_{k}': v for k, v in turnover_result.items()})
        
        # Capacity Analysis (se volumes disponíveis)
        if volumes is not None:
            capacity_result = self.calculate_capacity_analysis(returns, volumes)
            metrics.update({f'capacity_{k}': v for k, v in capacity_result.items()})
        
        # Estatísticas adicionais
        metrics['mean_return'] = float(returns.mean())
        metrics['std_return'] = float(returns.std())
        metrics['skewness'] = float(stats.skew(returns))
        metrics['kurtosis'] = float(stats.kurtosis(returns, fisher=True))
        metrics['max_return'] = float(returns.max())
        metrics['min_return'] = float(returns.min())
        metrics['positive_periods'] = int((returns > 0).sum())
        metrics['negative_periods'] = int((returns < 0).sum())
        metrics['win_rate'] = float((returns > 0).mean())
        
        log.info("all_metrics_calculated", n_metrics=len(metrics))
        
        return metrics