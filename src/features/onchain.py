"""On-chain features para análise de criptomoedas."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import structlog
from dataclasses import dataclass

log = structlog.get_logger()


@dataclass
class OnChainConfig:
    """Configuração para features on-chain."""
    
    nvt_window: int = 90
    mvrv_window: int = 365
    sopr_window: int = 7
    whale_threshold: float = 1000  # BTC
    exchange_flow_window: int = 24  # horas


class OnChainFeatures:
    """Extrator de features on-chain para criptomoedas."""
    
    def __init__(self, config: Optional[OnChainConfig] = None):
        """Inicializa o extrator de features on-chain.
        
        Args:
            config: Configuração para cálculo das features
        """
        self.config = config or OnChainConfig()
        
    def calculate_nvt_ratio(self, 
                           df: pd.DataFrame,
                           price_col: str = 'close',
                           volume_col: str = 'volume') -> pd.Series:
        """Calcula NVT (Network Value to Transactions) Ratio.
        
        NVT = Market Cap / Transaction Volume
        Alto NVT pode indicar sobrevalorização.
        
        Args:
            df: DataFrame com dados de preço e volume
            price_col: Nome da coluna de preço
            volume_col: Nome da coluna de volume on-chain
            
        Returns:
            Series com NVT ratio
        """
        # Usar volume como proxy para transaction volume
        # Em produção, usar dados reais de blockchain
        market_cap = df[price_col] * df[volume_col].rolling(7).mean()
        
        # Transaction volume (média móvel para suavizar)
        tx_volume = df[volume_col].rolling(self.config.nvt_window).mean()
        
        # Evitar divisão por zero
        nvt = market_cap / (tx_volume + 1e-10)
        
        # Aplicar transformação log para estabilizar
        nvt_log = np.log1p(nvt)
        
        log.info("nvt_calculated", 
                window=self.config.nvt_window,
                mean_nvt=float(nvt_log.mean()))
        
        return nvt_log
    
    def calculate_mvrv(self,
                      df: pd.DataFrame,
                      price_col: str = 'close') -> pd.Series:
        """Calcula MVRV (Market Value to Realized Value).
        
        MVRV = Market Cap / Realized Cap
        MVRV > 3.7 historicamente indica topo
        MVRV < 1 historicamente indica fundo
        
        Args:
            df: DataFrame com dados de preço
            price_col: Nome da coluna de preço
            
        Returns:
            Series com MVRV ratio
        """
        # Realized price (média móvel longa como proxy)
        # Em produção, usar UTXO data real
        realized_price = df[price_col].rolling(
            self.config.mvrv_window, min_periods=30
        ).mean()
        
        # MVRV ratio
        mvrv = df[price_col] / (realized_price + 1e-10)
        
        # Z-score para normalização
        mvrv_zscore = (mvrv - mvrv.rolling(90).mean()) / mvrv.rolling(90).std()
        
        log.info("mvrv_calculated",
                window=self.config.mvrv_window,
                current_mvrv=float(mvrv.iloc[-1]) if len(mvrv) > 0 else 0)
        
        return mvrv_zscore
    
    def calculate_sopr(self,
                      df: pd.DataFrame,
                      price_col: str = 'close') -> pd.Series:
        """Calcula SOPR (Spent Output Profit Ratio).
        
        SOPR > 1: Moedas vendidas com lucro
        SOPR < 1: Moedas vendidas com prejuízo
        SOPR = 1: Breakeven, possível suporte/resistência
        
        Args:
            df: DataFrame com dados de preço
            price_col: Nome da coluna de preço
            
        Returns:
            Series com SOPR
        """
        # Proxy: comparar preço atual com média móvel
        # Em produção, usar dados reais de UTXO
        cost_basis = df[price_col].rolling(
            self.config.sopr_window * 4, min_periods=7
        ).mean()
        
        sopr = df[price_col] / (cost_basis + 1e-10)
        
        # Adjusted SOPR (remove transações < 1h)
        # Simulado com suavização
        asopr = sopr.rolling(self.config.sopr_window).mean()
        
        log.info("sopr_calculated",
                window=self.config.sopr_window,
                mean_sopr=float(asopr.mean()))
        
        return asopr
    
    def calculate_active_addresses(self,
                                  df: pd.DataFrame,
                                  volume_col: str = 'volume') -> pd.Series:
        """Estima número de endereços ativos.
        
        Usa volume como proxy para atividade de rede.
        
        Args:
            df: DataFrame com dados
            volume_col: Nome da coluna de volume
            
        Returns:
            Series com estimativa de endereços ativos
        """
        # Normalizar volume e usar como proxy
        # Em produção, usar dados reais de blockchain
        volume_norm = df[volume_col] / df[volume_col].rolling(30).mean()
        
        # Simular contagem de endereços únicos
        # Assumir relação não-linear com volume
        active_addresses = np.power(volume_norm, 0.7) * 100000
        
        # Suavizar com média móvel
        active_smooth = active_addresses.rolling(7).mean()
        
        return active_smooth
    
    def calculate_hash_rate_proxy(self,
                                 df: pd.DataFrame,
                                 high_col: str = 'high',
                                 low_col: str = 'low') -> pd.Series:
        """Calcula proxy para hash rate / network security.
        
        Usa volatilidade e preço como proxy para segurança da rede.
        
        Args:
            df: DataFrame com dados OHLC
            high_col: Nome da coluna high
            low_col: Nome da coluna low
            
        Returns:
            Series com hash rate proxy
        """
        # Difficulty proxy: inverso da volatilidade
        # Maior segurança = menor volatilidade
        volatility = (df[high_col] - df[low_col]) / df[low_col]
        hash_proxy = 1 / (volatility.rolling(14).mean() + 0.01)
        
        # Normalizar
        hash_normalized = (hash_proxy - hash_proxy.rolling(90).mean()) / \
                         hash_proxy.rolling(90).std()
        
        return hash_normalized
    
    def detect_whale_movements(self,
                              df: pd.DataFrame,
                              volume_col: str = 'volume',
                              price_col: str = 'close') -> pd.Series:
        """Detecta movimentos de baleias (grandes transações).
        
        Args:
            df: DataFrame com dados
            volume_col: Nome da coluna de volume
            price_col: Nome da coluna de preço
            
        Returns:
            Series com indicador de movimento de baleias
        """
        # Volume em USD
        volume_usd = df[volume_col] * df[price_col]
        
        # Detectar volumes anormais (possíveis baleias)
        volume_mean = volume_usd.rolling(30).mean()
        volume_std = volume_usd.rolling(30).std()
        
        # Z-score do volume
        volume_zscore = (volume_usd - volume_mean) / (volume_std + 1e-10)
        
        # Movimento de baleia quando z-score > 2
        whale_movement = (volume_zscore > 2).astype(float)
        
        # Suavizar sinal
        whale_signal = whale_movement.rolling(3).mean()
        
        log.info("whale_movements_detected",
                total_movements=int(whale_movement.sum()))
        
        return whale_signal
    
    def calculate_exchange_flows(self,
                                df: pd.DataFrame,
                                volume_col: str = 'volume') -> Dict[str, pd.Series]:
        """Calcula fluxos de/para exchanges.
        
        Inflow alto: pressão de venda
        Outflow alto: acumulação/HODLing
        
        Args:
            df: DataFrame com dados
            volume_col: Nome da coluna de volume
            
        Returns:
            Dict com inflow e outflow estimados
        """
        # Simular com base em padrões de volume
        # Em produção, usar dados reais de blockchain
        
        # Volume crescente + preço caindo = inflow
        volume_change = df[volume_col].pct_change(5)
        price_change = df['close'].pct_change(5)
        
        # Estimar inflow/outflow
        inflow = pd.Series(0.0, index=df.index)
        outflow = pd.Series(0.0, index=df.index)
        
        # Condições para inflow (venda)
        sell_pressure = (volume_change > 0) & (price_change < 0)
        inflow[sell_pressure] = volume_change[sell_pressure].abs()
        
        # Condições para outflow (acumulação)
        accumulation = (volume_change > 0) & (price_change > 0)
        outflow[accumulation] = volume_change[accumulation].abs()
        
        # Suavizar sinais
        inflow_smooth = inflow.rolling(self.config.exchange_flow_window).mean()
        outflow_smooth = outflow.rolling(self.config.exchange_flow_window).mean()
        
        # Netflow
        netflow = outflow_smooth - inflow_smooth
        
        return {
            'exchange_inflow': inflow_smooth,
            'exchange_outflow': outflow_smooth,
            'exchange_netflow': netflow
        }
    
    def calculate_puell_multiple(self,
                                df: pd.DataFrame,
                                close_col: str = 'close',
                                volume_col: str = 'volume') -> pd.Series:
        """Calcula Puell Multiple.
        
        Puell = Daily Issuance Value / 365 MA of Daily Issuance Value
        
        Args:
            df: DataFrame com dados
            close_col: Nome da coluna de preço
            volume_col: Nome da coluna de volume
            
        Returns:
            Series com Puell Multiple
        """
        # Simular daily issuance value
        # Em produção, usar dados reais de mineração
        daily_issuance = df[volume_col] * 0.01  # 1% do volume como proxy
        issuance_value = daily_issuance * df[close_col]
        
        # MA de 365 dias
        ma_365 = issuance_value.rolling(365, min_periods=30).mean()
        
        # Puell Multiple
        puell = issuance_value / (ma_365 + 1e-10)
        
        return puell
    
    def calculate_dormancy_flow(self,
                               df: pd.DataFrame,
                               close_col: str = 'close') -> pd.Series:
        """Calcula Entity-Adjusted Dormancy Flow.
        
        Mede a razão entre coin days destroyed e market cap.
        
        Args:
            df: DataFrame com dados
            close_col: Nome da coluna de preço
            
        Returns:
            Series com dormancy flow
        """
        # Simular coin days destroyed
        # Em produção, usar dados reais de UTXO
        price_ma = df[close_col].rolling(90, min_periods=30).mean()
        price_std = df[close_col].rolling(90, min_periods=30).std()
        
        # Proxy: grandes movimentos de preço indicam coins antigas se movendo
        price_zscore = (df[close_col] - price_ma) / (price_std + 1e-10)
        dormancy = np.abs(price_zscore) * df['volume']
        
        # Normalizar por market cap (proxy)
        market_cap_proxy = df[close_col] * df['volume'].rolling(7).mean()
        dormancy_flow = dormancy / (market_cap_proxy + 1e-10)
        
        return dormancy_flow
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todas as features on-chain.
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com features on-chain adicionadas
        """
        log.info("calculating_all_onchain_features", rows=len(df))
        
        # Copiar DataFrame
        result = df.copy()
        
        # NVT Ratio
        result['nvt_ratio'] = self.calculate_nvt_ratio(df)
        
        # MVRV
        result['mvrv_zscore'] = self.calculate_mvrv(df)
        
        # SOPR
        result['asopr'] = self.calculate_sopr(df)
        
        # Active Addresses
        result['active_addresses'] = self.calculate_active_addresses(df)
        
        # Hash Rate Proxy
        result['hash_rate_proxy'] = self.calculate_hash_rate_proxy(df)
        
        # Whale Movements
        result['whale_signal'] = self.detect_whale_movements(df)
        
        # Exchange Flows
        flows = self.calculate_exchange_flows(df)
        for key, value in flows.items():
            result[key] = value
        
        # Puell Multiple
        result['puell_multiple'] = self.calculate_puell_multiple(df)
        
        # Dormancy Flow
        result['dormancy_flow'] = self.calculate_dormancy_flow(df)
        
        # Fill NaN values
        result = result.fillna(method='ffill').fillna(0)
        
        log.info("onchain_features_calculated",
                features_added=len(result.columns) - len(df.columns))
        
        return result
    
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """Retorna pesos de importância sugeridos para features on-chain.
        
        Returns:
            Dict com pesos por feature
        """
        return {
            'nvt_ratio': 0.15,
            'mvrv_zscore': 0.20,
            'asopr': 0.15,
            'active_addresses': 0.10,
            'hash_rate_proxy': 0.05,
            'whale_signal': 0.10,
            'exchange_netflow': 0.15,
            'puell_multiple': 0.05,
            'dormancy_flow': 0.05
        }