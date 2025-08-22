"""Feature engineering sem vazamento temporal."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import ta
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import structlog
import warnings

warnings.filterwarnings("ignore")
log = structlog.get_logger()


class FeatureEngineer:
    """Engenharia de features com garantia de não-vazamento temporal."""

    def __init__(
        self,
        lookback_periods: List[int] = None,
        technical_indicators: List[str] = None,
        scaler_type: str = "robust",
        include_microstructure: bool = False,
        include_derivatives: bool = False,
    ):
        """Inicializa o feature engineer.
        
        Args:
            lookback_periods: Períodos para features de janela
            technical_indicators: Lista de indicadores técnicos
            scaler_type: Tipo de scaler (standard, robust, none)
            include_microstructure: Incluir features de microestrutura avançadas
            include_derivatives: Incluir features de derivativos
        """
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 100, 200]
        self.technical_indicators = technical_indicators or [
            "rsi", "macd", "bbands", "atr", "obv", "adx", "cci", "stoch"
        ]
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_names = []
        self.include_microstructure = include_microstructure
        self.include_derivatives = include_derivatives
        
        # Configurar scaler
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline completo de criação de features.
        
        Args:
            df: DataFrame com OHLCV
            
        Returns:
            DataFrame com todas as features
        """
        log.info("creating_features", initial_shape=df.shape)
        
        # Garantir que temos as colunas básicas
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Coluna obrigatória ausente: {col}")
        
        # Criar cópia para não modificar original
        df = df.copy()
        
        # 1. Features de preço e retorno
        df = self.create_price_features(df)
        
        # 2. Features de volatilidade
        df = self.create_volatility_features(df)
        
        # 3. Indicadores técnicos
        df = self.create_technical_indicators(df)
        
        # 4. Features de microestrutura
        df = self.create_microstructure_features(df)
        
        # 5. Features de microestrutura avançadas (opcional)
        if self.include_microstructure:
            from .microstructure import calculate_microstructure_features
            micro_features = calculate_microstructure_features(
                df, 
                lookback_periods=[10, 30, 60],
                include_order_book=False  # Ativar se tivermos dados de order book
            )
            df = pd.concat([df, micro_features], axis=1)
        
        # 6. Features de derivativos (opcional)
        if self.include_derivatives:
            from .derivatives import calculate_derivatives_features
            deriv_features = calculate_derivatives_features(
                df,
                symbol="BTCUSDT",
                fetch_live_data=False,  # Usar sintético por enquanto
                lookback_periods=[24, 96, 288]
            )
            df = pd.concat([df, deriv_features], axis=1)
        
        # 7. Features de regime
        df = self.create_regime_features(df)
        
        # 8. Features de calendário
        df = self.create_calendar_features(df)
        
        # Remover NaN iniciais (devido a janelas)
        initial_rows = len(df)
        df = df.dropna()
        rows_dropped = initial_rows - len(df)
        
        # Salvar nomes das features
        self.feature_names = [
            col for col in df.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        
        log.info(
            "features_created",
            final_shape=df.shape,
            rows_dropped=rows_dropped,
            n_features=len(self.feature_names),
        )
        
        return df

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features baseadas em preço e retornos."""
        
        # Retornos logarítmicos
        df["returns"] = np.log(df["close"] / df["close"].shift(1))
        
        # Retornos para múltiplos períodos
        for period in self.lookback_periods:
            # Retornos acumulados
            df[f"returns_{period}"] = df["returns"].rolling(period).sum()
            
            # Média móvel simples
            df[f"sma_{period}"] = df["close"].rolling(period).mean()
            
            # Média móvel exponencial
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
            
            # Distância do preço para SMA
            df[f"price_to_sma_{period}"] = (
                df["close"] - df[f"sma_{period}"]
            ) / df[f"sma_{period}"]
            
            # Momentum
            df[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1
        
        # Z-scores de preço
        for period in [20, 50, 100]:
            mean = df["close"].rolling(period).mean()
            std = df["close"].rolling(period).std()
            df[f"zscore_{period}"] = (df["close"] - mean) / (std + 1e-10)
        
        # Cruzamentos de médias móveis (apenas para períodos disponíveis)
        if "sma_20" in df.columns and "sma_50" in df.columns:
            df["sma_cross_20_50"] = (
                (df["sma_20"] > df["sma_50"]).astype(int) -
                (df["sma_20"].shift(1) > df["sma_50"].shift(1)).astype(int)
            )
        
        if "sma_50" in df.columns and "sma_100" in df.columns:
            df["sma_cross_50_100"] = (
                (df["sma_50"] > df["sma_100"]).astype(int) -
                (df["sma_50"].shift(1) > df["sma_100"].shift(1)).astype(int)
            )
        
        return df

    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de volatilidade."""
        
        # Volatilidade histórica (desvio padrão dos retornos)
        for period in self.lookback_periods:
            df[f"volatility_{period}"] = df["returns"].rolling(period).std()
            
            # Volatilidade anualizada
            df[f"volatility_ann_{period}"] = (
                df[f"volatility_{period}"] * np.sqrt(365 * 24 * 4)  # 15min bars
            )
        
        # Parkinson volatility (usando high-low)
        for period in [10, 20, 50]:
            hl_ratio = np.log(df["high"] / df["low"])
            df[f"parkinson_vol_{period}"] = (
                hl_ratio.rolling(period).apply(
                    lambda x: np.sqrt(np.sum(x**2) / (4 * period * np.log(2)))
                )
            )
        
        # Garman-Klass volatility
        for period in [10, 20, 50]:
            hl = np.log(df["high"] / df["low"]) ** 2
            co = np.log(df["close"] / df["open"]) ** 2
            df[f"gk_vol_{period}"] = np.sqrt(
                0.5 * hl.rolling(period).mean() -
                (2 * np.log(2) - 1) * co.rolling(period).mean()
            )
        
        # Volatility ratio (short/long)
        df["vol_ratio_10_50"] = df["volatility_10"] / (df["volatility_50"] + 1e-10)
        df["vol_ratio_20_100"] = df["volatility_20"] / (df["volatility_100"] + 1e-10)
        
        return df

    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicadores técnicos clássicos."""
        
        # RSI (Relative Strength Index)
        if "rsi" in self.technical_indicators:
            for period in [7, 14, 21]:
                df[f"rsi_{period}"] = ta.momentum.RSIIndicator(
                    df["close"], window=period
                ).rsi()
                
                # RSI overbought/oversold
                df[f"rsi_{period}_overbought"] = (df[f"rsi_{period}"] > 70).astype(int)
                df[f"rsi_{period}_oversold"] = (df[f"rsi_{period}"] < 30).astype(int)
        
        # MACD
        if "macd" in self.technical_indicators:
            macd = ta.trend.MACD(df["close"])
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_diff"] = macd.macd_diff()
            
            # MACD crossover
            df["macd_crossover"] = (
                (df["macd"] > df["macd_signal"]).astype(int) -
                (df["macd"].shift(1) > df["macd_signal"].shift(1)).astype(int)
            )
        
        # Bollinger Bands
        if "bbands" in self.technical_indicators:
            for period in [20, 50]:
                bb = ta.volatility.BollingerBands(
                    df["close"], window=period, window_dev=2
                )
                df[f"bb_high_{period}"] = bb.bollinger_hband()
                df[f"bb_low_{period}"] = bb.bollinger_lband()
                df[f"bb_mid_{period}"] = bb.bollinger_mavg()
                df[f"bb_width_{period}"] = (
                    df[f"bb_high_{period}"] - df[f"bb_low_{period}"]
                )
                df[f"bb_position_{period}"] = (
                    (df["close"] - df[f"bb_low_{period}"]) /
                    (df[f"bb_width_{period}"] + 1e-10)
                )
                
                # Bollinger squeeze
                df[f"bb_squeeze_{period}"] = (
                    df[f"bb_width_{period}"] /
                    df[f"bb_width_{period}"].rolling(100).mean()
                )
        
        # ATR (Average True Range)
        if "atr" in self.technical_indicators:
            for period in [7, 14, 21]:
                df[f"atr_{period}"] = ta.volatility.AverageTrueRange(
                    df["high"], df["low"], df["close"], window=period
                ).average_true_range()
                
                # ATR percentage
                df[f"atr_pct_{period}"] = df[f"atr_{period}"] / df["close"]
        
        # OBV (On Balance Volume)
        if "obv" in self.technical_indicators:
            df["obv"] = ta.volume.OnBalanceVolumeIndicator(
                df["close"], df["volume"]
            ).on_balance_volume()
            
            # OBV momentum
            for period in [10, 20]:
                df[f"obv_momentum_{period}"] = (
                    df["obv"] / df["obv"].shift(period) - 1
                )
        
        # ADX (Average Directional Index)
        if "adx" in self.technical_indicators:
            adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
            df["adx"] = adx.adx()
            df["adx_pos"] = adx.adx_pos()
            df["adx_neg"] = adx.adx_neg()
            
            # Trend strength
            df["trend_strength"] = df["adx"] / 100
        
        # CCI (Commodity Channel Index)
        if "cci" in self.technical_indicators:
            for period in [14, 20]:
                df[f"cci_{period}"] = ta.trend.CCIIndicator(
                    df["high"], df["low"], df["close"], window=period
                ).cci()
        
        # Stochastic Oscillator
        if "stoch" in self.technical_indicators:
            stoch = ta.momentum.StochasticOscillator(
                df["high"], df["low"], df["close"]
            )
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()
            
            # Stochastic crossover
            df["stoch_crossover"] = (
                (df["stoch_k"] > df["stoch_d"]).astype(int) -
                (df["stoch_k"].shift(1) > df["stoch_d"].shift(1)).astype(int)
            )
        
        return df

    def create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de microestrutura de mercado."""
        
        # Volume features
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-10)
        df["volume_trend"] = (
            df["volume"].rolling(10).mean() /
            (df["volume"].rolling(50).mean() + 1e-10)
        )
        
        # Dollar volume
        df["dollar_volume"] = df["close"] * df["volume"]
        df["dollar_volume_20"] = df["dollar_volume"].rolling(20).mean()
        
        # VWAP (Volume Weighted Average Price)
        for period in [20, 50]:
            df[f"vwap_{period}"] = (
                (df["close"] * df["volume"]).rolling(period).sum() /
                df["volume"].rolling(period).sum()
            )
            df[f"vwap_distance_{period}"] = (
                (df["close"] - df[f"vwap_{period}"]) / df[f"vwap_{period}"]
            )
        
        # High-Low spread
        df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
        df["hl_spread_20"] = df["hl_spread"].rolling(20).mean()
        
        # Close position in bar
        df["close_position"] = (
            (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)
        )
        
        # Amihud illiquidity
        df["amihud_illiq"] = (
            np.abs(df["returns"]) / (df["dollar_volume"] + 1e-10)
        )
        df["amihud_illiq_20"] = df["amihud_illiq"].rolling(20).mean()
        
        # Kyle's lambda (simplified)
        for period in [20, 50]:
            df[f"kyle_lambda_{period}"] = (
                df["returns"].rolling(period).std() /
                (df["volume"].rolling(period).std() + 1e-10)
            )
        
        return df

    def create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de regime de mercado."""
        
        # Volatility regime
        vol_20 = df["returns"].rolling(20).std()
        vol_100 = df["returns"].rolling(100).std()
        
        # Volatility percentile
        df["vol_percentile_252"] = (
            vol_20.rolling(252).rank(pct=True)
        )
        
        # High/Low volatility regime
        df["high_vol_regime"] = (df["vol_percentile_252"] > 0.8).astype(int)
        df["low_vol_regime"] = (df["vol_percentile_252"] < 0.2).astype(int)
        
        # Trend regime (using ADX if available)
        if "adx" in df.columns:
            df["trending_regime"] = (df["adx"] > 25).astype(int)
            df["ranging_regime"] = (df["adx"] < 20).astype(int)
        
        # Momentum regime
        mom_20 = df["returns"].rolling(20).sum()
        df["momentum_percentile"] = mom_20.rolling(252).rank(pct=True)
        df["high_momentum"] = (df["momentum_percentile"] > 0.8).astype(int)
        df["low_momentum"] = (df["momentum_percentile"] < 0.2).astype(int)
        
        # Market phase (accumulation, markup, distribution, markdown)
        # Simplified version using price and volume
        price_trend = df["close"].rolling(50).apply(
            lambda x: stats.linregress(range(len(x)), x)[0]
        )
        volume_trend = df["volume"].rolling(50).apply(
            lambda x: stats.linregress(range(len(x)), x)[0]
        )
        
        df["accumulation_phase"] = (
            (price_trend < 0) & (volume_trend > 0)
        ).astype(int)
        df["markup_phase"] = (
            (price_trend > 0) & (volume_trend > 0)
        ).astype(int)
        df["distribution_phase"] = (
            (price_trend > 0) & (volume_trend < 0)
        ).astype(int)
        df["markdown_phase"] = (
            (price_trend < 0) & (volume_trend < 0)
        ).astype(int)
        
        return df

    def create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de calendário e tempo."""
        
        # Extrair componentes de tempo
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        
        # Features cíclicas (seno/cosseno para capturar periodicidade)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        # Sessões de trading (UTC)
        # Asian: 00:00-08:00
        # European: 08:00-16:00
        # American: 16:00-00:00
        df["asian_session"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
        df["european_session"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(int)
        df["american_session"] = ((df["hour"] >= 16) & (df["hour"] < 24)).astype(int)
        
        # Overlap de sessões
        df["session_overlap"] = (
            ((df["hour"] >= 7) & (df["hour"] < 9)) |  # Asia-Europe
            ((df["hour"] >= 14) & (df["hour"] < 17))   # Europe-America
        ).astype(int)
        
        # Fim de semana
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        
        # Início/fim de mês
        df["month_start"] = (df["day_of_month"] <= 5).astype(int)
        df["month_end"] = (df["day_of_month"] >= 25).astype(int)
        
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler e transforma features.
        
        Usar apenas em dados de treino!
        """
        # Criar features
        df = self.create_all_features(df)
        
        # Fit e transform scaler se configurado
        if self.scaler is not None:
            feature_cols = self.feature_names
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
            log.info("scaler_fitted", scaler_type=self.scaler_type)
        
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforma features usando scaler já fitado.
        
        Usar em dados de validação/teste!
        """
        # Criar features
        df = self.create_all_features(df)
        
        # Transform com scaler já fitado
        if self.scaler is not None:
            if not hasattr(self.scaler, "mean_") and not hasattr(self.scaler, "center_"):
                raise ValueError("Scaler não foi fitado! Use fit_transform primeiro.")
            
            feature_cols = self.feature_names
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        return df

    def get_feature_importance(self, model, feature_names: List[str] = None) -> pd.DataFrame:
        """Extrai importância das features do modelo.
        
        Args:
            model: Modelo treinado (XGBoost, RandomForest, etc)
            feature_names: Nomes das features (usa self.feature_names se None)
            
        Returns:
            DataFrame com importâncias ordenadas
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            raise ValueError("Modelo não tem atributo de importância")
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        # Normalizar para percentual
        importance_df["importance_pct"] = (
            importance_df["importance"] / importance_df["importance"].sum() * 100
        )
        
        return importance_df