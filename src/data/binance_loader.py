"""Binance data loader com validação temporal e caching."""

import os
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import ccxt
import yfinance as yf
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import structlog
from pathlib import Path

# Configurar logging
log = structlog.get_logger()

# Schema de validação
BarsSchema = DataFrameSchema(
    {
        "open": Column(float, Check.greater_than(0)),
        "high": Column(float, Check.greater_than(0)),
        "low": Column(float, Check.greater_than(0)),
        "close": Column(float, Check.greater_than(0)),
        "volume": Column(float, Check.greater_than_or_equal_to(0)),
    },
    index=pa.Index("datetime64[ns, UTC]", name="timestamp"),
    strict=True,
    coerce=True,
)


class CryptoDataLoader:
    """Loader de dados de criptomoedas com cache e fallback."""

    def __init__(
        self,
        exchange: str = "binance",
        cache_dir: str = "data/raw",
        use_cache: bool = True,
    ):
        """Inicializa o loader.
        
        Args:
            exchange: Exchange para buscar dados (binance)
            cache_dir: Diretório para cache local
            use_cache: Se deve usar cache local
        """
        self.exchange_name = exchange
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        
        # Inicializar exchange
        try:
            self.exchange = ccxt.binance(
                {
                    "rateLimit": 1200,
                    "enableRateLimit": True,
                    "timeout": 30000,
                }
            )
            log.info("binance_api_initialized")
        except Exception as e:
            log.warning("binance_api_failed", error=str(e))
            self.exchange = None

    def _get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Retorna o caminho do cache para o símbolo/timeframe."""
        filename = f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        return self.cache_dir / filename

    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """Calcula hash SHA256 dos dados."""
        data_str = df.to_json()
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        validate: bool = True,
    ) -> pd.DataFrame:
        """Busca dados OHLCV com cache e fallback.
        
        Args:
            symbol: Par de trading (ex: BTCUSDT)
            timeframe: Timeframe (5m, 15m, 1h, etc)
            start_date: Data inicial (YYYY-MM-DD)
            end_date: Data final (YYYY-MM-DD)
            validate: Se deve validar com pandera
            
        Returns:
            DataFrame com OHLCV indexado por timestamp UTC
        """
        cache_path = self._get_cache_path(symbol, timeframe)
        
        # Tentar carregar do cache
        if self.use_cache and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                log.info(
                    "data_loaded_from_cache",
                    symbol=symbol,
                    timeframe=timeframe,
                    rows=len(df),
                )
                
                # Filtrar por datas se necessário
                df = self._filter_by_dates(df, start_date, end_date)
                
                if validate:
                    df = self.validate_data(df)
                    
                return df
                
            except Exception as e:
                log.warning("cache_load_failed", error=str(e))
        
        # Buscar dados da API
        df = self._fetch_from_api(symbol, timeframe, start_date, end_date)
        
        # Fallback para yfinance se Binance falhar
        if df is None or df.empty:
            log.warning("binance_fetch_failed_trying_yfinance")
            df = self._fetch_from_yfinance(symbol, timeframe, start_date, end_date)
        
        if df is None or df.empty:
            raise ValueError(f"Não foi possível obter dados para {symbol}")
        
        # Validar dados
        if validate:
            df = self.validate_data(df)
        
        # Salvar no cache
        if self.use_cache:
            try:
                df.to_parquet(cache_path)
                log.info("data_saved_to_cache", path=str(cache_path))
            except Exception as e:
                log.warning("cache_save_failed", error=str(e))
        
        return df

    def _fetch_from_api(
        self, symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Busca dados da Binance API."""
        if self.exchange is None:
            return None
        
        try:
            # Converter datas para timestamps
            start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
            end_ts = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)
            
            all_data = []
            current_ts = start_ts
            
            # Mapear timeframe para milissegundos
            tf_ms = self._timeframe_to_ms(timeframe)
            
            while current_ts < end_ts:
                try:
                    # Buscar batch (máximo 1000 barras)
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol, timeframe, since=current_ts, limit=1000
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_data.extend(ohlcv)
                    
                    # Próximo timestamp
                    last_ts = ohlcv[-1][0]
                    if last_ts == current_ts:
                        break
                    current_ts = last_ts + tf_ms
                    
                    log.debug(
                        "batch_fetched",
                        symbol=symbol,
                        count=len(ohlcv),
                        last_date=pd.Timestamp(last_ts, unit="ms"),
                    )
                    
                except Exception as e:
                    log.error("api_fetch_error", error=str(e))
                    break
            
            if not all_data:
                return None
            
            # Converter para DataFrame
            df = pd.DataFrame(
                all_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            
            # Converter timestamp para datetime UTC
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            
            # Remover duplicatas e ordenar
            df = df[~df.index.duplicated(keep="first")]
            df.sort_index(inplace=True)
            
            log.info(
                "data_fetched_from_api",
                symbol=symbol,
                timeframe=timeframe,
                rows=len(df),
                start=df.index[0],
                end=df.index[-1],
            )
            
            return df
            
        except Exception as e:
            log.error("binance_api_error", error=str(e))
            return None

    def _fetch_from_yfinance(
        self, symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fallback para yfinance."""
        try:
            # Converter símbolo (BTCUSDT -> BTC-USD)
            yf_symbol = self._convert_symbol_to_yfinance(symbol)
            
            # Mapear timeframe
            yf_interval = self._convert_timeframe_to_yfinance(timeframe)
            
            # Baixar dados
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval,
                auto_adjust=False,
            )
            
            if df.empty:
                return None
            
            # Ajustar colunas
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["open", "high", "low", "close", "volume"]
            
            # Garantir timezone UTC
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            
            df.index.name = "timestamp"
            
            log.info(
                "data_fetched_from_yfinance",
                symbol=yf_symbol,
                rows=len(df),
                start=df.index[0],
                end=df.index[-1],
            )
            
            return df
            
        except Exception as e:
            log.error("yfinance_error", error=str(e))
            return None

    def _filter_by_dates(
        self, df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Filtra DataFrame por datas."""
        start = pd.Timestamp(start_date, tz="UTC")
        end = pd.Timestamp(end_date, tz="UTC")
        
        mask = (df.index >= start) & (df.index <= end)
        return df[mask]

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Converte timeframe para milissegundos."""
        mapping = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }
        return mapping.get(timeframe, 60 * 1000)

    def _convert_symbol_to_yfinance(self, symbol: str) -> str:
        """Converte símbolo Binance para yfinance."""
        # BTCUSDT -> BTC-USD
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            return f"{base}-USD"
        return symbol

    def _convert_timeframe_to_yfinance(self, timeframe: str) -> str:
        """Converte timeframe para yfinance."""
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "1h",  # yfinance não tem 4h
            "1d": "1d",
        }
        return mapping.get(timeframe, "1h")

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida dados com pandera.
        
        Args:
            df: DataFrame para validar
            
        Returns:
            DataFrame validado
            
        Raises:
            SchemaError: Se validação falhar
        """
        try:
            # Validar schema
            df = BarsSchema.validate(df)
            
            # Verificações adicionais
            assert df.index.is_monotonic_increasing, "Índice não é monotônico crescente"
            assert not df.index.has_duplicates, "Índice tem duplicatas"
            assert df["high"].ge(df["low"]).all(), "High < Low detectado"
            assert df["high"].ge(df["open"]).all(), "High < Open detectado"
            assert df["low"].le(df["open"]).all(), "Low > Open detectado"
            
            # Log hash dos dados
            data_hash = self._calculate_data_hash(df)
            log.info(
                "data_validated",
                rows=len(df),
                hash=data_hash,
                start=df.index[0],
                end=df.index[-1],
            )
            
            return df
            
        except pa.errors.SchemaError as e:
            log.error("schema_validation_failed", error=str(e))
            raise
        except AssertionError as e:
            log.error("data_validation_failed", error=str(e))
            raise

    def get_available_symbols(self) -> List[str]:
        """Retorna lista de símbolos disponíveis."""
        if self.exchange is None:
            return []
        
        try:
            markets = self.exchange.load_markets()
            symbols = [
                symbol
                for symbol in markets.keys()
                if "/" in symbol and symbol.endswith("USDT")
            ]
            return sorted(symbols)
        except Exception as e:
            log.error("get_symbols_failed", error=str(e))
            return []

    def check_data_quality(self, df: pd.DataFrame) -> Dict:
        """Verifica qualidade dos dados.
        
        Returns:
            Dicionário com métricas de qualidade
        """
        # Calcular gaps temporais
        time_diffs = df.index.to_series().diff()
        median_diff = time_diffs.median()
        
        # Detectar gaps (> 2x mediana)
        gaps = time_diffs[time_diffs > 2 * median_diff]
        
        # Calcular métricas
        metrics = {
            "total_rows": len(df),
            "start_date": df.index[0],
            "end_date": df.index[-1],
            "missing_values": df.isnull().sum().to_dict(),
            "gaps_detected": len(gaps),
            "largest_gap": gaps.max() if len(gaps) > 0 else pd.Timedelta(0),
            "data_hash": self._calculate_data_hash(df),
            "monotonic": df.index.is_monotonic_increasing,
            "duplicates": df.index.has_duplicates,
        }
        
        return metrics


# Backward compatibility alias
class BinanceDataLoader(CryptoDataLoader):
    """Alias for backward compatibility.

    Older code imports `BinanceDataLoader` from this module. The implementation
    was generalized to `CryptoDataLoader`; this class preserves the old name
    without changing behavior.
    """
    pass
