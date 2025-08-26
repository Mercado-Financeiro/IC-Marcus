"""
Binance Klines fetcher with rate limiting and public data support.
Aligned with PRD section 2.1 - Data Ingestion requirements.
"""

import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import requests
import ccxt
from tqdm import tqdm
import structlog
import hashlib
import json

log = structlog.get_logger()


class BinanceKlinesFetcher:
    """
    Fetcher for Binance klines with:
    - Rate limit handling (PRD section 12)
    - Binance public data support for bulk historical
    - REST API for recent data
    - Automatic validation with GE
    """
    
    # Binance public data base URL
    PUBLIC_DATA_BASE = "https://data.binance.vision"
    
    # REST API limits
    WEIGHT_LIMIT = 1200  # Per minute
    REQUEST_WEIGHT = 1    # Weight per klines request
    
    def __init__(
        self,
        output_dir: str = "data/raw",
        use_public_data: bool = True,
        validate_data: bool = True
    ):
        """Initialize fetcher with configuration."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_public_data = use_public_data
        self.validate_data = validate_data
        
        # Initialize CCXT exchange
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'rateLimit': 50,  # ms between requests
            'options': {
                'defaultType': 'spot',
            }
        })
        
        # Track rate limits
        self.requests_made = 0
        self.weight_used = 0
        self.reset_time = time.time() + 60
        
    def _check_rate_limit(self, weight: int = 1) -> None:
        """Check and handle rate limits."""
        current_time = time.time()
        
        # Reset counters if minute passed
        if current_time > self.reset_time:
            self.weight_used = 0
            self.reset_time = current_time + 60
            
        # Check if we would exceed limit
        if self.weight_used + weight > self.WEIGHT_LIMIT:
            sleep_time = self.reset_time - current_time
            if sleep_time > 0:
                log.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time + 1)
                self.weight_used = 0
                self.reset_time = time.time() + 60
                
        self.weight_used += weight
        
    def fetch_from_public_data(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical data from Binance public data repository.
        More efficient for large historical datasets.
        """
        
        # Convert dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        all_data = []
        
        # Generate monthly URLs
        current = start
        while current <= end:
            year = current.year
            month = str(current.month).zfill(2)
            
            # Construct URL for monthly data
            filename = f"{symbol}-{interval}-{year}-{month}.zip"
            url = f"{self.PUBLIC_DATA_BASE}/data/spot/monthly/klines/{symbol}/{interval}/{filename}"
            
            try:
                log.info(f"Fetching {filename} from public data")
                
                # Download and extract
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    # Save temporarily
                    temp_file = self.output_dir / filename
                    with open(temp_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Extract and read CSV
                    import zipfile
                    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                        zip_ref.extractall(self.output_dir)
                    
                    # Read the CSV file
                    csv_file = temp_file.with_suffix('.csv')
                    if csv_file.exists():
                        df = pd.read_csv(csv_file, header=None)
                        df.columns = [
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                            'taker_buy_quote', 'ignore'
                        ]
                        all_data.append(df)
                        
                        # Cleanup
                        csv_file.unlink()
                    
                    temp_file.unlink()
                    
                else:
                    log.warning(f"Public data not available for {year}-{month}")
                    
            except Exception as e:
                log.error(f"Error fetching public data: {e}")
                
            # Move to next month
            current = current + pd.DateOffset(months=1)
            
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume', 'trades']]
            df = df.astype(float)
            
            # Filter date range
            df = df[(df.index >= start) & (df.index <= end)]
            
            return df
        
        return pd.DataFrame()
    
    def fetch_from_rest_api(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch data using REST API (for recent data).
        Respects rate limits as per PRD.
        """
        
        # Convert interval to CCXT format
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m',
            '1h': '1h', '4h': '4h', '1d': '1d'
        }
        
        timeframe = interval_map.get(interval, '15m')
        
        # Convert dates to timestamps
        since = int(pd.Timestamp(start_date).timestamp() * 1000)
        
        all_ohlcv = []
        
        log.info(f"Fetching {symbol} {interval} from REST API")
        
        while True:
            try:
                # Check rate limit
                self._check_rate_limit(self.REQUEST_WEIGHT)
                
                # Fetch OHLCV
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol.replace('USDT', '/USDT'),
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                # Update since for next iteration
                since = ohlcv[-1][0] + 1
                
                # Check if we've reached end date
                if end_date:
                    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
                    if since > end_ts:
                        break
                        
                # Break if we got less than limit (no more data)
                if len(ohlcv) < limit:
                    break
                    
            except Exception as e:
                log.error(f"Error fetching from API: {e}")
                break
                
        if all_ohlcv:
            df = pd.DataFrame(
                all_ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['trades'] = 0  # API doesn't provide this
            
            return df
            
        return pd.DataFrame()
    
    def fetch_and_save(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        force_refresh: bool = False
    ) -> Path:
        """
        Main method to fetch and save data with caching.
        """
        
        # Generate cache filename
        cache_file = self.output_dir / f"{symbol}_{interval}_{start_date}_{end_date}.parquet"
        
        # Check cache
        if cache_file.exists() and not force_refresh:
            log.info(f"Using cached data: {cache_file}")
            return cache_file
            
        # Determine fetch strategy
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        days_ago = (pd.Timestamp.now() - start).days
        
        df = pd.DataFrame()
        
        if self.use_public_data and days_ago > 30:
            # Use public data for historical (>30 days old)
            log.info("Using Binance public data for historical fetch")
            df = self.fetch_from_public_data(symbol, interval, start_date, end_date)
            
        if df.empty or days_ago <= 30:
            # Use REST API for recent data
            log.info("Using REST API for recent data fetch")
            df = self.fetch_from_rest_api(symbol, interval, start_date, end_date)
            
        if df.empty:
            log.error(f"No data fetched for {symbol} {interval}")
            return None
            
        # Validate if enabled
        if self.validate_data:
            try:
                from scripts.validate.ge_checks import validate_crypto_data
                is_valid, report = validate_crypto_data(df, symbol, interval)
                if not is_valid:
                    log.warning(f"Data validation failed: {report}")
            except ImportError:
                log.warning("Great Expectations not available, skipping validation")
                
        # Save to parquet
        df.to_parquet(cache_file)
        log.info(f"Saved {len(df)} rows to {cache_file}")
        
        return cache_file
    
    def fetch_multiple(
        self,
        symbols: List[str],
        intervals: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, Path]:
        """Fetch multiple symbol/interval combinations."""
        
        results = {}
        
        total = len(symbols) * len(intervals)
        with tqdm(total=total, desc="Fetching data") as pbar:
            for symbol in symbols:
                for interval in intervals:
                    key = f"{symbol}_{interval}"
                    try:
                        path = self.fetch_and_save(
                            symbol, interval, start_date, end_date
                        )
                        results[key] = path
                    except Exception as e:
                        log.error(f"Failed to fetch {key}: {e}")
                        results[key] = None
                    pbar.update(1)
                    
        return results


def main():
    """CLI interface aligned with PRD."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch Binance klines data")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"], help="Symbols to fetch")
    parser.add_argument("--intervals", nargs="+", default=["15m"], help="Intervals (1m, 5m, 15m, 1h, 4h)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--no-public", action="store_true", help="Don't use public data")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    parser.add_argument("--force", action="store_true", help="Force refresh (ignore cache)")
    
    args = parser.parse_args()
    
    # Initialize fetcher
    fetcher = BinanceKlinesFetcher(
        output_dir=args.output,
        use_public_data=not args.no_public,
        validate_data=not args.no_validate
    )
    
    # Fetch data
    results = fetcher.fetch_multiple(
        symbols=args.symbols,
        intervals=args.intervals,
        start_date=args.start,
        end_date=args.end
    )
    
    # Print results
    print("\nFetch Results:")
    print("-" * 50)
    for key, path in results.items():
        if path:
            size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ {key}: {path.name} ({size:.2f} MB)")
        else:
            print(f"✗ {key}: Failed")
            
    # Summary
    success = sum(1 for p in results.values() if p)
    print(f"\nSuccess: {success}/{len(results)}")
    

if __name__ == "__main__":
    main()