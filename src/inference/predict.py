#!/usr/bin/env python3
"""
Production inference script for crypto ML models.
Loads trained model and generates trading signals.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import yaml
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.binance_loader import CryptoDataLoader
from src.features.engineering import FeatureEngineer
from src.backtest.engine import BacktestEngine
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CryptoPredictor:
    """Production inference for crypto trading models."""
    
    def __init__(self, model_path: str, config_path: str = None):
        """Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved model pickle
            config_path: Optional path to config YAML
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        
        # Load model and components
        self.model = None
        self.calibrator = None
        self.thresholds = None
        self.feature_cols = None
        
        self._load_model()
        self._load_config()
        
        # Initialize components
        self.data_loader = CryptoDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.backtest_engine = BacktestEngine()
        
        logger.info(f"Predictor initialized with model: {model_path}")
    
    def _load_model(self):
        """Load trained model from pickle file."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data.get('model')
        self.calibrator = model_data.get('calibrator')
        self.thresholds = model_data.get('thresholds', {})
        self.feature_cols = model_data.get('feature_cols', [])
        
        logger.info(f"Model loaded with {len(self.feature_cols)} features")
        logger.info(f"Thresholds: {self.thresholds}")
    
    def _load_config(self):
        """Load configuration from YAML."""
        if self.config_path and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default config
            self.config = {
                'threshold_long': self.thresholds.get('long', 0.65),
                'threshold_short': self.thresholds.get('short', 0.35),
                'lookback_days': 30,
                'min_data_points': 1000
            }
    
    def fetch_latest_data(self, symbol: str, timeframe: str, lookback_days: int = None):
        """Fetch latest market data.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '15m', '1h')
            lookback_days: Days of historical data to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        if lookback_days is None:
            lookback_days = self.config.get('lookback_days', 30)
        
        logger.info(f"Fetching {symbol} {timeframe} data (last {lookback_days} days)")
        
        # Fetch data
        df = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=(datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        logger.info(f"Fetched {len(df)} data points")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction.
        
        Args:
            df: Raw OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        
        # Apply feature engineering
        df_features = self.feature_engineer.create_features(
            df,
            include_microstructure=False,  # Keep it simple for production
            include_derivatives=False
        )
        
        # Ensure we have all required features
        missing_features = set(self.feature_cols) - set(df_features.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features as zeros (not ideal but prevents crashes)
            for feat in missing_features:
                df_features[feat] = 0
        
        # Select only model features
        df_features = df_features[self.feature_cols]
        
        logger.info(f"Features prepared: {df_features.shape}")
        return df_features
    
    def predict(self, df_features: pd.DataFrame, use_calibration: bool = True) -> dict:
        """Generate predictions and trading signals.
        
        Args:
            df_features: DataFrame with features
            use_calibration: Whether to use calibrated probabilities
            
        Returns:
            Dictionary with predictions and signals
        """
        # Remove NaNs
        df_clean = df_features.dropna()
        
        if len(df_clean) == 0:
            logger.warning("No valid data after cleaning")
            return None
        
        logger.info(f"Predicting on {len(df_clean)} samples")
        
        # Get predictions
        if use_calibration and self.calibrator is not None:
            probas = self.calibrator.predict_proba(df_clean)[:, 1]
        else:
            probas = self.model.predict_proba(df_clean)[:, 1]
        
        # Generate signals using double threshold
        signals = self.backtest_engine.generate_signals_with_thresholds(
            probas,
            threshold_long=self.config['threshold_long'],
            threshold_short=self.config['threshold_short'],
            mode='double'
        )
        
        # Calculate statistics
        abstention_rate = (signals == 0).mean()
        long_rate = (signals == 1).mean()
        short_rate = (signals == -1).mean()
        
        # Get latest prediction
        latest_proba = probas[-1]
        latest_signal = signals[-1]
        
        result = {
            'timestamp': df_clean.index[-1],
            'probability': latest_proba,
            'signal': latest_signal,
            'signal_str': ['SHORT', 'NEUTRAL', 'LONG'][latest_signal + 1],
            'confidence': abs(latest_proba - 0.5) * 2,  # 0 to 1 scale
            'statistics': {
                'abstention_rate': abstention_rate,
                'long_rate': long_rate,
                'short_rate': short_rate,
                'total_predictions': len(signals)
            },
            'thresholds': {
                'long': self.config['threshold_long'],
                'short': self.config['threshold_short']
            },
            'all_probas': probas,
            'all_signals': signals
        }
        
        return result
    
    def predict_next(self, symbol: str, timeframe: str) -> dict:
        """Complete pipeline: fetch data → features → prediction.
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
            
        Returns:
            Prediction result dictionary
        """
        # Fetch latest data
        df = self.fetch_latest_data(symbol, timeframe)
        
        # Prepare features
        df_features = self.prepare_features(df)
        
        # Generate prediction
        result = self.predict(df_features)
        
        if result:
            logger.info(f"Latest signal: {result['signal_str']} "
                       f"(p={result['probability']:.3f}, "
                       f"confidence={result['confidence']:.1%})")
        
        return result
    
    def batch_predict(self, symbols: list, timeframe: str) -> pd.DataFrame:
        """Generate predictions for multiple symbols.
        
        Args:
            symbols: List of trading pairs
            timeframe: Timeframe
            
        Returns:
            DataFrame with predictions for all symbols
        """
        results = []
        
        for symbol in symbols:
            logger.info(f"Processing {symbol}...")
            try:
                pred = self.predict_next(symbol, timeframe)
                if pred:
                    results.append({
                        'symbol': symbol,
                        'timestamp': pred['timestamp'],
                        'signal': pred['signal_str'],
                        'probability': pred['probability'],
                        'confidence': pred['confidence']
                    })
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        df_results = pd.DataFrame(results)
        return df_results


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description='Crypto ML Inference')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='15m', help='Timeframe')
    parser.add_argument('--config', help='Optional config file')
    parser.add_argument('--batch', nargs='+', help='Batch predict multiple symbols')
    parser.add_argument('--output', help='Output file for predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CryptoPredictor(args.model, args.config)
    
    if args.batch:
        # Batch prediction
        df_results = predictor.batch_predict(args.batch, args.timeframe)
        print("\n📊 Batch Predictions:")
        print(df_results.to_string(index=False))
        
        if args.output:
            df_results.to_csv(args.output, index=False)
            print(f"\n💾 Saved to {args.output}")
    else:
        # Single prediction
        result = predictor.predict_next(args.symbol, args.timeframe)
        
        if result:
            print("\n" + "="*60)
            print(f"🎯 PREDICTION for {args.symbol} {args.timeframe}")
            print("="*60)
            print(f"📅 Timestamp: {result['timestamp']}")
            print(f"📊 Probability: {result['probability']:.4f}")
            print(f"🚦 Signal: {result['signal_str']}")
            print(f"💪 Confidence: {result['confidence']:.1%}")
            print(f"\n📈 Statistics:")
            print(f"  • Abstention Rate: {result['statistics']['abstention_rate']:.1%}")
            print(f"  • Long Rate: {result['statistics']['long_rate']:.1%}")
            print(f"  • Short Rate: {result['statistics']['short_rate']:.1%}")
            print(f"\n🎚️ Thresholds:")
            print(f"  • Long: {result['thresholds']['long']:.3f}")
            print(f"  • Short: {result['thresholds']['short']:.3f}")
            
            if args.output:
                # Save detailed result
                import json
                with open(args.output, 'w') as f:
                    json.dump({
                        'symbol': args.symbol,
                        'timeframe': args.timeframe,
                        'timestamp': str(result['timestamp']),
                        'signal': result['signal_str'],
                        'probability': float(result['probability']),
                        'confidence': float(result['confidence']),
                        'statistics': result['statistics'],
                        'thresholds': result['thresholds']
                    }, f, indent=2)
                print(f"\n💾 Saved to {args.output}")


if __name__ == "__main__":
    main()