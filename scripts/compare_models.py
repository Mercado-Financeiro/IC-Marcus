#!/usr/bin/env python3
"""
Script to compare performance of different models (XGBoost, LSTM, Ensemble).

Usage:
    python compare_models.py --data BTCUSDT_15m
    python compare_models.py --create-ensemble
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, brier_score_loss
import mlflow

# Import modules
from src.data.binance_loader import BinanceDataLoader
from src.features.engineering import FeatureEngineer
from src.models.ensemble import VotingEnsemble, WeightedEnsemble, StackingEnsemble, EnsembleOptimizer
from src.backtest.engine import BacktestEngine

# Setup
SEED = 42
np.random.seed(SEED)
mlflow.set_tracking_uri("artifacts/mlruns")
mlflow.set_experiment("model_comparison")


class ModelComparator:
    """Compare different ML models and create ensembles."""
    
    def __init__(self):
        """Initialize comparator."""
        self.models = {}
        self.results = {}
        self.data_loader = BinanceDataLoader()
    
    def load_models(self):
        """Load all available models from artifacts."""
        
        models_dir = Path("artifacts/models")
        
        # Load XGBoost model
        xgb_path = models_dir / "xgboost_optimized.pkl"
        if xgb_path.exists():
            print(f"📂 Loading XGBoost model from {xgb_path}")
            with open(xgb_path, 'rb') as f:
                xgb_data = pickle.load(f)
                self.models['xgboost'] = {
                    'model': xgb_data.get('calibrator', xgb_data.get('model')),
                    'threshold_f1': xgb_data.get('thresholds', {}).get('f1', 0.5),
                    'threshold_ev': xgb_data.get('thresholds', {}).get('ev', 0.5),
                    'params': xgb_data.get('params', {})
                }
                print("✅ XGBoost model loaded")
        else:
            print(f"⚠️ XGBoost model not found at {xgb_path}")
        
        # Load LSTM model
        lstm_path = models_dir / "lstm_optimized.pkl"
        if lstm_path.exists():
            print(f"📂 Loading LSTM model from {lstm_path}")
            with open(lstm_path, 'rb') as f:
                lstm_data = pickle.load(f)
                self.models['lstm'] = {
                    'model': lstm_data.get('calibrator', lstm_data.get('model')),
                    'threshold_f1': lstm_data.get('thresholds', {}).get('f1', 0.5),
                    'threshold_ev': lstm_data.get('thresholds', {}).get('ev', 0.5),
                    'params': lstm_data.get('params', {}),
                    'scaler': lstm_data.get('scaler')
                }
                print("✅ LSTM model loaded")
        else:
            print(f"⚠️ LSTM model not found at {lstm_path}")
        
        # Load ensemble if exists
        ensemble_path = models_dir / "ensemble_optimized.pkl"
        if ensemble_path.exists():
            print(f"📂 Loading Ensemble model from {ensemble_path}")
            with open(ensemble_path, 'rb') as f:
                ensemble_data = pickle.load(f)
                self.models['ensemble'] = {
                    'model': ensemble_data.get('ensemble'),
                    'score': ensemble_data.get('score'),
                    'results': ensemble_data.get('results', {})
                }
                print("✅ Ensemble model loaded")
        
        print(f"\n📊 Total models loaded: {len(self.models)}")
        return self.models
    
    def prepare_test_data(self, symbol="BTCUSDT", timeframe="15m"):
        """Prepare test data for comparison."""
        
        print(f"\n📊 Loading test data for {symbol} {timeframe}...")
        
        # Load recent data for testing
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = "2024-01-01"  # Last year
        
        df = self.data_loader.fetch_ohlcv(symbol, timeframe, start_date, end_date)
        df = self.data_loader.validate_data(df)
        
        # Feature engineering
        print("🔧 Creating features...")
        feature_eng = FeatureEngineer(lookback_periods=[5, 10, 20, 50, 100])
        df = feature_eng.create_price_features(df)
        df = feature_eng.create_technical_indicators(df)
        
        # Simple labels for comparison
        df['returns'] = df['close'].pct_change()
        df['label'] = (df['returns'].shift(-1) > 0).astype(int)
        df = df.dropna()
        
        # Prepare features
        feature_cols = [c for c in df.columns 
                       if c not in ['open', 'high', 'low', 'close', 'volume', 
                                   'returns', 'label']]
        
        X = df[feature_cols]
        y = df['label']
        
        # Use last 20% as test
        test_size = int(len(X) * 0.2)
        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]
        
        print(f"✅ Test data prepared: {X_test.shape}")
        
        return X_test, y_test, df.iloc[-test_size:]
    
    def evaluate_model(self, model_name: str, model_data: dict, 
                      X_test: pd.DataFrame, y_test: pd.Series, 
                      df_test: pd.DataFrame):
        """Evaluate a single model."""
        
        print(f"\n🎯 Evaluating {model_name}...")
        
        model = model_data['model']
        
        # Handle LSTM scaler if needed
        X_eval = X_test.copy()
        if 'scaler' in model_data and model_data['scaler'] is not None:
            X_eval = pd.DataFrame(
                model_data['scaler'].transform(X_test),
                index=X_test.index,
                columns=X_test.columns
            )
        
        # Make predictions
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_eval)
                if len(y_proba.shape) > 1:
                    y_proba = y_proba[:, 1]
            else:
                y_proba = model.predict(X_eval)
            
            # Use threshold
            threshold = model_data.get('threshold_f1', 0.5)
            y_pred = (y_proba >= threshold).astype(int)
            
        except Exception as e:
            print(f"❌ Error predicting with {model_name}: {e}")
            return None
        
        # Calculate ML metrics
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        
        ml_metrics = {
            'f1_score': f1_score(y_test, y_pred),
            'pr_auc': auc(recall, precision),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'brier_score': brier_score_loss(y_test, y_proba),
            'accuracy': (y_pred == y_test).mean()
        }
        
        print(f"  F1 Score: {ml_metrics['f1_score']:.4f}")
        print(f"  PR-AUC: {ml_metrics['pr_auc']:.4f}")
        print(f"  ROC-AUC: {ml_metrics['roc_auc']:.4f}")
        
        # Run backtest
        print(f"  Running backtest...")
        signals = pd.Series(y_pred, index=X_test.index)
        
        bt = BacktestEngine(initial_capital=100000, fee_bps=5, slippage_bps=5)
        bt_results = bt.run_backtest(df_test, signals)
        bt_metrics = bt.calculate_metrics(bt_results)
        
        # Combine metrics
        results = {**ml_metrics, **bt_metrics}
        self.results[model_name] = results
        
        print(f"  Sharpe Ratio: {bt_metrics.get('sharpe_ratio', 0):.4f}")
        print(f"  Total Return: {bt_metrics.get('total_return', 0):.2%}")
        
        return results
    
    def create_ensemble(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Create and optimize ensemble from loaded models."""
        
        if len(self.models) < 2:
            print("⚠️ Need at least 2 models to create ensemble")
            return None
        
        print("\n🔄 Creating ensemble...")
        
        # Get base models (exclude existing ensemble)
        base_models = {}
        for name, data in self.models.items():
            if name != 'ensemble':
                base_models[name] = data['model']
        
        # Split test data for ensemble optimization
        val_size = int(len(X_test) * 0.5)
        X_val = X_test.iloc[:val_size]
        y_val = y_test.iloc[:val_size]
        X_test_final = X_test.iloc[val_size:]
        y_test_final = y_test.iloc[val_size:]
        
        # Optimize ensemble
        optimizer = EnsembleOptimizer(base_models)
        results = optimizer.optimize(X_val, y_val, X_test_final, y_test_final)
        
        print(f"\n✅ Best ensemble type: {max(results, key=lambda k: results[k].get('f1', 0))}")
        print(f"   Best F1 Score: {optimizer.best_score:.4f}")
        
        # Save ensemble
        save_path = "artifacts/models/ensemble_optimized.pkl"
        optimizer.save_best_ensemble(save_path)
        print(f"💾 Ensemble saved to {save_path}")
        
        return optimizer.best_ensemble
    
    def compare_all(self, symbol="BTCUSDT", timeframe="15m"):
        """Compare all models."""
        
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON")
        print("="*60)
        
        # Load models
        self.load_models()
        
        if not self.models:
            print("❌ No models found to compare")
            return
        
        # Prepare test data
        X_test, y_test, df_test = self.prepare_test_data(symbol, timeframe)
        
        # Evaluate each model
        for model_name, model_data in self.models.items():
            self.evaluate_model(model_name, model_data, X_test, y_test, df_test)
        
        # Print comparison table
        self.print_comparison_table()
        
        # Log to MLflow
        self.log_to_mlflow()
        
        return self.results
    
    def print_comparison_table(self):
        """Print formatted comparison table."""
        
        if not self.results:
            return
        
        print("\n" + "="*60)
        print("📋 COMPARISON RESULTS")
        print("="*60)
        
        # Create DataFrame for easy display
        df = pd.DataFrame(self.results).T
        
        # Select key metrics
        key_metrics = ['f1_score', 'pr_auc', 'roc_auc', 'brier_score', 
                      'sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
        
        available_metrics = [m for m in key_metrics if m in df.columns]
        
        print("\n🎯 ML Metrics:")
        ml_metrics = ['f1_score', 'pr_auc', 'roc_auc', 'brier_score']
        print(df[[m for m in ml_metrics if m in df.columns]].round(4))
        
        print("\n💰 Trading Metrics:")
        trading_metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
        print(df[[m for m in trading_metrics if m in df.columns]].round(4))
        
        # Find best model
        if 'f1_score' in df.columns:
            best_ml = df['f1_score'].idxmax()
            print(f"\n🏆 Best ML Performance: {best_ml}")
        
        if 'sharpe_ratio' in df.columns:
            best_trading = df['sharpe_ratio'].idxmax()
            print(f"🏆 Best Trading Performance: {best_trading}")
        
        print("="*60)
    
    def log_to_mlflow(self):
        """Log comparison results to MLflow."""
        
        with mlflow.start_run(run_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log each model's metrics
            for model_name, metrics in self.results.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{model_name}_{metric_name}", value)
            
            # Log comparison summary
            if self.results:
                df = pd.DataFrame(self.results).T
                
                # Save comparison table
                comparison_path = "artifacts/comparison_results.csv"
                df.to_csv(comparison_path)
                mlflow.log_artifact(comparison_path)
                
                # Log best scores
                if 'f1_score' in df.columns:
                    mlflow.log_metric("best_f1_score", df['f1_score'].max())
                    mlflow.set_tag("best_ml_model", df['f1_score'].idxmax())
                
                if 'sharpe_ratio' in df.columns:
                    mlflow.log_metric("best_sharpe_ratio", df['sharpe_ratio'].max())
                    mlflow.set_tag("best_trading_model", df['sharpe_ratio'].idxmax())
            
            print("\n✅ Results logged to MLflow")


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='Compare ML models performance')
    parser.add_argument('--data', default='BTCUSDT_15m', 
                       help='Data to use (format: SYMBOL_TIMEFRAME)')
    parser.add_argument('--create-ensemble', action='store_true',
                       help='Create optimized ensemble from models')
    
    args = parser.parse_args()
    
    # Parse data argument
    if '_' in args.data:
        symbol, timeframe = args.data.split('_')
    else:
        symbol, timeframe = 'BTCUSDT', '15m'
    
    print(f"\n🚀 Starting model comparison")
    print(f"📊 Data: {symbol} {timeframe}")
    print(f"🕐 Time: {datetime.now()}")
    
    # Create comparator
    comparator = ModelComparator()
    
    try:
        if args.create_ensemble:
            # Load models and create ensemble
            comparator.load_models()
            
            if len(comparator.models) >= 2:
                X_test, y_test, _ = comparator.prepare_test_data(symbol, timeframe)
                ensemble = comparator.create_ensemble(X_test, y_test)
                
                if ensemble:
                    print("\n✅ Ensemble created successfully!")
                    # Re-run comparison with ensemble
                    comparator.compare_all(symbol, timeframe)
            else:
                print("⚠️ Not enough models to create ensemble")
        else:
            # Just compare existing models
            results = comparator.compare_all(symbol, timeframe)
            
            if results:
                print("\n✅ Comparison complete!")
                print("📊 Check MLflow UI for detailed results: mlflow ui")
            else:
                print("\n⚠️ No results to compare")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Comparison interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())