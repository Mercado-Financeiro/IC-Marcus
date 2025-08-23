#!/usr/bin/env python3
"""
Execute full optimization pipeline with 100+ trials for production.
This is the production-ready version with extensive hyperparameter search.
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Suppress warnings
warnings.filterwarnings('ignore')

# Set deterministic environment
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, brier_score_loss
import mlflow

# Import our modules  
from src.data.binance_loader import BinanceDataLoader
from src.data.splits import PurgedKFold
from src.features.engineering import FeatureEngineer
from src.models.xgb_optuna import XGBoostOptuna
from src.models.lstm_optuna import LSTMOptuna
from src.backtest.engine import BacktestEngine

# Set random seeds
SEED = 42
np.random.seed(SEED)

# Configure MLflow
mlflow.set_tracking_uri("artifacts/mlruns")
mlflow.set_experiment("crypto_ml_production")

def run_xgboost_optimization():
    """Execute XGBoost optimization with 100+ trials."""
    
    print("=" * 80)
    print("XGBOOST FULL OPTIMIZATION - PRODUCTION")
    print("=" * 80)
    print(f"Start: {datetime.now()}\n")
    
    # Configuration
    symbol = "BTCUSDT"
    timeframe = "15m"
    start_date = "2020-01-01"  # 4 years of data
    end_date = "2024-12-31"
    test_size = 0.2
    
    # 1. Load Data
    print("📊 Loading 4 years of data...")
    loader = BinanceDataLoader()
    df = loader.fetch_ohlcv(symbol, timeframe, start_date, end_date)
    df = loader.validate_data(df)
    print(f"✅ Loaded {len(df)} bars (4 years)\n")
    
    # 2. Feature Engineering
    print("🔧 Creating comprehensive features...")
    feature_eng = FeatureEngineer(lookback_periods=[5, 10, 20, 50, 100, 200])
    df = feature_eng.create_price_features(df)
    df = feature_eng.create_technical_indicators(df)
    df = feature_eng.create_microstructure_features(df)
    print(f"✅ Created {df.shape[1]} features\n")
    
    # 3. Labeling Direcional Simples (Subir/Descer)
    print("🏷️ Creating directional labels (up/down)...")
    
    # Configuração de labeling direcional
    horizon_minutes = 15  # Horizonte de predição
    min_return_threshold = 0.0  # Threshold mínimo
    
    # Calcular retorno futuro
    future_returns = df['returns'].shift(-1)  # 1 barra à frente
    
    # Criar labels binários (subir=1, descer=0)
    df['label'] = (future_returns > min_return_threshold).astype(int)
    
    # Sample weights simples (sem volatilidade por enquanto)
    sample_weights = None
    
    # Análise da distribuição
    label_distribution = df['label'].value_counts().to_dict()
    total_samples = len(df)
    up_count = label_distribution.get(1, 0)
    down_count = label_distribution.get(0, 0)
    
    print(f"📊 Label Distribution:")
    print(f"  • Up (1): {up_count} ({up_count/total_samples*100:.1f}%)")
    print(f"  • Down (0): {down_count} ({down_count/total_samples*100:.1f}%)")
    print(f"  • Total: {total_samples} samples")
    print(f"✅ Labels created\n")
    
    # 4. Prepare ML Data
    print("📋 Preparing dataset...")
    feature_cols = [col for col in df.columns 
                   if col not in ['label', 'open', 'high', 'low', 'close', 
                                 'volume', 'returns', 'future_return']]
    X = df[feature_cols].dropna()
    y = df['label'].dropna()
    
    # Align
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    # Also align sample weights
    if sample_weights is not None:
        sample_weights = sample_weights.loc[common_idx]
    
    print(f"✅ Shape: X={X.shape}, y={y.shape}\n")
    
    # 5. Split
    print("✂️ Temporal split (80/20)...")
    test_n = int(len(X) * test_size)
    X_train, X_test = X.iloc[:-test_n], X.iloc[-test_n:]
    y_train, y_test = y.iloc[:-test_n], y.iloc[-test_n:]
    
    if sample_weights is not None:
        weights_train = sample_weights.iloc[:-test_n]
    else:
        weights_train = None
    
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"✅ Train period: {X_train.index[0]} to {X_train.index[-1]}")
    print(f"✅ Test period: {X_test.index[0]} to {X_test.index[-1]}\n")
    
    # 6. Verify no leakage
    assert X_train.index.max() < X_test.index.min()
    print("🔒 ✅ No temporal leakage\n")
    
    # 7. XGBoost Optimization with 100 trials
    print("🎯 Running XGBoost optimization (100 trials)...")
    print("⏳ This will take approximately 30-60 minutes...\n")
    
    start_time = time.time()
    
    xgb_opt = XGBoostOptuna(
        n_trials=100,  # PRODUCTION: 100 trials
        cv_folds=5,    # PRODUCTION: 5 folds
        embargo=20,    # PRODUCTION: 20 bars embargo
        pruner_type='hyperband',  # Best pruner for many trials
        use_mlflow=True,
        seed=SEED
    )
    
    # Run optimization with sample weights
    study, model = xgb_opt.optimize(X_train, y_train, sample_weights=weights_train)
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Optimization complete in {elapsed_time/60:.2f} minutes")
    print(f"✅ Best score: {study.best_value:.4f}")
    print(f"✅ Best params: {xgb_opt.best_params}\n")
    
    # 8. Predictions
    print("🎲 Making predictions on test set...")
    y_pred_proba = xgb_opt.predict_proba(X_test)
    y_pred_f1 = xgb_opt.predict(X_test, use_ev_threshold=False)
    y_pred_ev = xgb_opt.predict(X_test, use_ev_threshold=True)
    
    # 9. ML Metrics
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    ml_metrics = {
        'F1_threshold_f1': f1_score(y_test, y_pred_f1),
        'F1_threshold_ev': f1_score(y_test, y_pred_ev),
        'PR-AUC': auc(recall, precision),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
        'Brier': brier_score_loss(y_test, y_pred_proba),
        'Threshold_F1': xgb_opt.threshold_f1,
        'Threshold_EV': xgb_opt.threshold_ev
    }
    
    print("📈 ML Metrics:")
    for k, v in ml_metrics.items():
        print(f"   {k}: {v:.4f}")
    
    # 10. Backtest with both thresholds
    print("\n💰 Running backtest (F1 threshold)...")
    signals_f1 = pd.Series(y_pred_f1, index=X_test.index)
    bt_df = df.loc[X_test.index]
    
    bt = BacktestEngine(initial_capital=100000, fee_bps=5, slippage_bps=5)
    results_f1 = bt.run_backtest(bt_df, signals_f1)
    metrics_f1 = bt.calculate_metrics(results_f1)
    
    print("📊 Trading Metrics (F1 threshold):")
    for k in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
        if k in metrics_f1:
            print(f"   {k}: {metrics_f1[k]:.4f}")
    
    print("\n💰 Running backtest (EV threshold)...")
    signals_ev = pd.Series(y_pred_ev, index=X_test.index)
    results_ev = bt.run_backtest(bt_df, signals_ev)
    metrics_ev = bt.calculate_metrics(results_ev)
    
    print("📊 Trading Metrics (EV threshold):")
    for k in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
        if k in metrics_ev:
            print(f"   {k}: {metrics_ev[k]:.4f}")
    
    # 11. Save best model
    print("\n💾 Saving best model...")
    import pickle
    model_path = "artifacts/models/xgboost_production_100trials.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': xgb_opt.best_model,
            'calibrator': xgb_opt.calibrator,
            'threshold_f1': xgb_opt.threshold_f1,
            'threshold_ev': xgb_opt.threshold_ev,
            'best_params': xgb_opt.best_params,
            'feature_names': list(X_train.columns)
        }, f)
    print(f"✅ Model saved to {model_path}")
    
    # 12. Log to MLflow
    with mlflow.start_run(run_name="xgboost_production_100trials"):
        # Log parameters
        mlflow.log_params(xgb_opt.best_params)
        mlflow.log_param("n_trials", 100)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("embargo", 20)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        # Log metrics
        for k, v in ml_metrics.items():
            mlflow.log_metric(f"ml_{k}", v)
        for k, v in metrics_f1.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"trading_f1_{k}", v)
        for k, v in metrics_ev.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"trading_ev_{k}", v)
        
        # Log model
        mlflow.sklearn.log_model(xgb_opt.calibrator, "model")
        
        # Log artifacts
        mlflow.log_artifact(model_path)
        
        run_id = mlflow.active_run().info.run_id
        print(f"\n✅ Results logged to MLflow run: {run_id}")
    
    print("\n" + "=" * 80)
    
    # Check against targets
    ml_ok = ml_metrics['F1_threshold_f1'] > 0.6 and ml_metrics['Brier'] < 0.25
    trade_ok = metrics_f1.get('sharpe_ratio', 0) > 1.0 or metrics_ev.get('sharpe_ratio', 0) > 1.0
    
    if ml_ok and trade_ok:
        print("✅ PRODUCTION READY - All targets achieved!")
    else:
        print("⚠️ Some targets not met - consider further optimization")
    
    print(f"End: {datetime.now()}")
    print("=" * 80)
    
    return {
        'ml_metrics': ml_metrics,
        'trading_metrics_f1': metrics_f1,
        'trading_metrics_ev': metrics_ev,
        'model_path': model_path,
        'run_id': run_id
    }

def run_lstm_optimization():
    """Execute LSTM optimization with 50 trials."""
    
    print("\n" + "=" * 80)
    print("LSTM FULL OPTIMIZATION - PRODUCTION")
    print("=" * 80)
    print(f"Start: {datetime.now()}\n")
    
    # Similar to XGBoost but with LSTM-specific parameters
    # Using fewer trials (50) due to computational cost
    
    # Configuration
    symbol = "BTCUSDT"
    timeframe = "15m"
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    test_size = 0.2
    
    # 1. Load Data
    print("📊 Loading data for LSTM...")
    loader = BinanceDataLoader()
    df = loader.fetch_ohlcv(symbol, timeframe, start_date, end_date)
    df = loader.validate_data(df)
    print(f"✅ Loaded {len(df)} bars\n")
    
    # 2. Simplified features for LSTM
    print("🔧 Creating LSTM-optimized features...")
    # LSTM works better with fewer, normalized features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Add some technical indicators
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    from ta.volatility import BollingerBands, AverageTrueRange
    
    df['rsi'] = RSIIndicator(df['close']).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    bb = BollingerBands(df['close'])
    df['bb_width'] = bb.bollinger_wband()
    
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    # Simple binary labels
    df['future_return'] = df['returns'].shift(-1)
    df['label'] = (df['future_return'] > 0).astype(int)
    
    df = df.dropna()
    print(f"✅ Created LSTM features\n")
    
    # 3. Prepare data
    feature_cols = ['returns', 'log_returns', 'volume_ratio', 'rsi', 
                   'macd', 'macd_signal', 'macd_diff', 'bb_width', 'atr']
    
    X = df[feature_cols]
    y = df['label']
    
    # 4. Split
    test_n = int(len(X) * test_size)
    X_train, X_test = X.iloc[:-test_n], X.iloc[-test_n:]
    y_train, y_test = y.iloc[:-test_n], y.iloc[-test_n:]
    
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}\n")
    
    # 5. LSTM Optimization
    print("🎯 Running LSTM optimization (50 trials)...")
    print("⏳ This will take approximately 60-120 minutes...\n")
    
    start_time = time.time()
    
    lstm_opt = LSTMOptuna(
        n_trials=50,  # PRODUCTION: 50 trials for LSTM
        cv_folds=3,   # Fewer folds due to computational cost
        embargo=20,
        pruner_type='hyperband',
        early_stopping_patience=10,
        seed=SEED
    )
    
    # Run optimization
    study = lstm_opt.optimize(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Optimization complete in {elapsed_time/60:.2f} minutes")
    print(f"✅ Best score: {study.best_value:.4f}")
    print(f"✅ Best params: {lstm_opt.best_params}\n")
    
    # 6. Train final model
    print("🚀 Training final LSTM model...")
    lstm_opt.fit_final_model(X_train, y_train)
    
    # 7. Predictions
    y_pred_proba = lstm_opt.predict_proba(X_test)
    y_pred = lstm_opt.predict(X_test)
    
    # 8. Metrics
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    ml_metrics = {
        'F1': f1_score(y_test, y_pred),
        'PR-AUC': auc(recall, precision),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
        'Brier': brier_score_loss(y_test, y_pred_proba),
        'Threshold_F1': lstm_opt.threshold_f1,
        'Threshold_EV': lstm_opt.threshold_ev
    }
    
    print("📈 ML Metrics (LSTM):")
    for k, v in ml_metrics.items():
        print(f"   {k}: {v:.4f}")
    
    # 9. Backtest
    print("\n💰 Running backtest...")
    signals = pd.Series(y_pred, index=X_test.index)
    bt_df = df.loc[X_test.index]
    
    bt = BacktestEngine(initial_capital=100000, fee_bps=5, slippage_bps=5)
    results = bt.run_backtest(bt_df, signals)
    metrics = bt.calculate_metrics(results)
    
    print("📊 Trading Metrics (LSTM):")
    for k in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
        if k in metrics:
            print(f"   {k}: {metrics[k]:.4f}")
    
    # 10. Save model
    print("\n💾 Saving LSTM model...")
    model_path = "artifacts/models/lstm_production_50trials.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': lstm_opt.best_model,
            'calibrator': lstm_opt.calibrator,
            'threshold_f1': lstm_opt.threshold_f1,
            'threshold_ev': lstm_opt.threshold_ev,
            'best_params': lstm_opt.best_params,
            'feature_names': feature_cols,
            'scaler': lstm_opt.scaler
        }, f)
    print(f"✅ Model saved to {model_path}")
    
    print("\n" + "=" * 80)
    print(f"End: {datetime.now()}")
    print("=" * 80)
    
    return {
        'ml_metrics': ml_metrics,
        'trading_metrics': metrics,
        'model_path': model_path
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run full optimization pipeline')
    parser.add_argument('--model', choices=['xgboost', 'lstm', 'both'], 
                       default='xgboost', help='Which model to optimize')
    args = parser.parse_args()
    
    try:
        if args.model in ['xgboost', 'both']:
            print("\n🚀 Starting XGBoost optimization...")
            xgb_results = run_xgboost_optimization()
            print("\n✅ XGBoost optimization complete!")
            
        if args.model in ['lstm', 'both']:
            print("\n🚀 Starting LSTM optimization...")
            lstm_results = run_lstm_optimization()
            print("\n✅ LSTM optimization complete!")
            
        print("\n🎉 FULL OPTIMIZATION PIPELINE COMPLETE!")
        print("📊 Check MLflow UI for detailed results: mlflow ui")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
