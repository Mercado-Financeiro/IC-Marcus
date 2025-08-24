#!/usr/bin/env python3
"""
Enhanced LSTM training script with state-of-the-art Bayesian optimization.

Features:
- ASHA and SuccessiveHalving pruning
- Temperature scaling calibration  
- Walk-forward outer validation
- ECE and comprehensive calibration metrics
- Full determinism and reproducibility
- Expanded hyperparameter search space
- MLflow integration with comprehensive tracking

Usage:
    python src/training/train_lstm_enhanced.py [options]
    
Examples:
    # Basic training with ASHA pruning
    python src/training/train_lstm_enhanced.py --trials 50 --pruner asha
    
    # Full production optimization
    python src/training/train_lstm_enhanced.py --trials 200 --pruner asha --outer-cv 5
    
    # Fast development mode
    python src/training/train_lstm_enhanced.py --fast --trials 10
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import mlflow
import joblib
from sklearn.metrics import classification_report

# Import our enhanced modules
from src.models.lstm.optuna.optimizer_v2 import EnhancedLSTMOptuna
from src.models.lstm.optuna.config import LSTMOptunaConfig
from src.models.optuna.advanced_optimizer import AdvancedOptimizerConfig
from src.data.binance_loader import CryptoDataLoader
from src.features.engineering import FeatureEngineer
from src.utils.determinism_enhanced import set_full_determinism, assert_determinism
from src.utils.logging import log as logger


def create_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.002) -> pd.Series:
    """Create simple binary labels based on future returns."""
    future_returns = df['close'].pct_change(horizon).shift(-horizon)
    labels = (future_returns > threshold).astype(int)
    return labels.dropna()


def setup_mlflow(args):
    """Setup MLflow tracking with comprehensive tags."""
    mlflow.set_tracking_uri('artifacts/mlruns')
    
    experiment_name = 'lstm_enhanced_optimization'
    if args.fast:
        experiment_name += '_fast'
    
    mlflow.set_experiment(experiment_name)
    
    run_name = f'lstm_enhanced_{datetime.now():%Y%m%d_%H%M%S}'
    if args.fast:
        run_name += '_fast'
    
    run = mlflow.start_run(run_name=run_name)
    
    # Log comprehensive tags
    mlflow.set_tags({
        'model_type': 'lstm',
        'optimization': 'bayesian',
        'pruner': args.pruner,
        'sampler': args.sampler,
        'calibration': args.calibration,
        'outer_cv': str(args.outer_cv),
        'mode': 'fast' if args.fast else 'full',
        'deterministic': 'true',
        'git_commit': os.popen('git rev-parse HEAD').read().strip()[:8],
        'script': 'train_lstm_enhanced.py',
        'framework': 'pytorch',
        'data_source': 'binance'
    })
    
    # Log configuration parameters
    mlflow.log_params({
        'trials': args.trials,
        'timeout': args.timeout,
        'pruner_type': args.pruner,
        'sampler_type': args.sampler,
        'calibration_method': args.calibration,
        'outer_cv_splits': args.outer_cv,
        'inner_cv_splits': args.inner_cv,
        'embargo': args.embargo,
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'start_date': args.start,
        'end_date': args.end,
        'label_horizon': args.label_horizon,
        'label_threshold': args.label_threshold,
        'seed': args.seed
    })
    
    return run


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced LSTM training with Bayesian optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Basic training parameters
    parser.add_argument('--trials', type=int, default=50, 
                       help='Number of Optuna trials (default: 50)')
    parser.add_argument('--timeout', type=int, default=7200,
                       help='Optimization timeout in seconds (default: 7200)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # Optimization parameters  
    parser.add_argument('--pruner', choices=['asha', 'successive_halving', 'hyperband', 'median', 'percentile'],
                       default='asha', help='Pruning algorithm (default: asha)')
    parser.add_argument('--sampler', choices=['tpe', 'random', 'cmaes'], 
                       default='tpe', help='Sampling algorithm (default: tpe)')
    
    # Validation parameters
    parser.add_argument('--outer-cv', type=int, default=3,
                       help='Number of outer CV folds (default: 3)')
    parser.add_argument('--inner-cv', type=int, default=3,
                       help='Number of inner CV folds (default: 3)')
    parser.add_argument('--embargo', type=int, default=10,
                       help='Embargo period for time series validation (default: 10)')
    
    # Calibration parameters
    parser.add_argument('--calibration', choices=['temperature', 'vector', 'isotonic', 'platt'],
                       default='temperature', help='Calibration method (default: temperature)')
    
    # Data parameters
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default='15m',
                       help='Timeframe (default: 15m)')
    parser.add_argument('--start', type=str, default='2023-01-01',
                       help='Start date YYYY-MM-DD (default: 2023-01-01)')
    parser.add_argument('--end', type=str, default='2024-08-23',
                       help='End date YYYY-MM-DD (default: 2024-08-23)')
    parser.add_argument('--refresh', action='store_true',
                       help='Force refresh data from Binance')
    
    # Label parameters
    parser.add_argument('--label-horizon', type=int, default=5,
                       help='Label horizon in bars (default: 5)')
    parser.add_argument('--label-threshold', type=float, default=0.002,
                       help='Return threshold for labels (default: 0.002)')
    
    # Storage parameters
    parser.add_argument('--storage-url', type=str,
                       help='Optuna storage URL (default: in-memory)')
    parser.add_argument('--study-name', type=str,
                       help='Optuna study name (default: auto-generated)')
    
    # Execution modes
    parser.add_argument('--fast', action='store_true',
                       help='Fast development mode (fewer trials)')
    parser.add_argument('--no-outer-cv', action='store_true',
                       help='Disable outer cross-validation')
    parser.add_argument('--dry-run', action='store_true',
                       help='Setup only, no training')
    
    # Output parameters
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--save-model', action='store_true', default=True,
                       help='Save trained model')
    
    args = parser.parse_args()
    
    # Fast mode overrides
    if args.fast:
        args.trials = min(args.trials, 10)
        args.timeout = min(args.timeout, 1800)  # 30 minutes
        args.outer_cv = min(args.outer_cv, 2)
        args.inner_cv = min(args.inner_cv, 2)
    
    print("=" * 80)
    print("ENHANCED LSTM TRAINING WITH BAYESIAN OPTIMIZATION")  
    print("=" * 80)
    print(f"Start: {datetime.now()}")
    print(f"Mode: {'FAST' if args.fast else 'FULL'} | Trials: {args.trials} | Pruner: {args.pruner}")
    print(f"Calibration: {args.calibration} | Outer CV: {args.outer_cv} | Seed: {args.seed}")
    
    if args.dry_run:
        print("DRY RUN MODE - Setup only, no training")
    
    # 1. Set deterministic environment
    print("\n1. Setting up deterministic environment...")
    determinism_results = set_full_determinism(seed=args.seed, verify=True)
    
    try:
        assert_determinism(determinism_results, raise_on_fail=True)
        print("   ✅ Full determinism verified")
    except RuntimeError as e:
        print(f"   ⚠️  Partial determinism: {e}")
        if not args.fast:
            response = input("Continue with partial determinism? [y/N]: ")
            if response.lower() != 'y':
                sys.exit(1)
    
    # 2. Setup MLflow
    print("\n2. Setting up MLflow tracking...")
    mlflow_run = setup_mlflow(args)
    print(f"   ✅ MLflow run: {mlflow_run.info.run_id}")
    
    # Log determinism results
    mlflow.log_metrics({
        'determinism_score': sum(1 for v in determinism_results.get('verification', {}).values() 
                               if v is True) / max(1, len(determinism_results.get('verification', {})))
    })
    
    # 3. Load data
    print("\n3. Loading market data...")
    loader = CryptoDataLoader(use_cache=not args.refresh)
    
    try:
        df = loader.fetch_ohlcv(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end
        )
        print(f"   ✅ Data loaded: {len(df):,} bars")
        
        # Log data statistics
        mlflow.log_metrics({
            'n_samples': len(df),
            'date_range_days': (df.index[-1] - df.index[0]).days,
            'data_completeness': (len(df) / ((df.index[-1] - df.index[0]).total_seconds() / 
                                           (15 * 60 if args.timeframe == '15m' else 3600)))
        })
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        mlflow.end_run(status='FAILED')
        sys.exit(1)
    
    # 4. Feature engineering
    print("\n4. Engineering features...")
    engineer = FeatureEngineer(scaler_type="minmax")
    features_df = engineer.create_all_features(df)
    print(f"   ✅ Features created: {features_df.shape[1]} features")
    
    # 5. Create labels
    print("\n5. Creating labels...")
    labels = create_labels(df, horizon=args.label_horizon, threshold=args.label_threshold)
    print(f"   ✅ Labels created: {len(labels):,} samples")
    
    # 6. Align data
    print("\n6. Aligning features and labels...")
    common_index = features_df.index.intersection(labels.index)
    X = features_df.loc[common_index]
    y = labels.loc[common_index]
    
    # Remove NaN and infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X[mask]
    y = y[mask]
    
    print(f"   ✅ Final dataset: {len(y):,} samples ({y.mean():.2%} positive)")
    
    # Log data quality metrics
    mlflow.log_metrics({
        'final_samples': len(y),
        'positive_rate': float(y.mean()),
        'feature_completeness': float(mask.mean()),
        'n_features_final': X.shape[1]
    })
    
    if len(y) < 1000:
        print("   ⚠️  Warning: Very few samples available")
        if not args.fast:
            response = input("Continue with limited data? [y/N]: ")
            if response.lower() != 'y':
                mlflow.end_run(status='FAILED')
                sys.exit(1)
    
    if args.dry_run:
        print("\n✅ DRY RUN COMPLETED - Setup successful")
        mlflow.end_run(status='FINISHED')
        return
    
    # 7. Create enhanced optimizer
    print("\n7. Creating enhanced optimizer...")
    
    # Create configuration
    lstm_config = LSTMOptunaConfig(
        seed=args.seed,
        verbose=args.verbose
    )
    
    # Create optimizer
    optimizer = EnhancedLSTMOptuna(lstm_config)
    
    # Override optimizer configuration with command line arguments
    optimizer.optuna_config.n_trials = args.trials
    optimizer.optuna_config.timeout = args.timeout
    optimizer.optuna_config.pruner_type = args.pruner
    optimizer.optuna_config.sampler_type = args.sampler
    optimizer.optuna_config.calibration_method = args.calibration
    optimizer.optuna_config.outer_cv_splits = args.outer_cv
    optimizer.optuna_config.inner_cv_splits = args.inner_cv
    optimizer.optuna_config.embargo = args.embargo
    optimizer.optuna_config.use_outer_cv = not args.no_outer_cv
    optimizer.optuna_config.storage_url = args.storage_url
    optimizer.optuna_config.study_name = args.study_name
    optimizer.optuna_config.verbose = args.verbose
    
    print(f"   ✅ Optimizer configured:")
    print(f"      - Pruner: {args.pruner}")
    print(f"      - Sampler: {args.sampler}")  
    print(f"      - Calibration: {args.calibration}")
    print(f"      - Outer CV: {args.outer_cv if not args.no_outer_cv else 'Disabled'}")
    
    # 8. Run optimization
    print("\n8. Starting Bayesian optimization...")
    print("=" * 50)
    
    try:
        study = optimizer.optimize(X, y)
        print("=" * 50)
        print("   ✅ Optimization completed successfully!")
        
        # Log optimization results
        mlflow.log_metrics({
            'best_score': study.best_value,
            'n_trials_completed': len(study.trials),
            'n_trials_pruned': len([t for t in study.trials if t.state.name == 'PRUNED']),
            'n_trials_failed': len([t for t in study.trials if t.state.name == 'FAIL']),
            'optimization_efficiency': len([t for t in study.trials if t.state.name == 'COMPLETE']) / len(study.trials)
        })
        
        # Log best parameters
        for param_name, param_value in study.best_params.items():
            mlflow.log_param(f'best_{param_name}', param_value)
            
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        mlflow.log_metric('optimization_failed', 1)
        mlflow.end_run(status='FAILED')
        raise
    
    # 9. Train final model with calibration
    print("\n9. Training final model with calibration...")
    optimizer.fit_final_model(X, y)
    
    # 10. Evaluate final model
    print("\n10. Evaluating final model...")
    
    # Get predictions
    y_pred_proba = optimizer.predict_proba(X)
    y_pred = optimizer.predict(X)
    y_pred_ev = optimizer.predict(X, use_ev_threshold=True)
    
    # Classification report
    print(f"\n📊 Classification Report (F1 threshold):")
    print(classification_report(y, y_pred, digits=4))
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'f1_score': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba[:, 1]),
        'pr_auc': average_precision_score(y, y_pred_proba[:, 1]),
        'f1_ev_threshold': f1_score(y, y_pred_ev, zero_division=0)
    }
    
    print(f"\n📈 Final Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"   {metric_name}: {metric_value:.4f}")
    
    # Log final metrics
    mlflow.log_metrics(metrics)
    
    # Get calibration metrics
    try:
        calibration_curve = optimizer.get_calibration_curve(X, y)
        mlflow.log_metrics({
            'calibration_ece': calibration_curve['ece'],
            'calibration_temperature': calibration_curve.get('temperature', 1.0)
        })
        print(f"   📊 Calibration ECE: {calibration_curve['ece']:.4f}")
        print(f"   🌡️  Temperature: {calibration_curve.get('temperature', 1.0):.4f}")
    except Exception as e:
        logger.warning(f"Failed to compute calibration curve: {e}")
    
    # 11. Save model
    if args.save_model:
        print("\n11. Saving model artifacts...")
        
        # Create artifacts directory
        model_dir = Path('artifacts/models')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = model_dir / f'lstm_enhanced_{timestamp}.pkl'
        
        joblib.dump(optimizer, model_path)
        mlflow.log_artifact(str(model_path))
        
        print(f"   ✅ Model saved: {model_path}")
        
        # Save optimization summary
        summary = optimizer.get_optimization_summary()
        summary_path = model_dir / f'lstm_enhanced_summary_{timestamp}.json'
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        mlflow.log_artifact(str(summary_path))
        print(f"   ✅ Summary saved: {summary_path}")
    
    # 12. Final summary
    print("\n" + "=" * 80)
    print("🎉 ENHANCED LSTM TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📈 Best Score: {study.best_value:.4f}")
    print(f"🔥 Final F1: {metrics['f1_score']:.4f}")
    print(f"📊 PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"🌡️  Temperature: {calibration_curve.get('temperature', 1.0):.4f}")
    print(f"⏱️  Duration: {datetime.now() - datetime.fromisoformat(mlflow_run.info.start_time.replace('Z', '+00:00').replace('+00:00', ''))}")
    print(f"🔬 MLflow Run: {mlflow_run.info.run_id}")
    print("=" * 80)
    
    # End MLflow run
    mlflow.end_run(status='FINISHED')


if __name__ == "__main__":
    main()