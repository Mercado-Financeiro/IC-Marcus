#!/usr/bin/env python3
"""
Enhanced XGBoost training script with state-of-the-art Bayesian optimization.

Features:
- ASHA and HyperBand pruning with XGBoostPruningCallback
- Tree-specific calibration (Platt/Isotonic/Beta automatic selection)
- Walk-forward outer validation
- ECE and comprehensive calibration metrics
- Full determinism including GPU support
- Expanded hyperparameter search space
- MLflow integration with comprehensive tracking

Usage:
    python src/training/train_xgb_enhanced.py [options]
    
Examples:
    # Basic training with ASHA pruning
    python src/training/train_xgb_enhanced.py --trials 100 --pruner asha
    
    # Full production optimization with GPU
    python src/training/train_xgb_enhanced.py --trials 300 --tree-method gpu_hist --outer-cv 5
    
    # Fast development mode
    python src/training/train_xgb_enhanced.py --fast --trials 20
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
from sklearn.metrics import classification_report, average_precision_score

# Import our enhanced modules
from src.models.xgb.optuna.optimizer_enhanced import EnhancedXGBoostOptuna, XGBoostOptunaConfig
from src.data.binance_loader import CryptoDataLoader
from src.data.quality_pipeline import DataQualityPipeline
from src.data.splits import temporal_train_test_split
from src.features.engineering import FeatureEngineer
from src.features.filters import FeatureFilter
from src.utils.determinism_enhanced import set_full_determinism, assert_determinism
from src.utils.logging import log as logger


def get_git_commit_safe() -> str:
    """Safely get git commit hash without command injection risks."""
    try:
        import subprocess
        # Security: Use subprocess with explicit args (no shell=True)
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,  # Prevent hanging
            check=True  # Raise on non-zero exit
        )
        return result.stdout.strip()[:8]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        # Fallback: return a safe default if git is not available or fails
        return "unknown"


def create_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.002) -> pd.Series:
    """Create simple binary labels based on future returns."""
    future_returns = df['close'].pct_change(horizon).shift(-horizon)
    labels = (future_returns > threshold).astype(int)
    return labels.dropna()


def setup_mlflow(args):
    """Setup MLflow tracking with comprehensive tags."""
    mlflow.set_tracking_uri('artifacts/mlruns')
    
    experiment_name = 'xgboost_enhanced_optimization'
    if args.fast:
        experiment_name += '_fast'
    
    mlflow.set_experiment(experiment_name)
    
    run_name = f'xgb_enhanced_{datetime.now():%Y%m%d_%H%M%S}'
    if args.fast:
        run_name += '_fast'
    
    run = mlflow.start_run(run_name=run_name)
    
    # Log comprehensive tags
    mlflow.set_tags({
        'model_type': 'xgboost',
        'optimization': 'bayesian',
        'pruner': args.pruner,
        'sampler': args.sampler,
        'calibration': args.calibration,
        'tree_method': args.tree_method,
        'outer_cv': str(args.outer_cv),
        'mode': 'fast' if args.fast else 'full',
        'deterministic': 'true',
        'git_commit': get_git_commit_safe(),
        'script': 'train_xgb_enhanced.py',
        'framework': 'xgboost',
        'data_source': 'binance'
    })
    
    # Log configuration parameters
    mlflow.log_params({
        'trials': args.trials,
        'timeout': args.timeout,
        'pruner_type': args.pruner,
        'sampler_type': args.sampler,
        'calibration_method': args.calibration,
        'tree_method': args.tree_method,
        'device': args.device,
        'outer_cv_splits': args.outer_cv,
        'inner_cv_splits': args.inner_cv,
        'embargo': args.embargo,
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'start_date': args.start,
        'end_date': args.end,
        'label_horizon': args.label_horizon,
        'label_threshold': args.label_threshold,
        'seed': args.seed,
        'early_stopping_rounds': args.early_stopping
    })
    
    return run


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced XGBoost training with Bayesian optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Basic training parameters
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of Optuna trials (default: 100)')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Optimization timeout in seconds (default: 3600)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # XGBoost specific parameters
    parser.add_argument('--tree-method', choices=['hist', 'gpu_hist', 'exact'],
                       default='hist', help='Tree construction algorithm (default: hist)')
    parser.add_argument('--device', choices=['cpu', 'gpu', 'cuda'], default='cpu',
                       help='Device to use (default: cpu)')
    parser.add_argument('--early-stopping', type=int, default=100,
                       help='Early stopping rounds (default: 100)')
    
    # Optimization parameters
    parser.add_argument('--pruner', choices=['asha', 'hyperband', 'successive_halving', 'median', 'percentile'],
                       default='asha', help='Pruning algorithm (default: asha)')
    parser.add_argument('--sampler', choices=['tpe', 'random', 'cmaes'],
                       default='tpe', help='Sampling algorithm (default: tpe)')
    
    # Validation parameters
    parser.add_argument('--outer-cv', type=int, default=3,
                       help='Number of outer CV folds (default: 3)')
    parser.add_argument('--inner-cv', type=int, default=5,
                       help='Number of inner CV folds (default: 5)')
    parser.add_argument('--embargo', type=int, default=40,
                       help='Embargo period for time series validation (default: 40 bars = 10 hours for 15m)')
    
    # Calibration parameters
    parser.add_argument('--calibration', choices=['auto', 'platt', 'isotonic', 'beta'],
                       default='auto', help='Calibration method (default: auto)')
    parser.add_argument('--calibration-metric', choices=['brier', 'ece'],
                       default='brier', help='Metric for calibration selection (default: brier)')
    
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
    
    # Data quality parameters
    parser.add_argument('--enable-quality-pipeline', action='store_true', default=True,
                       help='Enable data quality pipeline with zombie removal (default: True)')
    parser.add_argument('--disable-quality-pipeline', dest='enable_quality_pipeline', 
                       action='store_false',
                       help='Disable data quality pipeline')
    parser.add_argument('--aggressive-filtering', action='store_true', default=True,
                       help='Use aggressive feature filtering (default: True)')
    parser.add_argument('--max-features', type=int, default=None,
                       help='Maximum number of features after filtering (default: no limit)')
    parser.add_argument('--variance-threshold', type=float, default=0.02,
                       help='Minimum variance threshold for features (default: 0.02)')
    parser.add_argument('--correlation-threshold', type=float, default=0.90,
                       help='Maximum correlation between features (default: 0.90)')
    
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
        args.trials = min(args.trials, 20)
        args.timeout = min(args.timeout, 1800)  # 30 minutes
        args.outer_cv = min(args.outer_cv, 2)
        args.inner_cv = min(args.inner_cv, 3)
        args.early_stopping = min(args.early_stopping, 50)
    
    print("=" * 80)
    print("ENHANCED XGBOOST TRAINING WITH BAYESIAN OPTIMIZATION")
    print("=" * 80)
    print(f"Start: {datetime.now()}")
    print(f"Mode: {'FAST' if args.fast else 'FULL'} | Trials: {args.trials} | Pruner: {args.pruner}")
    print(f"Tree method: {args.tree_method} | Device: {args.device}")
    print(f"Calibration: {args.calibration} | Outer CV: {args.outer_cv} | Seed: {args.seed}")
    
    if args.dry_run:
        print("DRY RUN MODE - Setup only, no training")
    
    # 1. Set deterministic environment
    print("\n1. Setting up deterministic environment...")
    determinism_results = set_full_determinism(seed=args.seed, verify=True)
    
    try:
        assert_determinism(determinism_results, raise_on_fail=True)
        print("   [OK] Full determinism verified")
    except RuntimeError as e:
        print(f"   [WARNING]  Partial determinism: {e}")
        if not args.fast:
            response = input("Continue with partial determinism? [y/N]: ")
            if response.lower() != 'y':
                sys.exit(1)
    
    # 2. Setup MLflow
    print("\n2. Setting up MLflow tracking...")
    mlflow_run = setup_mlflow(args)
    print(f"   [OK] MLflow run: {mlflow_run.info.run_id}")
    
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
        print(f"   [OK] Data loaded: {len(df):,} bars")
        
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
    print(f"   [OK] Features created: {features_df.shape[1]} features")
    
    # 5. Create labels
    print("\n5. Creating labels...")
    labels = create_labels(df, horizon=args.label_horizon, threshold=args.label_threshold)
    print(f"   [OK] Labels created: {len(labels):,} samples")
    
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
    
    print(f"   [OK] Initial dataset: {len(y):,} samples ({y.mean():.2%} positive)")
    print(f"       Initial features: {X.shape[1]} features")
    
    # 6a. Apply Data Quality Pipeline if enabled
    if args.enable_quality_pipeline:
        print("\n6a. Applying Data Quality Pipeline...")
        print(f"    - Embargo: {args.embargo} bars (~{args.embargo * 0.25:.1f} hours for 15m)")
        print(f"    - Variance threshold: {args.variance_threshold}")
        print(f"    - Correlation threshold: {args.correlation_threshold}")
        
        # Create quality pipeline
        quality_pipeline = DataQualityPipeline(
            symbol=args.symbol,
            timeframe=args.timeframe,
            embargo_bars=args.embargo,
            train_ratio=0.6,
            val_ratio=0.2,
            variance_threshold=args.variance_threshold,
            correlation_threshold=args.correlation_threshold,
            max_features=args.max_features,
            strict_mode=False,
            save_reports=True
        )
        
        try:
            # Process through quality pipeline
            pipeline_result = quality_pipeline.process(
                df=df,
                features=X,
                target=y
            )
            
            # Extract cleaned data
            X_train = pipeline_result['X_train']
            X_val = pipeline_result['X_val']
            X_test = pipeline_result['X_test']
            y_train = pipeline_result['y_train']
            y_val = pipeline_result['y_val']
            y_test = pipeline_result['y_test']
            
            # Combine for optimizer (it will do its own splits)
            X = pd.concat([X_train, X_val, X_test])
            y = pd.concat([y_train, y_val, y_test])
            
            # Log pipeline metrics
            metrics = pipeline_result['metrics']
            print(f"   [OK] Quality pipeline completed:")
            print(f"       - Features reduced: {metrics['original_features']} → {metrics['filtered_features']} ({metrics['feature_reduction_pct']:.1f}% reduction)")
            print(f"       - Zombie features removed: {metrics['zombie_features_removed']}")
            print(f"       - Validation pass rate: {metrics['validation_pass_rate']:.1f}%")
            
            mlflow.log_metrics({
                'features_original': metrics['original_features'],
                'features_filtered': metrics['filtered_features'],
                'feature_reduction_pct': metrics['feature_reduction_pct'],
                'zombie_features_removed': metrics['zombie_features_removed'],
                'embargo_bars': metrics['embargo_bars'],
                'validation_pass_rate': metrics['validation_pass_rate']
            })
            
        except Exception as e:
            print(f"   [WARNING] Quality pipeline failed: {e}")
            print("   Falling back to simple filtering...")
            
            # Fallback to simple feature filtering
            feature_filter = FeatureFilter(
                variance_threshold=args.variance_threshold,
                correlation_threshold=args.correlation_threshold,
                remove_zombie_features=True
            )
            feature_filter.fit(X, y)
            X = feature_filter.transform(X)
            
            print(f"   [OK] Simple filtering applied: {X.shape[1]} features retained")
    
    else:
        print("\n6a. Data quality pipeline disabled - using raw features")
    
    print(f"\n   [OK] Final dataset: {len(y):,} samples, {X.shape[1]} features")
    print(f"       Class balance: {y.mean():.2%} positive")
    
    # Log final data quality metrics
    mlflow.log_metrics({
        'final_samples': len(y),
        'positive_rate': float(y.mean()),
        'n_features_final': X.shape[1],
        'quality_pipeline_enabled': int(args.enable_quality_pipeline)
    })
    
    if len(y) < 1000:
        print("   [WARNING]  Warning: Very few samples available")
        if not args.fast:
            response = input("Continue with limited data? [y/N]: ")
            if response.lower() != 'y':
                mlflow.end_run(status='FAILED')
                sys.exit(1)
    
    if args.dry_run:
        print("\n[OK] DRY RUN COMPLETED - Setup successful")
        mlflow.end_run(status='FINISHED')
        return
    
    # 7. Create enhanced optimizer
    print("\n7. Creating enhanced XGBoost optimizer...")
    
    # Create configuration
    config = XGBoostOptunaConfig(
        n_trials=args.trials,
        timeout=args.timeout,
        seed=args.seed,
        tree_method=args.tree_method,
        device=args.device,
        sampler_type=args.sampler,
        pruner_type=args.pruner,
        calibration_method=args.calibration,
        calibration_selection_metric=args.calibration_metric,
        outer_cv_splits=args.outer_cv,
        inner_cv_splits=args.inner_cv,
        embargo=args.embargo,
        use_outer_cv=not args.no_outer_cv,
        storage_url=args.storage_url,
        study_name=args.study_name,
        early_stopping_rounds=args.early_stopping,
        verbose=args.verbose,
        deterministic=True,
        n_jobs=1  # For determinism
    )
    
    optimizer = EnhancedXGBoostOptuna(config)
    
    print(f"   [OK] Optimizer configured:")
    print(f"      - Pruner: {args.pruner} (with XGBoostPruningCallback)")
    print(f"      - Sampler: {args.sampler}")
    print(f"      - Tree method: {args.tree_method}")
    print(f"      - Calibration: {args.calibration}")
    print(f"      - Outer CV: {args.outer_cv if not args.no_outer_cv else 'Disabled'}")
    
    # 8. Run optimization
    print("\n8. Starting Bayesian optimization...")
    print("=" * 50)
    
    try:
        study = optimizer.optimize(X, y)
        print("=" * 50)
        print("   [OK] Optimization completed successfully!")
        
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
    print(f"\n[REPORT] Classification Report (F1 threshold):")
    print(classification_report(y, y_pred, digits=4))
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'f1_score': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba[:, 1]),
        'pr_auc': average_precision_score(y, y_pred_proba[:, 1]),
        'f1_ev_threshold': f1_score(y, y_pred_ev, zero_division=0)
    }
    
    print(f"\n[METRICS] Final Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"   {metric_name}: {metric_value:.4f}")
    
    # Log final metrics
    mlflow.log_metrics(metrics)
    
    # Get calibration info
    try:
        cal_info = optimizer.calibrator.get_calibration_info()
        selected_method = cal_info.get('selected_method', args.calibration)
        print(f"   [REPORT] Calibration method: {selected_method}")
        
        if 'method_scores' in cal_info:
            print(f"   [TARGET] Calibration scores: {cal_info['method_scores']}")
            
    except Exception as e:
        logger.warning(f"Failed to get calibration info: {e}")
    
    # Feature importance
    try:
        feature_importance = optimizer.get_feature_importance()
        print(f"\n[FEATURES] Top 10 Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
    except Exception as e:
        logger.warning(f"Failed to get feature importance: {e}")
    
    # 11. Save model
    if args.save_model:
        print("\n11. Saving model artifacts...")
        
        # Create artifacts directory
        model_dir = Path('artifacts/models')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = model_dir / f'xgb_enhanced_{timestamp}.pkl'
        
        joblib.dump(optimizer, model_path)
        mlflow.log_artifact(str(model_path))
        
        print(f"   [OK] Model saved: {model_path}")
        
        # Save optimization summary
        summary = optimizer.get_optimization_summary()
        summary_path = model_dir / f'xgb_enhanced_summary_{timestamp}.json'
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        mlflow.log_artifact(str(summary_path))
        print(f"   [OK] Summary saved: {summary_path}")
        
        # Save feature importance
        if hasattr(optimizer, 'feature_importances_'):
            importance_path = model_dir / f'xgb_feature_importance_{timestamp}.csv'
            feature_importance = optimizer.get_feature_importance()
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(str(importance_path))
            print(f"   [OK] Feature importance saved: {importance_path}")
    
    # 12. Final summary
    print("\n" + "=" * 80)
    print("[SUCCESS] ENHANCED XGBOOST TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"[METRICS] Best Score: {study.best_value:.4f}")
    print(f"[HOT] Final PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"[TARGET] Final F1: {metrics['f1_score']:.4f}")
    print(f"[REPORT] Calibration: {selected_method}")
    print(f"[TREE] Tree method: {args.tree_method}")
    print(f"[TIME]  Duration: {datetime.now() - datetime.fromisoformat(mlflow_run.info.start_time.replace('Z', '+00:00').replace('+00:00', ''))}")
    print(f"[LAB] MLflow Run: {mlflow_run.info.run_id}")
    print("=" * 80)
    
    # End MLflow run
    mlflow.end_run(status='FINISHED')


if __name__ == "__main__":
    main()
