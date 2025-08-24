"""
Performance validation script with gates that block PRs if performance regresses.
No poetry, just hard numbers.
"""
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any

from src.models.xgb import XGBoostModel
from src.validation.walkforward import validate_model_temporal
from src.metrics.core import calculate_comprehensive_metrics, find_optimal_threshold_by_metric
from src.data.loader import EfficientDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data(n_samples=5000, n_features=30, seed=42) -> tuple:
    """Create realistic test dataset for validation."""
    np.random.seed(seed)
    
    # Create features with different characteristics
    features = {}
    
    # Price-based features (trending)
    for i in range(5):
        trend = np.cumsum(np.random.randn(n_samples) * 0.01)
        features[f'price_feature_{i}'] = trend + np.random.randn(n_samples) * 0.1
    
    # Technical indicators (mean-reverting)
    for i in range(10):
        features[f'tech_feature_{i}'] = np.random.randn(n_samples) * 0.5
    
    # Volume features (log-normal)
    for i in range(5):
        features[f'volume_feature_{i}'] = np.random.lognormal(0, 1, n_samples)
    
    # Random features
    for i in range(10):
        features[f'random_feature_{i}'] = np.random.randn(n_samples)
    
    X = pd.DataFrame(features).astype('float32')
    
    # Create target with realistic signal-to-noise ratio
    signal = (X['price_feature_0'] + X['tech_feature_0'] * 0.3 + 
             X['volume_feature_0'] * 0.1)
    noise = np.random.randn(n_samples) * 2.0
    y = ((signal + noise) > np.percentile(signal + noise, 60)).astype(int)
    y = pd.Series(y, name='target')
    
    # Create synthetic returns for EV calculation
    returns = np.random.randn(n_samples) * 0.02 + y * 0.005  # Positive signal
    
    return X, y, returns


def validate_single_model(model_name: str, 
                         config: Dict[str, Any],
                         X: pd.DataFrame,
                         y: pd.Series,
                         returns: np.ndarray,
                         costs: Dict[str, float]) -> Dict[str, float]:
    """Validate a single model and return comprehensive metrics."""
    logger.info(f"Validating {model_name} model...")
    
    if model_name == 'xgboost':
        model = XGBoostModel(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Run temporal cross-validation
    fold_scores, fold_predictions, fold_labels = validate_model_temporal(
        model, X, y, cv_method='purged_kfold', n_splits=5,
        embargo_pct=0.02, purge_pct=0.01
    )
    
    # Combine all folds for overall metrics
    all_predictions = np.concatenate(fold_predictions)
    all_labels = np.concatenate(fold_labels)
    
    # Find optimal threshold
    threshold, _ = find_optimal_threshold_by_metric(
        all_labels, all_predictions, metric='ev', costs=costs
    )
    
    # Calculate binary predictions
    binary_predictions = (all_predictions >= threshold).astype(int)
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(
        all_labels, binary_predictions, all_predictions, costs
    )
    
    # Add cross-validation specific metrics
    metrics.update({
        'cv_mean_score': float(np.mean(fold_scores)),
        'cv_std_score': float(np.std(fold_scores)),
        'optimal_threshold': float(threshold),
        'n_folds': len(fold_scores)
    })
    
    logger.info(f"{model_name} validation complete:")
    logger.info(f"  EV: {metrics['ev']:.4f}")
    logger.info(f"  MCC: {metrics['mcc']:.4f}")
    logger.info(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
    
    return metrics


def load_baseline_metrics(baseline_path: str) -> Dict[str, float]:
    """Load baseline metrics from previous successful runs."""
    baseline_path = Path(baseline_path)
    
    if not baseline_path.exists():
        logger.warning(f"Baseline file not found: {baseline_path}")
        return {}
    
    try:
        with open(baseline_path, 'r') as f:
            baselines = json.load(f)
        logger.info(f"Loaded baseline metrics from {baseline_path}")
        return baselines
    except Exception as e:
        logger.error(f"Failed to load baselines: {e}")
        return {}


def compare_with_baseline(current_metrics: Dict[str, float],
                         baseline_metrics: Dict[str, float],
                         model_name: str) -> Dict[str, Any]:
    """Compare current metrics with baseline and determine if regression occurred."""
    
    # Key metrics to check for regression
    key_metrics = ['ev', 'mcc', 'auc_pr']
    
    # Acceptable degradation thresholds (relative)
    thresholds = {
        'ev': 0.05,      # 5% degradation allowed
        'mcc': 0.10,     # 10% degradation allowed
        'auc_pr': 0.05   # 5% degradation allowed
    }
    
    comparison = {
        'model': model_name,
        'passed': True,
        'details': {}
    }
    
    for metric in key_metrics:
        baseline_key = f"{model_name}_{metric}"
        
        if baseline_key not in baseline_metrics:
            logger.warning(f"No baseline for {baseline_key}")
            continue
        
        current_val = current_metrics.get(metric, 0)
        baseline_val = baseline_metrics[baseline_key]
        
        # Calculate relative change
        if baseline_val != 0:
            relative_change = (current_val - baseline_val) / abs(baseline_val)
        else:
            relative_change = 0
        
        # Check if degradation exceeds threshold
        degradation = -relative_change  # Positive degradation = worse performance
        threshold = thresholds[metric]
        
        passed = degradation <= threshold
        
        comparison['details'][metric] = {
            'current': float(current_val),
            'baseline': float(baseline_val),
            'change_pct': float(relative_change * 100),
            'degradation_pct': float(degradation * 100),
            'threshold_pct': float(threshold * 100),
            'passed': passed
        }
        
        if not passed:
            comparison['passed'] = False
            logger.error(f"{model_name} {metric}: {degradation*100:.2f}% degradation "
                        f"exceeds {threshold*100:.1f}% threshold")
        else:
            logger.info(f"{model_name} {metric}: {relative_change*100:+.2f}% change (OK)")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Validate model performance')
    parser.add_argument('--model', required=True, help='Model to validate')
    parser.add_argument('--baseline', help='Path to baseline metrics JSON')
    parser.add_argument('--output', default='artifacts/validation_results.json',
                       help='Output path for results')
    parser.add_argument('--quick', action='store_true', help='Quick validation with less data')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load baseline metrics
    baseline_metrics = {}
    if args.baseline:
        baseline_metrics = load_baseline_metrics(args.baseline)
    
    # Create test data
    n_samples = 1000 if args.quick else 5000
    X, y, returns = create_test_data(n_samples=n_samples, seed=42)
    
    # Define cost structure (realistic for crypto trading)
    costs = {
        'tp': 0.005,      # 0.5% profit per correct signal
        'fp': -0.003,     # 0.3% loss per false signal
        'tn': 0.0,        # No cost for correct no-trade
        'fn': -0.002,     # 0.2% opportunity cost for missed signal
        'fee_bps': 5,     # 5 bps transaction fee
        'slippage_bps': 3 # 3 bps slippage
    }
    
    # Model configuration
    if args.model == 'xgboost':
        config = {
            'seed': 42,
            'n_estimators': 50 if args.quick else 200,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 10
        }
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Validate model
    metrics = validate_single_model(args.model, config, X, y, returns, costs)
    
    # Compare with baseline
    comparison = compare_with_baseline(metrics, baseline_metrics, args.model)
    
    # Prepare results
    results = {
        'model': args.model,
        'metrics': metrics,
        'comparison': comparison,
        'data_info': {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'target_rate': float(y.mean())
        },
        'config': config,
        'costs': costs
    }
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Validation results saved to {output_path}")
    
    # Exit with error code if validation failed
    if not comparison['passed']:
        logger.error("VALIDATION FAILED - Performance regression detected")
        exit(1)
    else:
        logger.info("VALIDATION PASSED - No performance regression")
        exit(0)


if __name__ == "__main__":
    main()