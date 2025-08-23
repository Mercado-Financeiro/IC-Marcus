"""Multi-horizon pipeline for training and evaluation across different time horizons."""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
import joblib

# ML imports
# Use temporal validator with embargo
from src.features.validation.temporal import TemporalValidator, TemporalValidationConfig
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV

# XGBoost and Optuna
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    from optuna.integration import XGBoostPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# MLflow
try:
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Local imports
from .xgb_optuna import XGBoostOptuna
from ..features.adaptive_labeling import AdaptiveLabeler, resolve_funding_minutes
from ..features.calendar_features import Crypto24x7Features

warnings.filterwarnings('ignore')


class MultiHorizonPipeline:
    """
    Multi-horizon training and evaluation pipeline.
    
    Trains models across multiple time horizons with:
    - Adaptive volatility-based labeling
    - Optuna optimization per horizon
    - Mandatory calibration and threshold optimization
    - MLflow tracking with funding-aware features
    """
    
    def __init__(
        self,
        horizons: List[str] = None,
        symbol: str = "BTCUSDT",
        test_size: float = 0.2,
        val_size: float = 0.2,
        n_trials: int = 50,
        vol_estimator: str = 'yang_zhang',
        artifacts_path: str = "artifacts",
        use_mlflow: bool = True
    ):
        """
        Initialize multi-horizon pipeline.
        
        Args:
            horizons: List of horizons to process ['15m', '30m', '60m', '120m']
            symbol: Trading symbol for funding period resolution
            test_size: Proportion for test set
            val_size: Proportion for validation set
            n_trials: Number of Optuna trials per horizon
            vol_estimator: Volatility estimator for adaptive labeling
            artifacts_path: Path to save artifacts
            use_mlflow: Enable MLflow tracking
        """
        self.horizons = horizons or ['15m', '30m', '60m', '120m']
        self.symbol = symbol
        self.test_size = test_size
        self.val_size = val_size
        self.n_trials = n_trials
        self.vol_estimator = vol_estimator
        self.artifacts_path = artifacts_path
        self.use_mlflow = use_mlflow
        
        # Create artifacts directory
        Path(self.artifacts_path).mkdir(parents=True, exist_ok=True)
        Path(f"{self.artifacts_path}/models").mkdir(parents=True, exist_ok=True)
        Path(f"{self.artifacts_path}/reports").mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.crypto_features = Crypto24x7Features()
        
        # Resolve funding period for symbol
        self.funding_period_minutes = resolve_funding_minutes(self.symbol)
        
        print(f"✅ MultiHorizonPipeline initialized")
        print(f"  Symbol: {self.symbol} (funding: {self.funding_period_minutes}min)")
        print(f"  Horizons: {self.horizons}")
        print(f"  Trials per horizon: {self.n_trials}")
    
    def prepare_data_splits(self, df: pd.DataFrame) -> Tuple:
        """
        Create temporal data splits.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (train_idx, val_idx, test_idx)
        """
        n_samples = len(df)
        test_start = int(n_samples * (1 - self.test_size))
        val_start = int(n_samples * (1 - self.test_size - self.val_size))
        
        train_idx = slice(0, val_start)
        val_idx = slice(val_start, test_start)
        test_idx = slice(test_start, n_samples)
        
        print(f"📊 Data splits:")
        print(f"  Train: {val_start} samples ({val_start/n_samples:.1%})")
        print(f"  Val:   {test_start - val_start} samples ({self.val_size:.1%})")
        print(f"  Test:  {n_samples - test_start} samples ({self.test_size:.1%})")
        
        return train_idx, val_idx, test_idx
    
    def prepare_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features with funding-aware enhancements.
        
        Args:
            df: OHLC DataFrame
            features: Base features DataFrame
            
        Returns:
            Enhanced features DataFrame
        """
        # Add funding features
        features_enhanced = self.crypto_features.create_funding_features(
            df, features, funding_period_minutes=self.funding_period_minutes
        )
        
        print(f"✅ Features enhanced: {features_enhanced.shape[1]} total features")
        return features_enhanced
    
    def run_horizon_optimization(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        horizon: str,
        train_idx: slice,
        val_idx: slice,
        test_idx: slice
    ) -> Dict:
        """
        Run optimization for single horizon.
        
        Args:
            df: OHLC DataFrame
            features: Features DataFrame
            horizon: Target horizon
            train_idx: Training indices
            val_idx: Validation indices
            test_idx: Test indices
            
        Returns:
            Results dictionary for horizon
        """
        print(f"\n{'='*60}")
        print(f"⏱️ Processing horizon: {horizon}")
        print(f"{'='*60}")
        
        # Start MLflow run if enabled
        run_context = None
        if self.use_mlflow and MLFLOW_AVAILABLE:
            experiment_name = f"multi_horizon_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow.set_experiment(experiment_name)
            run_context = mlflow.start_run(run_name=f"horizon_{horizon}")
        
        try:
            # 1. Create adaptive labeler for this horizon
            labeler = AdaptiveLabeler(
                vol_estimator=self.vol_estimator,
                funding_period_minutes=self.funding_period_minutes
            )
            
            # Set horizon bars
            horizon_bars = labeler.horizon_map[horizon]
            labeler.horizon_bars = horizon_bars
            
            if self.use_mlflow and MLFLOW_AVAILABLE:
                mlflow.log_param("horizon", horizon)
                mlflow.log_param("horizon_bars", horizon_bars)
                mlflow.log_param("vol_estimator", self.vol_estimator)
                mlflow.log_param("funding_period", self.funding_period_minutes)
            
            # 2. Optimize k parameter for this horizon
            print(f"\n🔍 Optimizing k for horizon {horizon}...")
            optimal_k = labeler.optimize_k_for_horizon(
                df[train_idx], 
                features[train_idx],
                horizon=horizon,
                cv_splits=3,
                metric='pr_auc'
            )
            
            # 3. Create labels with optimized k
            labeler.k = optimal_k
            labels = labeler.create_labels(df)
            
            if self.use_mlflow and MLFLOW_AVAILABLE:
                mlflow.log_metric(f"optimal_k_{horizon}", optimal_k)
            
            # 4. Prepare clean datasets
            X_train = features[train_idx]
            y_train = labels[train_idx]
            X_val = features[val_idx]
            y_val = labels[val_idx]
            X_test = features[test_idx]
            y_test = labels[test_idx]
            
            # Remove NaN values
            mask_train = ~(X_train.isna().any(axis=1) | y_train.isna())
            mask_val = ~(X_val.isna().any(axis=1) | y_val.isna())
            mask_test = ~(X_test.isna().any(axis=1) | y_test.isna())
            
            X_train_clean = X_train[mask_train]
            y_train_clean = y_train[mask_train]
            X_val_clean = X_val[mask_val]
            y_val_clean = y_val[mask_val]
            X_test_clean = X_test[mask_test]
            y_test_clean = y_test[mask_test]
            
            # Convert to binary (1: up, 0: down/neutral)
            y_train_binary = (y_train_clean > 0).astype(int)
            y_val_binary = (y_val_clean > 0).astype(int)
            y_test_binary = (y_test_clean > 0).astype(int)
            
            # Log class distribution
            train_pos_pct = y_train_binary.mean()
            val_pos_pct = y_val_binary.mean()
            test_pos_pct = y_test_binary.mean()
            
            print(f"\n📈 Class distribution:")
            print(f"  Train: {train_pos_pct:.2%} positive")
            print(f"  Val:   {val_pos_pct:.2%} positive")
            print(f"  Test:  {test_pos_pct:.2%} positive")
            
            if self.use_mlflow and MLFLOW_AVAILABLE:
                mlflow.log_metric("train_positive_pct", train_pos_pct)
                mlflow.log_metric("val_positive_pct", val_pos_pct)
                mlflow.log_metric("test_positive_pct", test_pos_pct)
            
            # 5. Train XGBoost with Optuna
            print(f"\n🎯 Training XGBoost with Optuna...")
            
            xgb_optimizer = XGBoostOptuna(
                n_trials=self.n_trials,
                cv_folds=3,  # Reduced for speed
                embargo=10,
                use_mlflow=False,  # We handle MLflow logging here
                seed=42
            )
            
            # Run optimization
            study, best_model = xgb_optimizer.optimize(X_train_clean, y_train_binary)
            
            # Train final model with proper validation split
            final_model = xgb_optimizer.fit_final_model(
                X_train_clean, y_train_binary,
                X_val_clean, y_val_binary
            )
            
            # 6. Get calibrated predictions on test set
            y_test_pred_proba = xgb_optimizer.predict_proba(X_test_clean)
            
            # Apply optimized thresholds
            y_test_pred_f1 = xgb_optimizer.predict(X_test_clean, threshold_type='f1')
            y_test_pred_profit = xgb_optimizer.predict(X_test_clean, threshold_type='profit')
            
            # 7. Calculate comprehensive metrics
            test_pr_auc = average_precision_score(y_test_binary, y_test_pred_proba)
            test_f1 = f1_score(y_test_binary, y_test_pred_f1)
            test_mcc = matthews_corrcoef(y_test_binary, y_test_pred_f1)
            
            # Confusion matrix
            cm = confusion_matrix(y_test_binary, y_test_pred_f1, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print(f"\n📈 Test metrics for {horizon}:")
            print(f"  PR-AUC:      {test_pr_auc:.4f}")
            print(f"  F1 Score:    {test_f1:.4f}")
            print(f"  MCC:         {test_mcc:.4f}")
            print(f"  Precision:   {precision_val:.4f}")
            print(f"  Recall:      {recall_val:.4f}")
            print(f"  Specificity: {specificity:.4f}")
            
            # Log to MLflow
            if self.use_mlflow and MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    f"test_pr_auc_{horizon}": test_pr_auc,
                    f"test_f1_{horizon}": test_f1,
                    f"test_mcc_{horizon}": test_mcc,
                    f"test_precision_{horizon}": precision_val,
                    f"test_recall_{horizon}": recall_val,
                    f"test_specificity_{horizon}": specificity,
                    f"threshold_f1_{horizon}": xgb_optimizer.threshold_f1,
                    f"threshold_profit_{horizon}": xgb_optimizer.threshold_profit
                })
                
                # Log best parameters
                for key, value in xgb_optimizer.best_params.items():
                    mlflow.log_param(f"xgb_{key}_{horizon}", value)
            
            # 8. Feature importance analysis
            feature_importance = pd.DataFrame({
                'feature': X_train_clean.columns,
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔝 Top 10 features for {horizon}:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']:30s}: {row['importance']:.4f}")
            
            # 9. Save model artifacts
            model_path = f"{self.artifacts_path}/models/xgb_{horizon}.pkl"
            joblib.dump({
                'model': final_model,
                'calibrator': xgb_optimizer.calibrator,
                'threshold_f1': xgb_optimizer.threshold_f1,
                'threshold_profit': xgb_optimizer.threshold_profit,
                'labeler': labeler,
                'feature_names': list(X_train_clean.columns)
            }, model_path)
            
            if self.use_mlflow and MLFLOW_AVAILABLE:
                mlflow.log_artifact(model_path)
            
            # Return results
            return {
                'horizon': horizon,
                'model': final_model,
                'calibrator': xgb_optimizer.calibrator,
                'labeler': labeler,
                'optimal_k': optimal_k,
                'thresholds': {
                    'f1': xgb_optimizer.threshold_f1,
                    'profit': xgb_optimizer.threshold_profit
                },
                'metrics': {
                    'pr_auc': test_pr_auc,
                    'f1': test_f1,
                    'mcc': test_mcc,
                    'precision': precision_val,
                    'recall': recall_val,
                    'specificity': specificity
                },
                'confusion_matrix': cm,
                'feature_importance': feature_importance,
                'predictions': {
                    'proba': y_test_pred_proba,
                    'f1': y_test_pred_f1,
                    'profit': y_test_pred_profit
                },
                'labels': y_test_binary,
                'test_indices': X_test_clean.index,
                'study': study
            }
        
        finally:
            if self.use_mlflow and MLFLOW_AVAILABLE and run_context:
                mlflow.end_run()
    
    def run_pipeline(
        self, 
        df: pd.DataFrame, 
        features: pd.DataFrame
    ) -> Dict:
        """
        Run complete multi-horizon pipeline.
        
        Args:
            df: OHLC DataFrame
            features: Base features DataFrame
            
        Returns:
            Dictionary with results for all horizons
        """
        print("="*80)
        print("🚀 STARTING MULTI-HORIZON PIPELINE")
        print("="*80)
        
        # Validate inputs
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost not available")
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available")
        if self.use_mlflow and not MLFLOW_AVAILABLE:
            print("⚠️ MLflow not available - tracking disabled")
            self.use_mlflow = False
        
        # Prepare data
        train_idx, val_idx, test_idx = self.prepare_data_splits(df)
        features_enhanced = self.prepare_features(df, features)
        
        # Process each horizon
        results = {}
        for horizon in self.horizons:
            horizon_results = self.run_horizon_optimization(
                df, features_enhanced, horizon,
                train_idx, val_idx, test_idx
            )
            results[horizon] = horizon_results
        
        # Comparative analysis
        self.analyze_results(results)
        
        return results
    
    def analyze_results(self, results: Dict) -> None:
        """
        Perform comparative analysis across horizons.
        
        Args:
            results: Results dictionary from all horizons
        """
        print(f"\n{'='*80}")
        print("📊 MULTI-HORIZON COMPARATIVE ANALYSIS")
        print(f"{'='*80}")
        
        # Create comparison DataFrame
        comparison_data = {}
        for horizon, result in results.items():
            comparison_data[horizon] = {
                'PR-AUC': result['metrics']['pr_auc'],
                'F1': result['metrics']['f1'],
                'MCC': result['metrics']['mcc'],
                'Precision': result['metrics']['precision'],
                'Recall': result['metrics']['recall'],
                'Optimal_k': result['optimal_k'],
                'Threshold_F1': result['thresholds']['f1'],
                'Threshold_Profit': result['thresholds']['profit']
            }
        
        comparison_df = pd.DataFrame(comparison_data).T
        
        print("\n📈 Performance Comparison:")
        print(comparison_df.round(4))
        
        # Identify best performers
        best_pr_auc = comparison_df['PR-AUC'].idxmax()
        best_f1 = comparison_df['F1'].idxmax()
        best_mcc = comparison_df['MCC'].idxmax()
        
        print(f"\n🏆 Best performers:")
        print(f"  PR-AUC: {best_pr_auc} ({comparison_df.loc[best_pr_auc, 'PR-AUC']:.4f})")
        print(f"  F1:     {best_f1} ({comparison_df.loc[best_f1, 'F1']:.4f})")
        print(f"  MCC:    {best_mcc} ({comparison_df.loc[best_mcc, 'MCC']:.4f})")
        
        # Prediction correlations
        print(f"\n🔗 Prediction correlations between horizons:")
        pred_matrix = pd.DataFrame({
            horizon: results[horizon]['predictions']['proba']
            for horizon in self.horizons
        })
        
        corr_matrix = pred_matrix.corr()
        print(corr_matrix.round(3))
        
        # Save comparison
        comparison_path = f"{self.artifacts_path}/reports/horizon_comparison.csv"
        comparison_df.to_csv(comparison_path)
        
        print(f"\n💾 Results saved to {self.artifacts_path}/")
        print("✅ Multi-horizon pipeline completed successfully!")


def run_multi_horizon_pipeline(
    df: pd.DataFrame,
    features: pd.DataFrame,
    horizons: List[str] = None,
    symbol: str = "BTCUSDT",
    test_size: float = 0.2,
    val_size: float = 0.2,
    n_trials: int = 50,
    artifacts_path: str = "artifacts"
) -> Dict:
    """
    Convenience function to run multi-horizon pipeline.
    
    Args:
        df: OHLC DataFrame
        features: Features DataFrame
        horizons: Time horizons to process
        symbol: Trading symbol
        test_size: Test set proportion
        val_size: Validation set proportion
        n_trials: Optuna trials per horizon
        artifacts_path: Artifacts directory
        
    Returns:
        Results dictionary
    """
    pipeline = MultiHorizonPipeline(
        horizons=horizons,
        symbol=symbol,
        test_size=test_size,
        val_size=val_size,
        n_trials=n_trials,
        artifacts_path=artifacts_path
    )
    
    return pipeline.run_pipeline(df, features)