"""
Base trainer class for all model training scripts.
Provides common functionality for data loading, preprocessing, and training orchestration.
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any, Callable
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import mlflow
import joblib
from sklearn.metrics import classification_report

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.binance_loader import CryptoDataLoader
from src.data.quality_pipeline import DataQualityPipeline
from src.data.splits import temporal_train_test_split
from src.features.engineering import FeatureEngineer
from src.features.filters import FeatureFilter
from src.utils.determinism_enhanced import set_full_determinism
from src.utils.logging import log as logger
from src.utils.memory_manager import memory_manager


class BaseTrainer(ABC):
    """
    Abstract base class for all model trainers.
    Provides common functionality for training pipeline.
    """
    
    def __init__(self, model_type: str = "base"):
        """
        Initialize base trainer.
        
        Args:
            model_type: Type of model (lstm, xgboost, etc.)
        """
        self.model_type = model_type
        self.args = None
        self.data_loader = None
        self.feature_engineer = None
        self.model_optimizer = None
        self.df = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description=f'{self.model_type.upper()} training with Bayesian optimization'
        )
        
        # Common arguments
        parser.add_argument('--symbol', type=str, default='BTCUSDT',
                          help='Trading symbol')
        parser.add_argument('--interval', type=str, default='15m',
                          help='Candle interval')
        parser.add_argument('--start-date', type=str, default='2023-01-01',
                          help='Start date for data')
        parser.add_argument('--end-date', type=str, default='2024-01-01',
                          help='End date for data')
        
        # Optimization arguments
        parser.add_argument('--trials', type=int, default=100,
                          help='Number of Optuna trials')
        parser.add_argument('--timeout', type=int, default=None,
                          help='Optimization timeout in seconds')
        parser.add_argument('--pruner', type=str, default='asha',
                          choices=['asha', 'hyperband', 'median', 'percentile'],
                          help='Pruner type')
        parser.add_argument('--sampler', type=str, default='tpe',
                          choices=['tpe', 'random', 'cmaes'],
                          help='Sampler type')
        
        # Validation arguments
        parser.add_argument('--outer-cv', type=int, default=3,
                          help='Number of outer CV splits')
        parser.add_argument('--inner-cv', type=int, default=5,
                          help='Number of inner CV splits')
        parser.add_argument('--embargo', type=int, default=10,
                          help='Embargo period in bars')
        
        # Calibration arguments
        parser.add_argument('--calibration', type=str, default='auto',
                          help='Calibration method')
        
        # Feature engineering
        parser.add_argument('--max-features', type=int, default=100,
                          help='Maximum number of features to select')
        parser.add_argument('--feature-selection', type=str, default='mutual_info',
                          choices=['mutual_info', 'f_classif', 'chi2', 'all'],
                          help='Feature selection method')
        
        # Output arguments
        parser.add_argument('--output-dir', type=str, default='artifacts/models',
                          help='Output directory for models')
        parser.add_argument('--experiment-name', type=str, default=None,
                          help='MLflow experiment name')
        
        # Mode arguments
        parser.add_argument('--fast', action='store_true',
                          help='Fast mode with reduced trials')
        parser.add_argument('--seed', type=int, default=42,
                          help='Random seed')
        parser.add_argument('--verbose', action='store_true',
                          help='Verbose output')
        
        # Model-specific arguments
        self._add_model_specific_args(parser)
        
        args = parser.parse_args()
        
        # Adjust for fast mode
        if args.fast:
            args.trials = min(args.trials, 20)
            args.outer_cv = min(args.outer_cv, 2)
            args.inner_cv = min(args.inner_cv, 3)
            args.max_features = min(args.max_features, 50)
        
        self.args = args
        return args
    
    @abstractmethod
    def _add_model_specific_args(self, parser: argparse.ArgumentParser):
        """Add model-specific arguments to parser."""
        pass
    
    def setup_environment(self):
        """Setup deterministic environment and logging."""
        logger.info(
            f"setting_up_{self.model_type}_training_environment",
            seed=self.args.seed,
            mode='fast' if self.args.fast else 'full'
        )
        
        # Set determinism
        set_full_determinism(self.args.seed, verify=True)
        
        # Create output directory
        self.output_dir = Path(self.args.output_dir) / self.model_type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate data."""
        logger.info("loading_data",
                   symbol=self.args.symbol,
                   interval=self.args.interval,
                   start=self.args.start_date,
                   end=self.args.end_date)
        
        with memory_manager.memory_context("data_loading"):
            # Initialize data loader
            self.data_loader = CryptoDataLoader(
                symbol=self.args.symbol,
                interval=self.args.interval
            )
            
            # Load data
            self.df = self.data_loader.load_data(
                start_date=self.args.start_date,
                end_date=self.args.end_date
            )
            
            # Validate data quality
            pipeline = DataQualityPipeline()
            self.df = pipeline.process(self.df)
            
            # Optimize memory
            self.df = memory_manager.optimize_dataframe(self.df)
            
            logger.info("data_loaded",
                       shape=self.df.shape,
                       memory_mb=f"{self.df.memory_usage(deep=True).sum() / 1024**2:.1f}")
        
        return self.df
    
    def create_features(self) -> pd.DataFrame:
        """Create and select features."""
        logger.info("creating_features", max_features=self.args.max_features)
        
        with memory_manager.memory_context("feature_engineering"):
            # Initialize feature engineer
            self.feature_engineer = FeatureEngineer(
                include_microstructure=True,
                include_technical=True,
                include_statistical=True
            )
            
            # Create features
            df_features = self.feature_engineer.create_features(self.df)
            
            # Optimize memory
            df_features = memory_manager.optimize_dataframe(df_features)
            
            logger.info("features_created",
                       n_features=len(df_features.columns),
                       memory_mb=f"{df_features.memory_usage(deep=True).sum() / 1024**2:.1f}")
        
        return df_features
    
    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create labels for training.
        Can be overridden for custom labeling strategies.
        """
        horizon = 5  # 5 bars ahead
        threshold = 0.002  # 0.2% return threshold
        
        future_returns = df['close'].pct_change(horizon).shift(-horizon)
        labels = (future_returns > threshold).astype(int)
        
        return labels.dropna()
    
    def select_features(self, df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """Select top features based on importance."""
        logger.info("selecting_features",
                   method=self.args.feature_selection,
                   max_features=self.args.max_features)
        
        with memory_manager.memory_context("feature_selection"):
            # Initialize feature filter
            feature_filter = FeatureFilter(
                max_features=self.args.max_features,
                method=self.args.feature_selection
            )
            
            # Remove price columns and select features
            feature_cols = [col for col in df.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            # Align data
            common_index = df.index.intersection(labels.index)
            df_aligned = df.loc[common_index, feature_cols]
            labels_aligned = labels.loc[common_index]
            
            # Select features
            selected_features = feature_filter.filter_features(
                df_aligned, labels_aligned
            )
            
            logger.info("features_selected", n_selected=len(selected_features))
        
        return df[selected_features]
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data into train/val/test sets."""
        logger.info("splitting_data", embargo=self.args.embargo)
        
        # Calculate split ratios
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        
        # Temporal split with embargo
        (X_train, X_temp, y_train, y_temp) = temporal_train_test_split(
            X, y,
            test_size=(val_ratio + test_ratio),
            embargo=self.args.embargo
        )
        
        # Split validation and test
        val_size = val_ratio / (val_ratio + test_ratio)
        (X_val, X_test, y_val, y_test) = temporal_train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size),
            embargo=self.args.embargo
        )
        
        # Store splits
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        
        logger.info("data_split",
                   train_size=len(X_train),
                   val_size=len(X_val),
                   test_size=len(X_test))
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def setup_mlflow(self):
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri('artifacts/mlruns')
        
        experiment_name = (self.args.experiment_name or 
                          f'{self.model_type}_optimization')
        if self.args.fast:
            experiment_name += '_fast'
        
        mlflow.set_experiment(experiment_name)
        
        run_name = f'{self.model_type}_{datetime.now():%Y%m%d_%H%M%S}'
        if self.args.fast:
            run_name += '_fast'
        
        mlflow.start_run(run_name=run_name)
        
        # Log tags
        mlflow.set_tags({
            'model_type': self.model_type,
            'optimization': 'bayesian',
            'pruner': self.args.pruner,
            'sampler': self.args.sampler,
            'calibration': self.args.calibration,
            'mode': 'fast' if self.args.fast else 'full',
            'deterministic': 'true',
            'symbol': self.args.symbol,
            'interval': self.args.interval
        })
        
        # Log parameters
        mlflow.log_params({
            'trials': self.args.trials,
            'outer_cv': self.args.outer_cv,
            'inner_cv': self.args.inner_cv,
            'embargo': self.args.embargo,
            'max_features': self.args.max_features,
            'feature_selection': self.args.feature_selection,
            'seed': self.args.seed
        })
    
    @abstractmethod
    def create_optimizer(self) -> Any:
        """Create model-specific optimizer."""
        pass
    
    @abstractmethod
    def train_model(self):
        """Train the model with optimization."""
        pass
    
    def evaluate_model(self):
        """Evaluate trained model on test set."""
        if self.model_optimizer is None or self.model_optimizer.best_model is None:
            raise ValueError("Model not trained yet")
        
        logger.info("evaluating_model_on_test_set")
        
        # Get predictions
        y_pred = self.model_optimizer.predict(self.X_test)
        y_pred_proba = self.model_optimizer.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, average_precision_score
        )
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'pr_auc': average_precision_score(self.y_test, y_pred_proba)
        }
        
        # Log to MLflow
        mlflow.log_metrics({f'test_{k}': v for k, v in metrics.items()})
        
        # Print classification report
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['No Signal', 'Signal']))
        
        # Print metrics
        print("\nMETRICS:")
        for metric, value in metrics.items():
            print(f"  {metric:12s}: {value:.4f}")
        
        return metrics
    
    def save_model(self):
        """Save trained model and artifacts."""
        if self.model_optimizer is None or self.model_optimizer.best_model is None:
            raise ValueError("Model not trained yet")
        
        logger.info("saving_model", output_dir=str(self.output_dir))
        
        # Create timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = self.output_dir / f'{self.model_type}_model_{timestamp}.pkl'
        joblib.dump(self.model_optimizer, model_path)
        
        # Log to MLflow
        mlflow.log_artifact(str(model_path))
        
        # Save optimization report
        report_path = self.output_dir / f'{self.model_type}_report_{timestamp}.json'
        report = self.model_optimizer.get_optimization_summary()
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        mlflow.log_artifact(str(report_path))
        
        logger.info("model_saved", path=str(model_path))
        
        return model_path
    
    def run(self):
        """Run complete training pipeline."""
        try:
            # Setup
            self.parse_arguments()
            self.setup_environment()
            self.setup_mlflow()
            
            # Load and prepare data
            self.load_data()
            df_features = self.create_features()
            labels = self.create_labels(self.df)
            
            # Select features
            X = self.select_features(df_features, labels)
            
            # Align data
            common_index = X.index.intersection(labels.index)
            X = X.loc[common_index]
            y = labels.loc[common_index]
            
            # Split data
            self.split_data(X, y)
            
            # Train model
            self.train_model()
            
            # Evaluate
            test_metrics = self.evaluate_model()
            
            # Save model
            model_path = self.save_model()
            
            # Success message
            print("\n" + "="*60)
            print(f"✅ {self.model_type.upper()} TRAINING COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Model saved to: {model_path}")
            print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
            print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
            
        except Exception as e:
            logger.error(f"training_failed", error=str(e))
            raise
        finally:
            # Cleanup
            mlflow.end_run()
            memory_manager.clean_memory(force=True)


# Export
__all__ = ['BaseTrainer']