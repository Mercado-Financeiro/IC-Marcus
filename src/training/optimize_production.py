#!/usr/bin/env python
"""
Production training script with quality gates.

This script:
1. Optimizes models using Bayesian HPO with Optuna
2. Validates against quality gates (PR-AUC, Brier, ECE, MCC)
3. Automatically applies best calibration method
4. Saves models only if they pass all gates
5. Falls back to MONITOR_ONLY mode if gates fail
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import data and feature modules
from src.data.binance_loader import BinanceDataLoader
from src.features.engineering import FeatureEngineer
from src.data.splits import create_temporal_split

# Import model modules
from src.models.xgb_optuna import XGBoostOptuna
from src.models.lstm.optuna.optimizer import LSTMOptuna
from src.models.lstm.optuna.config import LSTMOptunaConfig

# Import validation modules
from src.models.validation.model_validator import ModelValidator
from src.models.metrics.quality_gates import QualityGates
from src.models.calibration.beta import BetaCalibration, AdaptiveBetaCalibration

# Import utilities
from src.utils.determinism import set_deterministic_environment
import structlog

# Setup logging
log = structlog.get_logger()


class ProductionTrainer:
    """Production model training with quality gates."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: str = "models/production",
        artifacts_dir: str = "artifacts/production"
    ):
        """
        Initialize production trainer.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory for saving models
            artifacts_dir: Directory for artifacts
        """
        self.config = self._load_config(config_path) if config_path else self._get_default_config()
        self.output_dir = Path(output_dir)
        self.artifacts_dir = Path(artifacts_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.validator = ModelValidator(
            gates=QualityGates(),
            save_plots=True,
            plot_dir=str(self.artifacts_dir / "validation"),
            verbose=True
        )
        
        # Track results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'decisions': {}
        }
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'data': {
                'symbol': 'BTCUSDT',
                'interval': '1h',
                'start_date': '2020-01-01',
                'end_date': None
            },
            'split': {
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15
            },
            'xgboost': {
                'n_trials': 100,
                'cv_folds': 5,
                'embargo': 30,
                'pruner_type': 'hyperband'
            },
            'lstm': {
                'n_trials': 50,
                'cv_folds': 3,
                'max_epochs': 100
            },
            'gates': {
                'pr_auc_threshold': 1.2,
                'brier_improvement': 0.9,
                'ece_threshold': 0.05,
                'mcc_threshold': 0.0
            }
        }
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare data."""
        log.info("Loading data", config=self.config['data'])
        
        # Load data
        loader = BinanceDataLoader(
            data_dir="data/binance",
            cache_dir="data/cache"
        )
        
        df = loader.load_data(
            symbol=self.config['data']['symbol'],
            interval=self.config['data']['interval'],
            start_date=self.config['data']['start_date'],
            end_date=self.config['data'].get('end_date')
        )
        
        # Engineer features
        log.info("Engineering features")
        engineer = FeatureEngineer()
        df = engineer.create_features(df)
        
        # Create target (next period return > 0)
        df['target'] = (df['returns'].shift(-1) > 0).astype(int)
        
        # Remove NaN and prepare
        df = df.dropna()
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in 
                       ['target', 'open', 'high', 'low', 'close', 'volume', 
                        'returns', 'log_returns']]
        
        X = df[feature_cols]
        y = df['target']
        
        log.info(f"Data loaded: {len(X)} samples, {X.shape[1]} features")
        log.info(f"Target distribution: {y.value_counts().to_dict()}")
        log.info(f"Prevalence: {y.mean():.3f}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data into train/val/test sets."""
        log.info("Splitting data", ratios=self.config['split'])
        
        X_train, X_val, X_test, y_train, y_val, y_test = create_temporal_split(
            X, y,
            train_ratio=self.config['split']['train_ratio'],
            val_ratio=self.config['split']['val_ratio']
        )
        
        log.info(f"Train: {len(X_train)} samples")
        log.info(f"Val: {len(X_val)} samples")
        log.info(f"Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Any:
        """Train XGBoost with Optuna optimization."""
        log.info("Training XGBoost with Bayesian HPO")
        
        # Initialize optimizer
        xgb_opt = XGBoostOptuna(
            n_trials=self.config['xgboost']['n_trials'],
            cv_folds=self.config['xgboost']['cv_folds'],
            embargo=self.config['xgboost']['embargo'],
            pruner_type=self.config['xgboost']['pruner_type'],
            seed=42
        )
        
        # Run optimization
        study, model = xgb_opt.optimize(X_train, y_train)
        
        # Store results
        self.results['models']['xgboost'] = {
            'best_params': xgb_opt.best_params,
            'best_score': xgb_opt.best_score,
            'n_trials': len(study.trials),
            'n_pruned': len([t for t in study.trials if t.state.name == 'PRUNED'])
        }
        
        log.info(f"XGBoost optimization complete: {self.results['models']['xgboost']}")
        
        return xgb_opt
    
    def train_lstm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Any:
        """Train LSTM with Optuna optimization."""
        log.info("Training LSTM with Bayesian HPO")
        
        # Configure LSTM
        lstm_config = LSTMOptunaConfig(
            n_trials=self.config['lstm']['n_trials'],
            cv_folds=self.config['lstm']['cv_folds'],
            max_epochs=self.config['lstm']['max_epochs']
        )
        
        # Initialize optimizer
        lstm_opt = LSTMOptuna(config=lstm_config)
        
        # Run optimization
        study = lstm_opt.optimize(X_train, y_train)
        
        # Fit final model
        lstm_opt.fit_final_model(X_train, y_train)
        
        # Store results
        self.results['models']['lstm'] = {
            'best_params': lstm_opt.best_params,
            'best_score': lstm_opt.best_score,
            'n_trials': len(study.trials),
            'n_pruned': len([t for t in study.trials if t.state.name == 'PRUNED'])
        }
        
        log.info(f"LSTM optimization complete: {self.results['models']['lstm']}")
        
        return lstm_opt
    
    def validate_and_save(
        self,
        model: Any,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> str:
        """
        Validate model and save if passes gates.
        
        Args:
            model: Trained model
            model_name: Name for saving
            X_test: Test features
            y_test: Test labels
            X_val: Validation features (for recalibration)
            y_val: Validation labels (for recalibration)
            
        Returns:
            Model mode (PRODUCTION_READY, MONITOR_ONLY, etc.)
        """
        log.info(f"Validating {model_name}")
        
        # Validate on test set
        gate_results, metrics = self.validator.validate_model(
            model, X_test, y_test, model_name
        )
        
        # Store validation results
        self.results['models'][model_name]['validation'] = {
            'pr_auc': metrics['pr_auc'],
            'pr_auc_normalized': metrics['pr_auc_normalized'],
            'brier_score': metrics['brier_score'],
            'ece': metrics['ece'],
            'mcc': metrics['mcc'],
            'gates_passed': gate_results['summary']['all_passed'],
            'mode': gate_results['summary']['mode']
        }
        
        mode = gate_results['summary']['mode']
        
        # Handle different modes
        if mode == 'PRODUCTION_READY':
            # Save model
            model_path = self.output_dir / f"{model_name}_production_{datetime.now():%Y%m%d_%H%M%S}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            log.info(f"✅ Model saved to {model_path}")
            
        elif mode == 'NEEDS_RECALIBRATION' and X_val is not None:
            log.info(f"⚠️ {model_name} needs recalibration")
            
            # Try Beta calibration
            try:
                # Get uncalibrated probabilities
                y_val_proba = model.predict_proba(X_val)
                if y_val_proba.ndim > 1:
                    y_val_proba = y_val_proba[:, 1]
                
                # Apply Beta calibration
                beta_cal = AdaptiveBetaCalibration(cv=3, metric='brier')
                beta_cal.fit(y_val_proba, y_val)
                
                # Create calibrated model wrapper
                class CalibratedModel:
                    def __init__(self, base_model, calibrator):
                        self.base_model = base_model
                        self.calibrator = calibrator
                    
                    def predict_proba(self, X):
                        proba = self.base_model.predict_proba(X)
                        if proba.ndim > 1:
                            proba = proba[:, 1]
                        calibrated = self.calibrator.transform(proba)
                        return np.column_stack([1 - calibrated, calibrated])
                    
                    def predict(self, X, threshold=0.5):
                        proba = self.predict_proba(X)[:, 1]
                        return (proba >= threshold).astype(int)
                
                calibrated_model = CalibratedModel(model, beta_cal)
                
                # Re-validate
                gate_results_cal, metrics_cal = self.validator.validate_model(
                    calibrated_model, X_test, y_test, f"{model_name}_calibrated"
                )
                
                if gate_results_cal['summary']['all_passed']:
                    # Save calibrated model
                    model_path = self.output_dir / f"{model_name}_calibrated_production_{datetime.now():%Y%m%d_%H%M%S}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(calibrated_model, f)
                    log.info(f"✅ Calibrated model saved to {model_path}")
                    mode = 'PRODUCTION_READY'
                else:
                    log.warning(f"Calibration did not resolve all issues")
                    
            except Exception as e:
                log.error(f"Calibration failed: {e}")
        
        elif mode == 'MONITOR_ONLY':
            log.warning(f"⚠️ {model_name} in MONITOR_ONLY mode - decisions will be neutralized")
            # Save with warning
            model_path = self.output_dir / f"{model_name}_monitor_only_{datetime.now():%Y%m%d_%H%M%S}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            log.info(f"Model saved for monitoring: {model_path}")
            
        else:
            log.error(f"❌ {model_name} failed quality gates - not suitable for production")
        
        self.results['decisions'][model_name] = mode
        return mode
    
    def run(self):
        """Run complete production training pipeline."""
        log.info("="*80)
        log.info("PRODUCTION TRAINING PIPELINE")
        log.info("="*80)
        
        # Set deterministic environment
        set_deterministic_environment(42)
        
        # Load data
        X, y = self.load_data()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Train XGBoost
        log.info("\n" + "="*60)
        log.info("XGBOOST TRAINING")
        log.info("="*60)
        xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)
        xgb_mode = self.validate_and_save(
            xgb_model, "xgboost", X_test, y_test, X_val, y_val
        )
        
        # Train LSTM
        log.info("\n" + "="*60)
        log.info("LSTM TRAINING")
        log.info("="*60)
        lstm_model = self.train_lstm(X_train, y_train, X_val, y_val)
        lstm_mode = self.validate_and_save(
            lstm_model, "lstm", X_test, y_test, X_val, y_val
        )
        
        # Save results summary
        results_path = self.artifacts_dir / f"training_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Print final summary
        log.info("\n" + "="*80)
        log.info("TRAINING COMPLETE")
        log.info("="*80)
        log.info(f"XGBoost: {xgb_mode}")
        log.info(f"LSTM: {lstm_mode}")
        
        # Determine overall status
        if xgb_mode == 'PRODUCTION_READY' or lstm_mode == 'PRODUCTION_READY':
            log.info("🎉 At least one model is PRODUCTION READY!")
            return 0
        elif xgb_mode == 'MONITOR_ONLY' or lstm_mode == 'MONITOR_ONLY':
            log.warning("⚠️ Models in MONITOR_ONLY mode - decisions neutralized")
            return 1
        else:
            log.error("❌ No models passed quality gates")
            return 2


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Production model training with quality gates")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='models/production',
                       help='Directory for saving models')
    parser.add_argument('--artifacts-dir', type=str, default='artifacts/production',
                       help='Directory for artifacts')
    args = parser.parse_args()
    
    # Run trainer
    trainer = ProductionTrainer(
        config_path=args.config,
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir
    )
    
    return trainer.run()


if __name__ == "__main__":
    sys.exit(main())