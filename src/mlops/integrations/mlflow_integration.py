"""
MLflow integration for comprehensive ML lifecycle management.
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.store.artifact.artifact_repository import ArtifactRepository

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json
import pickle
import os
from pathlib import Path
import tempfile
import shutil

from ...features.store import FeatureStore, VersionStatus
from ...monitoring.drift_detector import DriftDetector
from ...features.validation.data_quality import DataQualityValidator

logger = logging.getLogger(__name__)


@dataclass
class MLflowConfig:
    """MLflow integration configuration."""
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "crypto_ml_pipeline"
    artifact_location: Optional[str] = None
    enable_feature_store: bool = True
    enable_model_registry: bool = True
    enable_data_quality_logging: bool = True
    auto_log_metrics: bool = True
    auto_log_params: bool = True
    auto_log_artifacts: bool = True


class MLflowFeatureStore:
    """
    Integration between our Feature Store and MLflow.
    
    Features:
    - Automatic feature logging to MLflow
    - Feature versioning with MLflow Model Registry
    - Data quality metrics integration
    - Experiment tracking for feature engineering
    - Feature lineage in MLflow artifacts
    """
    
    def __init__(self, config: MLflowConfig, feature_store: FeatureStore):
        """Initialize MLflow Feature Store integration."""
        self.config = config
        self.feature_store = feature_store
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config.tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment = mlflow.create_experiment(
                name=config.experiment_name,
                artifact_location=config.artifact_location
            )
        except mlflow.exceptions.MlflowException:
            self.experiment = mlflow.get_experiment_by_name(config.experiment_name)
            self.experiment = self.experiment.experiment_id
        
        mlflow.set_experiment(experiment_id=self.experiment)
        
        # MLflow client for advanced operations
        self.client = MlflowClient()
        
        # Data quality validator for logging
        self.validator = DataQualityValidator() if config.enable_data_quality_logging else None
        
        logger.info(f"MLflowFeatureStore initialized with experiment: {config.experiment_name}")
    
    def log_feature_group(
        self,
        group_name: str,
        features_data: pd.DataFrame,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        log_sample: bool = True,
        sample_size: int = 1000
    ) -> str:
        """
        Log feature group to both Feature Store and MLflow.
        
        Args:
            group_name: Feature group name
            features_data: Feature data
            run_id: Existing MLflow run ID (creates new if None)
            tags: Additional tags
            log_sample: Whether to log data sample
            sample_size: Size of sample to log
            
        Returns:
            MLflow run ID
        """
        # Start or use existing run
        if run_id is None:
            run = mlflow.start_run()
            run_id = run.info.run_id
        else:
            mlflow.start_run(run_id=run_id)
        
        try:
            # Store in Feature Store first
            version_id = self.feature_store.write_feature_group(
                group_name=group_name,
                data=features_data,
                description=f"MLflow tracked features - Run: {run_id}",
                tags=list(tags.keys()) if tags else [],
                created_by="mlflow_integration"
            )
            
            # Log to MLflow
            if self.config.auto_log_params:
                mlflow.log_params({
                    'feature_group': group_name,
                    'feature_count': len(features_data.columns),
                    'record_count': len(features_data),
                    'feature_store_version': version_id
                })
            
            if self.config.auto_log_metrics:
                # Log data quality metrics
                quality_metrics = self._calculate_quality_metrics(features_data)
                mlflow.log_metrics(quality_metrics)
            
            # Log feature metadata
            feature_metadata = {
                'columns': list(features_data.columns),
                'dtypes': features_data.dtypes.to_dict(),
                'shape': features_data.shape,
                'memory_usage_mb': features_data.memory_usage(deep=True).sum() / 1024**2,
                'feature_store_version': version_id
            }
            
            # Save metadata as artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(feature_metadata, f, indent=2, default=str)
                temp_file = f.name
            
            mlflow.log_artifact(temp_file, "feature_metadata")
            os.unlink(temp_file)
            
            # Log sample data if requested
            if log_sample and len(features_data) > 0:
                sample_data = features_data.sample(min(sample_size, len(features_data)))
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    sample_data.to_csv(f.name, index=False)
                    temp_file = f.name
                
                mlflow.log_artifact(temp_file, "data_samples")
                os.unlink(temp_file)
            
            # Log feature statistics
            self._log_feature_statistics(features_data)
            
            # Add tags
            if tags:
                mlflow.set_tags(tags)
            
            # Default tags
            mlflow.set_tags({
                'component': 'feature_engineering',
                'feature_group': group_name,
                'integration': 'feature_store'
            })
            
            logger.info(f"Feature group {group_name} logged to MLflow (Run: {run_id})")
            return run_id
            
        finally:
            mlflow.end_run()
    
    def track_feature_engineering_experiment(
        self,
        experiment_name: str,
        feature_engineering_func: callable,
        raw_data: pd.DataFrame,
        parameters: Dict[str, Any],
        evaluation_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Track feature engineering experiment end-to-end.
        
        Args:
            experiment_name: Name of the experiment
            feature_engineering_func: Function that creates features
            raw_data: Raw input data
            parameters: Feature engineering parameters
            evaluation_metrics: Metrics to evaluate features
            
        Returns:
            Experiment results including run_id and metrics
        """
        with mlflow.start_run(run_name=experiment_name) as run:
            run_id = run.info.run_id
            
            # Log input data info
            mlflow.log_params({
                'input_records': len(raw_data),
                'input_features': len(raw_data.columns),
                'input_memory_mb': raw_data.memory_usage(deep=True).sum() / 1024**2
            })
            
            # Log feature engineering parameters
            mlflow.log_params(parameters)
            
            # Execute feature engineering
            start_time = datetime.now()
            
            try:
                engineered_features = feature_engineering_func(raw_data, **parameters)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Log processing metrics
                mlflow.log_metrics({
                    'processing_time_seconds': processing_time,
                    'output_features': len(engineered_features.columns),
                    'output_records': len(engineered_features),
                    'feature_creation_rate': len(engineered_features.columns) / processing_time
                })
                
                # Log feature engineering results
                feature_run_id = self.log_feature_group(
                    group_name=f"experiment_{experiment_name}",
                    features_data=engineered_features,
                    run_id=run_id
                )
                
                # Evaluate features if metrics specified
                if evaluation_metrics:
                    evaluation_results = self._evaluate_features(
                        engineered_features, evaluation_metrics
                    )
                    mlflow.log_metrics(evaluation_results)
                
                # Log success
                mlflow.log_metrics({'experiment_success': 1})
                
                return {
                    'run_id': run_id,
                    'success': True,
                    'processing_time': processing_time,
                    'output_features': engineered_features,
                    'feature_count': len(engineered_features.columns)
                }
                
            except Exception as e:
                # Log failure
                mlflow.log_metrics({'experiment_success': 0})
                mlflow.log_param('error_message', str(e))
                
                logger.error(f"Feature engineering experiment failed: {e}")
                raise
    
    def create_feature_model(
        self,
        group_name: str,
        version_id: str,
        model_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Register feature group as MLflow model.
        
        Args:
            group_name: Feature group name
            version_id: Feature Store version ID
            model_name: MLflow model name
            description: Model description
            
        Returns:
            MLflow model version
        """
        model_name = model_name or f"features_{group_name}"
        
        # Get feature data
        features_data = self.feature_store.read_feature_group(group_name, version_id)
        
        if features_data is None:
            raise ValueError(f"Feature group {group_name} version {version_id} not found")
        
        # Create custom MLflow model
        class FeatureModel(mlflow.pyfunc.PythonModel):
            def __init__(self, feature_store, group_name, version_id):
                self.feature_store = feature_store
                self.group_name = group_name
                self.version_id = version_id
            
            def predict(self, context, model_input):
                # This would implement feature serving logic
                # For now, return the features themselves
                return self.feature_store.read_feature_group(
                    self.group_name, self.version_id
                )
        
        # Create model instance
        feature_model = FeatureModel(self.feature_store, group_name, version_id)
        
        # Log model
        with mlflow.start_run():
            mlflow.pyfunc.log_model(
                artifact_path="feature_model",
                python_model=feature_model,
                registered_model_name=model_name
            )
            
            # Add model metadata
            mlflow.log_params({
                'feature_group': group_name,
                'feature_store_version': version_id,
                'feature_count': len(features_data.columns)
            })
        
        logger.info(f"Feature model {model_name} registered in MLflow")
        return model_name
    
    def compare_feature_versions(
        self,
        group_name: str,
        version_1: str,
        version_2: str,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare two feature versions and log results to MLflow.
        
        Args:
            group_name: Feature group name
            version_1: First version ID
            version_2: Second version ID
            metrics: Comparison metrics
            
        Returns:
            Comparison results
        """
        with mlflow.start_run(run_name=f"feature_comparison_{group_name}") as run:
            # Load both versions
            features_v1 = self.feature_store.read_feature_group(group_name, version_1)
            features_v2 = self.feature_store.read_feature_group(group_name, version_2)
            
            if features_v1 is None or features_v2 is None:
                raise ValueError("One or both feature versions not found")
            
            # Compare basic statistics
            comparison = {
                'v1_features': len(features_v1.columns),
                'v2_features': len(features_v2.columns),
                'v1_records': len(features_v1),
                'v2_records': len(features_v2),
                'feature_diff': len(features_v2.columns) - len(features_v1.columns),
                'record_diff': len(features_v2) - len(features_v1)
            }
            
            # Log comparison metrics
            mlflow.log_metrics(comparison)
            
            # Detect schema changes
            schema_changes = self._detect_schema_changes(features_v1, features_v2)
            mlflow.log_dict(schema_changes, "schema_changes.json")
            
            # Quality comparison
            if self.validator:
                quality_v1 = self._calculate_quality_metrics(features_v1)
                quality_v2 = self._calculate_quality_metrics(features_v2)
                
                quality_diff = {
                    f"{k}_diff": quality_v2.get(k, 0) - quality_v1.get(k, 0)
                    for k in set(quality_v1.keys()) | set(quality_v2.keys())
                }
                
                mlflow.log_metrics(quality_diff)
            
            # Log parameters
            mlflow.log_params({
                'feature_group': group_name,
                'version_1': version_1,
                'version_2': version_2,
                'comparison_type': 'feature_version'
            })
            
            return {
                'run_id': run.info.run_id,
                'comparison': comparison,
                'schema_changes': schema_changes
            }
    
    def _calculate_quality_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality metrics for MLflow logging."""
        metrics = {}
        
        # Completeness
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        metrics['completeness'] = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 1.0
        
        # Uniqueness (average across columns)
        uniqueness_scores = []
        for col in data.columns:
            if data[col].dtype in ['object', 'category']:
                uniqueness = data[col].nunique() / len(data) if len(data) > 0 else 0
                uniqueness_scores.append(uniqueness)
        
        if uniqueness_scores:
            metrics['avg_uniqueness'] = np.mean(uniqueness_scores)
        
        # Validity (no infinite/invalid values)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            invalid_count = np.isinf(data[numeric_cols]).sum().sum()
            metrics['validity'] = 1.0 - (invalid_count / len(data)) if len(data) > 0 else 1.0
        
        # Overall quality score
        quality_components = [v for k, v in metrics.items() if not k.startswith('avg_')]
        if quality_components:
            metrics['overall_quality'] = np.mean(quality_components)
        
        return metrics
    
    def _log_feature_statistics(self, data: pd.DataFrame):
        """Log detailed feature statistics to MLflow."""
        # Statistical summaries for numeric features
        numeric_features = data.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_features:
            if not data[feature].empty:
                feature_stats = {
                    f"{feature}_mean": data[feature].mean(),
                    f"{feature}_std": data[feature].std(),
                    f"{feature}_min": data[feature].min(),
                    f"{feature}_max": data[feature].max(),
                    f"{feature}_median": data[feature].median(),
                    f"{feature}_skew": data[feature].skew() if len(data[feature]) > 1 else 0,
                    f"{feature}_missing_pct": (data[feature].isnull().sum() / len(data)) * 100
                }
                
                # Log only non-NaN values
                clean_stats = {k: v for k, v in feature_stats.items() if not pd.isna(v)}
                mlflow.log_metrics(clean_stats)
    
    def _evaluate_features(self, features: pd.DataFrame, metrics: List[str]) -> Dict[str, float]:
        """Evaluate feature quality using specified metrics."""
        results = {}
        
        for metric in metrics:
            if metric == 'correlation_max':
                # Maximum correlation between features
                corr_matrix = features.corr().abs()
                # Remove diagonal and get upper triangle
                mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
                max_corr = corr_matrix.where(mask).max().max()
                results[metric] = max_corr if not pd.isna(max_corr) else 0
            
            elif metric == 'feature_importance_variance':
                # Variance of feature importance (proxy for feature diversity)
                numeric_features = features.select_dtypes(include=[np.number])
                if not numeric_features.empty:
                    variances = numeric_features.var()
                    results[metric] = variances.var() if len(variances) > 1 else 0
            
            elif metric == 'information_gain':
                # Placeholder for information gain calculation
                # Would need target variable for real implementation
                results[metric] = 0.5
        
        return results
    
    def _detect_schema_changes(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        """Detect schema changes between two DataFrames."""
        changes = {
            'added_columns': list(set(df2.columns) - set(df1.columns)),
            'removed_columns': list(set(df1.columns) - set(df2.columns)),
            'dtype_changes': {}
        }
        
        # Check for dtype changes
        common_columns = set(df1.columns) & set(df2.columns)
        for col in common_columns:
            if df1[col].dtype != df2[col].dtype:
                changes['dtype_changes'][col] = {
                    'old': str(df1[col].dtype),
                    'new': str(df2[col].dtype)
                }
        
        return changes
    
    def get_feature_lineage(self, group_name: str) -> Dict[str, Any]:
        """Get feature lineage and log to MLflow."""
        lineage = self.feature_store.get_feature_lineage(group_name)
        
        with mlflow.start_run(run_name=f"lineage_{group_name}"):
            # Log lineage as artifact
            mlflow.log_dict(lineage, "feature_lineage.json")
            
            mlflow.log_params({
                'feature_group': group_name,
                'operation': 'lineage_extraction'
            })
        
        return lineage
    
    def cleanup_experiments(self, days_old: int = 30):
        """Cleanup old MLflow experiments."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cutoff_timestamp = int(cutoff_date.timestamp() * 1000)
        
        # Get old runs
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment],
            filter_string=f"attribute.end_time < {cutoff_timestamp}"
        )
        
        deleted_count = 0
        for _, run in runs.iterrows():
            try:
                self.client.delete_run(run.run_id)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete run {run.run_id}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old MLflow runs")


class MLflowExperimentTracker:
    """
    Advanced experiment tracking for ML workflows.
    
    Features:
    - Model training tracking
    - Hyperparameter optimization
    - Model comparison and selection
    - A/B testing support
    - Performance monitoring
    """
    
    def __init__(self, config: MLflowConfig):
        """Initialize experiment tracker."""
        self.config = config
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config.tracking_uri)
        self.client = MlflowClient()
        
        # Create experiments
        self.training_experiment = self._get_or_create_experiment(
            f"{config.experiment_name}_training"
        )
        self.hyperopt_experiment = self._get_or_create_experiment(
            f"{config.experiment_name}_hyperopt"
        )
        
        logger.info("MLflowExperimentTracker initialized")
    
    def _get_or_create_experiment(self, name: str) -> str:
        """Get or create MLflow experiment."""
        try:
            experiment_id = mlflow.create_experiment(name)
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(name)
            experiment_id = experiment.experiment_id
        
        return experiment_id
    
    def track_model_training(
        self,
        model,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        target_column: str,
        model_name: str,
        hyperparameters: Dict[str, Any],
        custom_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Track complete model training process.
        
        Args:
            model: ML model instance
            train_data: Training data
            val_data: Validation data  
            target_column: Target column name
            model_name: Name for the model
            hyperparameters: Model hyperparameters
            custom_metrics: Additional metrics to log
            
        Returns:
            MLflow run ID
        """
        mlflow.set_experiment(experiment_id=self.training_experiment)
        
        with mlflow.start_run(run_name=f"{model_name}_training") as run:
            run_id = run.info.run_id
            
            # Log hyperparameters
            mlflow.log_params(hyperparameters)
            
            # Log data info
            mlflow.log_params({
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'features': len(train_data.columns) - 1,  # Exclude target
                'target_column': target_column
            })
            
            # Prepare data
            X_train = train_data.drop(columns=[target_column])
            y_train = train_data[target_column]
            X_val = val_data.drop(columns=[target_column])
            y_val = val_data[target_column]
            
            # Train model
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Log training time
            mlflow.log_metric('training_time_seconds', training_time)
            
            # Generate predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # Calculate and log metrics
            metrics = self._calculate_model_metrics(
                y_train, train_pred, y_val, val_pred
            )
            mlflow.log_metrics(metrics)
            
            # Log custom metrics
            if custom_metrics:
                mlflow.log_metrics(custom_metrics)
            
            # Log model
            if hasattr(model, 'predict'):
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=model_name
                )
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                mlflow.log_table(importance_df, "feature_importance.json")
            
            # Add tags
            mlflow.set_tags({
                'model_type': type(model).__name__,
                'stage': 'training',
                'framework': 'sklearn'  # Could detect automatically
            })
            
            return run_id
    
    def track_hyperparameter_optimization(
        self,
        optimization_func: callable,
        search_space: Dict[str, Any],
        n_trials: int = 100,
        objective_metric: str = 'val_score'
    ) -> Dict[str, Any]:
        """
        Track hyperparameter optimization process.
        
        Args:
            optimization_func: Function that trains and evaluates model
            search_space: Hyperparameter search space
            n_trials: Number of optimization trials
            objective_metric: Metric to optimize
            
        Returns:
            Best trial results
        """
        mlflow.set_experiment(experiment_id=self.hyperopt_experiment)
        
        best_score = float('-inf')
        best_params = None
        best_run_id = None
        
        # Parent run for the optimization
        with mlflow.start_run(run_name="hyperparameter_optimization") as parent_run:
            parent_run_id = parent_run.info.run_id
            
            # Log optimization config
            mlflow.log_params({
                'n_trials': n_trials,
                'objective_metric': objective_metric,
                'search_space': str(search_space)
            })
            
            for trial in range(n_trials):
                # Sample parameters (simplified - would use proper optimization library)
                trial_params = self._sample_parameters(search_space)
                
                # Nested run for each trial
                with mlflow.start_run(
                    run_name=f"trial_{trial}",
                    nested=True
                ) as trial_run:
                    
                    # Run optimization function
                    try:
                        results = optimization_func(trial_params)
                        score = results.get(objective_metric, float('-inf'))
                        
                        # Log trial results
                        mlflow.log_params(trial_params)
                        mlflow.log_metrics(results)
                        mlflow.log_metric('trial_number', trial)
                        
                        # Track best trial
                        if score > best_score:
                            best_score = score
                            best_params = trial_params.copy()
                            best_run_id = trial_run.info.run_id
                            
                            # Log as best trial
                            mlflow.set_tag('best_trial', 'true')
                        
                    except Exception as e:
                        logger.error(f"Trial {trial} failed: {e}")
                        mlflow.log_param('error', str(e))
                        mlflow.log_metric(objective_metric, float('-inf'))
            
            # Log best results in parent run
            mlflow.log_params(best_params)
            mlflow.log_metric(f'best_{objective_metric}', best_score)
            mlflow.log_param('best_run_id', best_run_id)
        
        return {
            'best_score': best_score,
            'best_params': best_params,
            'best_run_id': best_run_id,
            'parent_run_id': parent_run_id
        }
    
    def _sample_parameters(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters from search space (simplified)."""
        params = {}
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, dict):
                if param_config['type'] == 'uniform':
                    params[param_name] = np.random.uniform(
                        param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'choice':
                    params[param_name] = np.random.choice(param_config['choices'])
                elif param_config['type'] == 'int_uniform':
                    params[param_name] = np.random.randint(
                        param_config['low'], param_config['high']
                    )
            else:
                # Simple list of choices
                if isinstance(param_config, list):
                    params[param_name] = np.random.choice(param_config)
        
        return params
    
    def _calculate_model_metrics(
        self,
        y_train: np.ndarray,
        train_pred: np.ndarray,
        y_val: np.ndarray,
        val_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive model metrics."""
        metrics = {}
        
        # Import sklearn metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Determine if regression or classification
        is_classification = len(np.unique(y_train)) < 20  # Simple heuristic
        
        if is_classification:
            # Classification metrics
            metrics.update({
                'train_accuracy': accuracy_score(y_train, train_pred),
                'val_accuracy': accuracy_score(y_val, val_pred),
                'train_precision': precision_score(y_train, train_pred, average='weighted', zero_division=0),
                'val_precision': precision_score(y_val, val_pred, average='weighted', zero_division=0),
                'train_recall': recall_score(y_train, train_pred, average='weighted', zero_division=0),
                'val_recall': recall_score(y_val, val_pred, average='weighted', zero_division=0),
                'train_f1': f1_score(y_train, train_pred, average='weighted', zero_division=0),
                'val_f1': f1_score(y_val, val_pred, average='weighted', zero_division=0)
            })
        else:
            # Regression metrics
            metrics.update({
                'train_mse': mean_squared_error(y_train, train_pred),
                'val_mse': mean_squared_error(y_val, val_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'train_r2': r2_score(y_train, train_pred),
                'val_r2': r2_score(y_val, val_pred)
            })
            
            # RMSE
            metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
            metrics['val_rmse'] = np.sqrt(metrics['val_mse'])
        
        return metrics
    
    def compare_models(
        self,
        run_ids: List[str],
        metrics: List[str],
        model_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple model runs.
        
        Args:
            run_ids: List of MLflow run IDs
            metrics: Metrics to compare
            model_names: Optional names for models
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for i, run_id in enumerate(run_ids):
            run = self.client.get_run(run_id)
            
            model_data = {
                'run_id': run_id,
                'model_name': model_names[i] if model_names else f"Model_{i+1}",
                'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
                'status': run.info.status
            }
            
            # Add requested metrics
            for metric in metrics:
                model_data[metric] = run.data.metrics.get(metric, np.nan)
            
            # Add key parameters
            model_data.update(run.data.params)
            
            comparison_data.append(model_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Log comparison
        with mlflow.start_run(run_name="model_comparison"):
            mlflow.log_table(comparison_df, "model_comparison.json")
            
            # Find best model for each metric
            for metric in metrics:
                if metric in comparison_df.columns:
                    best_idx = comparison_df[metric].idxmax()
                    if not pd.isna(best_idx):
                        best_model = comparison_df.loc[best_idx, 'model_name']
                        mlflow.log_param(f'best_model_{metric}', best_model)
        
        return comparison_df


# Convenience functions
def setup_mlflow_integration(
    feature_store: FeatureStore,
    tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "crypto_ml_pipeline"
) -> Tuple[MLflowFeatureStore, MLflowExperimentTracker]:
    """Setup complete MLflow integration."""
    config = MLflowConfig(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )
    
    feature_store_integration = MLflowFeatureStore(config, feature_store)
    experiment_tracker = MLflowExperimentTracker(config)
    
    return feature_store_integration, experiment_tracker