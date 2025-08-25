"""
SHAP (SHapley Additive exPlanations) analysis for model interpretability.

Provides comprehensive feature importance and interaction analysis using SHAP values.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
import structlog
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
log = structlog.get_logger()


class SHAPAnalyzer:
    """Comprehensive SHAP analysis for ML models."""
    
    def __init__(
        self,
        model: Any,
        model_type: str = "xgboost",
        background_samples: int = 100,
        max_display: int = 20
    ):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: Trained model to analyze
            model_type: Type of model (xgboost, lstm, ensemble)
            background_samples: Number of background samples for SHAP
            max_display: Maximum features to display in plots
        """
        self.model = model
        self.model_type = model_type
        self.background_samples = background_samples
        self.max_display = max_display
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        
    def create_explainer(self, X_train: Union[pd.DataFrame, np.ndarray]):
        """
        Create SHAP explainer based on model type.
        
        Args:
            X_train: Training data for background
        """
        log.info("creating_shap_explainer", model_type=self.model_type)
        
        # Sample background data if needed
        if len(X_train) > self.background_samples:
            if isinstance(X_train, pd.DataFrame):
                background = X_train.sample(n=self.background_samples, random_state=42)
            else:
                idx = np.random.RandomState(42).choice(
                    len(X_train), self.background_samples, replace=False
                )
                background = X_train[idx]
        else:
            background = X_train
        
        # Create appropriate explainer
        if self.model_type == "xgboost":
            # Use TreeExplainer for XGBoost (exact SHAP values)
            self.explainer = shap.TreeExplainer(self.model, background)
        elif self.model_type == "lstm" or self.model_type == "neural":
            # Use DeepExplainer for neural networks
            self.explainer = shap.DeepExplainer(self.model, background)
        elif self.model_type == "ensemble":
            # Use KernelExplainer for ensemble models
            if hasattr(self.model, 'predict_proba'):
                predict_fn = lambda x: self.model.predict_proba(x)[:, 1]
            else:
                predict_fn = self.model.predict
            self.explainer = shap.KernelExplainer(predict_fn, background)
        else:
            # Default to KernelExplainer
            if hasattr(self.model, 'predict_proba'):
                predict_fn = lambda x: self.model.predict_proba(x)[:, 1]
            else:
                predict_fn = self.model.predict
            self.explainer = shap.KernelExplainer(predict_fn, background)
    
    def calculate_shap_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        check_additivity: bool = False
    ) -> np.ndarray:
        """
        Calculate SHAP values for given data.
        
        Args:
            X: Data to explain
            check_additivity: Whether to check SHAP additivity
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        log.info("calculating_shap_values", n_samples=len(X))
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            # For multi-class, take the positive class
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Store values
        self.shap_values = shap_values
        self.expected_value = self.explainer.expected_value
        if isinstance(self.expected_value, list):
            self.expected_value = self.expected_value[1] if len(self.expected_value) > 1 else self.expected_value[0]
        
        # Check additivity if requested
        if check_additivity:
            self._check_additivity(X, shap_values)
        
        return shap_values
    
    def _check_additivity(self, X: Union[pd.DataFrame, np.ndarray], shap_values: np.ndarray):
        """Check if SHAP values satisfy additivity property."""
        # Get model predictions
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(X)[:, 1]
        else:
            predictions = self.model.predict(X)
        
        # Calculate sum of SHAP values + expected value
        shap_sum = shap_values.sum(axis=1) + self.expected_value
        
        # Check difference
        diff = np.abs(predictions - shap_sum)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        log.info(
            "shap_additivity_check",
            max_diff=max_diff,
            mean_diff=mean_diff,
            passed=max_diff < 0.01
        )
    
    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        importance_type: str = "mean_abs"
    ) -> pd.DataFrame:
        """
        Calculate feature importance from SHAP values.
        
        Args:
            feature_names: Names of features
            importance_type: Type of importance (mean_abs, mean, std)
            
        Returns:
            DataFrame with feature importances
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values first.")
        
        # Calculate importance based on type
        if importance_type == "mean_abs":
            importance = np.abs(self.shap_values).mean(axis=0)
        elif importance_type == "mean":
            importance = self.shap_values.mean(axis=0)
        elif importance_type == "std":
            importance = self.shap_values.std(axis=0)
        else:
            raise ValueError(f"Unknown importance type: {importance_type}")
        
        # Create DataFrame
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'importance_pct': importance / importance.sum()
        })
        
        # Sort by importance
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Add rank
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_feature_interactions(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate feature interaction effects.
        
        Args:
            X: Data for interaction calculation
            feature_names: Names of features
            
        Returns:
            DataFrame with interaction scores
        """
        log.info("calculating_feature_interactions")
        
        # Calculate interaction values (only for tree-based models)
        if self.model_type == "xgboost" and hasattr(self.explainer, 'shap_interaction_values'):
            interaction_values = self.explainer.shap_interaction_values(X)
            
            # Average over samples
            mean_interactions = np.abs(interaction_values).mean(axis=0)
            
            # Create DataFrame
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(mean_interactions.shape[0])]
            
            # Get top interactions
            interactions = []
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    interactions.append({
                        'feature_1': feature_names[i],
                        'feature_2': feature_names[j],
                        'interaction_strength': mean_interactions[i, j]
                    })
            
            df = pd.DataFrame(interactions)
            df = df.sort_values('interaction_strength', ascending=False).reset_index(drop=True)
            
            return df
        else:
            log.warning("interaction_values_not_available", model_type=self.model_type)
            return pd.DataFrame()
    
    def plot_summary(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        plot_type: str = "dot",
        save_path: Optional[str] = None
    ):
        """
        Create SHAP summary plot.
        
        Args:
            X: Data for plot
            feature_names: Names of features
            plot_type: Type of plot (dot, bar, violin)
            save_path: Path to save plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values first.")
        
        plt.figure(figsize=(10, 8))
        
        # Create summary plot
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=feature_names,
            plot_type=plot_type,
            max_display=self.max_display,
            show=False
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            log.info("plot_saved", path=save_path)
        
        plt.show()
    
    def plot_waterfall(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sample_idx: int,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            X: Data
            sample_idx: Index of sample to explain
            feature_names: Names of features
            save_path: Path to save plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values first.")
        
        # Get single sample
        if isinstance(X, pd.DataFrame):
            sample = X.iloc[sample_idx].values
        else:
            sample = X[sample_idx]
        
        shap_sample = self.shap_values[sample_idx]
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_sample,
            base_values=self.expected_value,
            data=sample,
            feature_names=feature_names
        )
        
        # Create waterfall plot
        shap.waterfall_plot(explanation, max_display=self.max_display, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            log.info("plot_saved", path=save_path)
        
        plt.show()
    
    def plot_force(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sample_idx: int,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Create SHAP force plot for a single prediction.
        
        Args:
            X: Data
            sample_idx: Index of sample to explain
            feature_names: Names of features
            save_path: Path to save plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values first.")
        
        # Get single sample
        if isinstance(X, pd.DataFrame):
            sample = X.iloc[sample_idx].values
        else:
            sample = X[sample_idx]
        
        shap_sample = self.shap_values[sample_idx]
        
        # Create force plot
        force_plot = shap.force_plot(
            self.expected_value,
            shap_sample,
            sample,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            log.info("plot_saved", path=save_path)
        
        return force_plot
    
    def plot_dependence(
        self,
        feature: Union[str, int],
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        interaction_feature: Optional[Union[str, int]] = "auto",
        save_path: Optional[str] = None
    ):
        """
        Create SHAP dependence plot for a feature.
        
        Args:
            feature: Feature to plot (name or index)
            X: Data
            feature_names: Names of features
            interaction_feature: Feature for interaction coloring
            save_path: Path to save plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values first.")
        
        plt.figure(figsize=(10, 6))
        
        # Create dependence plot
        shap.dependence_plot(
            feature,
            self.shap_values,
            X,
            feature_names=feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            log.info("plot_saved", path=save_path)
        
        plt.show()
    
    def get_decision_plot_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get data for decision plot.
        
        Args:
            X: Data
            feature_names: Names of features
            
        Returns:
            Dictionary with decision plot data
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values first.")
        
        # Calculate cumulative SHAP values
        cumulative_shap = np.cumsum(self.shap_values, axis=1)
        
        # Add expected value
        cumulative_shap = cumulative_shap + self.expected_value
        
        return {
            'cumulative_shap': cumulative_shap,
            'shap_values': self.shap_values,
            'expected_value': self.expected_value,
            'feature_names': feature_names,
            'feature_values': X
        }
    
    def analyze_misclassifications(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze SHAP values for misclassified samples.
        
        Args:
            X: Features
            y_true: True labels
            y_pred: Predicted labels
            feature_names: Names of features
            
        Returns:
            Dictionary with analysis results
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values first.")
        
        # Identify misclassifications
        misclassified = y_true != y_pred
        false_positives = (y_pred == 1) & (y_true == 0)
        false_negatives = (y_pred == 0) & (y_true == 1)
        
        results = {}
        
        # Analyze false positives
        if false_positives.sum() > 0:
            fp_shap = self.shap_values[false_positives]
            fp_importance = np.abs(fp_shap).mean(axis=0)
            
            results['false_positives'] = pd.DataFrame({
                'feature': feature_names or [f"feature_{i}" for i in range(len(fp_importance))],
                'mean_abs_shap': fp_importance,
                'mean_shap': fp_shap.mean(axis=0),
                'std_shap': fp_shap.std(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)
        
        # Analyze false negatives
        if false_negatives.sum() > 0:
            fn_shap = self.shap_values[false_negatives]
            fn_importance = np.abs(fn_shap).mean(axis=0)
            
            results['false_negatives'] = pd.DataFrame({
                'feature': feature_names or [f"feature_{i}" for i in range(len(fn_importance))],
                'mean_abs_shap': fn_importance,
                'mean_shap': fn_shap.mean(axis=0),
                'std_shap': fn_shap.std(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)
        
        # Compare correct vs incorrect
        correct = ~misclassified
        if correct.sum() > 0 and misclassified.sum() > 0:
            correct_shap = self.shap_values[correct]
            incorrect_shap = self.shap_values[misclassified]
            
            results['comparison'] = pd.DataFrame({
                'feature': feature_names or [f"feature_{i}" for i in range(self.shap_values.shape[1])],
                'correct_mean_abs': np.abs(correct_shap).mean(axis=0),
                'incorrect_mean_abs': np.abs(incorrect_shap).mean(axis=0),
                'difference': np.abs(incorrect_shap).mean(axis=0) - np.abs(correct_shap).mean(axis=0)
            }).sort_values('difference', ascending=False)
        
        return results
    
    def save_analysis(self, path: str):
        """
        Save SHAP analysis results.
        
        Args:
            path: Path to save results
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save SHAP values
        if self.shap_values is not None:
            np.save(path / "shap_values.npy", self.shap_values)
        
        # Save expected value
        if self.expected_value is not None:
            np.save(path / "expected_value.npy", self.expected_value)
        
        # Save explainer if possible
        try:
            joblib.dump(self.explainer, path / "explainer.pkl")
        except Exception as e:
            log.warning("explainer_save_failed", error=str(e))
        
        log.info("analysis_saved", path=str(path))
    
    @classmethod
    def load_analysis(cls, path: str, model: Any, model_type: str = "xgboost"):
        """
        Load saved SHAP analysis.
        
        Args:
            path: Path to load from
            model: Model object
            model_type: Type of model
            
        Returns:
            SHAPAnalyzer instance
        """
        path = Path(path)
        
        # Create instance
        analyzer = cls(model, model_type)
        
        # Load SHAP values
        if (path / "shap_values.npy").exists():
            analyzer.shap_values = np.load(path / "shap_values.npy")
        
        # Load expected value
        if (path / "expected_value.npy").exists():
            analyzer.expected_value = np.load(path / "expected_value.npy")
        
        # Load explainer if available
        if (path / "explainer.pkl").exists():
            try:
                analyzer.explainer = joblib.load(path / "explainer.pkl")
            except Exception as e:
                log.warning("explainer_load_failed", error=str(e))
        
        return analyzer


def perform_shap_analysis(
    model: Any,
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    model_type: str = "xgboost",
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive SHAP analysis.
    
    Args:
        model: Trained model
        X_train: Training data for background
        X_test: Test data to explain
        y_test: Test labels (optional)
        feature_names: Names of features
        model_type: Type of model
        save_dir: Directory to save results
        
    Returns:
        Dictionary with analysis results
    """
    log.info("performing_shap_analysis", model_type=model_type)
    
    # Create analyzer
    analyzer = SHAPAnalyzer(model, model_type)
    
    # Create explainer
    analyzer.create_explainer(X_train)
    
    # Calculate SHAP values
    shap_values = analyzer.calculate_shap_values(X_test, check_additivity=True)
    
    # Get feature importance
    importance_df = analyzer.get_feature_importance(feature_names)
    
    # Get interactions (if available)
    interactions_df = analyzer.get_feature_interactions(X_test[:100], feature_names)
    
    results = {
        'analyzer': analyzer,
        'shap_values': shap_values,
        'importance': importance_df,
        'interactions': interactions_df,
        'expected_value': analyzer.expected_value
    }
    
    # Analyze misclassifications if labels provided
    if y_test is not None and hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
        misclass_analysis = analyzer.analyze_misclassifications(
            X_test, y_test, y_pred, feature_names
        )
        results['misclassification_analysis'] = misclass_analysis
    
    # Save results if requested
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save analysis
        analyzer.save_analysis(save_path / "shap_analysis")
        
        # Save importance
        importance_df.to_csv(save_path / "feature_importance.csv", index=False)
        
        # Save interactions
        if not interactions_df.empty:
            interactions_df.to_csv(save_path / "feature_interactions.csv", index=False)
        
        # Create plots
        analyzer.plot_summary(X_test, feature_names, "dot", save_path / "summary_plot.png")
        analyzer.plot_summary(X_test, feature_names, "bar", save_path / "importance_plot.png")
        
        log.info("analysis_saved", path=str(save_path))
    
    return results