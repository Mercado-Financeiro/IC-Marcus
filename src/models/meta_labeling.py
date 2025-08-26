"""
Meta-Labeling: Second Layer Filter for Trading Decisions

Meta-labeling is a technique where a secondary model decides whether
to take a trade or not, given that the primary model has generated a signal.

This helps filter out false positives and size positions based on confidence.

References:
- López de Prado: "Meta-Labeling" in AFML
- Hudson & Thames: Meta-labeling implementations
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings

from src.data.splits import PurgedKFold
from src.models.threshold_optimizer import ThresholdOptimizer, TradingCosts
from src.utils.logging import log as logger


@dataclass
class MetaLabelConfig:
    """Configuration for meta-labeling."""
    
    # Model configuration
    base_estimator: str = 'random_forest'  # 'random_forest', 'xgboost', 'lightgbm'
    n_estimators: int = 100
    max_depth: int = 5
    
    # Features to use
    use_probability: bool = True  # Use primary model probability
    use_market_features: bool = True  # Volume, volatility, etc.
    use_technical_features: bool = True  # RSI, MACD, etc.
    use_regime_features: bool = True  # Market regime indicators
    
    # Validation
    cv_splits: int = 5
    embargo_bars: int = 40
    
    # Optimization target
    optimize_for: str = 'sharpe'  # 'sharpe', 'precision', 'profit_factor'
    
    # Position sizing
    enable_position_sizing: bool = True
    min_position_size: float = 0.1
    max_position_size: float = 1.0


class MetaLabeler(BaseEstimator, ClassifierMixin):
    """
    Meta-labeling model for trade filtering and position sizing.
    
    The meta-labeler takes the primary model's predictions and decides:
    1. Whether to take the trade (binary classification)
    2. What size to trade (regression for position sizing)
    
    This is particularly useful for:
    - Filtering out trades in adverse conditions
    - Reducing false positives
    - Dynamic position sizing based on confidence
    """
    
    def __init__(self, config: Optional[MetaLabelConfig] = None):
        """
        Initialize meta-labeler.
        
        Args:
            config: Configuration object
        """
        self.config = config or MetaLabelConfig()
        self.model = None
        self.feature_importance_ = None
        self.is_fitted = False
        self.threshold_optimizer = ThresholdOptimizer()
        
    def create_meta_features(
        self,
        X: pd.DataFrame,
        primary_proba: np.ndarray,
        market_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create features for meta-labeling.
        
        Args:
            X: Original features
            primary_proba: Probabilities from primary model
            market_data: Additional market data (volume, volatility, etc.)
            
        Returns:
            DataFrame with meta-features
        """
        meta_features = pd.DataFrame(index=X.index)
        
        # 1. Primary model confidence
        if self.config.use_probability:
            meta_features['primary_proba'] = primary_proba
            meta_features['primary_confidence'] = np.abs(primary_proba - 0.5) * 2
            meta_features['primary_entropy'] = -(
                primary_proba * np.log(primary_proba + 1e-10) +
                (1 - primary_proba) * np.log(1 - primary_proba + 1e-10)
            )
        
        # 2. Market microstructure features
        if self.config.use_market_features and market_data is not None:
            if 'volume' in market_data.columns:
                meta_features['volume_ratio'] = (
                    market_data['volume'] / market_data['volume'].rolling(20).mean()
                ).fillna(1)
            
            if 'volatility' in market_data.columns:
                meta_features['volatility'] = market_data['volatility']
                meta_features['volatility_regime'] = (
                    market_data['volatility'] / market_data['volatility'].rolling(60).mean()
                ).fillna(1)
            
            if 'spread' in market_data.columns:
                meta_features['spread_bps'] = market_data['spread'] * 10000
            
            if 'close' in market_data.columns:
                # Price momentum
                meta_features['momentum_5'] = market_data['close'].pct_change(5).fillna(0)
                meta_features['momentum_20'] = market_data['close'].pct_change(20).fillna(0)
        
        # 3. Technical indicators
        if self.config.use_technical_features and market_data is not None:
            if 'close' in market_data.columns:
                # RSI
                delta = market_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-10)
                meta_features['rsi'] = 100 - (100 / (1 + rs))
                
                # Distance from moving averages
                ma_20 = market_data['close'].rolling(20).mean()
                ma_50 = market_data['close'].rolling(50).mean()
                meta_features['dist_ma20'] = (market_data['close'] - ma_20) / ma_20
                meta_features['dist_ma50'] = (market_data['close'] - ma_50) / ma_50
        
        # 4. Regime features
        if self.config.use_regime_features and market_data is not None:
            if 'returns' in market_data.columns:
                # Rolling Sharpe
                rolling_returns = market_data['returns'].rolling(60)
                meta_features['rolling_sharpe'] = (
                    rolling_returns.mean() / rolling_returns.std() * np.sqrt(252)
                ).fillna(0)
                
                # Trend strength
                cumsum = market_data['returns'].rolling(20).sum()
                meta_features['trend_strength'] = cumsum / (
                    market_data['returns'].rolling(20).std() * np.sqrt(20) + 1e-10
                )
        
        # 5. Time features
        if hasattr(X.index, 'hour'):
            meta_features['hour'] = X.index.hour
            meta_features['day_of_week'] = X.index.dayofweek
        
        # Fill NaN values
        meta_features = meta_features.fillna(0)
        
        return meta_features
    
    def create_meta_labels(
        self,
        primary_signals: np.ndarray,
        actual_returns: np.ndarray,
        costs: Optional[TradingCosts] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create labels for meta-labeling.
        
        Meta-labels are 1 if the trade was profitable (after costs), 0 otherwise.
        We only label points where the primary model gave a signal.
        
        Args:
            primary_signals: Binary signals from primary model
            actual_returns: Actual returns that occurred
            costs: Trading costs to account for
            
        Returns:
            (meta_labels, meta_mask) where mask indicates which samples to use
        """
        costs = costs or TradingCosts()
        cost_pct = costs.total_pct
        
        # Only consider samples where primary model signaled
        meta_mask = primary_signals == 1
        
        # Meta-label is 1 if the trade would have been profitable
        meta_labels = np.zeros_like(actual_returns)
        meta_labels[meta_mask] = (actual_returns[meta_mask] > cost_pct).astype(int)
        
        return meta_labels, meta_mask
    
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        primary_proba: np.ndarray,
        market_data: Optional[pd.DataFrame] = None,
        actual_returns: Optional[np.ndarray] = None
    ):
        """
        Train meta-labeling model.
        
        Args:
            X: Original features
            y: Original labels (or actual returns for meta-label creation)
            primary_proba: Probabilities from primary model
            market_data: Additional market data
            actual_returns: Actual returns for creating meta-labels
        """
        logger.info("Training meta-labeling model")
        
        # Create meta-features
        meta_X = self.create_meta_features(X, primary_proba, market_data)
        
        # Create meta-labels
        if actual_returns is not None:
            # Create meta-labels from actual returns
            primary_signals = (primary_proba > 0.5).astype(int)
            meta_y, meta_mask = self.create_meta_labels(
                primary_signals, actual_returns
            )
            
            # Filter to only samples with primary signals
            meta_X_filtered = meta_X[meta_mask]
            meta_y_filtered = meta_y[meta_mask]
        else:
            # Use provided labels
            meta_X_filtered = meta_X
            meta_y_filtered = y
        
        # Select and train model
        if self.config.base_estimator == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42,
                n_jobs=-1
            )
        else:
            # Add other models as needed
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42
            )
        
        # Cross-validation
        cv = PurgedKFold(
            n_splits=self.config.cv_splits,
            embargo=self.config.embargo_bars
        )
        
        cv_scores = cross_val_score(
            self.model, meta_X_filtered, meta_y_filtered,
            cv=cv, scoring='precision'
        )
        
        logger.info(
            "Meta-model CV scores",
            mean_precision=f"{np.mean(cv_scores):.3f}",
            std_precision=f"{np.std(cv_scores):.3f}"
        )
        
        # Fit final model
        self.model.fit(meta_X_filtered, meta_y_filtered)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': meta_X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        self.is_fitted = True
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        primary_proba: np.ndarray,
        market_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Predict whether to take trades.
        
        Args:
            X: Original features
            primary_proba: Probabilities from primary model
            market_data: Additional market data
            
        Returns:
            Binary predictions (1 = take trade, 0 = skip)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Create meta-features
        meta_X = self.create_meta_features(X, primary_proba, market_data)
        
        # Get meta-predictions
        meta_predictions = self.model.predict(meta_X)
        
        # Only trade when both primary and meta models agree
        primary_signals = (primary_proba > 0.5).astype(int)
        final_signals = primary_signals * meta_predictions
        
        return final_signals
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        primary_proba: np.ndarray,
        market_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Get meta-model probabilities.
        
        Args:
            X: Original features
            primary_proba: Probabilities from primary model
            market_data: Additional market data
            
        Returns:
            Meta-model probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Create meta-features
        meta_X = self.create_meta_features(X, primary_proba, market_data)
        
        # Get meta-probabilities
        meta_proba = self.model.predict_proba(meta_X)[:, 1]
        
        return meta_proba
    
    def get_position_sizes(
        self,
        X: pd.DataFrame,
        primary_proba: np.ndarray,
        market_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Calculate position sizes based on meta-model confidence.
        
        Args:
            X: Original features
            primary_proba: Probabilities from primary model
            market_data: Additional market data
            
        Returns:
            Position sizes [0, 1]
        """
        if not self.config.enable_position_sizing:
            # Return binary sizes
            return self.predict(X, primary_proba, market_data).astype(float)
        
        # Get meta probabilities
        meta_proba = self.predict_proba(X, primary_proba, market_data)
        
        # Get primary signals
        primary_signals = (primary_proba > 0.5).astype(int)
        
        # Position size based on meta confidence
        # Only size positions where primary model signals
        position_sizes = np.zeros(len(X))
        
        mask = primary_signals == 1
        if np.any(mask):
            # Linear scaling based on meta probability
            # Could also use Kelly criterion or other sizing methods
            sizes = meta_proba[mask]
            
            # Scale to [min_size, max_size]
            sizes = (
                self.config.min_position_size +
                (self.config.max_position_size - self.config.min_position_size) * sizes
            )
            
            # Apply threshold (only trade if meta confidence > 0.5)
            sizes[meta_proba[mask] < 0.5] = 0
            
            position_sizes[mask] = sizes
        
        return position_sizes
    
    def analyze_filtering_impact(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        primary_proba: np.ndarray,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Analyze the impact of meta-labeling filter.
        
        Args:
            X: Features
            y: True labels
            primary_proba: Primary model probabilities
            market_data: Market data
            
        Returns:
            Analysis metrics
        """
        # Get predictions
        primary_signals = (primary_proba > 0.5).astype(int)
        meta_signals = self.predict(X, primary_proba, market_data)
        
        # Calculate metrics
        primary_tp = np.sum((primary_signals == 1) & (y == 1))
        primary_fp = np.sum((primary_signals == 1) & (y == 0))
        primary_precision = primary_tp / (primary_tp + primary_fp) if (primary_tp + primary_fp) > 0 else 0
        
        meta_tp = np.sum((meta_signals == 1) & (y == 1))
        meta_fp = np.sum((meta_signals == 1) & (y == 0))
        meta_precision = meta_tp / (meta_tp + meta_fp) if (meta_tp + meta_fp) > 0 else 0
        
        analysis = {
            'primary': {
                'n_signals': np.sum(primary_signals),
                'precision': primary_precision,
                'true_positives': primary_tp,
                'false_positives': primary_fp
            },
            'with_meta': {
                'n_signals': np.sum(meta_signals),
                'precision': meta_precision,
                'true_positives': meta_tp,
                'false_positives': meta_fp
            },
            'reduction': {
                'signals_filtered': np.sum(primary_signals) - np.sum(meta_signals),
                'false_positives_removed': primary_fp - meta_fp,
                'precision_improvement': meta_precision - primary_precision
            },
            'filtering_rate': 1 - (np.sum(meta_signals) / np.sum(primary_signals)) if np.sum(primary_signals) > 0 else 0
        }
        
        logger.info(
            "Meta-labeling impact",
            signals_before=analysis['primary']['n_signals'],
            signals_after=analysis['with_meta']['n_signals'],
            precision_before=f"{analysis['primary']['precision']:.2%}",
            precision_after=f"{analysis['with_meta']['precision']:.2%}",
            filtering_rate=f"{analysis['filtering_rate']:.2%}"
        )
        
        return analysis


def create_meta_labeling_pipeline(
    primary_model,
    config: Optional[MetaLabelConfig] = None
) -> Dict:
    """
    Create a complete meta-labeling pipeline.
    
    Args:
        primary_model: Trained primary model
        config: Meta-labeling configuration
        
    Returns:
        Pipeline components
    """
    config = config or MetaLabelConfig()
    
    pipeline = {
        'primary_model': primary_model,
        'meta_labeler': MetaLabeler(config),
        'threshold_optimizer': ThresholdOptimizer(),
        'config': config
    }
    
    logger.info(
        "Meta-labeling pipeline created",
        primary_model_type=type(primary_model).__name__,
        meta_model_type=config.base_estimator,
        optimize_for=config.optimize_for
    )
    
    return pipeline