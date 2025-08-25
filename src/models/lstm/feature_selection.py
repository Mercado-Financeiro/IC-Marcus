"""
Feature selection optimized for LSTM models.
Focus on temporal patterns and smooth features.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class LSTMFeatureSelector:
    """
    Feature selection specifically optimized for LSTM models.
    
    Key differences from general feature selection:
    - Prefer features with strong autocorrelation
    - Remove high-frequency noise
    - Keep smooth, trending features
    - Consider temporal dependencies
    """
    
    def __init__(self,
                 max_features: int = 50,
                 min_autocorr: float = 0.1,
                 smoothing_window: int = 5,
                 remove_noise: bool = True):
        """
        Initialize LSTM feature selector.
        
        Args:
            max_features: Maximum number of features to select
            min_autocorr: Minimum autocorrelation to keep feature
            smoothing_window: Window for smoothing features
            remove_noise: Whether to remove noisy features
        """
        self.max_features = max_features
        self.min_autocorr = min_autocorr
        self.smoothing_window = smoothing_window
        self.remove_noise = remove_noise
        
        self.selected_features = None
        self.feature_scores = {}
        self.removed_features = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LSTMFeatureSelector':
        """
        Fit feature selector on training data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Self for chaining
        """
        logger.info(f"LSTM feature selection: {len(X.columns)} initial features")
        
        self.removed_features = {
            'low_autocorr': [],
            'high_noise': [],
            'constant': [],
            'low_importance': []
        }
        
        # Start with all features
        remaining_features = list(X.columns)
        
        # Step 1: Remove constant features
        remaining_features = self._remove_constant_features(X, remaining_features)
        
        # Step 2: Calculate temporal characteristics
        temporal_scores = self._calculate_temporal_scores(X[remaining_features])
        
        # Step 3: Remove low autocorrelation features
        if self.min_autocorr > 0:
            remaining_features = self._filter_by_autocorrelation(
                temporal_scores, remaining_features
            )
        
        # Step 4: Remove noisy features
        if self.remove_noise:
            remaining_features = self._filter_noisy_features(
                X[remaining_features], temporal_scores, remaining_features
            )
        
        # Step 5: Rank by importance for LSTM
        feature_importance = self._calculate_lstm_importance(
            X[remaining_features], y
        )
        
        # Step 6: Select top features
        if len(remaining_features) > self.max_features:
            top_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.max_features]
            
            selected = [f for f, _ in top_features]
            removed = [f for f in remaining_features if f not in selected]
            self.removed_features['low_importance'] = removed
            remaining_features = selected
        
        self.selected_features = remaining_features
        self.feature_scores = feature_importance
        
        # Log summary
        logger.info(f"LSTM features selected: {len(self.selected_features)}")
        for reason, features in self.removed_features.items():
            if features:
                logger.info(f"  {reason}: {len(features)} removed")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data with selected features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed DataFrame with selected and smoothed features
        """
        if self.selected_features is None:
            raise ValueError("Selector must be fitted before transform")
        
        # Select features
        X_selected = X[self.selected_features].copy()
        
        # Apply smoothing for LSTM
        if self.smoothing_window > 1:
            X_selected = self._apply_smoothing(X_selected)
        
        return X_selected
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _remove_constant_features(self, X: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove constant or near-constant features."""
        remaining = []
        
        for col in features:
            if X[col].std() < 1e-10 or X[col].nunique() <= 1:
                self.removed_features['constant'].append(col)
            else:
                remaining.append(col)
        
        return remaining
    
    def _calculate_temporal_scores(self, X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate temporal characteristics of features."""
        scores = {}
        
        for col in X.columns:
            series = X[col].values
            
            # Autocorrelation at lag 1 (simple calculation)
            if len(series) > 10:
                try:
                    # Simple autocorrelation calculation
                    series_normalized = (series - np.mean(series)) / (np.std(series) + 1e-10)
                    autocorr = np.corrcoef(series_normalized[:-1], series_normalized[1:])[0, 1]
                except:
                    autocorr = 0
            else:
                autocorr = 0
            
            # Noise-to-signal ratio (using coefficient of variation)
            mean_val = np.mean(series)
            if abs(mean_val) > 1e-10:
                noise_ratio = np.std(series) / abs(mean_val)
            else:
                noise_ratio = float('inf')
            
            # Trend strength (using linear regression)
            x = np.arange(len(series))
            if len(series) > 2:
                slope, _, r_value, _, _ = stats.linregress(x, series)
                trend_strength = abs(r_value)
            else:
                trend_strength = 0
            
            scores[col] = {
                'autocorr': float(autocorr),
                'noise_ratio': float(noise_ratio),
                'trend_strength': float(trend_strength)
            }
        
        return scores
    
    def _filter_by_autocorrelation(self, 
                                  temporal_scores: Dict,
                                  features: List[str]) -> List[str]:
        """Filter features by autocorrelation threshold."""
        remaining = []
        
        for col in features:
            if col in temporal_scores:
                autocorr = temporal_scores[col]['autocorr']
                if abs(autocorr) >= self.min_autocorr:
                    remaining.append(col)
                else:
                    self.removed_features['low_autocorr'].append(col)
            else:
                remaining.append(col)
        
        return remaining
    
    def _filter_noisy_features(self,
                              X: pd.DataFrame,
                              temporal_scores: Dict,
                              features: List[str]) -> List[str]:
        """Remove features with high noise."""
        remaining = []
        noise_threshold = 5.0  # Coefficient of variation threshold
        
        for col in features:
            if col in temporal_scores:
                noise_ratio = temporal_scores[col]['noise_ratio']
                
                # Also check for extreme outliers
                z_scores = np.abs(stats.zscore(X[col].dropna()))
                outlier_ratio = (z_scores > 5).sum() / len(z_scores)
                
                if noise_ratio < noise_threshold and outlier_ratio < 0.01:
                    remaining.append(col)
                else:
                    self.removed_features['high_noise'].append(col)
            else:
                remaining.append(col)
        
        return remaining
    
    def _calculate_lstm_importance(self,
                                  X: pd.DataFrame,
                                  y: pd.Series) -> Dict[str, float]:
        """
        Calculate feature importance for LSTM.
        
        Uses correlation with target and temporal characteristics.
        """
        importance = {}
        
        for col in X.columns:
            # Correlation with target
            try:
                corr_with_target = abs(X[col].corr(y))
            except:
                corr_with_target = 0
            
            # Correlation with lagged target (predictive power)
            try:
                lagged_corr = abs(X[col].iloc[:-1].corr(y.iloc[1:]))
            except:
                lagged_corr = 0
            
            # Combined score
            importance[col] = (corr_with_target + lagged_corr) / 2
        
        return importance
    
    def _apply_smoothing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing to reduce noise for LSTM."""
        X_smooth = X.copy()
        
        for col in X.columns:
            # Use exponential weighted moving average
            X_smooth[col] = X[col].ewm(
                span=self.smoothing_window,
                min_periods=1
            ).mean()
        
        return X_smooth
    
    def get_feature_report(self) -> pd.DataFrame:
        """
        Get detailed feature selection report.
        
        Returns:
            DataFrame with feature scores and selection status
        """
        if not self.feature_scores:
            return pd.DataFrame()
        
        report_data = []
        for feature, score in self.feature_scores.items():
            report_data.append({
                'feature': feature,
                'importance_score': score,
                'selected': feature in self.selected_features
            })
        
        # Add removed features
        for reason, features in self.removed_features.items():
            for feature in features:
                if feature not in [r['feature'] for r in report_data]:
                    report_data.append({
                        'feature': feature,
                        'importance_score': 0,
                        'selected': False,
                        'removal_reason': reason
                    })
        
        df = pd.DataFrame(report_data)
        return df.sort_values('importance_score', ascending=False)


def select_lstm_features(X: pd.DataFrame, 
                        y: pd.Series,
                        max_features: int = 50) -> Tuple[pd.DataFrame, List[str]]:
    """
    Quick function to select features for LSTM.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        max_features: Maximum number of features
        
    Returns:
        Tuple of (selected_features_df, feature_names)
    """
    selector = LSTMFeatureSelector(
        max_features=max_features,
        min_autocorr=0.1,
        smoothing_window=5,
        remove_noise=True
    )
    
    X_selected = selector.fit_transform(X, y)
    
    logger.info(f"LSTM features: {len(X.columns)} → {len(X_selected.columns)}")
    
    return X_selected, selector.selected_features