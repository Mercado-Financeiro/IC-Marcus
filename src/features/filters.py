"""
Efficient feature filtering pipeline.
Removes redundant, constant, and highly correlated features.
Fast, simple, and effective.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Iterator
import logging
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from .circuit_breaker import FeatureProcessingBreaker, circuit_breaker
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


class FeatureFilter:
    """
    Multi-stage feature filtering to reduce dimensionality efficiently.
    
    Stages:
    1. Remove constant/near-zero variance features
    2. Remove highly correlated features using clustering
    3. Remove features with low mutual information
    4. Optional: VIF filtering for multicollinearity
    """
    
    def __init__(self,
                 variance_threshold: float = 0.01,
                 correlation_threshold: float = 0.90,  # More aggressive correlation filtering
                 mutual_info_threshold: float = 0.01,
                 vif_threshold: float = 10.0,
                 use_vif: bool = False,
                 remove_zombie_features: bool = True,
                 chunk_size: int = 1000,
                 use_chunks: bool = True,
                 use_circuit_breaker: bool = True):
        """
        Initialize feature filter.
        
        Args:
            variance_threshold: Minimum variance to keep feature
            correlation_threshold: Maximum correlation between features
            mutual_info_threshold: Minimum mutual information with target
            vif_threshold: Maximum VIF score (if use_vif=True)
            use_vif: Whether to use VIF filtering (slower)
        """
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.mutual_info_threshold = mutual_info_threshold
        self.vif_threshold = vif_threshold
        self.use_vif = use_vif
        self.remove_zombie_features = remove_zombie_features
        self.chunk_size = chunk_size
        self.use_chunks = use_chunks
        self.use_circuit_breaker = use_circuit_breaker
        
        # Circuit breaker for resilience
        self.breaker = FeatureProcessingBreaker() if use_circuit_breaker else None
        
        # Fitted state
        self.selected_features = None
        self.feature_scores = {}
        self.removed_features = {}
        self.zombie_features = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureFilter':
        """
        Fit the filter on training data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Self for chaining
        """
        logger.info(f"Starting feature filtering: {len(X.columns)} features")
        
        # Track removed features
        self.removed_features = {
            'constant': [],
            'low_variance': [],
            'high_correlation': [],
            'low_mutual_info': [],
            'high_vif': [],
            'zombie': []
        }
        self.zombie_features = []
        
        # Start with all features
        remaining_features = list(X.columns)
        X_filtered = X.copy()
        
        # Stage 1: Remove constant features
        remaining_features = self._filter_constant_features(X_filtered, remaining_features)
        X_filtered = X_filtered[remaining_features]
        
        # Stage 2: Remove low variance features
        remaining_features = self._filter_low_variance(X_filtered, remaining_features)
        X_filtered = X_filtered[remaining_features]
        
        # Stage 3: Remove highly correlated features with circuit breaker
        if self.use_circuit_breaker and self.breaker:
            def correlation_filter():
                return self._filter_high_correlation(X_filtered, remaining_features)
            
            def correlation_fallback():
                logger.warning("Correlation filter failed, using simple fallback")
                # Simple fallback: keep features with low pairwise correlation
                return self._simple_correlation_filter(X_filtered, remaining_features)
            
            remaining_features = self.breaker.protected_call(
                'correlation', correlation_filter, correlation_fallback
            )
        else:
            remaining_features = self._filter_high_correlation(X_filtered, remaining_features)
        X_filtered = X_filtered[remaining_features]
        
        # Stage 4: Remove low mutual information features with circuit breaker
        if self.use_circuit_breaker and self.breaker:
            def mi_filter():
                return self._filter_low_mutual_info(X_filtered, y, remaining_features)
            
            def mi_fallback():
                logger.warning("MI filter failed, keeping all features")
                return remaining_features
            
            remaining_features = self.breaker.protected_call(
                'mutual_info', mi_filter, mi_fallback
            )
        else:
            remaining_features = self._filter_low_mutual_info(X_filtered, y, remaining_features)
        X_filtered = X_filtered[remaining_features]
        
        # Stage 5: Optional VIF filtering
        if self.use_vif and len(remaining_features) > 1:
            remaining_features = self._filter_high_vif(X_filtered, remaining_features)
        
        # Stage 6: Remove zombie features (calendar-based and others with no predictive power)
        if self.remove_zombie_features:
            remaining_features = self._filter_zombie_features(X_filtered, y, remaining_features)
        
        self.selected_features = remaining_features
        
        # Log summary
        total_removed = len(X.columns) - len(self.selected_features)
        logger.info(f"Feature filtering complete: {len(self.selected_features)} features retained, "
                   f"{total_removed} removed")
        
        for stage, features in self.removed_features.items():
            if features:
                logger.info(f"  {stage}: {len(features)} features removed")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted filter.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Filtered DataFrame
        """
        if self.selected_features is None:
            raise ValueError("Filter must be fitted before transform")
        
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Filtered DataFrame
        """
        return self.fit(X, y).transform(X)
    
    def _filter_constant_features(self, X: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove constant features."""
        remaining = []
        
        for col in features:
            if X[col].nunique() <= 1:
                self.removed_features['constant'].append(col)
            else:
                remaining.append(col)
        
        return remaining
    
    def _filter_low_variance(self, X: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove low variance features."""
        if not features:
            return features
        
        # Use sklearn's VarianceThreshold
        selector = VarianceThreshold(threshold=self.variance_threshold)
        
        try:
            selector.fit(X[features])
            mask = selector.get_support()
            
            remaining = []
            for i, col in enumerate(features):
                if mask[i]:
                    remaining.append(col)
                else:
                    self.removed_features['low_variance'].append(col)
            
            return remaining
        except Exception as e:
            logger.warning(f"Variance filtering failed: {e}")
            return features
    
    def _filter_high_correlation(self, X: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove highly correlated features using optimized NumPy operations."""
        if len(features) <= 1:
            return features
        
        try:
            # Convert to NumPy array for faster computation
            X_np = X[features].values
            n_features = X_np.shape[1]
            
            # Standardize features for correlation calculation
            X_centered = X_np - np.nanmean(X_np, axis=0)
            X_std = np.nanstd(X_centered, axis=0)
            X_std[X_std == 0] = 1  # Avoid division by zero
            X_normalized = X_centered / X_std
            
            # Fast correlation computation using NumPy
            # Replace NaN with 0 for correlation calculation
            X_normalized = np.nan_to_num(X_normalized, nan=0.0)
            
            # Vectorized correlation calculation
            corr_matrix = np.abs(np.corrcoef(X_normalized.T))
            np.fill_diagonal(corr_matrix, 0)  # Remove self-correlation
            
            # Find highly correlated pairs efficiently
            removed_indices = set()
            variances = np.var(X_np, axis=0)
            
            # Process in order of variance (keep high variance features)
            variance_order = np.argsort(variances)[::-1]
            
            for i in variance_order:
                if i in removed_indices:
                    continue
                    
                # Find features correlated with current feature
                correlated = np.where(corr_matrix[i] > self.correlation_threshold)[0]
                
                for j in correlated:
                    if j not in removed_indices and j != i:
                        # Remove the feature with lower variance
                        if variances[j] <= variances[i]:
                            removed_indices.add(j)
                            self.removed_features['high_correlation'].append(features[j])
                        else:
                            removed_indices.add(i)
                            self.removed_features['high_correlation'].append(features[i])
                            break
            
            # Return features not removed
            remaining = [features[i] for i in range(n_features) if i not in removed_indices]
            
            if not remaining:
                # Edge case: all features removed, keep the one with highest variance
                best_idx = np.argmax(variances)
                remaining = [features[best_idx]]
                logger.warning("All features highly correlated, keeping highest variance feature")
            
            return remaining
            
        except Exception as e:
            logger.warning(f"Optimized correlation filtering failed: {e}. Falling back to simple method.")
            # Simple fallback: remove features with correlation > threshold
            try:
                corr_matrix = X[features].corr().abs()
                upper_tri = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                to_drop = [column for column in upper_tri.columns 
                          if any(upper_tri[column] > self.correlation_threshold)]
                self.removed_features['high_correlation'].extend(to_drop)
                remaining = [f for f in features if f not in to_drop]
                return remaining if remaining else features[:1]
            except:
                return features
    
    def _filter_low_mutual_info(self, X: pd.DataFrame, y: pd.Series, 
                                features: List[str]) -> List[str]:
        """Remove features with low mutual information using chunked processing."""
        if not features:
            return features
        
        try:
            mi_scores = None
            
            if self.use_chunks and len(features) > self.chunk_size:
                # Process features in chunks to reduce memory usage
                mi_scores = np.zeros(len(features))
                
                for chunk_start in range(0, len(features), self.chunk_size):
                    chunk_end = min(chunk_start + self.chunk_size, len(features))
                    chunk_features = features[chunk_start:chunk_end]
                    
                    # Process chunk
                    X_chunk = X[chunk_features].copy()
                    
                    # Fill NaN with median for MI calculation
                    for col in X_chunk.columns:
                        if X_chunk[col].isna().any():
                            median_val = X_chunk[col].median()
                            X_chunk[col].fillna(median_val, inplace=True)
                    
                    # Calculate mutual information for chunk
                    chunk_scores = mutual_info_classif(X_chunk, y, random_state=42)
                    mi_scores[chunk_start:chunk_end] = chunk_scores
                    
                    # Clear memory
                    del X_chunk
                    
            else:
                # Standard processing for small feature sets
                X_for_mi = X[features].copy()
                
                # Fill NaN with median for MI calculation
                for col in X_for_mi.columns:
                    if X_for_mi[col].isna().any():
                        X_for_mi[col].fillna(X_for_mi[col].median(), inplace=True)
                
                # Calculate mutual information
                mi_scores = mutual_info_classif(X_for_mi, y, random_state=42)
            
            # Store scores
            self.feature_scores['mutual_info'] = dict(zip(features, mi_scores))
            
        except Exception as e:
            logger.warning(f"Mutual information calculation failed: {e}. Skipping MI filter.")
            return features
        
        # Filter by threshold
        remaining = []
        for feature, score in zip(features, mi_scores):
            if score >= self.mutual_info_threshold:
                remaining.append(feature)
            else:
                self.removed_features['low_mutual_info'].append(feature)
        
        return remaining
    
    def _filter_high_vif(self, X: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove features with high VIF (multicollinearity)."""
        if len(features) <= 1:
            return features
        
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        # Standardize for VIF calculation
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X[features]),
            columns=features,
            index=X.index
        )
        
        # Iteratively remove high VIF features
        remaining = features.copy()
        removed_any = True
        
        while removed_any and len(remaining) > 1:
            removed_any = False
            vif_data = pd.DataFrame()
            vif_data["feature"] = remaining
            vif_data["VIF"] = [
                variance_inflation_factor(X_scaled[remaining].values, i)
                for i in range(len(remaining))
            ]
            
            # Find feature with highest VIF
            max_vif_idx = vif_data["VIF"].idxmax()
            max_vif = vif_data.loc[max_vif_idx, "VIF"]
            
            if max_vif > self.vif_threshold:
                feature_to_remove = vif_data.loc[max_vif_idx, "feature"]
                remaining.remove(feature_to_remove)
                self.removed_features['high_vif'].append(feature_to_remove)
                removed_any = True
        
        return remaining
    
    def _filter_zombie_features(self, X: pd.DataFrame, y: pd.Series,
                               features: List[str]) -> List[str]:
        """Remove zombie features with no predictive power.
        
        Zombie features include:
        - Calendar features with constant patterns
        - Features with extremely low variance relative to their scale
        - Features that are deterministic functions of time
        """
        if not features:
            return features
        
        # Calculate mutual information if not already done
        if 'mutual_info' not in self.feature_scores and len(features) > 0:
            try:
                from sklearn.feature_selection import mutual_info_classif
                # Handle NaN for MI calculation
                X_for_mi = X[features].copy()
                for col in X_for_mi.columns:
                    if X_for_mi[col].isna().any():
                        X_for_mi[col].fillna(X_for_mi[col].median(), inplace=True)
                
                mi_scores = mutual_info_classif(X_for_mi, y, random_state=42)
                self.feature_scores['mutual_info'] = dict(zip(features, mi_scores))
            except Exception as e:
                logger.debug(f"Could not calculate MI for zombie detection: {e}")
        
        remaining = []
        zombie_patterns = ['month_', 'weekday_', 'hour_', 'quarter_', 'dayofyear_', 'zombie_', 'constant_', 'lowvar_']
        
        for feature in features:
            # Check for calendar-based zombie features
            is_zombie = False
            
            # Pattern 1: Calendar/known zombie features with low variance
            for pattern in zombie_patterns:
                if pattern in feature.lower():
                    # Check if it has very low predictive power
                    if feature in self.feature_scores.get('mutual_info', {}):
                        mi_score = self.feature_scores['mutual_info'][feature]
                        if mi_score < 0.005:  # Very low MI threshold for calendar features
                            is_zombie = True
                            self.zombie_features.append(feature)
                            self.removed_features['zombie'].append(feature)
                            logger.debug(f"Removed zombie calendar feature: {feature} (MI={mi_score:.4f})")
                            break
            
            # Pattern 2: Features with near-zero variance relative to mean
            if not is_zombie:
                col_data = X[feature]
                if col_data.mean() != 0:
                    cv = col_data.std() / abs(col_data.mean())  # Coefficient of variation
                    if cv < 0.001:  # Extremely low relative variance
                        is_zombie = True
                        self.zombie_features.append(feature)
                        self.removed_features['zombie'].append(feature)
                        logger.debug(f"Removed zombie low-CV feature: {feature} (CV={cv:.6f})")
            
            # Pattern 3: Features that are perfect linear functions of index
            if not is_zombie and len(X) > 100:
                # Check correlation with index position
                index_corr = abs(pd.Series(range(len(X)), index=X.index).corr(col_data))
                if index_corr > 0.999:  # Nearly perfect correlation with time
                    is_zombie = True
                    self.zombie_features.append(feature)
                    self.removed_features['zombie'].append(feature)
                    logger.debug(f"Removed zombie time-dependent feature: {feature} (corr={index_corr:.4f})")
            
            if not is_zombie:
                remaining.append(feature)
        
        if self.zombie_features:
            logger.info(f"Removed {len(self.zombie_features)} zombie features")
        
        return remaining
    
    def _simple_correlation_filter(self, X: pd.DataFrame, features: List[str]) -> List[str]:
        """Simple fallback correlation filter for circuit breaker."""
        if len(features) <= 1:
            return features
        
        try:
            # Quick correlation check on subset
            sample_size = min(1000, len(X))
            X_sample = X[features].sample(n=sample_size, random_state=42)
            
            corr_matrix = X_sample.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = set()
            for column in upper_tri.columns:
                if any(upper_tri[column] > self.correlation_threshold):
                    to_drop.add(column)
            
            remaining = [f for f in features if f not in to_drop]
            return remaining if remaining else features[:max(1, len(features)//2)]
            
        except Exception as e:
            logger.error(f"Even simple correlation filter failed: {e}")
            # Ultra-fallback: keep half the features
            return features[:max(1, len(features)//2)]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature scores
        """
        if not self.feature_scores:
            return pd.DataFrame()
        
        # Combine all scores
        all_scores = []
        for metric, scores in self.feature_scores.items():
            for feature, score in scores.items():
                if feature in self.selected_features:
                    all_scores.append({
                        'feature': feature,
                        'metric': metric,
                        'score': score
                    })
        
        if not all_scores:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_scores)
        
        # Pivot to have metrics as columns
        df_pivot = df.pivot(index='feature', columns='metric', values='score')
        
        # Sort by mutual information if available
        if 'mutual_info' in df_pivot.columns:
            df_pivot = df_pivot.sort_values('mutual_info', ascending=False)
        
        return df_pivot
    
    def get_summary(self) -> Dict:
        """
        Get filtering summary.
        
        Returns:
            Dictionary with filtering statistics
        """
        return {
            'n_features_selected': len(self.selected_features) if self.selected_features else 0,
            'n_features_removed': sum(len(v) for v in self.removed_features.values()),
            'removal_breakdown': {k: len(v) for k, v in self.removed_features.items()},
            'selected_features': self.selected_features,
            'thresholds': {
                'variance': self.variance_threshold,
                'correlation': self.correlation_threshold,
                'mutual_info': self.mutual_info_threshold,
                'vif': self.vif_threshold if self.use_vif else None
            },
            'processing': {
                'use_chunks': self.use_chunks,
                'chunk_size': self.chunk_size,
                'use_circuit_breaker': self.use_circuit_breaker
            },
            'circuit_breaker_stats': self.breaker.get_all_stats() if self.breaker else None
        }


def quick_feature_filter(X: pd.DataFrame, y: pd.Series,
                        max_features: int = 100,
                        aggressive: bool = True) -> List[str]:
    """
    Quick feature filtering for immediate use.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        max_features: Maximum number of features to keep
        
    Returns:
        List of selected feature names
    """
    filter = FeatureFilter(
        variance_threshold=0.01 if not aggressive else 0.02,
        correlation_threshold=0.95 if not aggressive else 0.90,
        mutual_info_threshold=0.001 if not aggressive else 0.005,
        use_vif=False,  # Skip VIF for speed
        remove_zombie_features=aggressive
    )
    
    filter.fit(X, y)
    selected = filter.selected_features
    
    # If still too many features, keep top by mutual information
    if len(selected) > max_features:
        importance = filter.get_feature_importance()
        if not importance.empty and 'mutual_info' in importance.columns:
            selected = importance.nlargest(max_features, 'mutual_info').index.tolist()
    
    logger.info(f"Quick filter: {len(X.columns)} → {len(selected)} features")
    
    return selected