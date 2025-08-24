"""
Efficient feature filtering pipeline.
Removes redundant, constant, and highly correlated features.
Fast, simple, and effective.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
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
                 correlation_threshold: float = 0.95,
                 mutual_info_threshold: float = 0.01,
                 vif_threshold: float = 10.0,
                 use_vif: bool = False):
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
        
        # Fitted state
        self.selected_features = None
        self.feature_scores = {}
        self.removed_features = {}
        
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
            'high_vif': []
        }
        
        # Start with all features
        remaining_features = list(X.columns)
        X_filtered = X.copy()
        
        # Stage 1: Remove constant features
        remaining_features = self._filter_constant_features(X_filtered, remaining_features)
        X_filtered = X_filtered[remaining_features]
        
        # Stage 2: Remove low variance features
        remaining_features = self._filter_low_variance(X_filtered, remaining_features)
        X_filtered = X_filtered[remaining_features]
        
        # Stage 3: Remove highly correlated features
        remaining_features = self._filter_high_correlation(X_filtered, remaining_features)
        X_filtered = X_filtered[remaining_features]
        
        # Stage 4: Remove low mutual information features
        remaining_features = self._filter_low_mutual_info(X_filtered, y, remaining_features)
        X_filtered = X_filtered[remaining_features]
        
        # Stage 5: Optional VIF filtering
        if self.use_vif and len(remaining_features) > 1:
            remaining_features = self._filter_high_vif(X_filtered, remaining_features)
        
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
        """Remove highly correlated features using hierarchical clustering."""
        if len(features) <= 1:
            return features
        
        # Calculate correlation matrix
        corr_matrix = X[features].corr().abs()
        
        # Convert to distance matrix
        distance_matrix = 1 - corr_matrix
        condensed_distances = squareform(distance_matrix)
        
        # Hierarchical clustering
        linkage = hierarchy.linkage(condensed_distances, method='average')
        
        # Get clusters at correlation threshold
        cluster_ids = hierarchy.fcluster(linkage, 1 - self.correlation_threshold, 
                                        criterion='distance')
        
        # Keep one feature from each cluster (the one with highest variance)
        cluster_representatives = {}
        for i, cluster_id in enumerate(cluster_ids):
            feature = features[i]
            variance = X[feature].var()
            
            if cluster_id not in cluster_representatives:
                cluster_representatives[cluster_id] = (feature, variance)
            else:
                if variance > cluster_representatives[cluster_id][1]:
                    # Remove old representative
                    self.removed_features['high_correlation'].append(
                        cluster_representatives[cluster_id][0]
                    )
                    cluster_representatives[cluster_id] = (feature, variance)
                else:
                    # Remove this feature
                    self.removed_features['high_correlation'].append(feature)
        
        return [feat for feat, _ in cluster_representatives.values()]
    
    def _filter_low_mutual_info(self, X: pd.DataFrame, y: pd.Series, 
                                features: List[str]) -> List[str]:
        """Remove features with low mutual information."""
        if not features:
            return features
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X[features], y, random_state=42)
        
        # Store scores
        self.feature_scores['mutual_info'] = dict(zip(features, mi_scores))
        
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
            }
        }


def quick_feature_filter(X: pd.DataFrame, y: pd.Series,
                        max_features: int = 100) -> List[str]:
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
        variance_threshold=0.01,
        correlation_threshold=0.95,
        mutual_info_threshold=0.001,
        use_vif=False  # Skip VIF for speed
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