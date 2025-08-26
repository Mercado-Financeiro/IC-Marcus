"""
Enterprise Feature Store - Main API and orchestration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from .versioning import VersionManager, VersionStatus, VersionInfo
from .metadata import MetadataStore, FeatureMetadata, LineageType
from ..validation.data_quality import DataQualityValidator
from ..validation.exceptions import FeatureValidationError

logger = logging.getLogger(__name__)


@dataclass
class FeatureStoreConfig:
    """Feature store configuration."""
    store_path: str = "data/feature_store"
    auto_version: bool = True
    validate_on_write: bool = True
    track_lineage: bool = True
    enable_caching: bool = True
    default_ttl_days: int = 7
    max_versions_per_group: int = 100
    enable_metrics: bool = True


class FeatureStore:
    """
    Enterprise Feature Store with versioning, lineage, and quality management.
    
    Key capabilities:
    - Automatic versioning and schema evolution
    - Data lineage tracking  
    - Quality validation and monitoring
    - Search and discovery
    - Usage analytics
    - Performance optimization
    """
    
    def __init__(self, config: Optional[FeatureStoreConfig] = None):
        """Initialize feature store."""
        self.config = config or FeatureStoreConfig()
        
        # Initialize components
        self.version_manager = VersionManager(store_path=self.config.store_path)
        self.metadata_store = MetadataStore(store_path=self.config.store_path)
        
        if self.config.validate_on_write:
            self.validator = DataQualityValidator()
        else:
            self.validator = None
        
        logger.info("Feature store initialized")
    
    def write_feature_group(
        self,
        group_name: str,
        data: pd.DataFrame,
        description: str = "",
        created_by: str = "system",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        validation_config: Optional[Dict[str, Any]] = None,
        parent_features: Optional[List[str]] = None
    ) -> str:
        """
        Write feature group with automatic versioning and validation.
        
        Args:
            group_name: Feature group name
            data: Feature data
            description: Group description
            created_by: Creator identifier
            tags: Tags for categorization
            metadata: Additional metadata
            validation_config: Data quality validation config
            parent_features: Parent features for lineage
            
        Returns:
            Version ID of created feature group
        """
        try:
            # Step 1: Data validation
            if self.validator and validation_config:
                validation_result = self.validator.validate_features(
                    data, validation_config
                )
                if not validation_result.is_valid:
                    raise FeatureValidationError(
                        f"Feature validation failed: {validation_result.errors}"
                    )
            
            # Step 2: Register feature group in metadata store
            group_id = f"{group_name}"
            self.metadata_store.register_feature_group(
                group_id=group_id,
                name=group_name,
                description=description,
                owner=created_by,
                tags=tags,
                metadata=metadata
            )
            
            # Step 3: Create feature metadata
            features = []
            for column in data.columns:
                if column == 'timestamp':  # Skip timestamp column
                    continue
                
                # Calculate basic quality metrics
                quality_metrics = self._calculate_feature_quality(data[column])
                
                feature_meta = FeatureMetadata(
                    name=column,
                    dtype=str(data[column].dtype),
                    description=f"Feature: {column}",
                    tags=tags or [],
                    created_at=datetime.now(),
                    created_by=created_by,
                    business_meaning="",
                    computation_logic="",
                    dependencies=parent_features or [],
                    quality_metrics=quality_metrics,
                    usage_stats={}
                )
                features.append(feature_meta)
            
            # Step 4: Register features in metadata store
            self.metadata_store.register_features(group_id, features)
            
            # Step 5: Create version
            version_id = self.version_manager.create_version(
                feature_group=group_name,
                data=data,
                description=description,
                created_by=created_by,
                metadata=metadata,
                tags=tags
            )
            
            # Step 6: Add schema version tracking
            schema = {
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'shape': data.shape,
                'index_name': data.index.name
            }
            self.metadata_store.add_schema_version(
                group_id, version_id, schema
            )
            
            # Step 7: Track lineage if parent features specified
            if self.config.track_lineage and parent_features:
                for parent_feature in parent_features:
                    self.metadata_store.add_lineage(
                        source_id=parent_feature,
                        target_id=group_id,
                        lineage_type=LineageType.DERIVED_FROM,
                        metadata={'transformation': 'feature_engineering'}
                    )
            
            logger.info(f"Written feature group {group_name} with version {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to write feature group: {e}")
            raise
    
    def read_feature_group(
        self,
        group_name: str,
        version_id: Optional[str] = None,
        status: Optional[VersionStatus] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        features: Optional[List[str]] = None,
        user_id: str = "system"
    ) -> Optional[pd.DataFrame]:
        """
        Read feature group with usage tracking.
        
        Args:
            group_name: Feature group name
            version_id: Specific version (latest if None)
            status: Version status filter
            start_date: Start date filter
            end_date: End date filter
            features: Specific features to read
            user_id: User identifier for tracking
            
        Returns:
            Feature data or None
        """
        try:
            # Get version
            if version_id:
                result = self.version_manager.get_version(group_name, version_id)
                if result is None:
                    return None
                data, version_info = result
            else:
                result = self.version_manager.get_latest_version(
                    group_name, status=status
                )
                if result is None:
                    return None
                version_id, data, version_info = result
            
            # Apply date filters
            if start_date or end_date:
                if 'timestamp' in data.columns:
                    date_col = 'timestamp'
                elif data.index.name == 'timestamp':
                    date_col = data.index
                else:
                    logger.warning("No timestamp column found for date filtering")
                    date_col = None
                
                if date_col is not None:
                    if start_date:
                        data = data[data[date_col] >= pd.Timestamp(start_date)]
                    if end_date:
                        data = data[data[date_col] <= pd.Timestamp(end_date)]
            
            # Apply feature selection
            if features:
                available_features = [f for f in features if f in data.columns]
                if len(available_features) != len(features):
                    missing = set(features) - set(available_features)
                    logger.warning(f"Missing features: {missing}")
                
                if available_features:
                    # Keep timestamp if exists
                    cols_to_keep = available_features[:]
                    if 'timestamp' in data.columns:
                        cols_to_keep.append('timestamp')
                    data = data[cols_to_keep]
                else:
                    return None
            
            # Track usage
            group_id = f"{group_name}"
            self.metadata_store.track_usage(
                feature_id=group_id,
                user_id=user_id,
                access_type="read",
                context={
                    'version_id': version_id,
                    'features_requested': features,
                    'date_range': [start_date, end_date]
                }
            )
            
            logger.info(f"Read feature group {group_name}, version {version_id}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to read feature group: {e}")
            return None
    
    def search_features(
        self,
        query: str = "",
        tags: Optional[List[str]] = None,
        owner: Optional[str] = None,
        min_quality_score: float = 0.0,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search and discover features.
        
        Args:
            query: Search query
            tags: Required tags
            owner: Owner filter
            min_quality_score: Minimum quality score
            limit: Maximum results
            
        Returns:
            List of matching features
        """
        results = self.metadata_store.search_features(
            query=query,
            tags=tags,
            owner=owner,
            limit=limit
        )
        
        # Filter by quality score
        if min_quality_score > 0:
            results = [r for r in results if r.get('quality_score', 0) >= min_quality_score]
        
        return results
    
    def get_feature_lineage(
        self,
        feature_name: str,
        group_name: Optional[str] = None,
        depth: int = 3
    ) -> Dict[str, Any]:
        """
        Get lineage graph for a feature.
        
        Args:
            feature_name: Feature name
            group_name: Feature group name (if known)
            depth: Maximum depth to traverse
            
        Returns:
            Lineage graph data
        """
        if group_name:
            feature_id = f"{group_name}:{feature_name}"
        else:
            # Search for feature
            results = self.search_features(query=feature_name, limit=1)
            if not results:
                return {'nodes': [], 'edges': [], 'node_details': {}}
            feature_id = results[0]['id']
        
        return self.metadata_store.get_lineage_graph(feature_id, depth)
    
    def promote_version(
        self,
        group_name: str,
        version_id: str,
        target_status: VersionStatus
    ) -> bool:
        """
        Promote feature group version to new status.
        
        Args:
            group_name: Feature group name
            version_id: Version ID
            target_status: Target status
            
        Returns:
            True if successful
        """
        return self.version_manager.promote_version(
            group_name, version_id, target_status
        )
    
    def list_versions(
        self,
        group_name: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        List versions for a feature group.
        
        Args:
            group_name: Feature group name
            limit: Maximum versions to return
            
        Returns:
            List of version information
        """
        versions = self.version_manager.list_versions(group_name, limit=limit)
        
        result = []
        for version_id, version_info in versions:
            result.append({
                'version_id': version_id,
                'created_at': version_info.created_at.isoformat(),
                'created_by': version_info.created_by,
                'status': version_info.status.value,
                'description': version_info.description,
                'tags': version_info.tags
            })
        
        return result
    
    def get_feature_stats(
        self,
        feature_name: str,
        group_name: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get feature statistics and usage analytics.
        
        Args:
            feature_name: Feature name
            group_name: Feature group name
            days: Number of days for analytics
            
        Returns:
            Feature statistics
        """
        feature_id = f"{group_name}:{feature_name}"
        return self.metadata_store.get_feature_stats(feature_id, days)
    
    def create_branch(
        self,
        branch_name: str,
        from_branch: str = "main"
    ) -> bool:
        """
        Create feature development branch.
        
        Args:
            branch_name: New branch name
            from_branch: Source branch
            
        Returns:
            True if successful
        """
        return self.version_manager.create_branch(branch_name, from_branch)
    
    def switch_branch(self, branch_name: str) -> bool:
        """
        Switch to different branch.
        
        Args:
            branch_name: Branch name
            
        Returns:
            True if successful
        """
        return self.version_manager.switch_branch(branch_name)
    
    def validate_feature_group(
        self,
        group_name: str,
        version_id: Optional[str] = None,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate feature group quality.
        
        Args:
            group_name: Feature group name
            version_id: Version ID (latest if None)
            validation_config: Validation configuration
            
        Returns:
            Validation results
        """
        if not self.validator:
            return {'valid': True, 'message': 'Validation disabled'}
        
        # Read feature group
        data = self.read_feature_group(group_name, version_id)
        if data is None:
            return {'valid': False, 'message': 'Feature group not found'}
        
        # Run validation
        try:
            result = self.validator.validate_features(
                data, validation_config or {}
            )
            
            return {
                'valid': result.is_valid,
                'errors': result.errors,
                'warnings': result.warnings,
                'metrics': result.metrics
            }
        except Exception as e:
            return {'valid': False, 'message': f'Validation error: {e}'}
    
    def cleanup_old_versions(
        self,
        keep_production: bool = True,
        keep_days: int = 30,
        max_versions_per_group: int = None
    ) -> int:
        """
        Clean up old versions to save storage.
        
        Args:
            keep_production: Keep production versions
            keep_days: Keep versions from last N days
            max_versions_per_group: Max versions per group
            
        Returns:
            Number of versions deleted
        """
        max_versions = max_versions_per_group or self.config.max_versions_per_group
        
        return self.version_manager.garbage_collect(
            keep_production=keep_production,
            keep_days=keep_days,
            max_versions_per_group=max_versions
        )
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        version_stats = self.version_manager.get_storage_stats()
        
        # Add metadata store stats (approximate)
        total_features = 0
        total_groups = 0
        
        try:
            cursor = self.metadata_store.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM features")
            total_features = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM feature_groups")
            total_groups = cursor.fetchone()[0]
        except Exception as e:
            logger.warning(f"Failed to get metadata stats: {e}")
        
        return {
            **version_stats,
            'total_registered_features': total_features,
            'total_registered_groups': total_groups,
            'config': {
                'auto_version': self.config.auto_version,
                'validate_on_write': self.config.validate_on_write,
                'track_lineage': self.config.track_lineage,
                'max_versions_per_group': self.config.max_versions_per_group
            }
        }
    
    def _calculate_feature_quality(self, series: pd.Series) -> Dict[str, float]:
        """Calculate basic quality metrics for a feature."""
        try:
            metrics = {}
            
            # Basic completeness
            total_count = len(series)
            null_count = series.isnull().sum()
            metrics['completeness'] = 1.0 - (null_count / total_count) if total_count > 0 else 0.0
            
            # Uniqueness (for non-numeric)
            if not pd.api.types.is_numeric_dtype(series):
                unique_count = series.nunique()
                metrics['uniqueness'] = unique_count / total_count if total_count > 0 else 0.0
            
            # Validity (no infinite values for numeric)
            if pd.api.types.is_numeric_dtype(series):
                inf_count = np.isinf(series).sum()
                metrics['validity'] = 1.0 - (inf_count / total_count) if total_count > 0 else 0.0
                
                # Distribution metrics
                if total_count > 0:
                    metrics['mean'] = float(series.mean()) if not series.isnull().all() else 0.0
                    metrics['std'] = float(series.std()) if not series.isnull().all() else 0.0
            
            # Overall quality score
            base_metrics = ['completeness', 'validity'] if pd.api.types.is_numeric_dtype(series) else ['completeness', 'uniqueness']
            if base_metrics:
                metrics['overall_score'] = np.mean([metrics.get(m, 0.0) for m in base_metrics])
            else:
                metrics['overall_score'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to calculate quality metrics: {e}")
            return {'overall_score': 0.0}
    
    def close(self):
        """Close feature store connections."""
        if hasattr(self, 'metadata_store'):
            self.metadata_store.close()
        logger.info("Feature store closed")