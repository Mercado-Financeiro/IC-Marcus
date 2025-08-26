"""
Version management system for features - Git-like versioning.
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class VersionStatus(Enum):
    """Version status enum."""
    DRAFT = "draft"
    STAGING = "staging" 
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class VersionInfo:
    """Version information."""
    version_id: str
    parent_version: Optional[str]
    created_at: datetime
    created_by: str
    status: VersionStatus
    description: str
    schema_hash: str
    data_hash: str
    metadata: Dict[str, Any]
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        result['status'] = self.status.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionInfo':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['status'] = VersionStatus(data['status'])
        return cls(**data)


class VersionManager:
    """
    Git-like version management for features.
    
    Features:
    - Semantic versioning (major.minor.patch)
    - Branching and merging
    - Schema evolution tracking
    - Rollback capabilities
    - Garbage collection
    """
    
    def __init__(self, store_path: str = "data/feature_store"):
        """Initialize version manager."""
        self.store_path = Path(store_path)
        self.versions_path = self.store_path / "versions"
        self.metadata_path = self.store_path / "metadata" 
        self.branches_path = self.store_path / "branches"
        
        # Create directories
        for path in [self.versions_path, self.metadata_path, self.branches_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Version registry
        self.registry_file = self.metadata_path / "version_registry.json"
        self.registry = self._load_registry()
        
        # Current branch
        self.current_branch = "main"
        self.branch_file = self.branches_path / "current_branch.txt"
        self._load_current_branch()
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load version registry."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
        return {}
    
    def _save_registry(self):
        """Save version registry."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _load_current_branch(self):
        """Load current branch."""
        if self.branch_file.exists():
            try:
                with open(self.branch_file, 'r') as f:
                    self.current_branch = f.read().strip()
            except Exception as e:
                logger.warning(f"Failed to load current branch: {e}")
    
    def _save_current_branch(self):
        """Save current branch."""
        try:
            with open(self.branch_file, 'w') as f:
                f.write(self.current_branch)
        except Exception as e:
            logger.error(f"Failed to save current branch: {e}")
    
    def _generate_version_id(self, feature_group: str, data_hash: str) -> str:
        """Generate unique version ID."""
        timestamp = datetime.now().isoformat()
        content = f"{feature_group}_{data_hash}_{timestamp}_{self.current_branch}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of data content."""
        # Include shape, dtypes, and sample of data
        content_str = f"{data.shape}_{data.dtypes.to_dict()}_{data.head().to_string()}"
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def _calculate_schema_hash(self, schema: Dict[str, Any]) -> str:
        """Calculate hash of schema."""
        schema_str = json.dumps(schema, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
    
    def create_version(
        self,
        feature_group: str,
        data: pd.DataFrame,
        description: str = "",
        created_by: str = "system",
        parent_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create new version of features.
        
        Args:
            feature_group: Feature group name
            data: Feature data
            description: Version description
            created_by: Creator name
            parent_version: Parent version ID
            metadata: Additional metadata
            tags: Version tags
            
        Returns:
            Version ID
        """
        # Calculate hashes
        data_hash = self._calculate_data_hash(data)
        schema = {
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'shape': data.shape,
            'index_name': data.index.name
        }
        schema_hash = self._calculate_schema_hash(schema)
        
        # Generate version ID
        version_id = self._generate_version_id(feature_group, data_hash)
        
        # Check if version already exists
        if version_id in self.registry.get(feature_group, {}):
            logger.warning(f"Version {version_id} already exists")
            return version_id
        
        # Create version info
        version_info = VersionInfo(
            version_id=version_id,
            parent_version=parent_version,
            created_at=datetime.now(),
            created_by=created_by,
            status=VersionStatus.DRAFT,
            description=description,
            schema_hash=schema_hash,
            data_hash=data_hash,
            metadata=metadata or {},
            tags=tags or []
        )
        
        # Save version data
        version_dir = self.versions_path / feature_group / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save data
            data_file = version_dir / "data.parquet"
            data.to_parquet(data_file, compression='snappy')
            
            # Save schema
            schema_file = version_dir / "schema.json"
            with open(schema_file, 'w') as f:
                json.dump(schema, f, indent=2, default=str)
            
            # Save version metadata
            version_file = version_dir / "version.json"
            with open(version_file, 'w') as f:
                json.dump(version_info.to_dict(), f, indent=2, default=str)
            
            # Update registry
            if feature_group not in self.registry:
                self.registry[feature_group] = {}
            
            self.registry[feature_group][version_id] = {
                'version_info': version_info.to_dict(),
                'branch': self.current_branch,
                'file_size': data_file.stat().st_size if data_file.exists() else 0
            }
            
            self._save_registry()
            
            logger.info(f"Created version {version_id} for {feature_group}")
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to create version: {e}")
            # Clean up on failure
            if version_dir.exists():
                shutil.rmtree(version_dir)
            raise
    
    def get_version(self, feature_group: str, version_id: str) -> Optional[Tuple[pd.DataFrame, VersionInfo]]:
        """
        Get specific version of features.
        
        Args:
            feature_group: Feature group name  
            version_id: Version ID
            
        Returns:
            Tuple of (data, version_info) or None
        """
        if feature_group not in self.registry:
            return None
        
        if version_id not in self.registry[feature_group]:
            return None
        
        version_dir = self.versions_path / feature_group / version_id
        
        try:
            # Load data
            data_file = version_dir / "data.parquet"
            if not data_file.exists():
                logger.error(f"Data file not found for version {version_id}")
                return None
            
            data = pd.read_parquet(data_file)
            
            # Load version info
            version_file = version_dir / "version.json"
            with open(version_file, 'r') as f:
                version_data = json.load(f)
            
            version_info = VersionInfo.from_dict(version_data)
            
            return data, version_info
            
        except Exception as e:
            logger.error(f"Failed to load version {version_id}: {e}")
            return None
    
    def get_latest_version(
        self, 
        feature_group: str, 
        branch: Optional[str] = None,
        status: Optional[VersionStatus] = None
    ) -> Optional[Tuple[str, pd.DataFrame, VersionInfo]]:
        """
        Get latest version of features.
        
        Args:
            feature_group: Feature group name
            branch: Branch to search (default: current)
            status: Required status filter
            
        Returns:
            Tuple of (version_id, data, version_info) or None
        """
        if feature_group not in self.registry:
            return None
        
        branch = branch or self.current_branch
        versions = self.registry[feature_group]
        
        # Filter by branch and status
        candidates = []
        for version_id, version_data in versions.items():
            if version_data.get('branch') == branch:
                version_info = VersionInfo.from_dict(version_data['version_info'])
                if status is None or version_info.status == status:
                    candidates.append((version_id, version_info))
        
        if not candidates:
            return None
        
        # Sort by creation time (latest first)
        candidates.sort(key=lambda x: x[1].created_at, reverse=True)
        latest_version_id, latest_info = candidates[0]
        
        # Load data
        result = self.get_version(feature_group, latest_version_id)
        if result is None:
            return None
        
        data, version_info = result
        return latest_version_id, data, version_info
    
    def list_versions(
        self,
        feature_group: str,
        branch: Optional[str] = None,
        limit: int = 50
    ) -> List[Tuple[str, VersionInfo]]:
        """
        List versions for feature group.
        
        Args:
            feature_group: Feature group name
            branch: Branch filter
            limit: Max versions to return
            
        Returns:
            List of (version_id, version_info) tuples
        """
        if feature_group not in self.registry:
            return []
        
        versions = []
        for version_id, version_data in self.registry[feature_group].items():
            if branch is None or version_data.get('branch') == branch:
                version_info = VersionInfo.from_dict(version_data['version_info'])
                versions.append((version_id, version_info))
        
        # Sort by creation time (latest first)
        versions.sort(key=lambda x: x[1].created_at, reverse=True)
        
        return versions[:limit]
    
    def promote_version(self, feature_group: str, version_id: str, status: VersionStatus) -> bool:
        """
        Promote version to new status.
        
        Args:
            feature_group: Feature group name
            version_id: Version ID
            status: New status
            
        Returns:
            True if successful
        """
        if feature_group not in self.registry:
            return False
        
        if version_id not in self.registry[feature_group]:
            return False
        
        try:
            # Update status in version file
            version_dir = self.versions_path / feature_group / version_id
            version_file = version_dir / "version.json"
            
            with open(version_file, 'r') as f:
                version_data = json.load(f)
            
            version_data['status'] = status.value
            
            with open(version_file, 'w') as f:
                json.dump(version_data, f, indent=2, default=str)
            
            # Update registry
            self.registry[feature_group][version_id]['version_info']['status'] = status.value
            self._save_registry()
            
            logger.info(f"Promoted version {version_id} to {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote version: {e}")
            return False
    
    def create_branch(self, branch_name: str, from_branch: str = "main") -> bool:
        """
        Create new branch.
        
        Args:
            branch_name: New branch name
            from_branch: Source branch
            
        Returns:
            True if successful
        """
        try:
            branch_file = self.branches_path / f"{branch_name}.json"
            
            if branch_file.exists():
                logger.warning(f"Branch {branch_name} already exists")
                return False
            
            branch_info = {
                'name': branch_name,
                'created_at': datetime.now().isoformat(),
                'parent_branch': from_branch,
                'head_versions': {}  # feature_group -> version_id
            }
            
            # Copy head versions from parent branch
            if from_branch != "main":
                parent_file = self.branches_path / f"{from_branch}.json"
                if parent_file.exists():
                    with open(parent_file, 'r') as f:
                        parent_info = json.load(f)
                        branch_info['head_versions'] = parent_info.get('head_versions', {})
            
            with open(branch_file, 'w') as f:
                json.dump(branch_info, f, indent=2)
            
            logger.info(f"Created branch {branch_name} from {from_branch}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create branch: {e}")
            return False
    
    def switch_branch(self, branch_name: str) -> bool:
        """
        Switch to different branch.
        
        Args:
            branch_name: Branch name
            
        Returns:
            True if successful
        """
        if branch_name == "main":
            self.current_branch = branch_name
            self._save_current_branch()
            return True
        
        branch_file = self.branches_path / f"{branch_name}.json"
        if not branch_file.exists():
            logger.error(f"Branch {branch_name} does not exist")
            return False
        
        self.current_branch = branch_name
        self._save_current_branch()
        logger.info(f"Switched to branch {branch_name}")
        return True
    
    def delete_version(self, feature_group: str, version_id: str, force: bool = False) -> bool:
        """
        Delete version (with safety checks).
        
        Args:
            feature_group: Feature group name
            version_id: Version ID
            force: Force delete even if in use
            
        Returns:
            True if successful
        """
        if feature_group not in self.registry:
            return False
        
        if version_id not in self.registry[feature_group]:
            return False
        
        version_info = VersionInfo.from_dict(
            self.registry[feature_group][version_id]['version_info']
        )
        
        # Safety check - don't delete production versions without force
        if version_info.status == VersionStatus.PRODUCTION and not force:
            logger.error("Cannot delete production version without force=True")
            return False
        
        try:
            # Remove version directory
            version_dir = self.versions_path / feature_group / version_id
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            # Update registry
            del self.registry[feature_group][version_id]
            self._save_registry()
            
            logger.info(f"Deleted version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete version: {e}")
            return False
    
    def garbage_collect(self, 
                       keep_production: bool = True,
                       keep_days: int = 30,
                       max_versions_per_group: int = 100) -> int:
        """
        Clean up old versions.
        
        Args:
            keep_production: Keep production versions
            keep_days: Keep versions from last N days
            max_versions_per_group: Max versions per feature group
            
        Returns:
            Number of versions deleted
        """
        deleted_count = 0
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        for feature_group, versions in list(self.registry.items()):
            version_list = []
            
            for version_id, version_data in versions.items():
                version_info = VersionInfo.from_dict(version_data['version_info'])
                version_list.append((version_id, version_info))
            
            # Sort by creation time (oldest first for deletion)
            version_list.sort(key=lambda x: x[1].created_at)
            
            # Determine versions to delete
            to_delete = []
            
            for i, (version_id, version_info) in enumerate(version_list):
                should_delete = False
                
                # Keep production versions if specified
                if keep_production and version_info.status == VersionStatus.PRODUCTION:
                    continue
                
                # Delete old versions beyond cutoff
                if version_info.created_at < cutoff_date:
                    should_delete = True
                
                # Keep only max_versions_per_group newest
                if len(version_list) - i > max_versions_per_group:
                    should_delete = True
                
                if should_delete:
                    to_delete.append(version_id)
            
            # Delete identified versions
            for version_id in to_delete:
                if self.delete_version(feature_group, version_id, force=True):
                    deleted_count += 1
        
        logger.info(f"Garbage collection: deleted {deleted_count} versions")
        return deleted_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = 0
        total_versions = 0
        feature_groups = len(self.registry)
        
        for feature_group, versions in self.registry.items():
            for version_id, version_data in versions.items():
                total_versions += 1
                total_size += version_data.get('file_size', 0)
        
        return {
            'total_feature_groups': feature_groups,
            'total_versions': total_versions,
            'total_size_mb': total_size / 1e6,
            'storage_path': str(self.store_path),
            'current_branch': self.current_branch
        }