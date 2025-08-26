"""
Feature Store REST API for external access and integrations.
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Path
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path as PathLib

from .feature_store import FeatureStore, FeatureStoreConfig
from .versioning import VersionStatus
from ..validation.exceptions import FeatureValidationError

logger = logging.getLogger(__name__)


# Pydantic models for API
class FeatureGroupWrite(BaseModel):
    """Request model for writing feature groups."""
    group_name: str = Field(..., description="Feature group name")
    description: str = Field("", description="Group description")
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    validation_config: Optional[Dict[str, Any]] = Field(None, description="Validation config")
    parent_features: Optional[List[str]] = Field(None, description="Parent features for lineage")


class FeatureGroupRead(BaseModel):
    """Request model for reading feature groups."""
    group_name: str = Field(..., description="Feature group name")
    version_id: Optional[str] = Field(None, description="Specific version ID")
    status: Optional[str] = Field(None, description="Version status filter")
    start_date: Optional[str] = Field(None, description="Start date filter (ISO format)")
    end_date: Optional[str] = Field(None, description="End date filter (ISO format)")
    features: Optional[List[str]] = Field(None, description="Specific features to read")


class FeatureSearch(BaseModel):
    """Request model for feature search."""
    query: str = Field("", description="Search query")
    tags: Optional[List[str]] = Field(None, description="Required tags")
    owner: Optional[str] = Field(None, description="Owner filter")
    min_quality_score: float = Field(0.0, description="Minimum quality score")
    limit: int = Field(50, description="Maximum results")


class VersionPromotion(BaseModel):
    """Request model for version promotion."""
    group_name: str = Field(..., description="Feature group name")
    version_id: str = Field(..., description="Version ID")
    target_status: str = Field(..., description="Target status")


class BranchOperation(BaseModel):
    """Request model for branch operations."""
    branch_name: str = Field(..., description="Branch name")
    from_branch: str = Field("main", description="Source branch (for creation)")


# Custom JSON encoder for pandas/numpy types
class CustomJSONEncoder:
    @staticmethod
    def encode_pandas_types(obj):
        """Convert pandas/numpy types to JSON serializable."""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_list()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj


class FeatureStoreAPI:
    """
    Feature Store REST API.
    
    Provides HTTP endpoints for:
    - Reading/writing feature groups
    - Version management
    - Search and discovery
    - Lineage tracking
    - Analytics and monitoring
    """
    
    def __init__(self, 
                 config: Optional[FeatureStoreConfig] = None,
                 auth_enabled: bool = False):
        """Initialize API."""
        self.app = FastAPI(
            title="Feature Store API",
            description="Enterprise Feature Store with versioning and lineage",
            version="1.0.0"
        )
        
        self.feature_store = FeatureStore(config)
        self.auth_enabled = auth_enabled
        self.security = HTTPBearer() if auth_enabled else None
        
        self._setup_routes()
        logger.info("Feature Store API initialized")
    
    def _get_current_user(self, credentials: HTTPAuthorizationCredentials = None) -> str:
        """Get current user from auth token."""
        if not self.auth_enabled:
            return "anonymous"
        
        # Simplified auth - in production, validate JWT token
        if credentials and credentials.credentials:
            return credentials.credentials  # Use token as user ID for now
        
        raise HTTPException(status_code=401, detail="Authentication required")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/stats")
        async def get_storage_stats():
            """Get feature store statistics."""
            try:
                stats = self.feature_store.get_storage_stats()
                return {"success": True, "data": stats}
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/features/write")
        async def write_feature_group(
            request: FeatureGroupWrite,
            data: str,  # JSON string of DataFrame
            user: str = Depends(self._get_current_user if self.auth_enabled else lambda: "anonymous")
        ):
            """Write feature group with data."""
            try:
                # Parse DataFrame from JSON
                data_dict = json.loads(data)
                df = pd.DataFrame(data_dict)
                
                # Convert status string if provided
                version_id = self.feature_store.write_feature_group(
                    group_name=request.group_name,
                    data=df,
                    description=request.description,
                    created_by=user,
                    tags=request.tags,
                    metadata=request.metadata,
                    validation_config=request.validation_config,
                    parent_features=request.parent_features
                )
                
                return {
                    "success": True,
                    "version_id": version_id,
                    "message": f"Feature group {request.group_name} written successfully"
                }
                
            except FeatureValidationError as e:
                raise HTTPException(status_code=400, detail=f"Validation error: {e}")
            except Exception as e:
                logger.error(f"Failed to write feature group: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/features/read")
        async def read_feature_group(
            request: FeatureGroupRead,
            user: str = Depends(self._get_current_user if self.auth_enabled else lambda: "anonymous")
        ):
            """Read feature group data."""
            try:
                # Convert status string to enum
                status = None
                if request.status:
                    try:
                        status = VersionStatus(request.status)
                    except ValueError:
                        raise HTTPException(status_code=400, detail=f"Invalid status: {request.status}")
                
                df = self.feature_store.read_feature_group(
                    group_name=request.group_name,
                    version_id=request.version_id,
                    status=status,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    features=request.features,
                    user_id=user
                )
                
                if df is None:
                    raise HTTPException(status_code=404, detail="Feature group not found")
                
                # Convert DataFrame to JSON
                data = CustomJSONEncoder.encode_pandas_types(df)
                
                return {
                    "success": True,
                    "data": data,
                    "shape": df.shape,
                    "columns": list(df.columns)
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to read feature group: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/features/search")
        async def search_features(request: FeatureSearch):
            """Search and discover features."""
            try:
                results = self.feature_store.search_features(
                    query=request.query,
                    tags=request.tags,
                    owner=request.owner,
                    min_quality_score=request.min_quality_score,
                    limit=request.limit
                )
                
                return {
                    "success": True,
                    "data": results,
                    "count": len(results)
                }
                
            except Exception as e:
                logger.error(f"Failed to search features: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/features/{group_name}/versions")
        async def list_versions(
            group_name: str = Path(..., description="Feature group name"),
            limit: int = Query(20, description="Maximum versions to return")
        ):
            """List versions for a feature group."""
            try:
                versions = self.feature_store.list_versions(group_name, limit)
                
                return {
                    "success": True,
                    "data": versions,
                    "count": len(versions)
                }
                
            except Exception as e:
                logger.error(f"Failed to list versions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/versions/promote")
        async def promote_version(request: VersionPromotion):
            """Promote version to new status."""
            try:
                # Convert status string to enum
                try:
                    target_status = VersionStatus(request.target_status)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid status: {request.target_status}")
                
                success = self.feature_store.promote_version(
                    request.group_name,
                    request.version_id,
                    target_status
                )
                
                if not success:
                    raise HTTPException(status_code=400, detail="Failed to promote version")
                
                return {
                    "success": True,
                    "message": f"Version {request.version_id} promoted to {request.target_status}"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to promote version: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/lineage/{group_name}/{feature_name}")
        async def get_feature_lineage(
            group_name: str = Path(..., description="Feature group name"),
            feature_name: str = Path(..., description="Feature name"),
            depth: int = Query(3, description="Maximum lineage depth")
        ):
            """Get feature lineage graph."""
            try:
                lineage = self.feature_store.get_feature_lineage(
                    feature_name, group_name, depth
                )
                
                return {
                    "success": True,
                    "data": lineage
                }
                
            except Exception as e:
                logger.error(f"Failed to get lineage: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/features/{group_name}/{feature_name}/stats")
        async def get_feature_stats(
            group_name: str = Path(..., description="Feature group name"),
            feature_name: str = Path(..., description="Feature name"),
            days: int = Query(30, description="Number of days for analytics")
        ):
            """Get feature statistics and usage analytics."""
            try:
                stats = self.feature_store.get_feature_stats(
                    feature_name, group_name, days
                )
                
                return {
                    "success": True,
                    "data": CustomJSONEncoder.encode_pandas_types(stats)
                }
                
            except Exception as e:
                logger.error(f"Failed to get feature stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/features/{group_name}/validate")
        async def validate_feature_group(
            group_name: str = Path(..., description="Feature group name"),
            version_id: Optional[str] = Query(None, description="Version ID"),
            validation_config: Optional[str] = Query(None, description="Validation config JSON")
        ):
            """Validate feature group quality."""
            try:
                config = None
                if validation_config:
                    config = json.loads(validation_config)
                
                result = self.feature_store.validate_feature_group(
                    group_name, version_id, config
                )
                
                return {
                    "success": True,
                    "data": result
                }
                
            except Exception as e:
                logger.error(f"Failed to validate feature group: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/branches/create")
        async def create_branch(request: BranchOperation):
            """Create new branch."""
            try:
                success = self.feature_store.create_branch(
                    request.branch_name, request.from_branch
                )
                
                if not success:
                    raise HTTPException(status_code=400, detail="Failed to create branch")
                
                return {
                    "success": True,
                    "message": f"Branch {request.branch_name} created"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to create branch: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/branches/switch")
        async def switch_branch(request: BranchOperation):
            """Switch to different branch."""
            try:
                success = self.feature_store.switch_branch(request.branch_name)
                
                if not success:
                    raise HTTPException(status_code=400, detail="Failed to switch branch")
                
                return {
                    "success": True,
                    "message": f"Switched to branch {request.branch_name}"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to switch branch: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/maintenance/cleanup")
        async def cleanup_old_versions(
            keep_production: bool = Query(True, description="Keep production versions"),
            keep_days: int = Query(30, description="Keep versions from last N days"),
            max_versions_per_group: Optional[int] = Query(None, description="Max versions per group")
        ):
            """Clean up old versions."""
            try:
                deleted_count = self.feature_store.cleanup_old_versions(
                    keep_production=keep_production,
                    keep_days=keep_days,
                    max_versions_per_group=max_versions_per_group
                )
                
                return {
                    "success": True,
                    "deleted_versions": deleted_count,
                    "message": f"Cleaned up {deleted_count} old versions"
                }
                
            except Exception as e:
                logger.error(f"Failed to cleanup versions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the API server."""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port, **kwargs)
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI app instance."""
        return self.app


# Convenience functions for API deployment
def create_feature_store_app(config: Optional[FeatureStoreConfig] = None) -> FastAPI:
    """Create and configure Feature Store API app."""
    api = FeatureStoreAPI(config=config)
    return api.get_app()


if __name__ == "__main__":
    # Example usage
    config = FeatureStoreConfig(
        store_path="data/feature_store_api",
        validate_on_write=True,
        track_lineage=True
    )
    
    api = FeatureStoreAPI(config=config)
    api.run(port=8001)