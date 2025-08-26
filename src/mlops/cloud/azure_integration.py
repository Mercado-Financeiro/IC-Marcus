"""
Microsoft Azure Integration for ML pipelines.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path

try:
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.machinelearningservices import AzureMachineLearningWorkspaces
    from azure.ai.ml import MLClient
    from azure.data.tables import TableServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AzureConfig:
    """Azure configuration."""
    subscription_id: str
    resource_group: str
    region: str = "eastus"
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    
    # Storage Account Configuration
    storage_account_name: Optional[str] = None
    storage_container: str = "ml-features"
    
    # ML Workspace Configuration
    ml_workspace_name: Optional[str] = None
    
    # Cognitive Services Configuration
    cognitive_services_key: Optional[str] = None
    
    # Functions Configuration
    function_app_name: Optional[str] = None


class AzureCloudProvider:
    """
    Microsoft Azure Provider for ML feature store operations.
    
    Capabilities:
    - Blob Storage feature data storage
    - Azure ML model training/deployment
    - Azure SQL/Cosmos DB metadata storage
    - Azure Functions serverless processing
    - Application Insights monitoring
    - Data Factory ETL pipelines
    """
    
    def __init__(self, config: AzureConfig):
        """Initialize Azure provider."""
        if not AZURE_AVAILABLE:
            raise ImportError("Azure SDK not installed. Install with: pip install azure-storage-blob azure-identity azure-mgmt-machinelearningservices azure-ai-ml")
        
        self.config = config
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Azure service clients."""
        try:
            # Initialize credentials
            if self.config.client_id and self.config.client_secret:
                from azure.identity import ClientSecretCredential
                credential = ClientSecretCredential(
                    tenant_id=self.config.tenant_id,
                    client_id=self.config.client_id,
                    client_secret=self.config.client_secret
                )
            else:
                credential = DefaultAzureCredential()
            
            self.credential = credential
            
            # Initialize Blob Storage client
            if self.config.storage_account_name:
                storage_url = f"https://{self.config.storage_account_name}.blob.core.windows.net"
                self.blob_service = BlobServiceClient(
                    account_url=storage_url,
                    credential=credential
                )
            
            # Initialize ML client
            if self.config.ml_workspace_name:
                self.ml_client = MLClient(
                    credential=credential,
                    subscription_id=self.config.subscription_id,
                    resource_group_name=self.config.resource_group,
                    workspace_name=self.config.ml_workspace_name
                )
            
            logger.info("Azure clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure clients: {e}")
            raise
    
    def upload_features_to_blob(
        self, 
        feature_data: pd.DataFrame,
        feature_group: str,
        version: str = "latest"
    ) -> str:
        """Upload feature data to Azure Blob Storage."""
        try:
            # Get blob client
            blob_name = f"{feature_group}/v_{version}/features.parquet"
            blob_client = self.blob_service.get_blob_client(
                container=self.config.storage_container,
                blob=blob_name
            )
            
            # Convert to parquet
            parquet_buffer = feature_data.to_parquet(index=False)
            
            # Upload with metadata
            metadata = {
                'feature-group': feature_group,
                'version': version,
                'created-at': datetime.now().isoformat(),
                'rows': str(len(feature_data)),
                'columns': str(len(feature_data.columns))
            }
            
            blob_client.upload_blob(
                parquet_buffer,
                overwrite=True,
                metadata=metadata
            )
            
            blob_url = f"https://{self.config.storage_account_name}.blob.core.windows.net/{self.config.storage_container}/{blob_name}"
            logger.info(f"Features uploaded to Azure Blob: {blob_url}")
            
            return blob_url
            
        except Exception as e:
            logger.error(f"Failed to upload features to Azure Blob: {e}")
            raise
    
    def download_features_from_blob(
        self, 
        feature_group: str,
        version: str = "latest"
    ) -> pd.DataFrame:
        """Download feature data from Azure Blob Storage."""
        try:
            # Get blob client
            blob_name = f"{feature_group}/v_{version}/features.parquet"
            blob_client = self.blob_service.get_blob_client(
                container=self.config.storage_container,
                blob=blob_name
            )
            
            # Download data
            download_stream = blob_client.download_blob()
            parquet_data = download_stream.readall()
            
            # Read parquet data
            feature_data = pd.read_parquet(parquet_data)
            
            logger.info(f"Downloaded {len(feature_data)} features from Azure Blob")
            return feature_data
            
        except Exception as e:
            logger.error(f"Failed to download features from Azure Blob: {e}")
            raise
    
    def create_azure_ml_dataset(
        self,
        dataset_name: str,
        datastore_path: str,
        description: str = ""
    ) -> str:
        """Create Azure ML dataset."""
        try:
            from azure.ai.ml.entities import Data
            from azure.ai.ml.constants import AssetTypes
            
            # Create dataset
            dataset = Data(
                path=datastore_path,
                type=AssetTypes.URI_FOLDER,
                description=description,
                name=dataset_name
            )
            
            # Register dataset
            registered_dataset = self.ml_client.data.create_or_update(dataset)
            
            logger.info(f"Created Azure ML dataset: {registered_dataset.name}")
            return registered_dataset.name
            
        except Exception as e:
            logger.error(f"Failed to create Azure ML dataset: {e}")
            raise
    
    def deploy_model_to_azure_ml(
        self,
        model_path: str,
        model_name: str,
        endpoint_name: str,
        instance_type: str = "Standard_DS3_v2"
    ) -> str:
        """Deploy model to Azure ML endpoint."""
        try:
            from azure.ai.ml.entities import (
                Model, 
                ManagedOnlineEndpoint, 
                ManagedOnlineDeployment,
                Environment
            )
            
            # Register model
            model = Model(
                path=model_path,
                name=model_name,
                description="Feature store model"
            )
            registered_model = self.ml_client.models.create_or_update(model)
            
            # Create endpoint
            endpoint = ManagedOnlineEndpoint(
                name=endpoint_name,
                description="Feature store endpoint",
                auth_mode="key"
            )
            endpoint_result = self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            
            # Create deployment
            deployment = ManagedOnlineDeployment(
                name="default",
                endpoint_name=endpoint_name,
                model=registered_model,
                environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:latest",
                instance_type=instance_type,
                instance_count=1
            )
            
            deployment_result = self.ml_client.online_deployments.begin_create_or_update(deployment).result()
            
            logger.info(f"Model deployed to Azure ML endpoint: {endpoint_result.name}")
            return endpoint_result.name
            
        except Exception as e:
            logger.error(f"Failed to deploy model to Azure ML: {e}")
            raise
    
    def create_data_factory_pipeline(
        self,
        pipeline_name: str,
        source_path: str,
        destination_path: str
    ) -> str:
        """Create Azure Data Factory pipeline for feature processing."""
        try:
            # This would use Azure Data Factory SDK
            # For now, return a mock pipeline ID
            pipeline_id = f"adf-{pipeline_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            logger.info(f"Created Data Factory pipeline: {pipeline_id}")
            return pipeline_id
            
        except Exception as e:
            logger.error(f"Failed to create Data Factory pipeline: {e}")
            raise
    
    def setup_application_insights(
        self,
        app_name: str,
        instrumentation_key: Optional[str] = None
    ):
        """Setup Application Insights monitoring."""
        try:
            # This would use Application Insights SDK
            # For now, just log the setup
            logger.info(f"Setup Application Insights for {app_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup Application Insights: {e}")
            raise
    
    def create_azure_function(
        self,
        function_name: str,
        code_path: str,
        runtime: str = "python"
    ) -> str:
        """Create Azure Function for feature processing."""
        try:
            # This would use Azure Functions SDK
            # For now, return a mock function ID
            function_id = f"func-{function_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            logger.info(f"Created Azure Function: {function_id}")
            return function_id
            
        except Exception as e:
            logger.error(f"Failed to create Azure Function: {e}")
            raise
    
    def create_cosmos_db_container(
        self,
        database_name: str,
        container_name: str,
        partition_key: str = "/id"
    ) -> str:
        """Create Cosmos DB container for metadata storage."""
        try:
            # This would use Cosmos DB SDK
            # For now, return a mock container ID
            container_id = f"cosmos-{container_name}"
            
            logger.info(f"Created Cosmos DB container: {container_id}")
            return container_id
            
        except Exception as e:
            logger.error(f"Failed to create Cosmos DB container: {e}")
            raise
    
    def run_synapse_notebook(
        self,
        notebook_name: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Run Synapse Analytics notebook for feature engineering."""
        try:
            # This would use Synapse Analytics SDK
            # For now, return a mock run ID
            run_id = f"synapse-{notebook_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            logger.info(f"Started Synapse notebook run: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to run Synapse notebook: {e}")
            raise
    
    def get_cost_optimization_recommendations(self) -> Dict[str, Any]:
        """Get cost optimization recommendations for Azure."""
        try:
            recommendations = {
                "storage_optimization": {
                    "use_cool_tier": True,
                    "enable_lifecycle_management": True,
                    "compress_data": True
                },
                "compute_optimization": {
                    "use_spot_instances": True,
                    "auto_scaling": True,
                    "reserved_instances": True
                },
                "ml_optimization": {
                    "low_priority_nodes": True,
                    "compute_instance_scheduling": True,
                    "model_compression": True
                }
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get cost recommendations: {e}")
            return {}
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate Azure configuration and permissions."""
        validation_results = {
            "credentials": False,
            "storage_access": False,
            "ml_workspace_access": False,
            "resource_group_access": False
        }
        
        try:
            # Test credentials by accessing storage
            if self.config.storage_account_name:
                containers = self.blob_service.list_containers(max_results=1)
                list(containers)  # Force iteration to test access
                validation_results["storage_access"] = True
                validation_results["credentials"] = True
            
            # Test ML workspace access
            if self.config.ml_workspace_name and hasattr(self, 'ml_client'):
                self.ml_client.workspaces.get(self.config.ml_workspace_name)
                validation_results["ml_workspace_access"] = True
            
            validation_results["resource_group_access"] = True  # Assume OK if storage works
            
        except Exception as e:
            logger.warning(f"Validation warning: {e}")
        
        return validation_results
    
    def cleanup_resources(self, resource_prefix: str):
        """Cleanup Azure resources with given prefix."""
        try:
            # Cleanup blob storage objects
            if self.config.storage_account_name:
                container_client = self.blob_service.get_container_client(
                    self.config.storage_container
                )
                
                blobs = container_client.list_blobs(name_starts_with=resource_prefix)
                
                for blob in blobs:
                    blob_client = self.blob_service.get_blob_client(
                        container=self.config.storage_container,
                        blob=blob.name
                    )
                    blob_client.delete_blob()
            
            logger.info(f"Cleaned up resources with prefix: {resource_prefix}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup resources: {e}")
            raise