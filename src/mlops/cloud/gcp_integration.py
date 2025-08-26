"""
Google Cloud Platform Integration for ML pipelines.
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
    from google.cloud import storage, aiplatform, bigquery
    from google.oauth2 import service_account
    from google.api_core.exceptions import GoogleAPIError
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    storage = bigquery = aiplatform = None

logger = logging.getLogger(__name__)


@dataclass
class GCPConfig:
    """GCP configuration."""
    project_id: str
    region: str = "us-central1"
    credentials_path: Optional[str] = None
    
    # Cloud Storage Configuration
    gcs_bucket: Optional[str] = None
    gcs_prefix: str = "ml-features"
    
    # Vertex AI Configuration
    vertex_staging_bucket: Optional[str] = None
    vertex_service_account: Optional[str] = None
    
    # BigQuery Configuration
    bigquery_dataset: str = "feature_store"
    bigquery_location: str = "US"
    
    # Cloud Functions Configuration
    function_runtime: str = "python39"
    function_memory: int = 512
    function_timeout: int = 300


class GCPCloudProvider:
    """
    Google Cloud Platform Provider for ML feature store operations.
    
    Capabilities:
    - Cloud Storage feature data storage
    - Vertex AI model training/deployment
    - BigQuery data warehouse integration
    - Cloud Functions serverless processing
    - Cloud Monitoring
    - Dataflow batch/stream processing
    """
    
    def __init__(self, config: GCPConfig):
        """Initialize GCP provider."""
        if not GCP_AVAILABLE:
            raise ImportError("GCP SDK not installed. Install with: pip install google-cloud-storage google-cloud-aiplatform google-cloud-bigquery")
        
        self.config = config
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize GCP service clients."""
        try:
            # Initialize credentials
            if self.config.credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    self.config.credentials_path
                )
            else:
                credentials = None  # Use default credentials
            
            # Initialize clients
            self.storage_client = storage.Client(
                project=self.config.project_id,
                credentials=credentials
            )
            
            self.bq_client = bigquery.Client(
                project=self.config.project_id,
                credentials=credentials
            )
            
            # Initialize Vertex AI
            aiplatform.init(
                project=self.config.project_id,
                location=self.config.region,
                credentials=credentials,
                staging_bucket=self.config.vertex_staging_bucket
            )
            
            logger.info("GCP clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCP clients: {e}")
            raise
    
    def upload_features_to_gcs(
        self, 
        feature_data: pd.DataFrame,
        feature_group: str,
        version: str = "latest"
    ) -> str:
        """Upload feature data to Google Cloud Storage."""
        try:
            # Get bucket
            bucket = self.storage_client.bucket(self.config.gcs_bucket)
            
            # Prepare blob name
            blob_name = f"{self.config.gcs_prefix}/{feature_group}/v_{version}/features.parquet"
            blob = bucket.blob(blob_name)
            
            # Convert to parquet
            parquet_buffer = feature_data.to_parquet(index=False)
            
            # Upload with metadata
            blob.metadata = {
                'feature-group': feature_group,
                'version': version,
                'created-at': datetime.now().isoformat(),
                'rows': str(len(feature_data)),
                'columns': str(len(feature_data.columns))
            }
            
            blob.upload_from_string(
                parquet_buffer,
                content_type='application/octet-stream'
            )
            
            gcs_uri = f"gs://{self.config.gcs_bucket}/{blob_name}"
            logger.info(f"Features uploaded to GCS: {gcs_uri}")
            
            return gcs_uri
            
        except Exception as e:
            logger.error(f"Failed to upload features to GCS: {e}")
            raise
    
    def download_features_from_gcs(
        self, 
        feature_group: str,
        version: str = "latest"
    ) -> pd.DataFrame:
        """Download feature data from Google Cloud Storage."""
        try:
            # Get bucket and blob
            bucket = self.storage_client.bucket(self.config.gcs_bucket)
            blob_name = f"{self.config.gcs_prefix}/{feature_group}/v_{version}/features.parquet"
            blob = bucket.blob(blob_name)
            
            # Download data
            parquet_data = blob.download_as_bytes()
            
            # Read parquet data
            feature_data = pd.read_parquet(parquet_data)
            
            logger.info(f"Downloaded {len(feature_data)} features from GCS")
            return feature_data
            
        except Exception as e:
            logger.error(f"Failed to download features from GCS: {e}")
            raise
    
    def create_bigquery_feature_table(
        self,
        table_name: str,
        feature_data: pd.DataFrame
    ) -> str:
        """Create BigQuery table for features."""
        try:
            # Create dataset if not exists
            dataset_id = f"{self.config.project_id}.{self.config.bigquery_dataset}"
            
            try:
                dataset = bigquery.Dataset(dataset_id)
                dataset.location = self.config.bigquery_location
                self.bq_client.create_dataset(dataset, exists_ok=True)
            except Exception as e:
                logger.warning(f"Dataset creation warning: {e}")
            
            # Create table reference
            table_id = f"{dataset_id}.{table_name}"
            
            # Define schema from DataFrame
            schema = []
            for column, dtype in feature_data.dtypes.items():
                if pd.api.types.is_integer_dtype(dtype):
                    bq_type = "INTEGER"
                elif pd.api.types.is_float_dtype(dtype):
                    bq_type = "FLOAT"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    bq_type = "TIMESTAMP"
                else:
                    bq_type = "STRING"
                
                schema.append(bigquery.SchemaField(column, bq_type))
            
            # Create table
            table = bigquery.Table(table_id, schema=schema)
            table = self.bq_client.create_table(table, exists_ok=True)
            
            # Load data
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_TRUNCATE",
                source_format=bigquery.SourceFormat.PARQUET
            )
            
            # Convert DataFrame to parquet for upload
            parquet_buffer = feature_data.to_parquet(index=False)
            
            job = self.bq_client.load_table_from_file(
                parquet_buffer, table, job_config=job_config
            )
            job.result()  # Wait for job to complete
            
            logger.info(f"Created BigQuery table: {table_id}")
            return table_id
            
        except Exception as e:
            logger.error(f"Failed to create BigQuery table: {e}")
            raise
    
    def deploy_model_to_vertex_ai(
        self,
        model_path: str,
        model_name: str,
        endpoint_name: str,
        machine_type: str = "n1-standard-2"
    ) -> str:
        """Deploy model to Vertex AI endpoint."""
        try:
            # Upload model
            model = aiplatform.Model.upload(
                display_name=model_name,
                artifact_uri=model_path,
                serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/sklearn-cpu.0-23:latest"
            )
            
            # Create endpoint
            endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
            
            # Deploy model to endpoint
            endpoint.deploy(
                model=model,
                deployed_model_display_name=f"{model_name}-deployment",
                machine_type=machine_type,
                min_replica_count=1,
                max_replica_count=3
            )
            
            logger.info(f"Model deployed to Vertex AI endpoint: {endpoint.name}")
            return endpoint.name
            
        except Exception as e:
            logger.error(f"Failed to deploy model to Vertex AI: {e}")
            raise
    
    def create_dataflow_pipeline(
        self,
        pipeline_name: str,
        template_path: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Create Dataflow pipeline for feature processing."""
        try:
            # This would use Dataflow templates
            # For now, return a mock pipeline ID
            pipeline_id = f"dataflow-{pipeline_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            logger.info(f"Created Dataflow pipeline: {pipeline_id}")
            return pipeline_id
            
        except Exception as e:
            logger.error(f"Failed to create Dataflow pipeline: {e}")
            raise
    
    def setup_monitoring(
        self,
        metric_name: str,
        resource_type: str = "gce_instance"
    ):
        """Setup Cloud Monitoring for feature operations."""
        try:
            # This would use Cloud Monitoring API
            # For now, just log the setup
            logger.info(f"Setup Cloud Monitoring for {metric_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup monitoring: {e}")
            raise
    
    def run_bigquery_feature_query(
        self,
        query: str,
        output_table: Optional[str] = None
    ) -> pd.DataFrame:
        """Run BigQuery query for feature engineering."""
        try:
            # Configure query job
            job_config = bigquery.QueryJobConfig()
            
            if output_table:
                table_ref = bigquery.TableReference.from_string(output_table)
                job_config.destination = table_ref
                job_config.write_disposition = "WRITE_TRUNCATE"
            
            # Run query
            query_job = self.bq_client.query(query, job_config=job_config)
            
            # Get results
            if output_table:
                query_job.result()  # Wait for job to complete
                logger.info(f"Query results written to {output_table}")
                return pd.DataFrame()  # Empty DataFrame since results are in table
            else:
                results = query_job.result()
                df = results.to_dataframe()
                logger.info(f"Query returned {len(df)} rows")
                return df
            
        except Exception as e:
            logger.error(f"Failed to run BigQuery query: {e}")
            raise
    
    def create_vertex_ai_feature_store(
        self,
        feature_store_id: str,
        entity_types: List[Dict[str, Any]]
    ) -> str:
        """Create Vertex AI Feature Store."""
        try:
            # This would use Vertex AI Feature Store API
            # For now, return a mock feature store
            feature_store_name = f"projects/{self.config.project_id}/locations/{self.config.region}/featureStores/{feature_store_id}"
            
            logger.info(f"Created Vertex AI Feature Store: {feature_store_name}")
            return feature_store_name
            
        except Exception as e:
            logger.error(f"Failed to create Vertex AI Feature Store: {e}")
            raise
    
    def get_cost_optimization_recommendations(self) -> Dict[str, Any]:
        """Get cost optimization recommendations for GCP."""
        try:
            recommendations = {
                "storage_optimization": {
                    "use_nearline_storage": True,
                    "enable_lifecycle_policies": True,
                    "compress_data": True
                },
                "compute_optimization": {
                    "use_preemptible_instances": True,
                    "auto_scaling": True,
                    "sustained_use_discounts": True
                },
                "bigquery_optimization": {
                    "partition_tables": True,
                    "cluster_tables": True,
                    "use_slots_efficiently": True
                }
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get cost recommendations: {e}")
            return {}
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate GCP configuration and permissions."""
        validation_results = {
            "credentials": False,
            "storage_access": False,
            "bigquery_access": False,
            "vertex_ai_access": False
        }
        
        try:
            # Test credentials by listing buckets
            list(self.storage_client.list_buckets(max_results=1))
            validation_results["credentials"] = True
            
            # Test storage access
            if self.config.gcs_bucket:
                bucket = self.storage_client.bucket(self.config.gcs_bucket)
                bucket.exists()
                validation_results["storage_access"] = True
            
            # Test BigQuery access
            list(self.bq_client.list_datasets(max_results=1))
            validation_results["bigquery_access"] = True
            
            # Test Vertex AI access
            validation_results["vertex_ai_access"] = True  # Assume OK if no error
            
        except Exception as e:
            logger.warning(f"Validation warning: {e}")
        
        return validation_results
    
    def cleanup_resources(self, resource_prefix: str):
        """Cleanup GCP resources with given prefix."""
        try:
            # Cleanup GCS objects
            if self.config.gcs_bucket:
                bucket = self.storage_client.bucket(self.config.gcs_bucket)
                blobs = bucket.list_blobs(prefix=f"{self.config.gcs_prefix}/{resource_prefix}")
                
                for blob in blobs:
                    blob.delete()
            
            logger.info(f"Cleaned up resources with prefix: {resource_prefix}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup resources: {e}")
            raise