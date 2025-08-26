"""
AWS Cloud Integration for ML pipelines.
"""

import boto3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    boto3 = None

logger = logging.getLogger(__name__)


@dataclass
class AWSConfig:
    """AWS configuration."""
    region_name: str = "us-east-1"
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    
    # S3 Configuration
    s3_bucket: Optional[str] = None
    s3_prefix: str = "ml-features"
    
    # SageMaker Configuration
    sagemaker_execution_role: Optional[str] = None
    sagemaker_instance_type: str = "ml.t3.medium"
    
    # RDS Configuration
    rds_endpoint: Optional[str] = None
    rds_database: str = "feature_store"
    
    # Lambda Configuration
    lambda_runtime: str = "python3.9"
    lambda_memory: int = 512
    lambda_timeout: int = 300


class AWSCloudProvider:
    """
    AWS Cloud Provider for ML feature store operations.
    
    Capabilities:
    - S3 feature data storage
    - SageMaker model training/deployment
    - RDS metadata storage
    - Lambda serverless processing
    - CloudWatch monitoring
    - Glue ETL jobs
    """
    
    def __init__(self, config: AWSConfig):
        """Initialize AWS provider."""
        if not AWS_AVAILABLE:
            raise ImportError("AWS SDK (boto3) not installed. Install with: pip install boto3")
        
        self.config = config
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AWS service clients."""
        session_kwargs = {
            'region_name': self.config.region_name
        }
        
        if self.config.access_key_id:
            session_kwargs['aws_access_key_id'] = self.config.access_key_id
            
        if self.config.secret_access_key:
            session_kwargs['aws_secret_access_key'] = self.config.secret_access_key
            
        if self.config.session_token:
            session_kwargs['aws_session_token'] = self.config.session_token
        
        try:
            self.session = boto3.Session(**session_kwargs)
            self.s3 = self.session.client('s3')
            self.sagemaker = self.session.client('sagemaker')
            self.rds = self.session.client('rds')
            self.lambda_client = self.session.client('lambda')
            self.cloudwatch = self.session.client('cloudwatch')
            self.glue = self.session.client('glue')
            
            logger.info("AWS clients initialized successfully")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            raise
    
    def upload_features_to_s3(
        self, 
        feature_data: pd.DataFrame,
        feature_group: str,
        version: str = "latest"
    ) -> str:
        """Upload feature data to S3."""
        try:
            # Prepare S3 key
            s3_key = f"{self.config.s3_prefix}/{feature_group}/v_{version}/features.parquet"
            
            # Convert to parquet for efficient storage
            parquet_buffer = feature_data.to_parquet(index=False)
            
            # Upload to S3
            self.s3.put_object(
                Bucket=self.config.s3_bucket,
                Key=s3_key,
                Body=parquet_buffer,
                ContentType='application/octet-stream',
                Metadata={
                    'feature-group': feature_group,
                    'version': version,
                    'created-at': datetime.now().isoformat(),
                    'rows': str(len(feature_data)),
                    'columns': str(len(feature_data.columns))
                }
            )
            
            s3_uri = f"s3://{self.config.s3_bucket}/{s3_key}"
            logger.info(f"Features uploaded to S3: {s3_uri}")
            
            return s3_uri
            
        except Exception as e:
            logger.error(f"Failed to upload features to S3: {e}")
            raise
    
    def download_features_from_s3(
        self, 
        feature_group: str,
        version: str = "latest"
    ) -> pd.DataFrame:
        """Download feature data from S3."""
        try:
            s3_key = f"{self.config.s3_prefix}/{feature_group}/v_{version}/features.parquet"
            
            # Download from S3
            response = self.s3.get_object(
                Bucket=self.config.s3_bucket,
                Key=s3_key
            )
            
            # Read parquet data
            feature_data = pd.read_parquet(response['Body'])
            
            logger.info(f"Downloaded {len(feature_data)} features from S3")
            return feature_data
            
        except Exception as e:
            logger.error(f"Failed to download features from S3: {e}")
            raise
    
    def create_sagemaker_feature_group(
        self,
        group_name: str,
        feature_definitions: List[Dict[str, str]],
        record_identifier: str = "id",
        event_time_feature: str = "timestamp"
    ) -> str:
        """Create SageMaker Feature Group."""
        try:
            response = self.sagemaker.create_feature_group(
                FeatureGroupName=group_name,
                RecordIdentifierFeatureName=record_identifier,
                EventTimeFeatureName=event_time_feature,
                FeatureDefinitions=feature_definitions,
                OnlineStoreConfig={
                    'EnableOnlineStore': True
                },
                OfflineStoreConfig={
                    'S3StorageConfig': {
                        'S3Uri': f"s3://{self.config.s3_bucket}/{self.config.s3_prefix}/sagemaker/{group_name}/"
                    }
                },
                RoleArn=self.config.sagemaker_execution_role
            )
            
            feature_group_arn = response['FeatureGroupArn']
            logger.info(f"Created SageMaker Feature Group: {feature_group_arn}")
            
            return feature_group_arn
            
        except Exception as e:
            logger.error(f"Failed to create SageMaker Feature Group: {e}")
            raise
    
    def deploy_model_to_sagemaker(
        self,
        model_data_path: str,
        model_name: str,
        endpoint_name: str
    ) -> str:
        """Deploy model to SageMaker endpoint."""
        try:
            # Create model
            model_response = self.sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': f"683313688378.dkr.ecr.{self.config.region_name}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
                    'ModelDataUrl': model_data_path
                },
                ExecutionRoleArn=self.config.sagemaker_execution_role
            )
            
            # Create endpoint configuration
            config_name = f"{model_name}-config"
            self.sagemaker.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': self.config.sagemaker_instance_type
                    }
                ]
            )
            
            # Create endpoint
            endpoint_response = self.sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
            
            endpoint_arn = endpoint_response['EndpointArn']
            logger.info(f"Model deployed to SageMaker endpoint: {endpoint_arn}")
            
            return endpoint_arn
            
        except Exception as e:
            logger.error(f"Failed to deploy model to SageMaker: {e}")
            raise
    
    def create_lambda_feature_processor(
        self,
        function_name: str,
        code_path: str,
        handler: str = "lambda_function.lambda_handler"
    ) -> str:
        """Create Lambda function for feature processing."""
        try:
            # Read function code
            with open(code_path, 'rb') as f:
                zip_content = f.read()
            
            # Create Lambda function
            response = self.lambda_client.create_function(
                FunctionName=function_name,
                Runtime=self.config.lambda_runtime,
                Role=self.config.sagemaker_execution_role,  # Reuse SageMaker role
                Handler=handler,
                Code={'ZipFile': zip_content},
                Memory=self.config.lambda_memory,
                Timeout=self.config.lambda_timeout,
                Environment={
                    'Variables': {
                        'S3_BUCKET': self.config.s3_bucket,
                        'S3_PREFIX': self.config.s3_prefix
                    }
                }
            )
            
            function_arn = response['FunctionArn']
            logger.info(f"Created Lambda function: {function_arn}")
            
            return function_arn
            
        except Exception as e:
            logger.error(f"Failed to create Lambda function: {e}")
            raise
    
    def create_glue_etl_job(
        self,
        job_name: str,
        script_location: str,
        input_path: str,
        output_path: str
    ) -> str:
        """Create AWS Glue ETL job for feature processing."""
        try:
            response = self.glue.create_job(
                Name=job_name,
                Role=self.config.sagemaker_execution_role,
                Command={
                    'Name': 'glueetl',
                    'ScriptLocation': script_location,
                    'PythonVersion': '3'
                },
                DefaultArguments={
                    '--job-language': 'python',
                    '--input-path': input_path,
                    '--output-path': output_path
                },
                MaxRetries=2,
                Timeout=120,
                GlueVersion='3.0'
            )
            
            logger.info(f"Created Glue ETL job: {job_name}")
            return job_name
            
        except Exception as e:
            logger.error(f"Failed to create Glue ETL job: {e}")
            raise
    
    def setup_cloudwatch_monitoring(
        self,
        metric_name: str,
        namespace: str = "ML/FeatureStore"
    ):
        """Setup CloudWatch monitoring for feature operations."""
        try:
            # Create custom metric
            self.cloudwatch.put_metric_data(
                Namespace=namespace,
                MetricData=[
                    {
                        'MetricName': metric_name,
                        'Value': 1.0,
                        'Unit': 'Count',
                        'Timestamp': datetime.now()
                    }
                ]
            )
            
            logger.info(f"Setup CloudWatch monitoring for {metric_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup CloudWatch monitoring: {e}")
            raise
    
    def get_cost_optimization_recommendations(self) -> Dict[str, Any]:
        """Get cost optimization recommendations."""
        try:
            recommendations = {
                "s3_optimization": {
                    "use_intelligent_tiering": True,
                    "compress_data": True,
                    "lifecycle_policies": True
                },
                "compute_optimization": {
                    "spot_instances": True,
                    "auto_scaling": True,
                    "schedule_based_scaling": True
                },
                "storage_optimization": {
                    "parquet_format": True,
                    "partitioning": True,
                    "columnar_storage": True
                }
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get cost recommendations: {e}")
            return {}
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate AWS configuration and permissions."""
        validation_results = {
            "credentials": False,
            "s3_access": False,
            "sagemaker_access": False,
            "lambda_access": False
        }
        
        try:
            # Test credentials
            sts = self.session.client('sts')
            sts.get_caller_identity()
            validation_results["credentials"] = True
            
            # Test S3 access
            if self.config.s3_bucket:
                self.s3.head_bucket(Bucket=self.config.s3_bucket)
                validation_results["s3_access"] = True
            
            # Test SageMaker access
            self.sagemaker.list_feature_groups()
            validation_results["sagemaker_access"] = True
            
            # Test Lambda access
            self.lambda_client.list_functions()
            validation_results["lambda_access"] = True
            
        except Exception as e:
            logger.warning(f"Validation warning: {e}")
        
        return validation_results
    
    def cleanup_resources(self, resource_prefix: str):
        """Cleanup AWS resources with given prefix."""
        try:
            # List and delete S3 objects
            if self.config.s3_bucket:
                objects = self.s3.list_objects_v2(
                    Bucket=self.config.s3_bucket,
                    Prefix=f"{self.config.s3_prefix}/{resource_prefix}"
                )
                
                if 'Contents' in objects:
                    for obj in objects['Contents']:
                        self.s3.delete_object(
                            Bucket=self.config.s3_bucket,
                            Key=obj['Key']
                        )
            
            logger.info(f"Cleaned up resources with prefix: {resource_prefix}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup resources: {e}")
            raise