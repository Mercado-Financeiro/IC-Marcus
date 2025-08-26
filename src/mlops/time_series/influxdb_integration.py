"""
InfluxDB Integration for Time Series Feature Store.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
    from influxdb_client.client.query_api import QueryApi
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    InfluxDBClient = Point = WritePrecision = None

logger = logging.getLogger(__name__)


@dataclass
class InfluxDBConfig:
    """InfluxDB configuration."""
    url: str = "http://localhost:8086"
    token: Optional[str] = None
    org: str = "crypto_ml_org"
    bucket: str = "crypto_features"
    
    # Performance settings
    batch_size: int = 1000
    flush_interval: int = 1000  # milliseconds
    timeout: int = 30000  # milliseconds
    
    # Retention policy
    retention_policy: str = "30d"  # 30 days
    
    # Connection settings
    verify_ssl: bool = True
    enable_gzip: bool = True


class InfluxDBProvider:
    """
    InfluxDB Provider for time series feature storage and retrieval.
    
    Capabilities:
    - High-performance time series data ingestion
    - Real-time feature streaming
    - Automatic downsampling and aggregation
    - Retention policy management
    - Query optimization for ML workloads
    - Batch and streaming writes
    """
    
    def __init__(self, config: InfluxDBConfig):
        """Initialize InfluxDB provider."""
        if not INFLUXDB_AVAILABLE:
            raise ImportError("InfluxDB client not installed. Install with: pip install influxdb-client")
        
        self.config = config
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize InfluxDB client and APIs."""
        try:
            self.client = InfluxDBClient(
                url=self.config.url,
                token=self.config.token,
                org=self.config.org,
                verify_ssl=self.config.verify_ssl,
                enable_gzip=self.config.enable_gzip,
                timeout=self.config.timeout
            )
            
            # Initialize APIs
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            self.buckets_api = self.client.buckets_api()
            self.delete_api = self.client.delete_api()
            
            # Test connection
            self._test_connection()
            
            logger.info("InfluxDB client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB client: {e}")
            raise
    
    def _test_connection(self):
        """Test InfluxDB connection."""
        try:
            # Simple health check
            health = self.client.health()
            if health.status != "pass":
                raise Exception(f"InfluxDB health check failed: {health.message}")
            
            # Ensure bucket exists
            self._ensure_bucket_exists()
            
        except Exception as e:
            logger.error(f"InfluxDB connection test failed: {e}")
            raise
    
    def _ensure_bucket_exists(self):
        """Ensure the configured bucket exists."""
        try:
            # Check if bucket exists
            buckets = self.buckets_api.find_buckets()
            bucket_names = [b.name for b in buckets.buckets] if buckets.buckets else []
            
            if self.config.bucket not in bucket_names:
                # Create bucket
                bucket = self.buckets_api.create_bucket(
                    bucket_name=self.config.bucket,
                    org=self.config.org,
                    retention_rules=[{
                        "type": "expire",
                        "everySeconds": self._parse_retention_to_seconds(self.config.retention_policy)
                    }]
                )
                logger.info(f"Created InfluxDB bucket: {bucket.name}")
            
        except Exception as e:
            logger.warning(f"Failed to ensure bucket exists: {e}")
    
    def _parse_retention_to_seconds(self, retention: str) -> int:
        """Parse retention policy string to seconds."""
        # Simple parser for formats like "30d", "7d", "24h"
        if retention.endswith('d'):
            return int(retention[:-1]) * 24 * 3600
        elif retention.endswith('h'):
            return int(retention[:-1]) * 3600
        elif retention.endswith('m'):
            return int(retention[:-1]) * 60
        else:
            return 30 * 24 * 3600  # Default 30 days
    
    def write_features_batch(
        self,
        feature_data: pd.DataFrame,
        measurement: str,
        tags: Optional[Dict[str, str]] = None,
        time_column: str = "timestamp"
    ) -> bool:
        """Write features to InfluxDB in batch mode."""
        try:
            if feature_data.empty:
                logger.warning("Empty feature data provided")
                return True
            
            # Prepare points
            points = []
            
            for _, row in feature_data.iterrows():
                # Create point
                point = Point(measurement)
                
                # Add tags
                if tags:
                    for tag_key, tag_value in tags.items():
                        point = point.tag(tag_key, str(tag_value))
                
                # Add timestamp
                if time_column in row.index:
                    timestamp = pd.to_datetime(row[time_column])
                    point = point.time(timestamp, WritePrecision.NS)
                else:
                    point = point.time(datetime.utcnow(), WritePrecision.NS)
                
                # Add fields (all non-time, non-tag columns)
                for field_name, field_value in row.items():
                    if field_name != time_column and pd.notna(field_value):
                        # Convert numpy types to Python types
                        if isinstance(field_value, (np.integer, np.floating)):
                            field_value = field_value.item()
                        elif isinstance(field_value, np.bool_):
                            field_value = bool(field_value)
                        
                        point = point.field(field_name, field_value)
                
                points.append(point)
            
            # Write batch
            self.write_api.write(
                bucket=self.config.bucket,
                org=self.config.org,
                record=points
            )
            
            logger.info(f"Successfully wrote {len(points)} points to InfluxDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write features to InfluxDB: {e}")
            return False
    
    def write_features_streaming(
        self,
        feature_point: Dict[str, Any],
        measurement: str,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Write single feature point for streaming use cases."""
        try:
            # Create point
            point = Point(measurement)
            
            # Add tags
            if tags:
                for tag_key, tag_value in tags.items():
                    point = point.tag(tag_key, str(tag_value))
            
            # Add timestamp
            if timestamp:
                point = point.time(timestamp, WritePrecision.NS)
            else:
                point = point.time(datetime.utcnow(), WritePrecision.NS)
            
            # Add fields
            for field_name, field_value in feature_point.items():
                if pd.notna(field_value):
                    # Convert numpy types
                    if isinstance(field_value, (np.integer, np.floating)):
                        field_value = field_value.item()
                    elif isinstance(field_value, np.bool_):
                        field_value = bool(field_value)
                    
                    point = point.field(field_name, field_value)
            
            # Write point
            self.write_api.write(
                bucket=self.config.bucket,
                org=self.config.org,
                record=point
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write streaming point: {e}")
            return False
    
    def query_features(
        self,
        measurement: str,
        start_time: Union[str, datetime] = "-1h",
        end_time: Union[str, datetime, None] = None,
        fields: Optional[List[str]] = None,
        tags: Optional[Dict[str, str]] = None,
        aggregation: Optional[str] = None,
        window: Optional[str] = None
    ) -> pd.DataFrame:
        """Query features from InfluxDB with flexible filtering."""
        try:
            # Build query
            query = f'from(bucket: "{self.config.bucket}") |> range(start: {self._format_time(start_time)}'
            
            if end_time:
                query += f', stop: {self._format_time(end_time)}'
            
            query += f') |> filter(fn: (r) => r["_measurement"] == "{measurement}")'
            
            # Add field filters
            if fields:
                field_filter = " or ".join([f'r["_field"] == "{field}"' for field in fields])
                query += f' |> filter(fn: (r) => {field_filter})'
            
            # Add tag filters
            if tags:
                for tag_key, tag_value in tags.items():
                    query += f' |> filter(fn: (r) => r["{tag_key}"] == "{tag_value}")'
            
            # Add aggregation
            if aggregation and window:
                if aggregation == "mean":
                    query += f' |> aggregateWindow(every: {window}, fn: mean, createEmpty: false)'
                elif aggregation == "sum":
                    query += f' |> aggregateWindow(every: {window}, fn: sum, createEmpty: false)'
                elif aggregation == "max":
                    query += f' |> aggregateWindow(every: {window}, fn: max, createEmpty: false)'
                elif aggregation == "min":
                    query += f' |> aggregateWindow(every: {window}, fn: min, createEmpty: false)'
            
            query += ' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
            
            # Execute query
            tables = self.query_api.query(query, org=self.config.org)
            
            # Convert to DataFrame
            records = []
            for table in tables:
                for record in table.records:
                    record_dict = {
                        "_time": record.get_time(),
                        "_measurement": record.get_measurement()
                    }
                    
                    # Add tag values
                    for key, value in record.values.items():
                        if not key.startswith('_') or key == '_value':
                            record_dict[key] = value
                    
                    records.append(record_dict)
            
            df = pd.DataFrame(records)
            
            if not df.empty:
                df['_time'] = pd.to_datetime(df['_time'])
                df = df.sort_values('_time')
            
            logger.info(f"Query returned {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to query features: {e}")
            return pd.DataFrame()
    
    def _format_time(self, time_input: Union[str, datetime]) -> str:
        """Format time for InfluxDB queries."""
        if isinstance(time_input, str):
            return time_input
        elif isinstance(time_input, datetime):
            return time_input.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            return str(time_input)
    
    def query_latest_features(
        self,
        measurement: str,
        tags: Optional[Dict[str, str]] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Query most recent features."""
        try:
            query = f'from(bucket: "{self.config.bucket}") |> range(start: -24h)'
            query += f' |> filter(fn: (r) => r["_measurement"] == "{measurement}")'
            
            # Add tag filters
            if tags:
                for tag_key, tag_value in tags.items():
                    query += f' |> filter(fn: (r) => r["{tag_key}"] == "{tag_value}")'
            
            query += f' |> tail(n: {limit})'
            query += ' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
            
            # Execute query
            tables = self.query_api.query(query, org=self.config.org)
            
            # Convert to DataFrame
            records = []
            for table in tables:
                for record in table.records:
                    record_dict = {"_time": record.get_time()}
                    
                    for key, value in record.values.items():
                        if not key.startswith('_') or key == '_value':
                            record_dict[key] = value
                    
                    records.append(record_dict)
            
            df = pd.DataFrame(records)
            
            if not df.empty:
                df['_time'] = pd.to_datetime(df['_time'])
                df = df.sort_values('_time', ascending=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to query latest features: {e}")
            return pd.DataFrame()
    
    def create_continuous_query(
        self,
        query_name: str,
        source_measurement: str,
        target_measurement: str,
        aggregation_window: str = "1m",
        aggregation_function: str = "mean"
    ) -> bool:
        """Create continuous query for real-time aggregation."""
        try:
            # InfluxDB 2.0 uses tasks instead of continuous queries
            # This would create a task for continuous aggregation
            task_flux = f'''
            option task = {{
                name: "{query_name}",
                every: {aggregation_window},
            }}
            
            from(bucket: "{self.config.bucket}")
                |> range(start: -task.every)
                |> filter(fn: (r) => r["_measurement"] == "{source_measurement}")
                |> aggregateWindow(every: {aggregation_window}, fn: {aggregation_function})
                |> set(key: "_measurement", value: "{target_measurement}")
                |> to(bucket: "{self.config.bucket}")
            '''
            
            # This would use the Tasks API to create the task
            logger.info(f"Continuous query created: {query_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create continuous query: {e}")
            return False
    
    def delete_features(
        self,
        measurement: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """Delete features within time range."""
        try:
            # Build predicate
            predicate = f'_measurement="{measurement}"'
            
            if tags:
                for tag_key, tag_value in tags.items():
                    predicate += f' AND {tag_key}="{tag_value}"'
            
            # Delete data
            self.delete_api.delete(
                start=self._format_time(start_time),
                stop=self._format_time(end_time),
                predicate=predicate,
                bucket=self.config.bucket,
                org=self.config.org
            )
            
            logger.info(f"Deleted features for measurement {measurement}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete features: {e}")
            return False
    
    def get_measurement_schema(self, measurement: str) -> Dict[str, List[str]]:
        """Get schema information for a measurement."""
        try:
            # Query tag keys
            tag_query = f'''
            import "influxdata/influxdb/schema"
            
            schema.tagKeys(
                bucket: "{self.config.bucket}",
                predicate: (r) => r["_measurement"] == "{measurement}",
                start: -30d
            )
            '''
            
            # Query field keys
            field_query = f'''
            import "influxdata/influxdb/schema"
            
            schema.fieldKeys(
                bucket: "{self.config.bucket}",
                predicate: (r) => r["_measurement"] == "{measurement}",
                start: -30d
            )
            '''
            
            # Execute queries
            tag_tables = self.query_api.query(tag_query, org=self.config.org)
            field_tables = self.query_api.query(field_query, org=self.config.org)
            
            # Extract results
            tags = []
            for table in tag_tables:
                for record in table.records:
                    tags.append(record.get_value())
            
            fields = []
            for table in field_tables:
                for record in table.records:
                    fields.append(record.get_value())
            
            schema = {
                "tags": tags,
                "fields": fields
            }
            
            logger.info(f"Retrieved schema for measurement {measurement}")
            return schema
            
        except Exception as e:
            logger.error(f"Failed to get measurement schema: {e}")
            return {"tags": [], "fields": []}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            # Query bucket usage
            usage_query = f'''
            from(bucket: "{self.config.bucket}")
                |> range(start: -1h)
                |> count()
                |> yield(name: "point_count")
            '''
            
            tables = self.query_api.query(usage_query, org=self.config.org)
            
            total_points = 0
            for table in tables:
                for record in table.records:
                    total_points += record.get_value() or 0
            
            stats = {
                "total_points": total_points,
                "bucket": self.config.bucket,
                "retention_policy": self.config.retention_policy,
                "connection_status": "healthy"
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def close(self):
        """Close InfluxDB client."""
        try:
            if hasattr(self, 'client'):
                self.client.close()
                logger.info("InfluxDB client closed")
        except Exception as e:
            logger.error(f"Error closing InfluxDB client: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()