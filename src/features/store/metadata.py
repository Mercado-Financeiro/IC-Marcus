"""
Metadata store with lineage tracking and schema management.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import pandas as pd
from enum import Enum

logger = logging.getLogger(__name__)


class LineageType(Enum):
    """Type of lineage relationship."""
    DERIVED_FROM = "derived_from"
    TRANSFORMED_TO = "transformed_to"
    MERGED_FROM = "merged_from"
    SPLIT_TO = "split_to"
    AGGREGATED_FROM = "aggregated_from"
    FILTERED_FROM = "filtered_from"


@dataclass
class FeatureMetadata:
    """Metadata for a single feature."""
    name: str
    dtype: str
    description: str
    tags: List[str]
    created_at: datetime
    created_by: str
    business_meaning: str
    computation_logic: str
    dependencies: List[str]
    quality_metrics: Dict[str, float]
    usage_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureMetadata':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class LineageEdge:
    """Lineage relationship between features/feature groups."""
    source_id: str
    target_id: str
    lineage_type: LineageType
    created_at: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['lineage_type'] = self.lineage_type.value
        result['created_at'] = self.created_at.isoformat()
        return result


class MetadataStore:
    """
    Centralized metadata store with lineage tracking.
    
    Features:
    - Feature-level metadata
    - Schema evolution tracking
    - Data lineage graph
    - Quality metrics history
    - Usage analytics
    - Search and discovery
    """
    
    def __init__(self, store_path: str = "data/feature_store"):
        """Initialize metadata store."""
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # SQLite database for structured queries
        self.db_path = self.store_path / "metadata.db"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_database()
        
        # JSON files for complex metadata
        self.metadata_dir = self.store_path / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
    
    def _init_database(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()
        
        # Feature groups table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_groups (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                owner TEXT,
                created_at TEXT,
                updated_at TEXT,
                status TEXT,
                tags TEXT,
                metadata TEXT
            )
        """)
        
        # Features table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                id TEXT PRIMARY KEY,
                group_id TEXT,
                name TEXT NOT NULL,
                dtype TEXT,
                description TEXT,
                created_at TEXT,
                created_by TEXT,
                business_meaning TEXT,
                computation_logic TEXT,
                quality_score REAL,
                usage_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                tags TEXT,
                FOREIGN KEY (group_id) REFERENCES feature_groups (id)
            )
        """)
        
        # Schema versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT,
                version_id TEXT,
                schema_hash TEXT,
                schema_json TEXT,
                created_at TEXT,
                changes TEXT,
                FOREIGN KEY (group_id) REFERENCES feature_groups (id)
            )
        """)
        
        # Lineage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT,
                target_id TEXT,
                lineage_type TEXT,
                created_at TEXT,
                metadata TEXT
            )
        """)
        
        # Quality metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_id TEXT,
                version_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                created_at TEXT,
                FOREIGN KEY (feature_id) REFERENCES features (id)
            )
        """)
        
        # Usage analytics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_id TEXT,
                user_id TEXT,
                access_type TEXT,
                accessed_at TEXT,
                context TEXT,
                FOREIGN KEY (feature_id) REFERENCES features (id)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_features_group ON features (group_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineage_source ON lineage (source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineage_target ON lineage (target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_feature ON quality_metrics (feature_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_feature ON usage_analytics (feature_id)")
        
        self.conn.commit()
    
    def register_feature_group(
        self,
        group_id: str,
        name: str,
        description: str = "",
        owner: str = "system",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a new feature group.
        
        Args:
            group_id: Unique group identifier
            name: Human-readable name
            description: Group description
            owner: Group owner
            tags: Tags for categorization
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT OR REPLACE INTO feature_groups 
                (id, name, description, owner, created_at, updated_at, status, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                group_id, name, description, owner, now, now, "active",
                json.dumps(tags or []), json.dumps(metadata or {})
            ))
            
            self.conn.commit()
            logger.info(f"Registered feature group: {group_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register feature group: {e}")
            return False
    
    def register_features(
        self,
        group_id: str,
        features: List[FeatureMetadata]
    ) -> bool:
        """
        Register features for a group.
        
        Args:
            group_id: Feature group ID
            features: List of feature metadata
            
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            
            for feature in features:
                feature_id = f"{group_id}:{feature.name}"
                
                cursor.execute("""
                    INSERT OR REPLACE INTO features 
                    (id, group_id, name, dtype, description, created_at, created_by,
                     business_meaning, computation_logic, quality_score, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feature_id, group_id, feature.name, feature.dtype,
                    feature.description, feature.created_at.isoformat(),
                    feature.created_by, feature.business_meaning,
                    feature.computation_logic,
                    feature.quality_metrics.get('overall_score', 0.0),
                    json.dumps(feature.tags)
                ))
                
                # Store quality metrics
                for metric_name, metric_value in feature.quality_metrics.items():
                    cursor.execute("""
                        INSERT INTO quality_metrics 
                        (feature_id, metric_name, metric_value, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (feature_id, metric_name, metric_value, datetime.now().isoformat()))
            
            self.conn.commit()
            logger.info(f"Registered {len(features)} features for group {group_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register features: {e}")
            return False
    
    def add_schema_version(
        self,
        group_id: str,
        version_id: str,
        schema: Dict[str, Any],
        changes: Optional[List[str]] = None
    ) -> bool:
        """
        Add schema version for tracking evolution.
        
        Args:
            group_id: Feature group ID
            version_id: Version ID
            schema: Schema dictionary
            changes: List of changes made
            
        Returns:
            True if successful
        """
        try:
            import hashlib
            
            schema_json = json.dumps(schema, sort_keys=True)
            schema_hash = hashlib.sha256(schema_json.encode()).hexdigest()[:16]
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO schema_versions 
                (group_id, version_id, schema_hash, schema_json, created_at, changes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                group_id, version_id, schema_hash, schema_json,
                datetime.now().isoformat(), json.dumps(changes or [])
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to add schema version: {e}")
            return False
    
    def add_lineage(
        self,
        source_id: str,
        target_id: str,
        lineage_type: LineageType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add lineage relationship.
        
        Args:
            source_id: Source feature/group ID
            target_id: Target feature/group ID
            lineage_type: Type of relationship
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO lineage (source_id, target_id, lineage_type, created_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                source_id, target_id, lineage_type.value,
                datetime.now().isoformat(), json.dumps(metadata or {})
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to add lineage: {e}")
            return False
    
    def track_usage(
        self,
        feature_id: str,
        user_id: str,
        access_type: str = "read",
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Track feature usage for analytics.
        
        Args:
            feature_id: Feature ID
            user_id: User identifier
            access_type: Type of access (read, write, transform)
            context: Additional context
        """
        try:
            cursor = self.conn.cursor()
            
            # Add usage record
            cursor.execute("""
                INSERT INTO usage_analytics 
                (feature_id, user_id, access_type, accessed_at, context)
                VALUES (?, ?, ?, ?, ?)
            """, (
                feature_id, user_id, access_type,
                datetime.now().isoformat(), json.dumps(context or {})
            ))
            
            # Update usage count and last accessed
            cursor.execute("""
                UPDATE features 
                SET usage_count = usage_count + 1, last_accessed = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), feature_id))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to track usage: {e}")
    
    def search_features(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        owner: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search features by name, description, or tags.
        
        Args:
            query: Search query
            tags: Required tags
            owner: Feature owner filter
            limit: Maximum results
            
        Returns:
            List of matching features
        """
        try:
            cursor = self.conn.cursor()
            
            # Build query conditions
            conditions = []
            params = []
            
            if query:
                conditions.append("(f.name LIKE ? OR f.description LIKE ? OR f.business_meaning LIKE ?)")
                query_pattern = f"%{query}%"
                params.extend([query_pattern, query_pattern, query_pattern])
            
            if owner:
                conditions.append("fg.owner = ?")
                params.append(owner)
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            sql = f"""
                SELECT f.*, fg.name as group_name, fg.owner
                FROM features f
                JOIN feature_groups fg ON f.group_id = fg.id
                {where_clause}
                ORDER BY f.quality_score DESC, f.usage_count DESC
                LIMIT ?
            """
            params.append(limit)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            columns = [desc[0] for desc in cursor.description]
            results = []
            
            for row in rows:
                result = dict(zip(columns, row))
                
                # Filter by tags if specified
                if tags:
                    feature_tags = json.loads(result.get('tags', '[]'))
                    if not all(tag in feature_tags for tag in tags):
                        continue
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search features: {e}")
            return []
    
    def get_lineage_graph(self, feature_id: str, depth: int = 3) -> Dict[str, Any]:
        """
        Get lineage graph for a feature.
        
        Args:
            feature_id: Feature ID
            depth: Maximum depth to traverse
            
        Returns:
            Lineage graph data
        """
        try:
            cursor = self.conn.cursor()
            
            nodes = set()
            edges = []
            visited = set()
            
            def traverse(node_id: str, current_depth: int, direction: str):
                if current_depth > depth or node_id in visited:
                    return
                
                visited.add(node_id)
                nodes.add(node_id)
                
                # Get upstream/downstream relationships
                if direction in ['upstream', 'both']:
                    cursor.execute("""
                        SELECT source_id, lineage_type, metadata 
                        FROM lineage WHERE target_id = ?
                    """, (node_id,))
                    
                    for source_id, lineage_type, metadata in cursor.fetchall():
                        edges.append({
                            'source': source_id,
                            'target': node_id,
                            'type': lineage_type,
                            'metadata': json.loads(metadata or '{}')
                        })
                        traverse(source_id, current_depth + 1, direction)
                
                if direction in ['downstream', 'both']:
                    cursor.execute("""
                        SELECT target_id, lineage_type, metadata 
                        FROM lineage WHERE source_id = ?
                    """, (node_id,))
                    
                    for target_id, lineage_type, metadata in cursor.fetchall():
                        edges.append({
                            'source': node_id,
                            'target': target_id,
                            'type': lineage_type,
                            'metadata': json.loads(metadata or '{}')
                        })
                        traverse(target_id, current_depth + 1, direction)
            
            # Start traversal
            traverse(feature_id, 0, 'both')
            
            # Get node details
            node_details = {}
            if nodes:
                placeholders = ','.join('?' * len(nodes))
                cursor.execute(f"""
                    SELECT id, name, dtype, description, quality_score
                    FROM features WHERE id IN ({placeholders})
                """, list(nodes))
                
                for row in cursor.fetchall():
                    node_details[row[0]] = {
                        'id': row[0],
                        'name': row[1],
                        'type': 'feature',
                        'dtype': row[2],
                        'description': row[3],
                        'quality_score': row[4]
                    }
            
            return {
                'nodes': list(nodes),
                'edges': edges,
                'node_details': node_details,
                'center_node': feature_id
            }
            
        except Exception as e:
            logger.error(f"Failed to get lineage graph: {e}")
            return {'nodes': [], 'edges': [], 'node_details': {}}
    
    def get_feature_stats(self, feature_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get feature statistics and usage analytics.
        
        Args:
            feature_id: Feature ID
            days: Number of days for analytics
            
        Returns:
            Feature statistics
        """
        try:
            cursor = self.conn.cursor()
            
            # Basic feature info
            cursor.execute("""
                SELECT * FROM features WHERE id = ?
            """, (feature_id,))
            
            row = cursor.fetchone()
            if not row:
                return {}
            
            columns = [desc[0] for desc in cursor.description]
            feature_info = dict(zip(columns, row))
            
            # Quality metrics history
            cursor.execute("""
                SELECT metric_name, metric_value, created_at
                FROM quality_metrics 
                WHERE feature_id = ? AND created_at > ?
                ORDER BY created_at DESC
            """, (feature_id, (datetime.now() - timedelta(days=days)).isoformat()))
            
            quality_history = [
                {'metric': row[0], 'value': row[1], 'timestamp': row[2]}
                for row in cursor.fetchall()
            ]
            
            # Usage analytics
            cursor.execute("""
                SELECT COUNT(*) as usage_count,
                       COUNT(DISTINCT user_id) as unique_users,
                       access_type
                FROM usage_analytics 
                WHERE feature_id = ? AND accessed_at > ?
                GROUP BY access_type
            """, (feature_id, (datetime.now() - timedelta(days=days)).isoformat()))
            
            usage_stats = {
                row[2]: {'usage_count': row[0], 'unique_users': row[1]}
                for row in cursor.fetchall()
            }
            
            return {
                'feature_info': feature_info,
                'quality_history': quality_history,
                'usage_stats': usage_stats,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Failed to get feature stats: {e}")
            return {}
    
    def get_schema_evolution(self, group_id: str) -> List[Dict[str, Any]]:
        """
        Get schema evolution history for a feature group.
        
        Args:
            group_id: Feature group ID
            
        Returns:
            List of schema versions with changes
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT version_id, schema_hash, schema_json, created_at, changes
                FROM schema_versions
                WHERE group_id = ?
                ORDER BY created_at DESC
            """, (group_id,))
            
            versions = []
            for row in cursor.fetchall():
                versions.append({
                    'version_id': row[0],
                    'schema_hash': row[1],
                    'schema': json.loads(row[2]),
                    'created_at': row[3],
                    'changes': json.loads(row[4])
                })
            
            return versions
            
        except Exception as e:
            logger.error(f"Failed to get schema evolution: {e}")
            return []
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()