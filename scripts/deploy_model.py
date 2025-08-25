#!/usr/bin/env python3
"""
Deploy trained model to production.
Manages model versioning and champion/challenger setup.
"""

import sys
import os
from pathlib import Path
import shutil
import pickle
import yaml
import json
from datetime import datetime
import hashlib
import argparse

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from src.mlops.model_registry import ModelRegistry


class ModelDeployer:
    """Manage model deployment and versioning."""
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        """Initialize model deployer.
        
        Args:
            artifacts_dir: Directory for model artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = self.artifacts_dir / "models"
        self.production_dir = self.models_dir / "production"
        self.staging_dir = self.models_dir / "staging"
        self.archive_dir = self.models_dir / "archive"
        
        # Create directories
        for dir_path in [self.production_dir, self.staging_dir, self.archive_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.registry = ModelRegistry()
    
    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _load_model_metadata(self, model_path: Path) -> dict:
        """Load metadata from model file."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract metadata
        metadata = {
            'model_type': type(model_data.get('model')).__name__,
            'thresholds': model_data.get('thresholds', {}),
            'params': model_data.get('params', {}),
            'feature_count': len(model_data.get('feature_cols', [])),
            'file_hash': self._calculate_hash(model_path),
            'file_size_mb': model_path.stat().st_size / (1024 * 1024)
        }
        
        return metadata
    
    def deploy_to_staging(self, model_path: str, model_name: str = None) -> dict:
        """Deploy model to staging environment.
        
        Args:
            model_path: Path to trained model
            model_name: Optional model name
            
        Returns:
            Deployment info
        """
        source = Path(model_path)
        if not source.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Generate model name if not provided
        if model_name is None:
            model_name = f"{source.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load metadata
        metadata = self._load_model_metadata(source)
        
        # Copy to staging
        dest = self.staging_dir / f"{model_name}.pkl"
        shutil.copy2(source, dest)
        
        # Save metadata
        meta_path = self.staging_dir / f"{model_name}_metadata.json"
        metadata['model_name'] = model_name
        metadata['deployed_at'] = datetime.now().isoformat()
        metadata['environment'] = 'staging'
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model deployed to staging: {model_name}")
        print(f"  • Type: {metadata['model_type']}")
        print(f"  • Features: {metadata['feature_count']}")
        print(f"  • Size: {metadata['file_size_mb']:.2f} MB")
        print(f"  • Hash: {metadata['file_hash'][:8]}...")
        
        return metadata
    
    def promote_to_production(self, model_name: str, backup_current: bool = True) -> dict:
        """Promote staging model to production.
        
        Args:
            model_name: Name of model in staging
            backup_current: Whether to backup current production model
            
        Returns:
            Promotion info
        """
        # Check staging model exists
        staging_model = self.staging_dir / f"{model_name}.pkl"
        staging_meta = self.staging_dir / f"{model_name}_metadata.json"
        
        if not staging_model.exists():
            raise FileNotFoundError(f"Model not found in staging: {model_name}")
        
        # Backup current production model
        if backup_current:
            current_prod = list(self.production_dir.glob("*.pkl"))
            if current_prod:
                for model_file in current_prod:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    archive_name = f"{model_file.stem}_archived_{timestamp}{model_file.suffix}"
                    shutil.move(model_file, self.archive_dir / archive_name)
                    print(f"📦 Archived: {model_file.name} → {archive_name}")
        
        # Move to production
        prod_model = self.production_dir / f"{model_name}.pkl"
        prod_meta = self.production_dir / f"{model_name}_metadata.json"
        
        shutil.move(staging_model, prod_model)
        if staging_meta.exists():
            with open(staging_meta, 'r') as f:
                metadata = json.load(f)
            metadata['promoted_at'] = datetime.now().isoformat()
            metadata['environment'] = 'production'
            with open(prod_meta, 'w') as f:
                json.dump(metadata, f, indent=2)
            staging_meta.unlink()
        
        print(f"🚀 Model promoted to production: {model_name}")
        print(f"  • Location: {prod_model}")
        
        return metadata
    
    def rollback_production(self) -> dict:
        """Rollback to previous production model."""
        # Find most recent archived model
        archived = sorted(self.archive_dir.glob("*.pkl"), 
                         key=lambda x: x.stat().st_mtime, 
                         reverse=True)
        
        if not archived:
            raise ValueError("No archived models found for rollback")
        
        # Move current production to staging
        current_prod = list(self.production_dir.glob("*.pkl"))
        if current_prod:
            for model_file in current_prod:
                staging_dest = self.staging_dir / f"rollback_{model_file.name}"
                shutil.move(model_file, staging_dest)
                print(f"↩️ Moved current to staging: {model_file.name}")
        
        # Restore from archive
        restore_model = archived[0]
        prod_dest = self.production_dir / restore_model.name.replace("_archived", "")
        shutil.copy2(restore_model, prod_dest)
        
        print(f"✅ Rollback complete: {restore_model.name} → production")
        
        return {'restored_model': restore_model.name}
    
    def list_models(self) -> dict:
        """List all models in different environments."""
        models = {
            'production': [],
            'staging': [],
            'archive': []
        }
        
        for env, dir_path in [
            ('production', self.production_dir),
            ('staging', self.staging_dir),
            ('archive', self.archive_dir)
        ]:
            for model_file in dir_path.glob("*.pkl"):
                info = {
                    'name': model_file.stem,
                    'size_mb': model_file.stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                }
                
                # Load metadata if exists
                meta_file = dir_path / f"{model_file.stem}_metadata.json"
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        info.update(json.load(f))
                
                models[env].append(info)
        
        return models
    
    def compare_models(self, model1_path: str, model2_path: str) -> dict:
        """Compare two models.
        
        Args:
            model1_path: Path to first model
            model2_path: Path to second model
            
        Returns:
            Comparison results
        """
        meta1 = self._load_model_metadata(Path(model1_path))
        meta2 = self._load_model_metadata(Path(model2_path))
        
        comparison = {
            'model1': {
                'path': model1_path,
                'metadata': meta1
            },
            'model2': {
                'path': model2_path,
                'metadata': meta2
            },
            'differences': {
                'feature_count': meta2['feature_count'] - meta1['feature_count'],
                'size_diff_mb': meta2['file_size_mb'] - meta1['file_size_mb'],
                'same_type': meta1['model_type'] == meta2['model_type']
            }
        }
        
        return comparison


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Model Deployment Manager')
    parser.add_argument('action', choices=['deploy', 'promote', 'rollback', 'list', 'compare'],
                       help='Action to perform')
    parser.add_argument('--model', help='Model file path or name')
    parser.add_argument('--name', help='Model name for deployment')
    parser.add_argument('--model2', help='Second model for comparison')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup on promotion')
    
    args = parser.parse_args()
    
    deployer = ModelDeployer()
    
    if args.action == 'deploy':
        if not args.model:
            print("❌ Error: --model required for deploy")
            sys.exit(1)
        deployer.deploy_to_staging(args.model, args.name)
    
    elif args.action == 'promote':
        if not args.model:
            print("❌ Error: --model required for promote")
            sys.exit(1)
        deployer.promote_to_production(args.model, not args.no_backup)
    
    elif args.action == 'rollback':
        deployer.rollback_production()
    
    elif args.action == 'list':
        models = deployer.list_models()
        
        for env in ['production', 'staging', 'archive']:
            print(f"\n📁 {env.upper()}:")
            if models[env]:
                for model in models[env]:
                    print(f"  • {model['name']} ({model['size_mb']:.2f} MB)")
                    if 'model_type' in model:
                        print(f"    Type: {model['model_type']}, Features: {model.get('feature_count', 'N/A')}")
            else:
                print("  (empty)")
    
    elif args.action == 'compare':
        if not args.model or not args.model2:
            print("❌ Error: --model and --model2 required for compare")
            sys.exit(1)
        
        result = deployer.compare_models(args.model, args.model2)
        
        print("\n📊 MODEL COMPARISON:")
        print(f"\nModel 1: {args.model}")
        print(f"  • Type: {result['model1']['metadata']['model_type']}")
        print(f"  • Features: {result['model1']['metadata']['feature_count']}")
        print(f"  • Size: {result['model1']['metadata']['file_size_mb']:.2f} MB")
        
        print(f"\nModel 2: {args.model2}")
        print(f"  • Type: {result['model2']['metadata']['model_type']}")
        print(f"  • Features: {result['model2']['metadata']['feature_count']}")
        print(f"  • Size: {result['model2']['metadata']['file_size_mb']:.2f} MB")
        
        print(f"\nDifferences:")
        print(f"  • Feature count diff: {result['differences']['feature_count']:+d}")
        print(f"  • Size diff: {result['differences']['size_diff_mb']:+.2f} MB")
        print(f"  • Same type: {'✅' if result['differences']['same_type'] else '❌'}")


if __name__ == "__main__":
    main()