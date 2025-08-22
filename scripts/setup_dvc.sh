#!/bin/bash
# Setup DVC for data versioning

echo "🔧 Setting up DVC..."

# Install DVC if not already installed
if ! command -v dvc &> /dev/null; then
    echo "Installing DVC..."
    pip install --user dvc[s3]
fi

# Initialize DVC if not already initialized
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
fi

# Configure DVC
echo "Configuring DVC..."
dvc config core.analytics false
dvc config core.autostage true
dvc config cache.type symlink

# Add data directories to DVC
echo "Adding data directories to DVC..."
dvc add data/raw --desc "Raw market data from Binance"
dvc add data/processed --desc "Processed features and labels"
dvc add artifacts/models --desc "Trained ML models"

# Configure remote storage (example with S3)
read -p "Do you want to configure S3 remote storage? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter S3 bucket name: " bucket
    read -p "Enter S3 region (default: us-east-1): " region
    region=${region:-us-east-1}
    
    dvc remote add -d s3remote s3://$bucket/ml-trading-pipeline
    dvc remote modify s3remote region $region
    
    echo "✅ S3 remote configured: s3://$bucket/ml-trading-pipeline"
fi

# Create pipeline stages
echo "Creating DVC pipeline..."
cat > dvc.yaml << EOF
stages:
  download_data:
    cmd: python -m src.data.binance_loader --symbol BTCUSDT --timeframe 15m
    deps:
      - src/data/binance_loader.py
    outs:
      - data/raw/BTCUSDT_15m.parquet
    params:
      - configs/data.yaml:
          - symbol
          - timeframe
          - start_date
          - end_date

  create_features:
    cmd: python -m src.features.engineering --input data/raw --output data/processed
    deps:
      - src/features/engineering.py
      - data/raw/
    outs:
      - data/processed/features.parquet
    params:
      - configs/data.yaml:
          - features
          - lookback_periods

  train_xgboost:
    cmd: python run_optimization.py --model xgboost
    deps:
      - src/models/xgb_optuna.py
      - data/processed/features.parquet
    outs:
      - artifacts/models/xgboost_optimized.pkl
    params:
      - configs/xgb.yaml
    metrics:
      - artifacts/reports/xgb_metrics.json:
          cache: false

  train_lstm:
    cmd: python run_optimization.py --model lstm
    deps:
      - src/models/lstm_optuna.py
      - data/processed/features.parquet
    outs:
      - artifacts/models/lstm_optimized.pkl
    params:
      - configs/lstm.yaml
    metrics:
      - artifacts/reports/lstm_metrics.json:
          cache: false
EOF

echo "✅ DVC setup complete!"
echo ""
echo "Next steps:"
echo "1. Review and commit .dvc files: git add *.dvc dvc.yaml .dvcignore"
echo "2. Push data to remote: dvc push"
echo "3. Run pipeline: dvc repro"
echo "4. View pipeline: dvc dag"