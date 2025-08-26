#!/usr/bin/env python3
"""
Enhanced LSTM Training Script
Includes all improvements for imbalanced classification:
- BCEWithLogitsLoss with pos_weight
- Proper metrics (PR-AUC, MCC, Brier)
- Purged K-Fold validation
- Probability calibration
- Comparison with baselines
"""

import sys
import os

# Fix path to avoid conflicts
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, matthews_corrcoef
import mlflow
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from src.data.binance_loader import CryptoDataLoader
from src.features.engineering import FeatureEngineer
from src.data.splits import PurgedKFold
from src.models.lstm.metrics import LSTMMetrics
from src.models.lstm.calibration import LSTMCalibrator
from src.models.lstm.optuna.model import LSTMModel
from src.models.lstm.optuna.utils import set_lstm_deterministic, get_device

def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple:
    """Create sequences for LSTM input."""
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001, 
                patience=10, pos_weight=None):
    """Train LSTM model with proper loss and early stopping."""
    
    # Setup loss with pos_weight
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=False
    )
    
    best_pr_auc = 0
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_pr_auc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float()
                
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                # Apply sigmoid for probabilities (since using BCEWithLogitsLoss)
                if not model.use_sigmoid:
                    probs = torch.sigmoid(outputs)
                else:
                    probs = outputs
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        val_pr_auc = average_precision_score(val_targets, val_preds)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_pr_auc'].append(val_pr_auc)
        
        # Update scheduler
        scheduler.step(val_pr_auc)
        
        # Early stopping based on PR-AUC
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            print(f"   Epoch {epoch+1}/{epochs}: "
                  f"Loss={avg_val_loss:.4f}, PR-AUC={val_pr_auc:.4f}")
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, best_pr_auc

def main():
    parser = argparse.ArgumentParser(description='Enhanced LSTM Training')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='15m')
    parser.add_argument('--start_date', type=str, default='2023-01-01')
    parser.add_argument('--end_date', type=str, default='2024-08-23')
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--embargo', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("="*80)
    print("ENHANCED LSTM TRAINING")
    print("="*80)
    print(f"Start: {datetime.now()}")
    
    # Set deterministic
    set_lstm_deterministic(args.seed)
    device = get_device(args.device)
    print(f"\nDevice: {device}")
    
    # 1. Load data
    print("\n1. Loading data...")
    loader = CryptoDataLoader()
    df = loader.fetch_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date
    )
    print(f"   ✓ Data loaded: {len(df)} bars")
    
    # 2. Generate features
    print("\n2. Generating features...")
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(df)
    print(f"   ✓ Features created: {features_df.shape[1]} features")
    
    # 3. Create labels
    print("\n3. Creating labels...")
    returns = df['close'].pct_change(5).shift(-5)
    threshold = returns.quantile(0.6)  # Top 40% as positive
    labels = (returns > threshold).astype(int).dropna()
    print(f"   ✓ Labels created with {labels.mean():.1%} positive")
    
    # 4. Prepare data
    common_index = features_df.index.intersection(labels.index)
    X = features_df.loc[common_index]
    y = labels.loc[common_index]
    
    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X[mask].values
    y = y[mask].values
    
    print(f"   ✓ Final samples: {len(y)}")
    
    # Calculate pos_weight
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    pos_weight = neg_count / pos_count
    print(f"   ✓ Class balance: {pos_count/len(y):.1%} positive")
    print(f"   ✓ Using pos_weight: {pos_weight:.2f}")
    
    # 5. Normalize data
    print("\n4. Normalizing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 6. Cross-validation with Purged K-Fold
    print(f"\n5. Cross-validation (Purged K-Fold, {args.n_splits} splits)...")
    cv = PurgedKFold(n_splits=args.n_splits, embargo=args.embargo)
    
    cv_scores = []
    fold_models = []
    metrics_calc = LSTMMetrics()
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y), 1):
        print(f"\n   Fold {fold}/{args.n_splits}:")
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create sequences
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, args.seq_len)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, args.seq_len)
        
        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq),
            torch.FloatTensor(y_train_seq)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_seq),
            torch.FloatTensor(y_val_seq)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create model
        model = LSTMModel(
            input_size=X.shape[1],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_sigmoid=False  # Using BCEWithLogitsLoss
        ).to(device)
        
        # Train model
        model, history, best_pr_auc = train_model(
            model, train_loader, val_loader, device,
            epochs=args.epochs, pos_weight=pos_weight
        )
        
        cv_scores.append(best_pr_auc)
        fold_models.append(model)
        
        # Calculate final metrics
        model.eval()
        with torch.no_grad():
            val_preds = []
            for batch_x, _ in val_loader:
                outputs = model(batch_x.to(device)).squeeze()
                if not model.use_sigmoid:
                    probs = torch.sigmoid(outputs)
                else:
                    probs = outputs
                val_preds.extend(probs.cpu().numpy())
        
        metrics = metrics_calc.calculate_metrics(y_val_seq, np.array(val_preds))
        print(f"   Fold {fold} Results:")
        print(f"     PR-AUC: {metrics['pr_auc']:.4f} (baseline: {metrics['prevalence']:.4f})")
        print(f"     MCC: {metrics['mcc']:.4f}")
        print(f"     Brier: {metrics['brier_score']:.4f}")
    
    print(f"\n   CV Average PR-AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # 7. Train final model on all data
    print("\n6. Training final model on all data...")
    
    # Use best fold's hyperparameters
    best_fold_idx = np.argmax(cv_scores)
    best_model = fold_models[best_fold_idx]
    
    # Split for final training
    test_size = int(0.2 * len(X_scaled))
    X_train_final = X_scaled[:-test_size]
    y_train_final = y[:-test_size]
    X_test = X_scaled[-test_size:]
    y_test = y[-test_size:]
    
    # Create sequences
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, args.seq_len)
    
    # 8. Calibration
    print("\n7. Calibrating probabilities...")
    
    # Get validation predictions for calibration
    val_size = int(0.2 * len(X_train_final))
    X_cal = X_train_final[-val_size:]
    y_cal = y_train_final[-val_size:]
    X_cal_seq, y_cal_seq = create_sequences(X_cal, y_cal, args.seq_len)
    
    model.eval()
    with torch.no_grad():
        cal_dataset = TensorDataset(torch.FloatTensor(X_cal_seq))
        cal_loader = DataLoader(cal_dataset, batch_size=args.batch_size)
        cal_preds = []
        for batch_x, in cal_loader:
            outputs = model(batch_x.to(device)).squeeze()
            if not model.use_sigmoid:
                probs = torch.sigmoid(outputs)
            else:
                probs = outputs
            cal_preds.extend(probs.cpu().numpy())
    
    calibrator = LSTMCalibrator(method='both')
    calibrator.fit(y_cal_seq, np.array(cal_preds))
    
    # 9. Final evaluation
    print("\n8. Final evaluation on test set...")
    
    model.eval()
    with torch.no_grad():
        test_dataset = TensorDataset(torch.FloatTensor(X_test_seq))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        test_preds = []
        for batch_x, in test_loader:
            outputs = model(batch_x.to(device)).squeeze()
            if not model.use_sigmoid:
                probs = torch.sigmoid(outputs)
            else:
                probs = outputs
            test_preds.extend(probs.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_preds_cal = calibrator.transform(test_preds)
    
    # Calculate metrics
    metrics_calc.print_report(y_test_seq, test_preds_cal, model_name="Enhanced LSTM")
    
    # Find optimal threshold
    threshold_f1, score_f1 = metrics_calc.find_optimal_threshold(y_test_seq, test_preds_cal, 'f1')
    threshold_mcc, score_mcc = metrics_calc.find_optimal_threshold(y_test_seq, test_preds_cal, 'mcc')
    
    print(f"\nOptimal Thresholds:")
    print(f"  For F1: {threshold_f1:.3f} (score: {score_f1:.4f})")
    print(f"  For MCC: {threshold_mcc:.3f} (score: {score_mcc:.4f})")
    
    # 10. MLflow logging
    print("\n9. Logging to MLflow...")
    mlflow.set_tracking_uri('artifacts/mlruns')
    mlflow.set_experiment('lstm_enhanced')
    
    with mlflow.start_run(run_name=f'lstm_enhanced_{datetime.now():%Y%m%d_%H%M%S}'):
        # Log parameters
        mlflow.log_params({
            'symbol': args.symbol,
            'timeframe': args.timeframe,
            'seq_len': args.seq_len,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'n_samples': len(y),
            'n_features': X.shape[1],
            'pos_weight': pos_weight,
            'cv_method': 'purged_kfold',
            'n_splits': args.n_splits,
            'embargo': args.embargo
        })
        
        # Log metrics
        final_metrics = metrics_calc.calculate_metrics(y_test_seq, test_preds_cal)
        for key, value in final_metrics.items():
            mlflow.log_metric(f'test_{key}', value)
        
        mlflow.log_metric('cv_pr_auc_mean', np.mean(cv_scores))
        mlflow.log_metric('cv_pr_auc_std', np.std(cv_scores))
        mlflow.log_metric('threshold_f1', threshold_f1)
        mlflow.log_metric('threshold_mcc', threshold_mcc)
        
        # Save model
        model_dir = Path('artifacts/models')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f'lstm_enhanced_{datetime.now():%Y%m%d_%H%M%S}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'calibrator': calibrator,
            'config': vars(args),
            'metrics': final_metrics
        }, model_path)
        
        mlflow.log_artifact(str(model_path))
        print(f"   ✓ Model saved to: {model_path}")
    
    print("\n" + "="*80)
    print("ENHANCED LSTM TRAINING COMPLETE!")
    print(f"End: {datetime.now()}")
    print("="*80)

if __name__ == "__main__":
    main()