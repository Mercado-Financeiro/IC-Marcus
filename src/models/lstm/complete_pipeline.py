"""Complete LSTM pipeline with Optuna optimization and calibration."""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
import warnings
from pathlib import Path
import joblib

# ML imports
from sklearn.preprocessing import StandardScaler
# Use temporal validator with embargo
from src.features.validation.temporal import TemporalValidator, TemporalValidationConfig
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, precision_recall_curve, brier_score_loss
)
from sklearn.isotonic import IsotonicRegression

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available")

# Optuna
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available")

# MLflow
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

warnings.filterwarnings('ignore')


def make_sequences(X_df: pd.DataFrame, y_series: pd.Series, seq_len: int):
    """
    Create sequences for LSTM from tabular data.
    
    Args:
        X_df: Features DataFrame
        y_series: Labels Series
        seq_len: Sequence length
        
    Returns:
        X_seq: Array of sequences [n_samples, seq_len, n_features]
        y_seq: Array of labels
        idx_seq: Corresponding pandas indices
    """
    X = X_df.values.astype(np.float32)
    y = y_series.values.astype(np.int64)
    idx = X_df.index

    X_seq, y_seq, idx_seq = [], [], []
    for t in range(seq_len, len(X)):
        X_seq.append(X[t-seq_len:t])
        y_seq.append(y[t])
        idx_seq.append(idx[t])
    
    return np.array(X_seq), np.array(y_seq), pd.Index(idx_seq)


def train_val_test_split_time(X_df: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """
    Temporal split for train/validation/test.
    
    Args:
        X_df: Features
        y: Labels
        n_splits: Number of splits for TimeSeriesSplit
        
    Returns:
        tr_idx: Training indices
        va_idx: Validation indices
        te_idx: Test indices
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X_df))
    
    # Second to last split for validation
    (tr_idx, va_idx) = splits[-2]
    # Last split for test
    (tr2_idx, te_idx) = splits[-1]
    
    # Training = from start to end of second to last split
    tr_idx_full = np.arange(0, va_idx[-1] + 1)
    
    return tr_idx_full, va_idx, te_idx


def build_lstm_tensors(X_df: pd.DataFrame, y: pd.Series, seq_len: int, 
                      tr_idx, va_idx, te_idx):
    """
    Prepare tensors for LSTM with scaling.
    
    Args:
        X_df: Features
        y: Labels
        seq_len: Sequence length
        tr_idx, va_idx, te_idx: Split indices
        
    Returns:
        Tuple with train/val/test tensors, indices and scaler
    """
    # Scale features based on training data
    scaler = StandardScaler()
    X_train_df = X_df.iloc[tr_idx]
    scaler.fit(X_train_df.values)
    
    # Apply scaling
    Xs = pd.DataFrame(
        scaler.transform(X_df.values), 
        index=X_df.index, 
        columns=X_df.columns
    )
    
    # Generate sequences
    X_seq, y_seq, idx_seq = make_sequences(Xs, y, seq_len)
    
    # Map original indices to sequence indices
    idx_map = pd.Series(range(len(idx_seq)), index=idx_seq)
    
    # Get sequence indices for each split
    tr = idx_map[idx_seq.intersection(X_df.index[tr_idx])].dropna().astype(int).values
    va = idx_map[idx_seq.intersection(X_df.index[va_idx])].dropna().astype(int).values
    te = idx_map[idx_seq.intersection(X_df.index[te_idx])].dropna().astype(int).values
    
    # Separate tensors
    Xtr, ytr = X_seq[tr], y_seq[tr]
    Xva, yva = X_seq[va], y_seq[va]
    Xte, yte = X_seq[te], y_seq[te]
    
    return (Xtr, ytr, Xva, yva, Xte, yte, idx_seq[te], scaler)


if TORCH_AVAILABLE:
    class LSTMClassifier(nn.Module):
        """LSTM for binary time series classification."""
        
        def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                in_dim, hidden, 
                num_layers=layers, 
                batch_first=True, 
                dropout=dropout if layers > 1 else 0
            )
            self.head = nn.Linear(hidden, 1)
            
        def forward(self, x):
            # x: [batch, seq, feat]
            out, _ = self.lstm(x)
            # Use only last timestep
            logit = self.head(out[:, -1, :])
            return logit.squeeze(1)  # Return logits (no sigmoid)


if TORCH_AVAILABLE:
    def train_one_epoch(model, optimizer, loss_fn, loader, device):
        """
        Train model for one epoch.
        
        Args:
            model: LSTM model
            optimizer: Optimizer
            loss_fn: Loss function
            loader: DataLoader
            device: Device (cuda/cpu)
            
        Returns:
            Average loss for epoch
        """
        model.train()
        total_loss = 0.0
        
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).float()
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            
        return total_loss / len(loader.dataset)
    
    @torch.no_grad()
    def eval_ap(model, loader, device):
        """
        Evaluate model with Average Precision (PR-AUC).
        
        Args:
            model: LSTM model
            loader: DataLoader
            device: Device
            
        Returns:
            ap: Average Precision score
            probs: Predicted probabilities
            labels: True labels
        """
        model.eval()
        probs_list, labels_list = [], []
        
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
            labels_list.append(yb.numpy())
        
        labels = np.concatenate(labels_list)
        probs = np.concatenate(probs_list)
        
        ap = average_precision_score(labels, probs)
        return ap, probs, labels


if TORCH_AVAILABLE and OPTUNA_AVAILABLE:
    def objective_lstm(trial, X_train, y_train, X_val, y_val, seq_len, device):
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial
            X_train, y_train: Training data
            X_val, y_val: Validation data
            seq_len: Sequence length
            device: Device (cuda/cpu)
            
        Returns:
            Best Average Precision achieved
        """
        # Hyperparameters to optimize
        hidden = trial.suggest_categorical("hidden", [64, 128, 256])
        layers = trial.suggest_int("layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        
        # Create model
        model = LSTMClassifier(
            X_train.shape[-1], hidden, layers, dropout
        ).to(device)
        
        # BCEWithLogitsLoss with weight for imbalance
        pos_ratio = y_train.mean()
        pos_ratio = float(pos_ratio if pos_ratio > 0 else 1e-6)
        pos_weight = torch.tensor((1 - pos_ratio) / pos_ratio, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=wd
        )
        
        # DataLoaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=False,  # NEVER shuffle time series data!
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=512, 
            shuffle=False
        )
        
        # Train with early stopping via pruning
        best_ap = 0.0
        patience_counter = 0
        patience = 10
        
        for epoch in range(60):
            # Train
            train_loss = train_one_epoch(
                model, optimizer, loss_fn, train_loader, device
            )
            
            # Evaluate
            ap, _, _ = eval_ap(model, val_loader, device)
            
            # Report to Optuna
            trial.report(ap, epoch)
            
            # Pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Track best
            if ap > best_ap:
                best_ap = ap
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                break
                
        return best_ap


if TORCH_AVAILABLE and OPTUNA_AVAILABLE:
    def fit_lstm_with_optuna(Xtr, ytr, Xva, yva, n_trials=50, 
                            device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Train LSTM with Bayesian optimization.
        
        Args:
            Xtr, ytr: Training data
            Xva, yva: Validation data
            n_trials: Number of Optuna trials
            device: Device for training
            
        Returns:
            model: Trained model
            best_params: Best hyperparameters
        """
        print(f"\n🔍 Bayesian optimization with Optuna ({n_trials} trials)")
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Optimize
        study.optimize(
            lambda t: objective_lstm(t, Xtr, ytr, Xva, yva, Xtr.shape[1], device),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        print(f"✅ Best AP in validation: {study.best_value:.4f}")
        print(f"📊 Best parameters: {best_params}")
        
        # Re-train with best parameters on combined dataset (train + val)
        print(f"\n🏋️ Training final model...")
        
        model = LSTMClassifier(
            Xtr.shape[-1], 
            best_params['hidden'],
            best_params['layers'],
            best_params['dropout']
        ).to(device)
        
        # Combine train and validation
        X_combined = np.concatenate([Xtr, Xva])
        y_combined = np.concatenate([ytr, yva])
        
        # Loss with weight to maintain class balance
        pos_ratio = y_combined.mean()
        pos_ratio = float(pos_ratio if pos_ratio > 0 else 1e-6)
        pos_weight = torch.tensor((1 - pos_ratio) / pos_ratio, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=best_params['lr'],
            weight_decay=best_params['weight_decay']
        )
        
        combined_dataset = TensorDataset(
            torch.tensor(X_combined, dtype=torch.float32),
            torch.tensor(y_combined, dtype=torch.float32)
        )
        
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=best_params['batch_size'],
            shuffle=True,
            drop_last=True
        )
        
        # Train for more epochs (1.5x)
        for epoch in range(90):
            train_loss = train_one_epoch(
                model, optimizer, loss_fn, combined_loader, device
            )
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {train_loss:.4f}")
        
        return model, best_params


if TORCH_AVAILABLE:
    @torch.no_grad()
    def calibrate_and_choose_threshold(model, Xva, yva, device):
        """
        Calibrate probabilities and choose optimal threshold.
        
        Args:
            model: Trained LSTM model
            Xva, yva: Validation data
            device: Device
            
        Returns:
            calibrator: Isotonic calibrator
            threshold: Optimal threshold
        """
        model.eval()
        
        # Get probabilities
        Xva_tensor = torch.tensor(Xva, dtype=torch.float32).to(device)
        logits = model(Xva_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        
        # Isotonic calibration
        calibrator = IsotonicRegression(out_of_bounds='clip')
        probs_cal = calibrator.fit_transform(probs, yva)
        
        # Choose threshold that maximizes F1
        precision, recall, thresholds = precision_recall_curve(yva, probs_cal)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
        
        # Ignore last element (threshold = 1.0)
        best_idx = np.nanargmax(f1_scores[:-1])
        best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
        best_f1 = f1_scores[best_idx]
        
        print(f"\n📐 Calibration complete")
        print(f"  Optimal threshold: {best_threshold:.3f}")
        print(f"  F1 in validation: {best_f1:.3f}")
        
        # Calculate Brier score
        brier_before = brier_score_loss(yva, probs)
        brier_after = brier_score_loss(yva, probs_cal)
        print(f"  Brier Score: {brier_before:.4f} → {brier_after:.4f}")
        
        return calibrator, best_threshold
    
    @torch.no_grad()
    def predict_proba(model, X, device):
        """
        Predict probabilities for dataset.
        
        Args:
            model: LSTM model
            X: Features
            device: Device
            
        Returns:
            Predicted probabilities
        """
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X_tensor)
        return torch.sigmoid(logits).cpu().numpy()


if TORCH_AVAILABLE:
    def evaluate_on_test(model, Xte, yte, calibrator, threshold, test_index, device):
        """
        Evaluate model on test set.
        
        Args:
            model: Trained LSTM model
            Xte, yte: Test data
            calibrator: Isotonic calibrator
            threshold: Chosen threshold
            test_index: Test indices
            device: Device
            
        Returns:
            Dictionary with metrics and predictions
        """
        # Predictions
        probs = predict_proba(model, Xte, device)
        probs_cal = calibrator.transform(probs)
        preds = (probs_cal >= threshold).astype(int)
        
        # Metrics
        ap = average_precision_score(yte, probs_cal)
        f1 = f1_score(yte, preds)
        acc = accuracy_score(yte, preds)
        
        # MCC and Brier
        mcc = matthews_corrcoef(yte, preds)
        brier = brier_score_loss(yte, probs_cal)
        
        # Confusion matrix - force 2x2 shape
        cm = confusion_matrix(yte, preds, labels=[0, 1])
        
        # Safely unpack matrix
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Fallback
            tn = fp = fn = tp = 0
        
        # Signals for backtest (-1 for short, +1 for long)
        signals = pd.Series((preds * 2 - 1), index=test_index)
        
        print(f"\n📊 Test Results:")
        print(f"  AP (PR-AUC): {ap:.4f}")
        print(f"  F1 Score:    {f1:.4f}")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  MCC:         {mcc:.4f}")
        print(f"  Brier Score: {brier:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {tn:4d}  FP: {fp:4d}")
        print(f"    FN: {fn:4d}  TP: {tp:4d}")
        
        return {
            "ap": ap,
            "f1": f1,
            "accuracy": acc,
            "mcc": mcc,
            "brier": brier,
            "confusion_matrix": cm,
            "proba": probs_cal,
            "pred": preds,
            "signals": signals
        }


if TORCH_AVAILABLE:
    def export_torchscript(model, in_dim, seq_len, path="lstm_model.pt", device="cpu"):
        """
        Export model to TorchScript.
        
        Args:
            model: LSTM model
            in_dim: Input dimension
            seq_len: Sequence length
            path: Save path
            device: Device
            
        Returns:
            Path to saved file
        """
        model_cpu = model.to(device).eval()
        
        # Dummy input for tracing
        dummy_input = torch.randn(1, seq_len, in_dim).to(device)
        
        # Trace and save
        traced_model = torch.jit.trace(model_cpu, dummy_input)
        traced_model.save(path)
        
        print(f"✅ Model exported to: {path}")
        return path


def run_lstm_pipeline(
    X_df: pd.DataFrame, 
    y_series: pd.Series,
    seq_len: int = 64,
    n_trials: int = 50,
    device: str = None,
    horizon: str = "lstm",
    artifacts_path: str = "artifacts"
) -> Dict:
    """
    Complete LSTM pipeline with Optuna.
    
    Args:
        X_df: Features DataFrame
        y_series: Binary labels Series
        seq_len: Sequence length (default: 64 = 16 hours in 15min)
        n_trials: Number of Optuna trials
        device: Device for training (None = auto-detect)
        horizon: Horizon name for logging
        artifacts_path: Artifacts directory
        
    Returns:
        Results dictionary compatible with multi-horizon backtest
    """
    if not TORCH_AVAILABLE:
        print("⚠️ PyTorch not available - skipping LSTM")
        return {}
    
    if not OPTUNA_AVAILABLE:
        print("⚠️ Optuna not available - skipping LSTM optimization")
        return {}
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*80}")
    print(f"🚀 LSTM PIPELINE - Horizon: {horizon}")
    print(f"{'='*80}")
    print(f"📊 Dataset: {len(X_df)} samples, {X_df.shape[1]} features")
    print(f"⚙️  Device: {device}")
    print(f"📏 Sequence: {seq_len} timesteps")
    
    # Create artifacts directory
    Path(artifacts_path).mkdir(parents=True, exist_ok=True)
    Path(f"{artifacts_path}/models").mkdir(parents=True, exist_ok=True)
    
    # 1. Temporal split
    print(f"\n1️⃣ Preparing temporal splits...")
    tr_idx, va_idx, te_idx = train_val_test_split_time(X_df, y_series)
    print(f"  Train:    {len(tr_idx)} samples")
    print(f"  Validation: {len(va_idx)} samples")
    print(f"  Test:     {len(te_idx)} samples")
    
    # 2. Prepare tensors
    print(f"\n2️⃣ Generating sequences and scaling features...")
    (Xtr, ytr, Xva, yva, Xte, yte, test_index, scaler) = build_lstm_tensors(
        X_df, y_series, seq_len, tr_idx, va_idx, te_idx
    )
    print(f"  Training sequences: {Xtr.shape}")
    print(f"  Validation sequences:  {Xva.shape}")
    print(f"  Test sequences:  {Xte.shape}")
    
    # 3. Optuna optimization
    print(f"\n3️⃣ Bayesian optimization...")
    model, best_params = fit_lstm_with_optuna(
        Xtr, ytr, Xva, yva, n_trials=n_trials, device=device
    )
    
    # 4. Calibration and threshold
    print(f"\n4️⃣ Probability calibration...")
    calibrator, threshold = calibrate_and_choose_threshold(
        model, Xva, yva, device
    )
    
    # 5. Test evaluation
    print(f"\n5️⃣ Test set evaluation...")
    eval_results = evaluate_on_test(
        model, Xte, yte, calibrator, threshold, test_index, device
    )
    
    # 6. Export for production
    print(f"\n6️⃣ Exporting model...")
    model_path = f"{artifacts_path}/models/lstm_{horizon}.pt"
    
    export_path = export_torchscript(
        model, Xtr.shape[-1], seq_len, 
        path=model_path, device="cpu"
    )
    
    # 7. Prepare results in expected format
    results = {
        horizon: {
            "best_params": best_params,
            "threshold": threshold,
            "test_metrics": {
                "accuracy": eval_results["accuracy"],
                "precision": eval_results["confusion_matrix"][1,1] / 
                            (eval_results["confusion_matrix"][1,1] + 
                             eval_results["confusion_matrix"][0,1] + 1e-10),
                "recall": eval_results["confusion_matrix"][1,1] / 
                         (eval_results["confusion_matrix"][1,1] + 
                          eval_results["confusion_matrix"][1,0] + 1e-10),
                "f1": eval_results["f1"],
                "pr_auc": eval_results["ap"],
                "mcc": eval_results["mcc"],
                "brier": eval_results["brier"]
            },
            "test_indices": test_index.tolist(),
            "predictions": {
                "proba": eval_results["proba"],
                "binary": eval_results["pred"]
            },
            "signals": eval_results["signals"],
            "confusion_matrix": eval_results["confusion_matrix"],
            "artifact": export_path,
            "scaler": scaler,
            "model_type": "lstm"
        }
    }
    
    # MLflow logging if available
    if MLFLOW_AVAILABLE:
        try:
            import mlflow
            
            with mlflow.start_run(run_name=f"lstm_{horizon}"):
                mlflow.log_params({
                    f"lstm_seq_len_{horizon}": seq_len,
                    f"lstm_device_{horizon}": device,
                    f"lstm_n_trials_{horizon}": n_trials
                })
                
                # Log best parameters
                for key, value in best_params.items():
                    mlflow.log_param(f"lstm_{key}_{horizon}", value)
                
                # Log metrics
                for key, value in results[horizon]["test_metrics"].items():
                    mlflow.log_metric(f"lstm_test_{key}_{horizon}", value)
                
                mlflow.log_metric(f"lstm_threshold_{horizon}", threshold)
                
                # Log artifact
                mlflow.log_artifact(export_path)
                
        except Exception as e:
            print(f"⚠️ MLflow logging failed: {e}")
    
    print(f"\n✅ LSTM pipeline completed for horizon {horizon}")
    return results