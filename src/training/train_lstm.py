#!/usr/bin/env python3
"""
LSTM Training Script - Fixed Version
Corrige problemas críticos:
1. Loss balanceado com pos_weight
2. Métricas adequadas (PR-AUC, MCC, Brier)
3. Validação temporal com Purged K-Fold
4. Calibração de probabilidades
5. Comparação com baselines
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from datetime import datetime
import mlflow
import joblib
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, auc, matthews_corrcoef,
    brier_score_loss, log_loss, f1_score, confusion_matrix,
    average_precision_score, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import warnings
warnings.filterwarnings('ignore')

import argparse
from src.data.binance_loader import CryptoDataLoader
from src.features.engineering import FeatureEngineer
from src.data.splits import PurgedKFold

# Optuna imports for Bayesian optimization
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available, optimization disabled")

# Set deterministic behavior
def set_deterministic(seed=42):
    """Ensure reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class TimeSeriesDataset(Dataset):
    """Dataset for time series with proper temporal handling."""
    
    def __init__(self, X, y, window_size=60):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.window_size = window_size
        
    def __len__(self):
        return len(self.X) - self.window_size
    
    def __getitem__(self, idx):
        return (
            self.X[idx:idx+self.window_size],
            self.y[idx+self.window_size]
        )

class ImprovedLSTM(nn.Module):
    """LSTM model without final sigmoid (for BCEWithLogitsLoss)."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(ImprovedLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        # NO sigmoid here - BCEWithLogitsLoss handles it
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out.squeeze()

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate comprehensive metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Basic metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1': f1_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'brier': brier_score_loss(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'support_pos': int(y_true.sum()),
        'support_neg': int(len(y_true) - y_true.sum()),
        'baseline_acc': max(y_true.mean(), 1 - y_true.mean()),
        'prevalence': y_true.mean()
    }
    
    # Normalized PR-AUC
    p = metrics['prevalence']
    metrics['pr_auc_norm'] = (metrics['pr_auc'] - p) / (1 - p) if p < 1 else 0
    
    return metrics

def train_lstm_cv(model, X_train, y_train, X_val, y_val, 
                  epochs=50, lr=0.001, device='cpu', window_size=60,
                  batch_size=32, patience=10):
    """Train LSTM with proper loss weighting and early stopping on PR-AUC."""
    
    # Calculate class weights
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = neg_count / pos_count
    
    print(f"   Class balance: {pos_count/len(y_train):.2%} positive")
    print(f"   Using pos_weight: {pos_weight:.2f}")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, window_size)
    val_dataset = TimeSeriesDataset(X_val, y_val, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup training
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    
    best_pr_auc = 0
    best_model_state = None
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_preds = []
        train_targets = []
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_targets.extend(batch_y.cpu().numpy())
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        train_metrics = calculate_metrics(np.array(train_targets), np.array(train_preds))
        val_metrics = calculate_metrics(np.array(val_targets), np.array(val_preds))
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Update scheduler based on PR-AUC
        scheduler.step(val_metrics['pr_auc'])
        
        # Early stopping based on PR-AUC
        if val_metrics['pr_auc'] > best_pr_auc:
            best_pr_auc = val_metrics['pr_auc']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0 or patience_counter == 0:
            print(f"   Epoch {epoch+1}/{epochs}:")
            print(f"     Loss: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
            print(f"     PR-AUC: Train={train_metrics['pr_auc']:.4f}, Val={val_metrics['pr_auc']:.4f}")
            print(f"     MCC: Train={train_metrics['mcc']:.4f}, Val={val_metrics['mcc']:.4f}")
            print(f"     Brier: Train={train_metrics['brier']:.4f}, Val={val_metrics['brier']:.4f}")
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, history, best_pr_auc

class LSTMWrapper:
    """Wrapper for sklearn compatibility (needed for calibration)."""
    
    def __init__(self, model, scaler, window_size, device):
        self.model = model
        self.scaler = scaler
        self.window_size = window_size
        self.device = device
        # Mark as fitted classifier for sklearn
        self._is_fitted = True
        self.classes_ = np.array([0, 1])  # Binary classification
        self._estimator_type = "classifier"  # Mark as classifier
    
    def fit(self, X, y):
        """Dummy fit method for sklearn compatibility.
        Model is already trained, so just return self."""
        # Set sklearn fitted attributes
        self._is_fitted = True
        self.classes_ = np.array([0, 1])
        self._estimator_type = "classifier"
        return self
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
        
    def predict_proba(self, X):
        """Predict probabilities."""
        self.model.eval()
        
        # Handle both raw and scaled input
        if not isinstance(X, torch.Tensor):
            # Need to create windows
            X_scaled = self.scaler.transform(X)
            # TimeSeriesDataset expects y to be aligned with windowed output
            # So y should have length = len(X) initially
            # The dataset will handle the window indexing
            dummy_y = np.zeros(len(X_scaled))
            dataset = TimeSeriesDataset(X_scaled, dummy_y, self.window_size)
        else:
            # Already windowed
            dataset = TimeSeriesDataset(X, np.zeros(len(X)), self.window_size)
        
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs)
        
        probs = np.array(all_preds)
        # Return as 2D array for sklearn
        return np.column_stack([1 - probs, probs])

def compare_with_baselines(y_true, y_pred_proba):
    """Compare model with trivial baselines."""
    
    print("\n" + "="*60)
    print("COMPARAÇÃO COM BASELINES")
    print("="*60)
    
    prevalence = y_true.mean()
    
    # Baseline 1: Always predict majority class (0)
    baseline_zero = np.zeros_like(y_true)
    metrics_zero = {
        'accuracy': 1 - prevalence,
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'mcc': 0
    }
    
    # Baseline 2: Always predict minority class (1)
    baseline_one = np.ones_like(y_true)
    metrics_one = {
        'accuracy': prevalence,
        'precision': prevalence,
        'recall': 1.0,
        'f1': 2 * prevalence / (1 + prevalence),
        'mcc': 0
    }
    
    # Baseline 3: Random with class probability
    np.random.seed(42)
    baseline_random = np.random.random(len(y_true))
    metrics_random = calculate_metrics(y_true, baseline_random)
    
    # Model metrics
    metrics_model = calculate_metrics(y_true, y_pred_proba)
    
    # Print comparison
    print(f"\nPrevalência da classe positiva: {prevalence:.2%}")
    print(f"Baseline esperado para PR-AUC: {prevalence:.4f}")
    
    print("\n1. Baseline 'Sempre 0' (maioria):")
    print(f"   Acurácia: {metrics_zero['accuracy']:.4f}")
    print(f"   F1: {metrics_zero['f1']:.4f}")
    print(f"   MCC: {metrics_zero['mcc']:.4f}")
    
    print("\n2. Baseline 'Sempre 1' (minoria):")
    print(f"   Acurácia: {metrics_one['accuracy']:.4f}")
    print(f"   F1: {metrics_one['f1']:.4f}")
    print(f"   MCC: {metrics_one['mcc']:.4f}")
    
    print("\n3. Baseline 'Aleatório':")
    print(f"   Acurácia: {metrics_random['accuracy']:.4f}")
    print(f"   PR-AUC: {metrics_random['pr_auc']:.4f}")
    print(f"   MCC: {metrics_random['mcc']:.4f}")
    
    print("\n4. MODELO LSTM:")
    print(f"   Acurácia: {metrics_model['accuracy']:.4f}")
    print(f"   PR-AUC: {metrics_model['pr_auc']:.4f}")
    print(f"   PR-AUC Normalizado: {metrics_model['pr_auc_norm']:.4f}")
    print(f"   MCC: {metrics_model['mcc']:.4f}")
    print(f"   Brier Score: {metrics_model['brier']:.4f}")
    
    # Verdict
    print("\n" + "="*60)
    if metrics_model['pr_auc'] <= prevalence * 1.1:
        print("❌ MODELO NÃO SUPERA BASELINE TRIVIAL!")
        print(f"   PR-AUC do modelo ({metrics_model['pr_auc']:.4f}) ≈ prevalência ({prevalence:.4f})")
    elif metrics_model['mcc'] <= 0.1:
        print("⚠️  MODELO TEM CORRELAÇÃO FRACA (MCC < 0.1)")
    else:
        print("✓ Modelo supera baselines triviais")
        print(f"   Melhoria sobre baseline: {(metrics_model['pr_auc']/prevalence - 1)*100:.1f}%")
    
    return metrics_model

def optimize_lstm_optuna(X, y, args, device, scaler):
    """Optimize LSTM hyperparameters using Optuna (Bayesian optimization).
    
    Returns:
        best_params: Dictionary of best hyperparameters
        best_value: Best PR-AUC score
        best_model: Trained model with best parameters
    """
    
    best_model_state = None
    best_trial_score = -float('inf')
    
    def objective(trial):
        nonlocal best_model_state, best_trial_score
        
        # Suggest hyperparameters
        window_size = trial.suggest_int('window_size', 30, 120, step=10)
        hidden_size = trial.suggest_int('hidden_size', 32, 256, step=32)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Use Purged K-Fold for temporal validation
        cv = PurgedKFold(n_splits=3, embargo=args.embargo)  # Fewer splits for faster optimization
        cv_scores = []
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create model with suggested parameters
            model = ImprovedLSTM(
                input_size=X.shape[1],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
            
            # Train with early stopping on PR-AUC
            model, _, pr_auc = train_lstm_cv(
                model, X_train, y_train, X_val, y_val,
                epochs=20,  # Fewer epochs for optimization
                lr=learning_rate,
                device=device,
                window_size=window_size,
                batch_size=batch_size,
                patience=5
            )
            
            cv_scores.append(pr_auc)
            fold_models.append(model)
            
            # Report intermediate value for pruning
            trial.report(pr_auc, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        mean_score = np.mean(cv_scores)
        
        # Save best model state
        if mean_score > best_trial_score:
            best_trial_score = mean_score
            # Use the model from the best fold
            best_fold_idx = np.argmax(cv_scores)
            best_model_state = fold_models[best_fold_idx].state_dict()
            trial.set_user_attr('best_model_state', best_model_state)
            trial.set_user_attr('window_size', window_size)
            trial.set_user_attr('hidden_size', hidden_size)
            trial.set_user_attr('num_layers', num_layers)
            trial.set_user_attr('dropout', dropout)
        
        return mean_score
    
    # Create Optuna study with Bayesian optimization
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=args.seed),  # Tree-structured Parzen Estimator (Bayesian)
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name=f'lstm_{datetime.now():%Y%m%d_%H%M%S}'
    )
    
    # Optimize
    print("\n" + "="*60)
    print("OTIMIZAÇÃO BAYESIANA COM OPTUNA")
    print("="*60)
    print(f"Trials: {args.trials}")
    print(f"Timeout: {args.timeout}s")
    print(f"Sampler: TPESampler (Bayesian)")
    print(f"Pruner: MedianPruner")
    print("\nIniciando otimização...")
    
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial
    
    print("\n" + "="*60)
    print("MELHORES HIPERPARÂMETROS ENCONTRADOS")
    print("="*60)
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"\nBest PR-AUC (CV): {best_value:.4f}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    # Create and save the best model
    print("\nCriando modelo com melhores parâmetros...")
    best_model = ImprovedLSTM(
        input_size=X.shape[1],
        hidden_size=best_trial.user_attrs['hidden_size'],
        num_layers=best_trial.user_attrs['num_layers'],
        dropout=best_trial.user_attrs['dropout']
    )
    
    # Load best model state
    if 'best_model_state' in best_trial.user_attrs:
        best_model.load_state_dict(best_trial.user_attrs['best_model_state'])
        best_model = best_model.to(device)
    
    # Save study results and model
    study_path = Path('artifacts/optuna')
    study_path.mkdir(parents=True, exist_ok=True)
    model_dir = Path('artifacts/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save study to CSV
    df_trials = study.trials_dataframe()
    csv_path = study_path / f'lstm_optuna_{datetime.now():%Y%m%d_%H%M%S}.csv'
    df_trials.to_csv(csv_path, index=False)
    
    # Save best model
    model_path = model_dir / f'lstm_optuna_best_{datetime.now():%Y%m%d_%H%M%S}.pth'
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'best_params': best_params,
        'best_pr_auc': best_value,
        'scaler': scaler,
        'window_size': best_params['window_size'],
        'hidden_size': best_params['hidden_size'],
        'num_layers': best_params['num_layers'],
        'dropout': best_params['dropout'],
        'learning_rate': best_params['learning_rate'],
        'batch_size': best_params['batch_size'],
        'input_size': X.shape[1],
        'optuna_trials': len(study.trials),
        'timestamp': datetime.now().isoformat()
    }, model_path)
    
    print(f"   ✓ Modelo otimizado salvo em: {model_path}")
    print(f"   ✓ Resultados salvos em: {csv_path}")
    
    # Log to MLflow if available
    if mlflow:
        mlflow.log_params(best_params)
        mlflow.log_metric('optuna_best_pr_auc', best_value)
        mlflow.log_metric('optuna_n_trials', len(study.trials))
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(csv_path))
    
    return best_params, best_value, best_model

def main():
    print("=" * 80)
    print("TREINAMENTO LSTM - VERSÃO CORRIGIDA")
    print("=" * 80)
    print(f"Início: {datetime.now()}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=10)
    # Optuna optimization flags
    parser.add_argument("--optuna", action="store_true", help="Use Optuna for hyperparameter optimization")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=7200, help="Optimization timeout in seconds")
    args = parser.parse_args()
    
    # Set deterministic
    set_deterministic(args.seed)
    
    # Configure device
    if args.device == "cuda":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == "cpu":
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDispositivo: {device}")
    
    # 1. Load more data (1.5+ years)
    print("\n1. Carregando dados estendidos...")
    loader = CryptoDataLoader()
    df = loader.fetch_ohlcv(
        symbol='BTCUSDT',
        timeframe='15m',
        start_date='2023-01-01',  # Extended period
        end_date='2024-08-23'
    )
    print(f"   ✓ Dados carregados: {len(df)} barras ({len(df)/(4*24*365):.1f} anos)")
    
    # 2. Generate features
    print("\n2. Gerando features...")
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(df)
    print(f"   ✓ Features criadas: {features_df.shape[1]} features")
    
    # 3. Create labels with proper threshold
    print("\n3. Criando labels...")
    # More balanced threshold based on volatility
    returns = df['close'].pct_change(5).shift(-5)
    threshold = returns.quantile(0.6)  # Top 40% as positive
    labels = (returns > threshold).astype(int).dropna()
    print(f"   ✓ Labels criados: {len(labels)} amostras")
    print(f"   ✓ Threshold usado: {threshold:.4f}")
    
    # 4. Prepare data
    common_index = features_df.index.intersection(labels.index)
    X = features_df.loc[common_index]
    y = labels.loc[common_index]
    
    # Remove NaN and infinites
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X[mask].values
    y = y[mask].values
    
    print(f"   ✓ Dados finais: {len(y)} amostras ({y.mean():.2%} positivos)")
    
    # 5. Normalize data
    print("\n4. Normalizando dados...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check if using Optuna optimization
    if args.optuna and OPTUNA_AVAILABLE:
        # Run Bayesian optimization
        best_params, best_score, best_model_optuna = optimize_lstm_optuna(X_scaled, y, args, device, scaler)
        
        # Use best parameters from optimization
        window_size = best_params.get('window_size', args.window)
        hidden_size = best_params.get('hidden_size', 64)
        num_layers = best_params.get('num_layers', 2)
        dropout = best_params.get('dropout', 0.3)
        learning_rate = best_params.get('learning_rate', 0.001)
        batch_size = best_params.get('batch_size', 32)
        
        # Store that we used optimization
        used_optuna = True
        optuna_score = best_score
    else:
        # Use default parameters
        if args.optuna and not OPTUNA_AVAILABLE:
            print("\n⚠ Optuna não disponível, usando parâmetros padrão")
        
        window_size = args.window
        hidden_size = 64
        num_layers = 2
        dropout = 0.3
        learning_rate = 0.001
        batch_size = 32
        used_optuna = False
        optuna_score = None
    
    # 6. Temporal validation with Purged K-Fold
    print(f"\n5. Validação cruzada temporal (Purged K-Fold, {args.n_splits} splits)...")
    cv = PurgedKFold(n_splits=args.n_splits, embargo=args.embargo)
    
    cv_scores = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y), 1):
        print(f"\n   Fold {fold}/{args.n_splits}:")
        print(f"   Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create and train model with optimized or default parameters
        model = ImprovedLSTM(
            input_size=X_scaled.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        model, history, best_pr_auc = train_lstm_cv(
            model, X_train, y_train, X_val, y_val,
            epochs=args.epochs, lr=learning_rate, device=device,
            window_size=window_size, batch_size=batch_size, patience=args.patience
        )
        
        cv_scores.append(best_pr_auc)
        fold_models.append(model)
        
        print(f"   Best PR-AUC: {best_pr_auc:.4f}")
    
    print(f"\n   CV Results: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # 7. Final test on hold-out set
    print("\n6. Teste final em conjunto hold-out...")
    
    # Use last 20% as final test
    test_size = int(0.2 * len(X_scaled))
    X_train_final = X_scaled[:-test_size]
    y_train_final = y[:-test_size]
    X_test = X_scaled[-test_size:]
    y_test = y[-test_size:]
    
    print(f"   Train final: {len(y_train_final)} samples")
    print(f"   Test final: {len(y_test)} samples")
    
    # Train final model with optimized or default parameters
    model_final = ImprovedLSTM(
        input_size=X_scaled.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Split train into train/val for final training
    val_split = int(0.8 * len(X_train_final))
    X_tr, X_vl = X_train_final[:val_split], X_train_final[val_split:]
    y_tr, y_vl = y_train_final[:val_split], y_train_final[val_split:]
    
    model_final, _, _ = train_lstm_cv(
        model_final, X_tr, y_tr, X_vl, y_vl,
        epochs=args.epochs, lr=learning_rate, device=device,
        window_size=window_size, batch_size=batch_size, patience=args.patience
    )
    
    # 8. Calibration
    print("\n7. Calibrando probabilidades...")
    
    # Wrap model for sklearn
    wrapped_model = LSTMWrapper(model_final, scaler, args.window, device)
    
    # For calibration, we need to match the windowed output
    # The wrapper's predict_proba will handle the windowing
    # So we pass raw X_vl and adjust y_vl to match the output length
    y_vl_cal = y_vl[args.window:] if len(y_vl) > args.window else y_vl
    
    # Test both calibration methods
    calibrators = {}
    for method in ['isotonic', 'sigmoid']:
        try:
            # Since our model handles windowing internally and X/y have different lengths
            # after windowing, we can't use CalibratedClassifierCV directly
            # Instead, get predictions first and calibrate those
            val_probs = wrapped_model.predict_proba(X_vl)[:, 1]
            
            # Now calibrate the probabilities directly
            from sklearn.isotonic import IsotonicRegression
            from sklearn.linear_model import LogisticRegression
            
            if method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(val_probs, y_vl_cal)
                cal_probs = calibrator.transform(val_probs)
            else:  # sigmoid
                calibrator = LogisticRegression()
                calibrator.fit(val_probs.reshape(-1, 1), y_vl_cal)
                cal_probs = calibrator.predict_proba(val_probs.reshape(-1, 1))[:, 1]
            
            calibrators[method] = calibrator
            
            # Test calibration
            brier = brier_score_loss(y_vl_cal, cal_probs)
            print(f"   {method}: Brier Score = {brier:.4f}")
        except Exception as e:
            print(f"   {method}: Failed - {e}")
    
    # Use best calibrator if available
    if not calibrators:
        print("   ⚠ Calibração falhou, usando modelo sem calibração")
        best_calibrator = None
        best_method = 'none'
    else:
        # Get best method based on Brier score
        best_brier = float('inf')
        best_method = None
        for method, calibrator in calibrators.items():
            val_probs = wrapped_model.predict_proba(X_vl)[:, 1]
            if method == 'isotonic':
                cal_probs = calibrator.transform(val_probs)
            else:  # sigmoid
                cal_probs = calibrator.predict_proba(val_probs.reshape(-1, 1))[:, 1]
            brier = brier_score_loss(y_vl_cal, cal_probs)
            if brier < best_brier:
                best_brier = brier
                best_method = method
        
        best_calibrator = calibrators[best_method]
        print(f"   ✓ Melhor calibração: {best_method} (Brier: {best_brier:.4f})")
    
    # 9. Final evaluation
    print("\n8. Avaliação final no conjunto de teste...")
    
    # Get predictions
    test_dataset = TimeSeriesDataset(X_test, y_test, args.window)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model_final.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model_final(batch_x)
            probs = torch.sigmoid(outputs).cpu().numpy()
            test_preds.extend(probs)
            test_targets.extend(batch_y.numpy())
    
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    
    # Calculate and print metrics
    final_metrics = calculate_metrics(test_targets, test_preds)
    
    print("\n" + "="*60)
    print("MÉTRICAS FINAIS NO TESTE")
    print("="*60)
    print(f"Prevalência: {final_metrics['prevalence']:.2%}")
    print(f"Baseline Acc: {final_metrics['baseline_acc']:.4f}")
    print(f"Model Acc: {final_metrics['accuracy']:.4f}")
    print(f"PR-AUC: {final_metrics['pr_auc']:.4f}")
    print(f"PR-AUC Norm: {final_metrics['pr_auc_norm']:.4f}")
    print(f"ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print(f"MCC: {final_metrics['mcc']:.4f}")
    print(f"Brier: {final_metrics['brier']:.4f}")
    print(f"F1: {final_metrics['f1']:.4f}")
    
    # 10. Compare with baselines
    baseline_metrics = compare_with_baselines(test_targets, test_preds)
    
    # 11. MLflow logging
    print("\n9. Registrando no MLflow...")
    mlflow.set_tracking_uri('artifacts/mlruns')
    mlflow.set_experiment('lstm_fixed')
    
    with mlflow.start_run(run_name=f'lstm_fixed_{datetime.now():%Y%m%d_%H%M%S}'):
        # Log parameters
        mlflow.log_param('symbol', 'BTCUSDT')
        mlflow.log_param('timeframe', '15m')
        mlflow.log_param('n_samples', len(y))
        mlflow.log_param('n_features', X.shape[1])
        mlflow.log_param('window_size', window_size)
        mlflow.log_param('hidden_size', hidden_size)
        mlflow.log_param('num_layers', num_layers)
        mlflow.log_param('dropout', dropout)
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('device', str(device))
        mlflow.log_param('cv_method', 'purged_kfold')
        mlflow.log_param('n_splits', args.n_splits)
        mlflow.log_param('embargo', args.embargo)
        mlflow.log_param('calibration', best_method if best_method != 'none' else 'uncalibrated')
        mlflow.log_param('optimization', 'optuna_bayesian' if used_optuna else 'none')
        if used_optuna:
            mlflow.log_param('optuna_trials', args.trials)
            mlflow.log_metric('optuna_best_score', optuna_score)
        
        # Log metrics
        for k, v in final_metrics.items():
            mlflow.log_metric(f'test_{k}', v)
        
        mlflow.log_metric('cv_pr_auc_mean', np.mean(cv_scores))
        mlflow.log_metric('cv_pr_auc_std', np.std(cv_scores))
        
        # Save model
        model_dir = Path('artifacts/models')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Use different name if Optuna was used
        if used_optuna:
            model_path = model_dir / f'lstm_optuna_{datetime.now():%Y%m%d_%H%M%S}.pth'
        else:
            model_path = model_dir / f'lstm_fixed_{datetime.now():%Y%m%d_%H%M%S}.pth'
        
        torch.save({
            'model_state_dict': model_final.state_dict(),
            'scaler': scaler,
            'window_size': window_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'input_size': X.shape[1],
            'metrics': final_metrics,
            'calibrator': best_calibrator if 'best_calibrator' in locals() else None,
            'optimized_with_optuna': used_optuna,
            'optuna_best_score': optuna_score if used_optuna else None
        }, model_path)
        
        mlflow.log_artifact(str(model_path))
        print(f"   ✓ Modelo salvo em: {model_path}")
    
    print("\n" + "=" * 80)
    print("TREINAMENTO LSTM CORRIGIDO CONCLUÍDO!")
    print(f"Fim: {datetime.now()}")
    print("=" * 80)

if __name__ == "__main__":
    main()