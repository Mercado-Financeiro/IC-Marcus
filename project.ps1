# ============================================================================
# Project Management Script for Windows PowerShell
# ML Trading Pipeline - Cryptocurrency Price Prediction
# ============================================================================

param(
    [string]$Command = "help",
    [string]$Model = "",
    [int]$Trials = 100,
    [int]$Epochs = 50,
    [string]$Device = "auto",
    [string]$Symbol = "BTCUSDT",
    [string]$Timeframe = "15m",
    [int]$Years = 3,
    [switch]$Fast = $false,
    [switch]$Production = $false,
    [switch]$GPU = $false
)

# Set console encoding to UTF-8 for proper character display
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Colors for output
function Write-ColorOutput($ForegroundColor, $Text) {
    Write-Host $Text -ForegroundColor $ForegroundColor
}

function Write-Success($Text) { Write-ColorOutput Green "OK $Text" }
function Write-Info($Text) { Write-ColorOutput Yellow "-> $Text" }
function Write-Error($Text) { Write-ColorOutput Red "X $Text" }
function Write-Title($Text) { 
    Write-Host ""
    Write-ColorOutput Cyan "============================================================================"
    Write-ColorOutput Cyan "  $Text"
    Write-ColorOutput Cyan "============================================================================"
    Write-Host ""
}

# Check if virtual environment is activated
function Test-VirtualEnv {
    if (-not $env:VIRTUAL_ENV) {
        Write-Error "Virtual environment not activated!"
        Write-Info "Please activate your virtual environment first:"
        Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor White
        return $false
    }
    return $true
}

# Display help menu
function Show-Help {
    Write-Title "ML Trading Pipeline - Command Center"
    
    Write-ColorOutput Yellow "📥 DATA MANAGEMENT:"
    Write-Host "  project download-data         " -NoNewline; Write-Host "Download 3 years of historical data" -ForegroundColor DarkGray
    Write-Host "  project download-data-fast    " -NoNewline; Write-Host "Download 1 year of data for testing" -ForegroundColor DarkGray
    Write-Host "  project cache-info            " -NoNewline; Write-Host "View cache information" -ForegroundColor DarkGray
    Write-Host "  project optimize-cache        " -NoNewline; Write-Host "Optimize database" -ForegroundColor DarkGray
    Write-Host ""
    
    Write-ColorOutput Yellow "📊 MODEL TRAINING:"
    Write-Host "  project train-xgb             " -NoNewline; Write-Host "Train XGBoost model - basic" -ForegroundColor DarkGray
    Write-Host "  project train-xgb-enhanced    " -NoNewline; Write-Host "XGBoost with Bayesian Optimization" -ForegroundColor DarkGray
    Write-Host "  project train-xgb-production  " -NoNewline; Write-Host "Production XGBoost - 300 trials" -ForegroundColor DarkGray
    Write-Host "  project train-xgb-gpu         " -NoNewline; Write-Host "XGBoost with GPU acceleration" -ForegroundColor DarkGray
    Write-Host "  project train-lstm            " -NoNewline; Write-Host "Train LSTM model - basic" -ForegroundColor DarkGray
    Write-Host "  project train-lstm-enhanced   " -NoNewline; Write-Host "LSTM with Bayesian Optimization" -ForegroundColor DarkGray
    Write-Host "  project train-lstm-production " -NoNewline; Write-Host "Production LSTM - 200 trials" -ForegroundColor DarkGray
    Write-Host "  project train-all             " -NoNewline; Write-Host "Train all models" -ForegroundColor DarkGray
    Write-Host "  project train-fast            " -NoNewline; Write-Host "Quick training for testing" -ForegroundColor DarkGray
    Write-Host ""
    
    Write-ColorOutput Yellow "🔧 OPTIMIZATION:"
    Write-Host "  project optimize-xgb          " -NoNewline; Write-Host "Optimize XGBoost with Optuna" -ForegroundColor DarkGray
    Write-Host "  project optimize-lstm         " -NoNewline; Write-Host "Optimize LSTM with Optuna" -ForegroundColor DarkGray
    Write-Host "  project optimize-all          " -NoNewline; Write-Host "Optimize all models" -ForegroundColor DarkGray
    Write-Host ""
    
    Write-ColorOutput Yellow "📈 ANALYSIS:"
    Write-Host "  project walkforward           " -NoNewline; Write-Host "Run walk-forward analysis" -ForegroundColor DarkGray
    Write-Host "  project walkforward-fast      " -NoNewline; Write-Host "Quick walk-forward analysis" -ForegroundColor DarkGray
    Write-Host "  project analyze               " -NoNewline; Write-Host "Analyze model results" -ForegroundColor DarkGray
    Write-Host "  project compare               " -NoNewline; Write-Host "Compare model performance" -ForegroundColor DarkGray
    Write-Host ""
    
    Write-ColorOutput Yellow "🖥️  VISUALIZATION:"
    Write-Host "  project dashboard             " -NoNewline; Write-Host "Launch Streamlit dashboard" -ForegroundColor DarkGray
    Write-Host "  project mlflow                " -NoNewline; Write-Host "Launch MLflow UI" -ForegroundColor DarkGray
    Write-Host "  project notebook              " -NoNewline; Write-Host "Open Jupyter notebook" -ForegroundColor DarkGray
    Write-Host ""
    
    Write-ColorOutput Yellow "🧹 MAINTENANCE:"
    Write-Host "  project clean-cache           " -NoNewline; Write-Host "Clean data cache" -ForegroundColor DarkGray
    Write-Host "  project clean-models          " -NoNewline; Write-Host "Remove old models - keep last 5" -ForegroundColor DarkGray
    Write-Host "  project clean-logs            " -NoNewline; Write-Host "Clean log files" -ForegroundColor DarkGray
    Write-Host "  project clean-all             " -NoNewline; Write-Host "Complete cleanup" -ForegroundColor DarkGray
    Write-Host ""
    
    Write-ColorOutput Yellow "🔧 DEVELOPMENT:"
    Write-Host "  project install               " -NoNewline; Write-Host "Install dependencies" -ForegroundColor DarkGray
    Write-Host "  project test                  " -NoNewline; Write-Host "Run tests" -ForegroundColor DarkGray
    Write-Host "  project test-fast             " -NoNewline; Write-Host "Run quick tests" -ForegroundColor DarkGray
    Write-Host "  project format                " -NoNewline; Write-Host "Format code" -ForegroundColor DarkGray
    Write-Host "  project lint                  " -NoNewline; Write-Host "Check code quality" -ForegroundColor DarkGray
    Write-Host "  project deterministic         " -NoNewline; Write-Host "Configure deterministic environment" -ForegroundColor DarkGray
    Write-Host ""
    
    Write-Title "Quick Examples"
    Write-Host "  .\project.ps1 train-fast     " -NoNewline; Write-Host "# Quick test of models" -ForegroundColor DarkGray
    Write-Host "  .\project.ps1 train-all      " -NoNewline; Write-Host "# Train all models" -ForegroundColor DarkGray
    Write-Host "  .\project.ps1 dashboard      " -NoNewline; Write-Host "# View results" -ForegroundColor DarkGray
    Write-Host ""
}

# Installation commands
function Install-Dependencies {
    Write-Info "Installing dependencies..."
    pip install -r requirements.txt
    Write-Success "Dependencies installed"
}

function Install-Dev {
    Write-Info "Installing development dependencies..."
    pip install -r requirements-dev.txt
    pre-commit install
    Write-Success "Development environment configured"
}

# Data management
function Download-Data {
    Write-Info "Downloading historical data..."
    Write-Info "Symbol: $Symbol | Timeframe: $Timeframe | Period: $Years years"
    if ($Years -eq 1) {
        Write-Info "Quick download (1 year for testing)..."
    } else {
        Write-Info "Full download (estimated 30-60 minutes with rate limiting)..."
    }
    
    $script = "scripts/download_historical_data.py"
    if (Test-Path $script) {
        python $script --symbol $Symbol --timeframe $Timeframe --years $Years
        Write-Success "Data download completed!"
    } else {
        Write-Error "Script not found: $script"
    }
}

function Show-CacheInfo {
    Write-Info "Cache information..."
    python -c @"
from src.data.database_cache import MarketDataCache
cache = MarketDataCache()
info = cache.get_statistics()
print(f"Total bars: {info['total_rows']:,}")
print(f"Database size: {info['size_mb']} MB")
print(f"Symbols: {info['symbols']}")
print(f"Timeframes: {info['timeframes']}")
if info['date_range']:
    print(f"Period: {info['date_range']['first_date']} to {info['date_range']['last_date']}")
"@
    Write-Success "Cache information displayed"
}

function Optimize-Cache {
    Write-Info "Optimizing database..."
    python -c @"
from src.data.database_cache import MarketDataCache
cache = MarketDataCache()
cache.optimize_database()
print('Database optimized')
"@
    Write-Success "Cache optimized!"
}

# Training commands
function Train-XGB {
    Write-Info "Training XGBoost..."
    Write-Info "Configuration: $Symbol $Timeframe"
    
    $script = "src/training/train_xgb.py"
    if ($Fast) {
        python $script --trials 2
    } elseif ($Production) {
        python $script --trials 300 --pruner asha --calibration auto --outer-cv 5 --timeout 7200
    } elseif ($GPU) {
        python $script --trials 200 --pruner asha --tree-method gpu_hist --device gpu --calibration auto
    } elseif ($Model -eq "enhanced") {
        $script = "src/training/train_xgb_enhanced.py"
        python $script --trials $Trials --pruner asha --calibration auto
    } else {
        python $script
    }
    Write-Success "XGBoost training completed!"
}

function Train-LSTM {
    Write-Info "Training LSTM..."
    Write-Info "Configuration: $Epochs epochs | Device: $Device"
    
    $script = "src/training/train_lstm.py"
    if ($Fast) {
        python $script --epochs 10
    } elseif ($Production) {
        $script = "src/training/train_lstm_enhanced.py"
        python $script --trials 200 --pruner asha --calibration temperature --outer-cv 5 --timeout 14400
    } elseif ($Model -eq "enhanced") {
        $script = "src/training/train_lstm_enhanced.py"
        python $script --trials 50 --pruner asha --calibration temperature
    } else {
        python $script --epochs $Epochs --device $Device
    }
    Write-Success "LSTM training completed!"
}

# Optimization
function Optimize-XGB {
    Write-Info "Optimizing XGBoost with Optuna..."
    python src/training/train_xgb.py --trials $Trials
    Write-Success "XGBoost optimization completed!"
}

function Optimize-LSTM {
    Write-Info "Optimizing LSTM with Optuna..."
    python src/training/train_lstm.py --optuna --trials 50 --epochs 20
    Write-Success "LSTM optimization completed!"
}

# Analysis
function Run-WalkForward {
    Write-Info "Running walk-forward analysis..."
    if ($Fast) {
        python src/training/walkforward_fast.py
    } else {
        python src/training/walkforward.py
    }
    Write-Success "Walk-forward analysis completed!"
}

function Run-Analysis {
    Write-Info "Analyzing results..."
    python src/analysis/walkforward_report.py
    python src/analysis/model_report.py
    Write-Success "Analysis completed!"
}

# Visualization
function Start-Dashboard {
    Write-Info "Starting Streamlit dashboard..."
    Write-Host "Access at: " -NoNewline; Write-ColorOutput Green "http://localhost:8501"
    streamlit run src/dashboard/app.py
}

function Start-MLflow {
    Write-Info "Starting MLflow UI..."
    Write-Host "Access at: " -NoNewline; Write-ColorOutput Green "http://localhost:5000"
    mlflow ui --backend-store-uri artifacts/mlruns
}

function Start-Notebook {
    Write-Info "Opening Jupyter notebook..."
    Set-Location notebooks
    jupyter notebook
    Set-Location ..
}

# Cleaning
function Clean-Cache {
    Write-Info "Cleaning data cache..."
    Remove-Item -Path "data/cache/*.db" -ErrorAction SilentlyContinue
    Remove-Item -Path "data/raw/*.parquet" -ErrorAction SilentlyContinue
    Write-Success "Cache cleaned!"
}

function Clean-Models {
    Write-Info "Cleaning old models - keeping last 5..."
    $models = Get-ChildItem -Path "artifacts/models" -Include "*.pth", "*.pkl" -ErrorAction SilentlyContinue | 
              Sort-Object LastWriteTime -Descending
    if ($models.Count -gt 5) {
        $models[5..($models.Count-1)] | Remove-Item
    }
    Write-Success "Old models removed!"
}

function Clean-Logs {
    Write-Info "Cleaning logs..."
    Remove-Item -Path "logs/*.log" -ErrorAction SilentlyContinue
    Remove-Item -Path "artifacts/mlruns/.trash" -Recurse -ErrorAction SilentlyContinue
    Write-Success "Logs cleaned!"
}

function Clean-All {
    Clean-Cache
    Clean-Models
    Clean-Logs
    Remove-Item -Path "__pycache__" -Recurse -ErrorAction SilentlyContinue
    Remove-Item -Path ".pytest_cache" -Recurse -ErrorAction SilentlyContinue
    Get-ChildItem -Directory -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
    Get-ChildItem -File -Recurse -Filter "*.pyc" | Remove-Item
    Write-Success "Complete cleanup done!"
}

# Development
function Run-Tests {
    Write-Info "Running tests..."
    if ($Fast) {
        pytest tests/unit -v --tb=short -x
    } else {
        pytest tests/ -v --tb=short
    }
}

function Format-Code {
    Write-Info "Formatting code..."
    black src/ tests/
    isort src/ tests/
    Write-Success "Code formatted!"
}

function Lint-Code {
    Write-Info "Checking code quality..."
    ruff check src/
    mypy src/ --ignore-missing-imports
    Write-Success "Code checked!"
}

function Set-Deterministic {
    Write-Info "Configuring deterministic environment..."
    python -c @"
import os, random
os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
print('OK Random seed: 42')

try:
    import numpy as np
    np.random.seed(42)
    print('OK NumPy seed: 42')
except:
    pass

try:
    import torch
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('OK PyTorch deterministic')
except:
    pass

print('OK Deterministic environment configured')
"@
    Write-Success "Environment configured!"
}

# Main command dispatcher
if (-not (Test-VirtualEnv)) {
    exit 1
}

switch ($Command.ToLower()) {
    # Help
    "help" { Show-Help }
    
    # Installation
    "install" { Install-Dependencies }
    "install-dev" { Install-Dev }
    
    # Data
    "download-data" { Download-Data }
    "download-data-fast" { $Years = 1; Download-Data }
    "cache-info" { Show-CacheInfo }
    "optimize-cache" { Optimize-Cache }
    
    # Training
    "train-xgb" { Train-XGB }
    "train-xgb-enhanced" { $Model = "enhanced"; Train-XGB }
    "train-xgb-production" { $Production = $true; Train-XGB }
    "train-xgb-gpu" { $GPU = $true; Train-XGB }
    "train-lstm" { Train-LSTM }
    "train-lstm-enhanced" { $Model = "enhanced"; Train-LSTM }
    "train-lstm-production" { $Production = $true; Train-LSTM }
    "train-all" { 
        Set-Deterministic
        Train-XGB
        Train-LSTM 
    }
    "train-fast" { 
        $Fast = $true
        Train-XGB
        Train-LSTM 
    }
    
    # Optimization
    "optimize-xgb" { Optimize-XGB }
    "optimize-lstm" { Optimize-LSTM }
    "optimize-all" { 
        Optimize-XGB
        Optimize-LSTM 
    }
    
    # Analysis
    "walkforward" { Run-WalkForward }
    "walkforward-fast" { $Fast = $true; Run-WalkForward }
    "analyze" { Run-Analysis }
    "compare" { 
        Write-Info "Opening MLflow for comparison..."
        Start-MLflow 
    }
    
    # Visualization
    "dashboard" { Start-Dashboard }
    "mlflow" { Start-MLflow }
    "notebook" { Start-Notebook }
    
    # Cleaning
    "clean-cache" { Clean-Cache }
    "clean-models" { Clean-Models }
    "clean-logs" { Clean-Logs }
    "clean-all" { Clean-All }
    "clean" { Clean-All }
    
    # Development
    "test" { Run-Tests }
    "test-fast" { $Fast = $true; Run-Tests }
    "format" { Format-Code }
    "lint" { Lint-Code }
    "deterministic" { Set-Deterministic }
    
    # Unknown command
    default {
        Write-Error "Unknown command: $Command"
        Write-Info "Use 'project.ps1 help' to see available commands"
        exit 1
    }
}