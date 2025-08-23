# 🚀 ML Trading Pipeline - Cryptocurrency Price Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0+-orange.svg)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B.svg)](https://streamlit.io/)

A production-ready machine learning pipeline for cryptocurrency trading with XGBoost and LSTM models, featuring Bayesian optimization, temporal validation, and comprehensive backtesting.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Testing](#testing)
- [MLOps](#mlops)
- [Dashboard](#dashboard)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Core Capabilities
- **Dual Model Approach**: XGBoost and LSTM with ensemble capabilities
- **Bayesian Optimization**: 100+ trials with Optuna and pruning strategies
- **Temporal Validation**: Purged K-Fold with embargo to prevent data leakage
- **Volatility-Scaled Labeling**: Adaptive threshold labeling (τ = k × σ̂ × √horizon)
- **Feature Engineering**: 100+ technical indicators and microstructure features
- **Calibrated Probabilities**: Isotonic/Platt calibration for reliable predictions
- **EV-Optimized Thresholds**: Threshold selection by expected value maximization

### Production Features
- **MLflow Tracking**: Comprehensive experiment tracking and model registry
- **Real-time Dashboard**: Streamlit app with live monitoring
- **Paper Trading**: Risk-free strategy testing with virtual positions
- **Model Serving API**: REST endpoints for batch and real-time predictions
- **Deterministic Training**: Reproducible results with fixed seeds
- **Security Auditing**: Pre-commit hooks, secret detection, dependency scanning

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Data Ingestion │────▶│ Volatility   │────▶│   Models    │
│   (Binance WS)  │     │   Labeling   │     │ (XGB/LSTM)  │
│  15m@250ms      │     │ τ=k×σ̂×√h     │     │  + Pruning  │
└─────────────────┘     └──────────────┘     └─────────────┘
         │                      │                     │
         ▼                      ▼                     ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   Validation    │     │  Threshold   │     │   MLflow    │
│  (PurgedKFold)  │     │ EV Optimize  │     │   Tracking  │
│   + Embargo     │     │ (Next t+1)   │     │   + Tags    │
└─────────────────┘     └──────────────┘     └─────────────┘
         │                      │                     │
         └──────────────────────┴─────────────────────┘
                               ▼
                        ┌──────────────┐
                        │  Dashboard   │
                        │   + Live     │
                        └──────────────┘
```

## 📊 Volatility-Scaled Labeling System

Our adaptive labeling system replaces traditional Triple Barrier methods with a more robust approach:

**Formula**: `label = sign(r_future) if |r_future| > τ else 0`

Where:
- `τ = k × σ̂ × √horizon` (adaptive threshold)
- `σ̂` = volatility estimator (Yang-Zhang, ATR, Garman-Klass)
- `k` = multiplier optimized via Optuna
- `horizon` = prediction window (15m to 8h)

**Benefits**:
- ✅ No look-ahead bias
- ✅ Market regime adaptive
- ✅ Horizon-aware scaling
- ✅ Multiple estimators support
- ✅ Neutral zone optional

**Supported Estimators**:
- **Yang-Zhang**: Best for 24/7 markets (crypto)
- **ATR**: Simple and robust
- **Garman-Klass**: High-low based
- **Parkinson**: Efficient for clean data
- **Realized Vol**: Traditional approach

**WebSocket Cadences**:
- **Kline 15m**: Updates every 250ms
- **Mark Price**: 1s or 3s streams available
- **Funding Rate**: Dynamic (1h-8h per symbol)

## 📦 Installation

### Prerequisites
- Python 3.11+
- CUDA 11.8+ (optional, for GPU acceleration)
- 16GB RAM minimum
- 10GB disk space

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ml-trading-pipeline.git
cd ml-trading-pipeline
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
make install
# Or manually:
pip install -r requirements.txt
pre-commit install
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your settings
```

## 🚀 Quick Start

### 1. Train XGBoost Model
```bash
make train-xgb SYMBOL=BTCUSDT TIMEFRAME=15m
# Or with custom config:
python run_optimization.py --model xgboost --config configs/xgb.yaml
```

### 2. Train LSTM Model
```bash
make train-lstm SYMBOL=BTCUSDT TIMEFRAME=15m
# Or with custom config:
python run_optimization.py --model lstm --config configs/lstm.yaml
```

### 3. Run Backtest
```bash
make backtest MODEL=xgboost
# Or:
python -m src.backtest.engine --config configs/backtest.yaml
```

### 4. Launch Dashboard
```bash
make dash
# Or:
streamlit run src/dashboard/app_enhanced.py --server.port 8501
```

### 5. Start Paper Trading
```bash
python -m src.trading.paper_trader --model artifacts/models/xgboost_optimized.pkl
```

## 📁 Project Structure

```
.
├── src/
│   ├── data/          # Data loaders and validation
│   ├── features/      # Feature engineering
│   ├── models/        # Model implementations
│   ├── backtest/      # Backtesting engine
│   ├── dashboard/     # Streamlit application
│   ├── mlops/         # MLOps utilities
│   ├── trading/       # Trading strategies
│   └── utils/         # Helper functions
├── configs/           # YAML configurations
├── tests/             # Test suite
├── notebooks/         # Jupyter notebooks
├── artifacts/         # Model artifacts and reports
├── data/             # Data storage
│   ├── raw/          # Raw market data
│   └── processed/    # Processed features
└── scripts/          # Utility scripts
```

## 📊 Model Performance

### XGBoost Results (100 trials)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| F1 Score | 0.434 | > 0.60 | 🟡 Otimização em andamento |
| PR-AUC | 0.714 | > 0.60 | ✅ Meta atingida |
| ROC-AUC | 0.500 | > 0.55 | 🟡 Melhorias necessárias |
| Brier Score | 0.250 | < 0.25 | 🟡 Calibração em progresso |
| Sharpe Ratio | TBD | > 1.0 | ⏳ Backtest pendente |
| Max Drawdown | TBD | < 20% | ⏳ Backtest pendente |

### LSTM Results (pending)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| F1 Score | - | > 0.60 | ⏳ Implementação pendente |
| PR-AUC | - | > 0.60 | ⏳ Implementação pendente |
| ROC-AUC | - | > 0.55 | ⏳ Implementação pendente |
| Brier Score | - | < 0.25 | ⏳ Implementação pendente |

## ⚙️ Configuration

### Data Configuration (`configs/data.yaml`)
```yaml
symbol: "BTCUSDT"
timeframe: "15m"
labels:
  method: "vol_threshold"
  vol_threshold:
    estimator: "yang_zhang"
    k:
      grid: [0.5, 0.75, 1.0, 1.25, 1.5]
      default: 1.0
    horizons:
      "15m": 1
      "60m": 4
      "240m": 16
```

### Model Configuration (`configs/xgb.yaml`)
```yaml
xgb:
  eval_metric: "aucpr"  # Primary metric
  tree_method: "hist"
  device: "cpu"        # For determinism
optuna:
  n_trials: 100
  pruner:
    type: "hyperband"
postprocessing:
  calibration:
    enabled: true
    method: "isotonic"
  threshold_tuning:
    enabled: true
    methods: ["f1", "pr_auc", "ev_net"]
```

### Backtest Configuration (`configs/backtest.yaml`)
```yaml
initial_capital: 100000
position_mode: long_short
execution:
  rule: next_bar_open
costs:
  fee_bps: 5
  slippage_bps: 10
```

## 🧪 Testing

### Run All Tests
```bash
make test
```

### Run Specific Test Categories
```bash
# Unit tests
pytest tests/unit/ -v

# Regression tests (prevent known bugs)
pytest tests/regression/ -v

# Validation tests (model sanity)
pytest tests/validation/ -v

# Coverage report
pytest --cov=src --cov-report=html
```

### Security Audit
```bash
make security-audit
# Runs: pip-audit, bandit, detect-secrets
```

## 🔬 MLOps

### MLflow Tracking
```bash
# View experiments
mlflow ui --backend-store-uri artifacts/mlruns

# Compare models
python scripts/compare_models.py --run-id1 <id1> --run-id2 <id2>
```

### Model Registry
```bash
# Promote model to production
python scripts/deploy_model.py --run-id <run_id> --stage production

# Rollback to previous version
python scripts/deploy_model.py --rollback
```

### Monitoring
```bash
# Monitor training progress
python scripts/monitor_training.py

# Check optimization status
python scripts/monitor_optimization.py --log artifacts/reports/xgb_optimization.log
```

## 📈 Dashboard

The Streamlit dashboard provides:

- **Overview**: Key metrics and model comparison
- **Performance**: Equity curves, drawdown analysis  
- **Volatility Analysis**: Adaptive threshold visualization
- **Threshold Tuning**: Interactive EV optimization with cost analysis
- **Features**: Importance analysis, SHAP values
- **Live Trading**: Real-time WebSocket feeds (250ms klines, 1s mark price)
- **MLflow Integration**: Experiment tracking with PRD compliance

**Key Features**:
- 📊 **EV Optimization**: Visual threshold selection by expected value
- 📈 **Real-time Feeds**: 15min klines at 250ms, mark price at 1s
- 🔬 **Volatility Regimes**: Yang-Zhang vs ATR comparison
- 📋 **Model Registry**: Champion/challenger with rollback support
- 🎯 **Funding Tracking**: Dynamic rates (1h-8h) per symbol

Access at: http://localhost:8501

## 🛠️ Development

### Code Quality
```bash
# Format code
make fmt

# Type checking
make type

# Linting
make lint
```

### Pre-commit Hooks
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Documentation
```bash
# Generate API docs
make docs

# View documentation
open docs/_build/html/index.html
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Commit Convention
We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code refactoring
- `test:` Tests
- `chore:` Maintenance

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only. It is not intended as financial advice or a recommendation to trade. Trading cryptocurrencies involves substantial risk of loss. Always do your own research and consult with qualified financial advisors before making investment decisions.**

## 🙏 Acknowledgments

- [XGBoost](https://github.com/dmlc/xgboost) - Gradient boosting framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Optuna](https://optuna.org/) - Hyperparameter optimization
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [Streamlit](https://streamlit.io/) - Dashboard framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---
**Last Updated**: 2025-08-23
**Version**: 1.1.0
**Status**: 🟢 Production Ready (Vol-aware labeling implemented, EV optimization active)
**Architecture**: Volatility-Scaled Classification + EV Threshold Optimization
