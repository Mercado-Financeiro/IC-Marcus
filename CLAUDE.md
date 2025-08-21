# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a cryptocurrency prediction and automated trading system research project for academic purposes (IC - Iniciação Científica). The system uses Machine Learning models (LSTM and XGBoost) with Bayesian Optimization to predict price movements for major cryptocurrencies (BTC, ETH, BNB, SOL, XRP).

## Core Architecture

### Models Pipeline
1. **Feature Engineering**: Technical indicators, wavelet transforms, statistical features
2. **LSTM Model**: Deep learning for temporal pattern recognition
3. **XGBoost Model**: Gradient boosting for non-linear pattern detection
4. **Bayesian Optimization**: Hyperparameter tuning using Optuna
5. **Trading Strategy**: Signal generation and backtesting

### Key Technologies
- **ML Frameworks**: TensorFlow/Keras (LSTM), XGBoost, Optuna
- **Data Processing**: Pandas, NumPy, PyArrow, Polars
- **Technical Analysis**: TA-Lib, PyWavelets
- **Visualization**: Plotly, Altair, Matplotlib, Seaborn
- **Model Interpretability**: SHAP
- **API Integration**: Binance, KuCoin (for live trading)

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m fast         # Fast tests (<10s)
pytest -m "not slow"   # Exclude slow tests

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_feature_engineering.py
```

### Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Docker Operations
```bash
# Build and run with Docker Compose
docker-compose up -d

# Build collector service
docker build -f Dockerfile.collector -t crypto-collector .

# View logs
docker-compose logs -f
```

## Project Structure

```
├── src/               # Main source code
│   ├── lstm/         # LSTM model implementation
│   ├── xgboost/      # XGBoost model implementation
│   └── main.py       # Main pipeline orchestrator
├── raw/              # Raw cryptocurrency data
├── tests/            # Test suite
│   ├── unit/         # Unit tests
│   ├── integration/  # Integration tests
│   └── conftest.py   # Pytest fixtures
├── .github/workflows/  # CI/CD pipelines
│   ├── ci.yml        # Main CI pipeline
│   ├── coverage.yml  # Coverage reporting
│   └── lint.yml      # Linting checks
└── .claude/agents/   # Specialized Claude agents
```

## Important Files

- **IC.md**: Complete research documentation (Portuguese)
- **requirements.txt**: Python dependencies
- **pytest.ini**: Test configuration
- **docker-compose.yml**: Docker services configuration
- **.env**: Environment variables (API keys, etc.)

## CI/CD Pipeline

The project uses GitHub Actions with the following jobs:
- **Unit Tests**: Fast tests on multiple Python versions
- **Integration Tests**: Full pipeline validation
- **Coverage**: Code coverage reporting
- **Linting**: Code quality checks
- **Statistical Validation**: Model performance validation

## Key Considerations

1. **Data Processing**: The system processes large time-series datasets. Memory optimization is critical.

2. **Temporal Integrity**: When splitting data for training/validation, always respect temporal order to avoid data leakage.

3. **Feature Engineering**: Over 70 technical indicators are computed. Wavelet decomposition (db4) is used for noise reduction.

4. **Model Training**:
   - LSTM uses early stopping and dropout for regularization
   - XGBoost uses balanced class weights for imbalanced data
   - Both models are optimized using Optuna (Bayesian Optimization)

5. **Backtesting**: Always validate strategies on out-of-sample data with proper walk-forward analysis.

## Performance Targets

- **Model Accuracy**: >65% for direction prediction
- **F1-Score**: >0.60 for buy/sell signals
- **ROI**: Positive returns after transaction costs
- **Sharpe Ratio**: >1.0 for risk-adjusted returns
- **Maximum Drawdown**: <20% for risk management

## Common Workflows

### Training a New Model
```python
# Feature engineering
python src/feature_engineering.py --crypto BTC --timeframe 15m

# Train LSTM
python src/lstm/train.py --crypto BTC --timeframe 15m --epochs 100

# Train XGBoost
python src/xgboost/train.py --crypto BTC --timeframe 15m --n_estimators 1000

# Run backtesting
python src/backtesting/backtest.py --model lstm --crypto BTC --timeframe 15m
```

### Adding New Features
1. Update feature configuration in `src/config/features.yaml`
2. Implement feature calculation in appropriate processor
3. Add unit tests for new features
4. Validate feature importance with SHAP

## Specialized Claude Agents

The project includes specialized agents in `.claude/agents/`:
- **crypto-feature-engineer**: Feature engineering and technical indicators
- **lstm-crypto-expert**: LSTM architecture and training optimization
- **xgboost-crypto-expert**: XGBoost optimization and interpretability
- **crypto-trading-strategist**: Trading strategy development and backtesting
- **ic-pipeline-master**: Full pipeline orchestration
- **meta-orchestrator-agent**: Multi-agent coordination
- **research-analysis-expert**: Academic validation and reporting

## Research Context

This is an academic research project (Iniciação Científica) at PUC Goiás, focused on validating ML methods for cryptocurrency prediction. Key research questions:
1. Can LSTM capture temporal dependencies better than traditional methods?
2. How does XGBoost compare for non-linear pattern detection?
3. What is the impact of Bayesian Optimization on model performance?
4. Can the system generate profitable trading signals in real market conditions?

## Data Sources

- Historical OHLCV data from Binance API
- Timeframes: 1m, 5m, 15m, 1h, 4h, 1d
- Cryptocurrencies: BTC, ETH, BNB, SOL, XRP
- Period: 2020-2024 (varies by asset)

## Important Notes

- Always respect rate limits when fetching data from exchanges
- Use environment variables for API keys (never commit them)
- Run tests before committing changes
- Document significant model improvements in IC.md
- Follow PEP 8 style guidelines for Python code
