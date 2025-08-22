# CODE_MAP.md - Mapa de Código do Projeto

## Entrypoints Principais

```json
{
  "entrypoints": {
    "notebook_principal": "notebooks/IC_Crypto_Complete.ipynb",
    "train_xgb": "src/models/xgb_optuna.py:main",
    "train_lstm": "src/models/lstm_optuna.py:main",
    "backtest": "src/backtest/engine.py:run_backtest",
    "dashboard": "src/dashboard/app.py:main",
    "data_loader": "src/data/binance_loader.py:CryptoDataLoader",
    "feature_eng": "src/features/engineering.py:FeatureEngineer",
    "labeler": "src/features/labels.py:TripleBarrierLabeler",
    "cv_splitter": "src/data/splits.py:PurgedKFold"
  },
  
  "configs": [
    "configs/data.yaml",
    "configs/xgb.yaml", 
    "configs/lstm.yaml",
    "configs/optuna.yaml",
    "configs/backtest.yaml"
  ],
  
  "datasets": {
    "raw": "data/raw/<symbol>_<timeframe>.csv",
    "processed": "data/processed/<symbol>_<timeframe>_features.parquet",
    "labels": "data/processed/<symbol>_<timeframe>_labels.parquet"
  },
  
  "models": {
    "xgb_best": "artifacts/models/xgb_best.pkl",
    "lstm_best": "artifacts/models/lstm_best.pth",
    "calibrators": "artifacts/models/calibrators/"
  },
  
  "mlflow": {
    "tracking_uri": "artifacts/mlruns",
    "experiment": "crypto_ml_pipeline",
    "registry": "artifacts/model_registry"
  },
  
  "commands": {
    "setup": "make setup",
    "train": "make train MODEL=xgb",
    "backtest": "make backtest",
    "dashboard": "make dash",
    "notebook": "make notebook",
    "test": "make test",
    "format": "make fmt"
  },
  
  "structure": {
    "src/": "Código fonte principal",
    "notebooks/": "Notebooks Jupyter (principal: IC_Crypto_Complete.ipynb)",
    "configs/": "Arquivos de configuração YAML",
    "data/": "Dados raw e processados",
    "artifacts/": "Modelos, MLflow runs, relatórios",
    "tests/": "Testes unitários e de integração"
  },
  
  "dependencies": {
    "core": ["numpy", "pandas", "scikit-learn", "xgboost", "torch"],
    "ml": ["optuna", "mlflow", "shap"],
    "data": ["ccxt", "python-binance", "ta", "yfinance"],
    "validation": ["pandera", "pydantic-settings"],
    "viz": ["streamlit", "plotly", "matplotlib"],
    "dev": ["pytest", "ruff", "black", "mypy", "jupytext"]
  },
  
  "hashes": {
    "dataset": "pending",
    "libs_freeze": "pending",
    "config": "pending"
  },
  
  "version": "0.1.0",
  "updated": "2024-12-31"
}
```

## Fluxo de Execução

1. **Setup Inicial**
   ```bash
   make setup  # Instala deps + configura determinismo
   ```

2. **Desenvolvimento**
   ```bash
   make notebook  # Abre IC_Crypto_Complete.ipynb
   ```

3. **Treinamento**
   ```python
   # No notebook, executar célula:
   results = run_complete_pipeline("BTCUSDT", "15m")
   ```

4. **Backtest**
   ```python
   # Automático no pipeline ou:
   make backtest
   ```

5. **Dashboard**
   ```bash
   make dash  # Inicia Streamlit na porta 8501
   ```

## Classes e Funções Principais

### Data Pipeline
- `CryptoDataLoader.fetch_ohlcv()` - Busca dados OHLCV
- `FeatureEngineer.create_all_features()` - Gera features
- `TripleBarrierLabeler.apply_triple_barrier()` - Cria labels

### Validação
- `PurgedKFold.split()` - Split temporal sem vazamento
- `CombinatorialPurgedKFold.split()` - CPCV avançado

### Modelos
- `XGBoostOptuna.optimize()` - Otimização Bayesiana XGB
- `LSTMOptuna.optimize()` - Otimização Bayesiana LSTM

### Backtest
- `BacktestEngine.run_backtest()` - Simulação com custos
- `BacktestEngine.calculate_metrics()` - Métricas de performance

### MLOps
- `MLflowTracker.log_run()` - Tracking de experimentos
- `explain_with_shap()` - Interpretabilidade

## Testes Críticos

1. **Determinismo**: `verify_determinism()`
2. **Não-vazamento**: Asserts em `PurgedKFold`
3. **Calibração**: Brier score deve melhorar
4. **Execução t+1**: Verificação de timing

## Notebooks

- **IC_Crypto_Complete.ipynb**: Pipeline completo com 15 seções
  - Setup determinístico
  - Pipeline de dados
  - Feature engineering
  - Triple Barrier
  - Purged K-Fold
  - Otimização Bayesiana (XGB/LSTM)
  - Backtest
  - MLflow
  - Visualizações
  - SHAP
  - Testes