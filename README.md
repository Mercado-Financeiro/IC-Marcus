# Projeto IC — Pipeline LSTM para Previsão de Cripto

Pipeline de ponta-a-ponta para treinar LSTM em OHLCV (CSV local ou Binance API), com splits temporais sem vazamento, janelamento e métricas robustas (RMSE, MAE, sMAPE, MASE). Inclui avaliação opcional em rolling-origin.

## Quickstart

1) Instale as dependências (requer Python 3.10+):
```bash
pip install -r requirements.txt
```

2) Execute com CSV local (padrão lê de `src/data/raw`):
```bash
python src/main.py --data-dir src/data/raw --csv-pattern BTCUSDT --rule 1min --horizon 5 --lookback 128 --epochs 30 --batch 256 --lr 1e-3 --hidden 128 --layers 2 --dropout 0.2 --scaler standard
```

Também é possível apontar um arquivo específico:
```bash
python src/main.py --csv src/data/raw/BTCUSDT_1m.csv --rule 1min --horizon 5 --lookback 128
```

3) Ou baixe direto da Binance (Spot):
```bash
python src/main.py --binance --symbol BTCUSDT --interval 1m --rule 1m --horizon 5 --lookback 128 --epochs 20 --batch 256
```

4) Rolling-origin (opcional):
```bash
python src/main.py --csv /caminho/ohlcv.csv --rule 1min --horizon 5 --lookback 128 --rolling --ro_splits 5 --ro_step 1000 --ro_val 500 --ro_test 500
```

## Estrutura

```
src/
  data/
    dataset.py        # carregamento, resample, features, splits, scaler, janelamento
    metrics.py        # rmse, mae, smape, mase
    binance_ingest.py # ingestão de OHLCV via API Binance (Spot)
  lstm/
    lstm.py           # definição e compilação do modelo LSTM (Keras)
  main.py             # CLI de treino/avaliação
```

## Formato de entrada (CSV)
- Colunas reconhecidas: `timestamp|ts|time|date|datetime|open_time`, `open`, `high`, `low`, `close`, `volume`.
- Timestamps em UTC; o script reamostra para a frequência `--rule`.

## Métricas
- Regressão: RMSE, MAE, sMAPE, MASE (MASE normalizado por naïve m=1 via treino).

## Notas
- Sem vazamento: escalonadores ajustados apenas no treino; janelamento respeita lookback.
- Sementes fixas; `shuffle=False` no treino.
- Para produção, considere fixar versões em `requirements.txt` e monitorar drift.

## XGBoost — Uso

Classificação (triple-barrier → binário +1 vs resto):
```bash
time python -m src.xgboost.xgboost --task class --horizon 60 --rule 1min --data-dir src/data/raw --csv-pattern BTCUSDT --calibrate --calib_method sigmoid
```

Regressão (retorno H-passos):
```bash
time python -m src.xgboost.xgboost --task reg --horizon 60 --rule 1min --data-dir src/data/raw --csv-pattern BTCUSDT
```

Quantis (p10/p50/p90):
```bash
time python -m src.xgboost.xgboost --task quantile --horizon 60 --rule 1min --data-dir src/data/raw --csv-pattern BTCUSDT --quantiles 0.1 0.5 0.9
```

Parâmetros úteis: `--lags`, `--windows`, `--device cuda:0`, `--n_estimators`, `--learning_rate`, `--max_depth`, `--scale_pos_weight` (apenas class), `--early_stopping_rounds`.

## Otimização Bayesiana (Optuna)

LSTM (rolling-origin, métrica: RMSE no val):
```bash
python -m src.lstm.optuna_lstm --data-dir src/data/raw --csv-pattern BTCUSDT --rule 1min --horizon 5 --ro_splits 5 --ro_step 1000 --ro_val 500 --ro_test 500 --n_trials 60 --pruner median --study lstm_bo
```

XGBoost (purged walk-forward + embargo, métrica: PR-AUC ou F1):
```bash
python -m src.xgboost.optuna_xgb --data-dir src/data/raw --csv-pattern BTCUSDT --rule 1min --horizon 60 --n_splits 6 --train_min_points 2000 --val_points 1000 --embargo_points 60 --step 1000 --metric pr_auc --n_trials 80 --pruner asha --study xgb_bo --calibrate --calib_method sigmoid
```

Dicas:
- Ajuste `--metric` para `f1` se preferir otimizar F1 com threshold dentro do val.
- Use `--device cuda:0` (XGB) se houver GPU.
- Exporte resultados com `study.trials_dataframe()` nos scripts conforme necessidade.
