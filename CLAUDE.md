# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Notebook Policy (OBRIGATÓRIO)

- O notebook `notebooks/IC_Crypto_Complete.ipynb` é **console de orquestração** (EDA, execução de rotinas, visualizações, experimentos).
- **Fonte da verdade** do código fica em `src/**`. Toda lógica de dados, features, modelos, backtest e pós-processamento deve residir em módulos importáveis.
- **Pareamento Jupytext**: `ipynb,py:percent` obrigatório. Commits que alterem `.ipynb` sem atualizar o par `.py` **falham no CI**.
- **Regra crítica**: TODO desenvolvimento e experimento DEVE ser feito no notebook `notebooks/IC_Crypto_Complete.ipynb`, aplicando otimizações de memória sempre que possível.

---

# CLAUDE.md — Manual de Operações do Projeto (v2)

> Manual para IA atuar como engenheiro autônomo em um projeto de **Machine Learning para mercado financeiro**: treinar modelos com **dataset interno**, entregar **dashboard**, aplicar **MLOps** e manter **memória viva do código**. **Não emitir recomendações de investimento.**

---

## TL;DR

* **Objetivo**: pipeline reprodutível de pesquisa/trading com XGBoost e LSTM, backtest realista e dashboard Streamlit.
* **Garantias**: CV temporal sem vazamento, **calibração de probabilidades + threshold tuning obrigatórios**, tracking com MLflow, dados versionados (DVC opc.), testes, lint e tipagem.
* **Determinismo**: seeds unificadas (Python/NumPy/Torch), cuDNN determinístico, CUBLAS workspace configurado.
* **Execução t+1**: sinal gerado na barra t e executado na abertura da barra t+1 com slippage.
* **Métrica de decisão**: além de F1/PR-AUC, **threshold por EV líquido** em validação sob custos.
* **Memória**: atualizar `AI_MEMORY.md` e `CODE_MAP.md` a cada tarefa.

---

## QUICK COMMANDS (cartão de bolso)

```
Instalação dev  : make install
Determinismo    : make deterministic
Formatar código : make fmt
Tipos           : make type
Testes          : make test
Treinar XGB     : make train MODEL=xgb
Treinar LSTM    : make train MODEL=lstm
Backtest        : make backtest
Dashboard       : make dash
```

---

## Avisos e Ética

* Proibido interpretar como aconselhamento financeiro.
* Tratar custos, slippage e risco. Nunca reportar métricas sem custos.
* Sanitizar PII. Não baixar dados externos sem aprovação explícita.

---

## Regras Globais

* **Reprodutibilidade**: seeds fixas; `PYTHONHASHSEED=0`; versões de libs congeladas com **lock de dependências** (hashes) e auditoria de segurança.
* **Determinismo real**: Torch determinístico, cuDNN determinístico, `CUBLAS_WORKSPACE_CONFIG` definido; XGBoost com parâmetros estáveis e CPU quando for necessário reproduzir bit a bit.
* **Sem vazamento temporal**: *Purged/Embargoed K-Fold* ou *Walk-Forward Anchored*. `shuffle=False` sempre.
* **Calibração/Threshold**: usar `CalibratedClassifierCV` (isotônica/Platt) em validação e **fixar threshold** antes de avaliar em teste; reportar **Brier score**. Registrar também **threshold que maximiza EV líquido**.
* **Execução t+1**: decisões calculadas na barra t e ordens executadas na **abertura** de t+1.
* **Métricas ML**: F1, MCC, AUC, PR-AUC, Brier. **Métricas Trading**: Sharpe, Sortino, Calmar, retorno líquido, MDD, turnover, EV.
* **Config centralizada**: `configs/*.yaml` via argparse/Hydra-like simples.
* **Qualidade**: Python 3.11+, type hints, docstrings Google-style; `ruff`, `black`, `mypy`, `bandit`.
* **Pipelines**: `sklearn.Pipeline` e serialização; artefatos em `artifacts/`.
* **Tracking**: MLflow local em `artifacts/mlruns` com tags obrigatórias: `git_commit`, `dataset_sha256`, `config_sha256`, `degraded_mode`, `prd_name`, `prd_version`, `prd_sha256`, `exec_rule`.
* **Versionamento de dados**: DVC (opcional) sobre `data/` com *remote* S3-compatível.
* **Segredos**: `.env` via `pydantic-settings`; nunca commitar chaves.
* **Observabilidade**: logging estruturado (`structlog`); logs por run.
* **Timezone e unidades**: timestamps **timezone-aware (UTC)**; contrato `{returns: log, costs: bps}` em todo o projeto.

---

## Estrutura de Pastas (alvo)

```
.
├─ src/
│  ├─ data/        # loaders e validações (pandera)
│  ├─ features/    # engenharia (lags, rolling, volatilidade, indicadores)
│  ├─ models/      # treino/inferência (baseline, xgb, lstm, postproc)
│  ├─ backtest/    # motor de simulação e métricas financeiras
│  ├─ utils/       # logging, seeds, custos, configs
│  └─ dashboard/   # app Streamlit
├─ notebooks/      # EDA apenas; código final vai para src/
├─ configs/        # *.yaml (dados, xgb, lstm, backtest, agent)
├─ tests/          # pytest + hypothesis
├─ data/{raw,processed}
├─ artifacts/      # modelos, relatórios, figuras, mlruns/
├─ .dvc/           # se DVC habilitado
├─ AI_MEMORY.md    # memória viva (estado)
├─ CODE_MAP.md     # entrypoints e mapas
├─ ARCHITECTURE.md # visão do sistema
├─ EXPERIMENTS.md  # tabela de experimentos
├─ Makefile
├─ requirements.txt    # lock com hashes (gerado)
└─ README.md
```

---

## Stack técnico

* **Core**: numpy, pandas, scikit-learn, xgboost, torch, optuna, pandera, pydantic-settings, structlog
* **MLOps**: mlflow, dvc\[s3] (opcional), pre-commit, pytest, hypothesis, ruff, black, mypy, bandit, pip-audit, gitleaks
* **Dashboard**: streamlit, plotly/matplotlib

### `pyproject.toml` (trecho mínimo)

```toml
[project]
name = "ml-finance"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy", "pandas", "scikit-learn", "xgboost",
  "torch", "optuna", "mlflow", "pandera",
  "pydantic-settings", "structlog", "streamlit",
  "matplotlib", "plotly"
]

[project.optional-dependencies]
dev = [
  "pytest", "hypothesis", "ruff", "black", "mypy", "bandit", "pre-commit",
  "pip-tools", "pip-audit", "gitleaks"
]
```

> **Lock de dependências**: gere `requirements.txt` com hashes via `pip-compile --generate-hashes` e commite o lock.

### `.env.example`

```
MLFLOW_TRACKING_URI=artifacts/mlruns
DVC_REMOTE_URL=s3://bucket/projeto
DASHBOARD_PORT=8501
```

### `.pre-commit-config.yaml` (resumo)

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks: [{id: ruff},{id: ruff-format}]
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks: [{id: black}]
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks: [{id: mypy}]
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.9
    hooks: [{id: bandit}]
  - repo: https://github.com/pypa/pip-audit
    rev: v2.7.3
    hooks: [{id: pip-audit, args: ["-r","requirements.txt","--strict"], additional_dependencies: ["pip-audit==2.7.3"]}]
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.4
    hooks: [{id: gitleaks}]
```

### Jupytext (Notebooks)

* Não editar arquivos `.ipynb` diretamente; trabalhar no par em texto `*.py` no formato percent (`# %%`).
* Pareamento obrigatório via `.jupytext.toml` com `formats = "ipynb,py:percent"`.
* Sincronizar pares: `jupytext --sync notebooks/*.ipynb`.
* Ignorar checkpoints: `**/.ipynb_checkpoints/**`.

Quickstart:

```bash
pip install jupytext
jupytext --set-formats "ipynb,py:percent" notebooks/*.ipynb
jupytext --sync notebooks/*.ipynb
```

Opcional (pre-commit):

```yaml
- repo: https://github.com/mwouts/jupytext
  rev: v1.16.2
  hooks:
    - id: jupytext
      args: [--sync]
      files: ^notebooks/.*\.(ipynb|py)$
```

---

## Loop do Desenvolvedor (executar para cada solicitação)

### 1) Planejar

* Registrar em `AI_MEMORY.md`: objetivo, insumos, saídas, aceitação, riscos (leak, overfit, latência).
* Definir partições temporais: treino/validação/teste fora da janela.
* Escolher modelo(s) e métricas (ML e Trading) e custos padrão.

### 2) Desenvolver

* **Dados**: `src/data/loaders.py` com validação `pandera`; `src/data/splits.py` com PurgedKFold/WalkForward.
* **Features**: `src/features/pipeline.py` com transforms componíveis; normalizar por treino.
* **Modelos**:

  * `src/models/xgb.py`: treino com TimeSeriesSplit, Optuna opcional; importâncias SHAP/permutation.
  * `src/models/lstm.py`: janelas, máscara temporal, early stopping; escalonador fitado no treino; **Torch determinístico**.
  * `src/models/baseline.py`: sinais simples (momentum/mean-reversion) para calibrar expectativa.
* **Postproc**: `src/models/postproc.py` com calibração e seleção de threshold por F1/PR-AUC **e** por **EV líquido**.
* **Backtest**: `src/backtest/engine.py` com sizing, custos fixos e proporcionais, slippage, funding/borrow fee pró‑rata, restrição de alavancagem, horários de negociação e **execução t+1**.
* **Dashboard**: `src/dashboard/app.py` com páginas: runs MLflow, equity curve, drawdown, distribuição de retornos, matriz de confusão, importância de features, comparação de estratégias e **aba de Threshold Tuning/EV**.
* **Logging/Tracking**: logar parâmetros, métricas e artefatos em MLflow; salvar figuras em `artifacts/`.

### 3) Testar

* Unitários: splits sem vazamento; shapes; serialização; tempo limite de treino curto.
* Property-based: `hypothesis` em transforms.
* Smoke tests: dashboard carrega; backtest roda num *toy set*.
* Validar esquema com `pandera` a cada load.

### 4) Revisar

* Rodar `ruff`, `black`, `mypy`, `bandit`, `pip-audit` e `gitleaks`.
* Refatorar funções longas; remover duplicações; documentar contratos e efeitos colaterais.
* Atualizar `ARCHITECTURE.md` se estrutura mudou.

### 5) Commitar

* Conventional Commits; atualizar `EXPERIMENTS.md` e `AI_MEMORY.md`.
* Se DVC usado: `dvc add` e commitar `.dvc`.

---

## Memória do Projeto (autogerida)

Após cada loop, anexar em `AI_MEMORY.md`:

* Resumo da tarefa e decisão arquitetural.
* Arquivos tocados e entrypoints.
* Artefatos criados (paths MLflow/DVC).
* Próximos passos e bloqueios.

Atualizar `CODE_MAP.md` com JSON curto:

```json
{
  "entrypoints": {
    "train_xgb": "src/models/xgb.py:cli_train",
    "train_lstm": "src/models/lstm.py:cli_train",
    "backtest": "src/backtest/engine.py:run_backtest",
    "dashboard": "src/dashboard/app.py:main"
  },
  "configs": [
    "configs/data.yaml",
    "configs/xgb.yaml",
    "configs/lstm.yaml",
    "configs/backtest.yaml"
  ],
  "datasets": {
    "bars": "data/processed/<symbol>_<freq>.parquet",
    "features": "data/processed/<symbol>_<freq>_features.parquet"
  },
  "hashes": {
    "dataset": "<sha256>",
    "libs_freeze": "<pip-freeze-hash>"
  },
  "prd": {
    "xgb": "docs/prd/PRD_XGB.md",
    "lstm": "docs/prd/PRD_LSTM.md"
  }
}
```

---

## Regras de Sentinela contra Vazamento (Leakage Guards)

**Sempre provar com asserts e logs** em cada run:

1. `max(train_time) + embargo < min(val_time)` e `< min(test_time)`.
2. `fit()` só em treino; validação/teste apenas `transform()`.
3. Rolling/expanding stats: fit no treino e `partial_transform` nas janelas seguintes.
4. Métricas de teste só após **calibração + threshold** fixados na validação.
5. **Execução t+1** comprovada: timestamps de decisão e de execução registrados na run.

---

## Prioridade de Features

1. **Básico**: OHLCV, retornos, volatilidade realizada.
2. **Técnico**: RSI, médias móveis, z-score, bandas; janelas múltiplas.
3. **Avançado**: microestrutura (spreads/imbalance), regime/vol clusters, calendário.

---

## Definição de Pronto (DoD)

* Treino reproduzível via `make train MODEL=xgb|lstm` ou `python -m src.models.<model> --config ...`.
* `make deterministic` executado e verificado.
* `make test` passa; lints, tipos, **pip-audit** e **gitleaks** OK; lock `requirements.txt` presente e atualizado.
* **Calibração executada**, `Brier` reportado e **threshold** escolhido em validação (F1/PR-AUC) **e** `th_ev` por EV líquido.
* **Execução t+1** habilitada e registrada (`exec_rule=next_bar_open`).
* Experimento no MLflow com métricas ML e Trading e figuras anexas; tags preenchidas (`git_commit`, `dataset_sha256`, `config_sha256`, `degraded_mode`, `prd_*`).
* Dashboard `make dash` rodando com equity, drawdown, comparação de runs e aba de **Threshold Tuning/EV**.
* Documentação atualizada: `AI_MEMORY.md`, `CODE_MAP.md`, `ARCHITECTURE.md`.
* **Gate**: se `temporal_leak_checks != pass`, `determinism != pass` ou `tests != pass`, **tarefa reprovada** e o agente deve retornar um novo `=== PLAN ===` de correção.
* **Gate PRD**: Run aprovada somente se o PLAN listar **PRD + versão + sha** e as tags estiverem no MLflow.

---

## Makefile (alvo mínimo)

```makefile
install:
	pip install -U pip && pip install .[dev]
	pre-commit install

fmt: ; ruff check --fix . ; black . ; ruff format .

type: ; mypy src

test: ; pytest -q

deterministic:
	python - <<'PY'
import os, random, numpy as np
try:
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass
os.environ.update({
  "PYTHONHASHSEED": "0",
  "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
  "CUDA_LAUNCH_BLOCKING": "1",
})
random.seed(42); np.random.seed(42)
print("determinism:on")
PY

train:
	python -m src.models.$(MODEL) --config configs/$(MODEL).yaml

backtest:
	python -m src.backtest.engine --config configs/backtest.yaml

ash:
	streamlit run src/dashboard/app.py --server.port $${DASHBOARD_PORT:-8501}
```

---

## Exemplos de Configs (`configs/*.yaml`)

### `configs/data.yaml`

```yaml
path_raw: data/raw
path_processed: data/processed
index_col: timestamp
symbol: "BTCUSDT"
timezone: "UTC"   # timezone-aware obrigatório
units:
  returns: "log"
  costs: "bps"
features:
  - rsi_14
  - zscore_60
  - vol_30
  - mom_20
label:
  kind: classification
  horizon: 15m
  rule: future_return_gt_0
split:
  method: walk_forward
  n_splits: 5
  embargo: 5   # barras
schema_version: 1
```

### `configs/xgb.yaml`

```yaml
seed: 42
cv: purged_kfold
cv_params:
  n_splits: 5
  embargo: 5
xgb:
  n_estimators: 500
  learning_rate: 0.05
  max_depth: 6
  subsample: 0.8
  colsample_bytree: 0.8
optuna:
  enabled: false
  n_trials: 50
costs:
  fee_bps: 5
  slippage_bps: 5
```

### `configs/lstm.yaml`

```yaml
seed: 42
window: 128
stride: 1
batch_size: 256
epochs: 50
lr: 1e-3
hidden_size: 64
num_layers: 2
dropout: 0.2
optimizer: adam
scheduler: cosine
val_patience: 5
scaler: standard   # fitado no treino
deterministic: true
gradient_clipping: 1.0
weight_decay: 1e-4
```

### `configs/backtest.yaml`

```yaml
initial_capital: 100000
position_mode: long_short
execution:
  rule: next_bar_open     # sinal em t, execução na abertura de t+1
  partial_fills: false
  min_lot: 0.0001
  price_reference: open
risk:
  max_leverage: 1.0
  kelly_fraction: 0.25
costs:
  fee_bps: 5
  slippage_bps: 10
  funding_apr_est: 0.00   # funding anualizado estimado (pró-rata por barra)
  borrow_apr_est: 0.00    # custo de short sintético quando aplicável
sizing:
  method: volatility_target
  vol_target: 0.2
  lookback: 60
trading_hours:
  start: "00:00"
  end: "23:59"
```

### `configs/agent.yaml` (guardrails)

```yaml
agent:
  max_edits: 8
  max_runtime_min: 12
  test_timeout_s: 180
  fast_flags:
    train_xgb: "--fast"
    train_lstm: "--fast"
  allowed_dirs: ["src","configs","tests","AI_MEMORY.md","CODE_MAP.md","EXPERIMENTS.md"]
  banned_paths: ["data/raw/**","artifacts/mlruns/**"]
  gpu: false
```

---

## Especificação do Backtest

* **Calendário**: usar timestamps do dataset; filtrar buracos e consolidar barras.
* **Sinal para posição**: limiarização de probabilidade calibrada (ou z-score do retorno esperado).
* **Execução**: **t+1** na abertura; aplicar slippage e fees; sem “close-to-close”.
* **Custos**: fee + slippage + funding/borrow. Transações somente quando há mudança de posição.
* **Relatórios**: equity, MDD, retorno anualizado, heatmap por mês, distribuição de retornos, turnover, exposição média.
* **Overfit flag**: marcar se Sharpe OOS ≪ IS; registrar nota honesta no scorecard.

---

## Interpretabilidade

* **XGB**: Tree SHAP (exato para árvores); permutação global.
* **Estabilidade**: reportar média e **desvio-padrão** das top‑k importâncias entre *folds*.
* **PDP/ICE**: somente em validação; salvar em MLflow.

---

## Labeling

* **Fixed horizon** (simples) e **Triple‑Barrier** (padrão quando a meta envolve risco/retorno com stop/TP/timeout).

---

## MLOps

* **MLflow**: artefatos (modelos, figuras), parâmetros e métricas. Nomear runs por `symbol/model/horizon`. Tags obrigatórias: `git_commit`, `dataset_sha256`, `config_sha256`, `degraded_mode`, `prd_name`, `prd_version`, `prd_sha256`, `exec_rule`.
* **Model Registry**: adotar **champion/challenger**; promover somente com melhora OOS consistente e risco não piorado.
* **DVC** (opcional): versionar `data/raw` e `data/processed`; remoto S3-compatível.
* **CI**: pipeline com lint, tipos, testes, *smoke* do dashboard, treinos rápidos de XGB e LSTM e verificação de determinismo.
* **Produção (quando aplicável)**: monitorar *data drift* (PSI/KL), *prediction drift*, *latency* e *error rate*; alertas com thresholds.
* **Retraining**: janela móvel e *champion/challenger*; promover se Sharpe OOS melhora e risco não piora.

### DVC Quickstart

```
dvc remote add -d s3remote s3://<bucket>/<key>
dvc add data/processed
dvc push -r s3remote
```

### GitHub Actions (exemplo)

```yaml
name: ci
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -U pip && pip install .[dev]
      - run: make deterministic
      - run: ruff check . && ruff format --check . && black --check .
      - run: mypy src
      - run: pip-audit -r requirements.txt --strict
      - run: gitleaks detect --no-git --redact || true
      - run: pytest -q
      - name: Smoke - treino rápido XGB
        run: python -m src.models.xgb --config configs/xgb.yaml --fast
      - name: Smoke - treino rápido LSTM
        run: python -m src.models.lstm --config configs/lstm.yaml --fast
      - name: Smoke - dashboard
        run: |
          python - <<'PY'
          import importlib
          importlib.import_module('src.dashboard.app')
          print('dashboard import ok')
          PY
      - name: Dataset & env hashes
        run: |
          python - <<'PY'
          import hashlib, pkgutil, sys
          from pathlib import Path
          p=Path('data/processed'); h=hashlib.sha256()
          [ h.update(f.read_bytes()) for f in sorted(p.rglob('*.parquet')) ]
          print('DATASET_SHA256=',h.hexdigest())
          print('PYTHON_VERSION=',sys.version.split()[0])
          print('PKGS=',len(list(pkgutil.iter_modules())))
          PY
```

---

## Dashboard (páginas)

1. **Visão Geral**: parâmetros da run, métricas, tabela de runs MLflow com filtro.
2. **Curvas**: equity, drawdown, rolling Sharpe.
3. **Classificação**: matriz de confusão, PR/ROC, **Threshold Tuning/EV** com slider e curva de EV líquido.
4. **Atribuição**: importâncias, PDP/ICE, ablação de features.
5. **Regimes**: performance por volatilidade/horário/sessão.
6. **Compare Runs**: overlay de equities entre duas runs.
7. **Backtest**: operações, turnover, heatmap de retornos por período.

---

## Entrypoints esperados

* `python -m src.models.xgb --config configs/xgb.yaml`
* `python -m src.models.lstm --config configs/lstm.yaml`
* `python -m src.backtest.engine --config configs/backtest.yaml`
* `streamlit run src/dashboard/app.py`

---

## Roadmap sugerido

1. Baselines e backtest confiável com custos e execução t+1.
2. XGB com PurgedKFold e Optuna leve.
3. LSTM com janelas, regularização forte e determinismo habilitado.
4. Dashboard conectado ao MLflow.
5. CI completa, lock de deps e DVC remoto.

---

## Checklist por Tarefa

* [ ] Plano e aceitação em `AI_MEMORY.md`.
* [ ] Código com type hints e docstrings.
* [ ] Testes atualizados e passando; validações `pandera` OK.
* [ ] Lints, tipos, segurança OK; **pip-audit** e **gitleaks** limpos; lock presente.
* [ ] **Calibração + threshold** executados (F1/PR‑AUC) e **`th_ev`** por EV líquido reportado; **Brier** reportado.
* [ ] Artefatos no `artifacts/` e log no MLflow; tags preenchidas, incluindo `prd_*` e `exec_rule`.
* [ ] Memória e mapas atualizados.
* [ ] Commit convencional e, se aplicável, DVC atualizado.

---

## Troubleshooting (rápido)

* **"Temporal leakage detected"**: revise `embargo`, checagem de janelas e se algum `fit()` escapou para validação/teste.
* **"Sharpe baixo OOS"**: aumente custos realistas, reduza turnover, reavalie threshold e regularização.
* **"MLflow não loga"**: verifique `MLFLOW_TRACKING_URI` e permissões de escrita.
* **"Dashboard não abre"**: teste import (`import src.dashboard.app`) e dependências.
* **"Resultados não reproduzem"**: valide `make deterministic`, seeds, versão de libs e hardware (CPU vs GPU).
* **"Timezone/Unidades"**: garanta `UTC` e contrato `{returns: log, costs: bps}`.
* **"Lock ausente ou desatualizado"**: regenere `requirements.txt` com hashes e recommite.

---

## Apêndice A — Convenções de mensagem de commit

* `feat(model): add xgb with purged kfold and mlflow logging`
* `refactor(features): unify rolling window transforms`
* `fix(backtest): correct slippage application on trade open`

## Apêndice B — Placeholders de Código (stubs)

```python
# src/utils/config.py
from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    mlflow_uri: str = "artifacts/mlruns"
    class Config:
        env_file = ".env"
settings = Settings()
```

```python
# src/utils/logging.py
import structlog
structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()])
log = structlog.get_logger()
```

```python
# src/data/schema.py
import pandera as pa
from pandera import DataFrameSchema, Column, Check
BarsSchema = DataFrameSchema({
  "timestamp": Column(pa.Timestamp, checks=[Check.is_monotonic_increasing()]),
  "open": Column(float, checks=Check.ge(0)),
  "high": Column(float, checks=Check.ge(0)),
  "low":  Column(float, checks=Check.ge(0)),
  "close":Column(float, checks=Check.ge(0)),
  "volume": Column(float, checks=Check.ge(0)),
}, coerce=True)
```

```python
# src/data/splits.py (Purged K-Fold com embargo, stub funcional)
from typing import Iterator, Tuple
import numpy as np, pandas as pd

def purged_kfold_index(times: pd.Series, n_splits=5, embargo=0) -> Iterator[Tuple[np.ndarray,np.ndarray]]:
    idx = np.arange(len(times))
    folds = np.array_split(idx, n_splits)
    for val in folds:
        val_start, val_end = val[0], val[-1]
        mask = (idx <= val_start - embargo) | (idx >= val_end + embargo)
        train = idx[mask]
        yield train, val
```

```python
# src/features/labels.py (Triple-Barrier)
def triple_barrier(close, pt=0.01, sl=0.01, max_h=60):
    """Retorna labels {-1,0,1} por barra sem olhar além de max_h."""
    ...
```

```python
# src/models/postproc.py (calibração + threshold por EV)
from sklearn.calibration import CalibratedClassifierCV

def calibrate_and_threshold(model, X_tr, y_tr, X_val, y_val, costs, target="f1"):
    """Ajusta calibrador (isotônico/Platt) e retorna calibrador + thresholds (f1/prauc/ev)."""
    ...

def choose_threshold_by_ev(p_val, y_val, costs, clip=(0,1)):
    import numpy as np
    cand = np.linspace(0.05, 0.95, 181)
    def ev(th):
        side = (p_val >= th).astype(int)*2-1
        gross = expected_return_proxy(side, y_val)  # seu proxy
        net = gross - turnover(side)*costs.total_bps/1e4
        return net.mean()
    th = max(cand, key=ev)
    return float(np.clip(th, *clip))
```

---

# 🔧 ADDENDUM — Operação Claude Code sob Controle Total

## 1) Contrato de Resposta do Agente

O agente **sempre** responde neste formato por tarefa:

```
=== PLAN ===
goal: <uma frase verificável>
steps:
  - read: <arquivos-alvo>
  - edit: <arquivos/funções-alvo>
  - run: <comandos exatos de verificação>
risks: [leakage, custo, latência, flakiness]
done_when:
  - <checks objetivos observáveis>
refs:
  prd:
    file: docs/prd/PRD_XGB.md
    version: 1.2.0
    sha256: 4e6c…9b
=== /PLAN ===

=== DIFF ===
# ÚNICO patch unificado, edição mínima.
--- a/src/...
+++ b/src/...
@@ ...
=== /DIFF ===

=== RUN ===
make deterministic
make type
make test
python -m src.models.xgb --config configs/xgb.yaml --fast
python -m src.models.lstm --config configs/lstm.yaml --fast
python -m src.backtest.engine --config configs/backtest.yaml --fast
=== /RUN ===

=== MEMORY_UPDATE(JSON) ===
{ "ts": "<UTC ISO>", "task": "<slug>",
  "touched": ["src/..."], "entrypoints_added": [],
  "configs_changed": ["configs/..."],
  "mlflow": {"run_id": null, "tags": {"degraded_mode": false}},
  "notes": "porquês, trade-offs, TODO imediato" }
=== /MEMORY_UPDATE ===
```

## 2) Auto-QA do Agente (Scorecard obrigatório)

Salvar em `artifacts/reports/last_scorecard.txt` e imprimir no fim da resposta:

```
=== AGENT_SCORECARD ===
data_contracts: pass|fail
temporal_leak_checks: pass|fail
determinism: pass|fail
tests(pytest): <X> passed, <Y> failed
lint(type): ruff 0 warn, mypy 0 err, audit 0 vuln
ml_metrics: {"F1": ..., "PR-AUC": ..., "MCC": ..., "Brier": ...}
trading_metrics: {"Sharpe": ..., "MDD": ..., "turnover": ..., "EV_val_net": ...}
risk_notes: "<1 linha honesta>"
=== /AGENT_SCORECARD ===
```

Se falhar, o agente deve responder com um novo `=== PLAN ===` de correção.

---

## PRDs persistentes e Instruções de Projeto (Claude Projects)

1. **Base de conhecimento**: suba `PRD_XGB.md` e `PRD_LSTM.md` no **Project** para recuperação automática.
2. **Project Instructions**: cravar regra:

```
Regra de PRD (obrigatória):
- Sempre identificar o modelo-alvo (XGBoost ou LSTM) e CONSULTAR o PRD correspondente antes de planejar, editar ou treinar.
- O plano (=== PLAN ===) deve citar explicitamente: PRD: <arquivo> | versão <semver> | sha256 <hash curto>
- Se o PRD estiver ausente/desatualizado, parar e pedir correção.
- Não avaliar resultados de teste sem cumprir o PRD.
```

3. **Memória no repositório**: bloco fixo “PRD Registry” em `AI_MEMORY.md`:

```yaml
## PRD Registry
- name: PRD_XGB
  path: docs/prd/PRD_XGB.md
  version: 1.2.0
  sha256: 4e6c…9b
- name: PRD_LSTM
  path: docs/prd/PRD_LSTM.md
  version: 1.1.3
  sha256: a91d…07
```

4. **Model Registry/MLflow**: logar `prd_name`, `prd_version`, `prd_sha256` em toda run.

---

> **Resumo**: método acima de brilho. Com determinismo, execução t+1, custos completos e threshold por EV, não há Sharpe “de papel”. É pesquisa que se sustenta na segunda-feira de manhã.
