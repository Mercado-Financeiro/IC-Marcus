# Makefile para ML Finance Crypto Project

.PHONY: help install fmt type test deterministic train backtest dash clean

# Cores para output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Mostra este help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Instala dependências e configura ambiente
	@echo "$(YELLOW)📦 Instalando dependências...$(NC)"
	pip install -U pip setuptools wheel
	pip install -e .[dev]
	pre-commit install
	@echo "$(GREEN)✅ Instalação completa!$(NC)"

fmt: ## Formata código com ruff e black
	@echo "$(YELLOW)🎨 Formatando código...$(NC)"
	ruff check --fix src tests
	black src tests
	ruff format src tests
	@echo "$(GREEN)✅ Código formatado!$(NC)"

type: ## Verifica tipos com mypy
	@echo "$(YELLOW)🔍 Verificando tipos...$(NC)"
	mypy src
	@echo "$(GREEN)✅ Tipos verificados!$(NC)"

test: ## Roda testes com pytest
	@echo "$(YELLOW)🧪 Rodando testes...$(NC)"
	pytest -q tests/
	@echo "$(GREEN)✅ Testes passaram!$(NC)"

deterministic: ## Configura ambiente determinístico
	@echo "$(YELLOW)🎲 Configurando determinismo...$(NC)"
	@python - <<'PY'
	import os, random, numpy as np
	try:
	    import torch
	    torch.use_deterministic_algorithms(True)
	    torch.backends.cudnn.deterministic = True
	    torch.backends.cudnn.benchmark = False
	    print("✅ Torch configurado")
	except Exception:
	    print("⚠️ Torch não disponível")
	os.environ.update({
	    "PYTHONHASHSEED": "0",
	    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
	    "CUDA_LAUNCH_BLOCKING": "1",
	})
	random.seed(42)
	np.random.seed(42)
	print("✅ Determinismo configurado!")
	PY

train: ## Treina modelo (MODEL=xgb|lstm)
	@echo "$(YELLOW)🚀 Treinando modelo $(MODEL)...$(NC)"
	python -m src.models.$(MODEL) --config configs/$(MODEL).yaml

backtest: ## Roda backtest
	@echo "$(YELLOW)💰 Executando backtest...$(NC)"
	python -m src.backtest.engine --config configs/backtest.yaml

dash: ## Inicia dashboard Streamlit
	@echo "$(YELLOW)📊 Iniciando dashboard...$(NC)"
	streamlit run src/dashboard/app.py --server.port $${DASHBOARD_PORT:-8501}

notebook: ## Abre Jupyter Lab com o notebook principal
	@echo "$(YELLOW)📓 Abrindo Jupyter Lab...$(NC)"
	jupyter lab notebooks/IC_Crypto_Complete.ipynb

sync: ## Sincroniza notebooks com Jupytext
	@echo "$(YELLOW)🔄 Sincronizando notebooks...$(NC)"
	jupytext --sync notebooks/*.ipynb
	@echo "$(GREEN)✅ Notebooks sincronizados!$(NC)"

audit: ## Auditoria de segurança
	@echo "$(YELLOW)🔒 Auditando segurança...$(NC)"
	pip-audit -r requirements.txt --strict || true
	bandit -r src/ || true
	gitleaks detect --no-git --redact || true
	@echo "$(GREEN)✅ Auditoria completa!$(NC)"

clean: ## Limpa arquivos temporários
	@echo "$(YELLOW)🧹 Limpando arquivos temporários...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	find . -type f -name "temp_*.csv" -delete
	find . -type f -name "temp_*.pkl" -delete
	@echo "$(GREEN)✅ Limpeza completa!$(NC)"

# Comandos compostos
check: fmt type test ## Roda formatação, tipos e testes
	@echo "$(GREEN)✅ Todas as verificações passaram!$(NC)"

setup: install deterministic ## Setup completo do projeto
	@echo "$(GREEN)✅ Projeto configurado e pronto!$(NC)"

all: setup check ## Setup e verificações completas
	@echo "$(GREEN)🎉 Projeto totalmente configurado e verificado!$(NC)"

# Variáveis de ambiente default
export PYTHONPATH := $(shell pwd)
export MLFLOW_TRACKING_URI := artifacts/mlruns
export DASHBOARD_PORT ?= 8501