# Makefile para Projeto ML Trading de Criptomoedas
# Configurações e comandos padronizados

PYTHON = python3
SYMBOL = BTCUSDT
TIMEFRAME = 15m

# Instalação e Setup
.PHONY: install setup deterministic

setup:
	@echo "🔧 Configurando ambiente..."
	mkdir -p artifacts/models artifacts/reports data/raw data/processed
	@echo "✅ Estrutura de diretórios criada"

deterministic:
	@echo "🎯 Configurando determinismo..."
	@$(PYTHON) -c "\
import os, random, numpy as np; \
try: \
    import torch; \
    torch.use_deterministic_algorithms(True); \
    torch.backends.cudnn.deterministic = True; \
    torch.backends.cudnn.benchmark = False; \
    print('✅ Torch configurado para determinismo'); \
except Exception: \
    print('⚠️ Torch não disponível'); \
os.environ.update({ \
    'PYTHONHASHSEED': '0', \
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8', \
    'CUDA_LAUNCH_BLOCKING': '1' \
}); \
random.seed(42); \
np.random.seed(42); \
print('✅ Determinismo configurado (SEED=42)')"

# Execução de Modelos
.PHONY: train-xgb train-lstm train-both quick-test

train-xgb:
	@echo "🎯 Treinando XGBoost com $(SYMBOL) $(TIMEFRAME)..."
	$(PYTHON) run_optimization.py --model xgboost --symbol $(SYMBOL) --timeframe $(TIMEFRAME)

train-lstm:
	@echo "🧠 Treinando LSTM com $(SYMBOL) $(TIMEFRAME)..."
	$(PYTHON) run_optimization.py --model lstm --symbol $(SYMBOL) --timeframe $(TIMEFRAME)

train-both:
	@echo "🚀 Treinando XGBoost e LSTM com $(SYMBOL) $(TIMEFRAME)..."
	$(PYTHON) run_optimization.py --model both --symbol $(SYMBOL) --timeframe $(TIMEFRAME)

quick-test:
	@echo "⚡ Teste rápido com XGBoost..."
	$(PYTHON) run_optimization.py --quick --model xgboost --symbol $(SYMBOL) --timeframe $(TIMEFRAME)

# Pipeline Notebook (NOVO - Sistema Corrigido)
.PHONY: train-notebook quick-notebook validate-notebook

train-notebook:
	@echo "🚀 Pipeline Notebook PRODUÇÃO com $(SYMBOL) $(TIMEFRAME)..."
	$(PYTHON) run_notebook_pipeline.py --mode production --symbol $(SYMBOL) --timeframe $(TIMEFRAME) --trials 50

quick-notebook:
	@echo "⚡ Demo rápida Pipeline Notebook..."
	$(PYTHON) run_notebook_pipeline.py --mode quick

validate-notebook:
	@echo "🧪 Validação completa Pipeline Notebook..."
	$(PYTHON) run_notebook_pipeline.py --mode validation

# Dashboard e Visualização
.PHONY: dashboard mlflow-ui

dashboard:
	@echo "📊 Iniciando dashboard Streamlit..."
	streamlit run src/dashboard/app.py --server.port 8501

mlflow-ui:
	@echo "📈 Iniciando MLflow UI..."
	mlflow ui --backend-store-uri artifacts/mlruns --port 5000

# Segurança e Qualidade
.PHONY: security-audit detect-secrets validate-all pre-commit-run

security-audit:
	@echo "🛡️ Executando auditoria completa de segurança..."
	$(PYTHON) scripts/security_audit.py

detect-secrets:
	@echo "🔐 Detectando segredos no código..."
	$(PYTHON) scripts/detect_secrets.py src/

validate-all:
	@echo "✅ Executando todas as validações..."
	@$(MAKE) deterministic
	@$(MAKE) security-audit
	@$(MAKE) detect-secrets
	@echo "🎉 Todas as validações passaram!"

pre-commit-run:
	@echo "🔧 Executando pre-commit hooks..."
	pre-commit run --all-files

# MLOps e Model Registry
.PHONY: test-model-registry promote-model rollback-model

test-model-registry:
	@echo "🏆 Testando Model Registry..."
	$(PYTHON) scripts/test_model_registry.py

promote-model:
	@if [ -z "$(MODEL_NAME)" ] || [ -z "$(VERSION)" ]; then \
		echo "❌ Especifique MODEL_NAME e VERSION"; \
		echo "Uso: make promote-model MODEL_NAME=crypto_xgb VERSION=1"; \
		exit 1; \
	fi
	@echo "🚀 Promovendo modelo $(MODEL_NAME) versão $(VERSION)..."
	$(PYTHON) -c "\
from src.mlops.model_registry import get_model_registry; \
registry = get_model_registry(); \
result = registry.promote_to_production('$(MODEL_NAME)', '$(VERSION)'); \
print('✅ Promoção bem-sucedida!' if result else '❌ Falha na promoção')"

rollback-model:
	@if [ -z "$(MODEL_NAME)" ]; then \
		echo "❌ Especifique MODEL_NAME"; \
		echo "Uso: make rollback-model MODEL_NAME=crypto_xgb"; \
		exit 1; \
	fi
	@echo "⏪ Executando rollback do modelo $(MODEL_NAME)..."
	$(PYTHON) -c "\
from src.mlops.model_registry import get_model_registry; \
registry = get_model_registry(); \
result = registry.rollback_to_previous_champion('$(MODEL_NAME)'); \
print('✅ Rollback bem-sucedido!' if result else '❌ Falha no rollback')"

# Comandos informativos
.PHONY: help status

help:
	@echo "🚀 Projeto ML Trading de Criptomoedas - Sistema Completo"
	@echo ""
	@echo "📦 Setup:"
	@echo "  make setup          - Configurar ambiente"
	@echo "  make deterministic  - Configurar determinismo"
	@echo ""
	@echo "🎯 Treinamento:"
	@echo "  make train-xgb      - Treinar XGBoost"
	@echo "  make train-lstm     - Treinar LSTM"
	@echo "  make train-both     - Treinar ambos modelos"
	@echo "  make quick-test     - Teste rápido"
	@echo ""
	@echo "🚀 Pipeline Notebook (RECOMENDADO):"
	@echo "  make quick-notebook     - Demo rápida (5min)"
	@echo "  make train-notebook     - Produção completa (30-60min)"
	@echo "  make validate-notebook  - Testes de validação (15min)"
	@echo ""
	@echo "📊 Visualização:"
	@echo "  make dashboard      - Dashboard Streamlit"
	@echo "  make mlflow-ui      - MLflow UI"
	@echo ""
	@echo "🛡️ Segurança e Qualidade:"
	@echo "  make security-audit - Auditoria completa de segurança"
	@echo "  make detect-secrets - Detectar segredos no código"
	@echo "  make validate-all   - Executar todas as validações"
	@echo "  make pre-commit-run - Executar pre-commit hooks"
	@echo ""
	@echo "🏆 MLOps:"
	@echo "  make test-model-registry                    - Testar Model Registry"
	@echo "  make promote-model MODEL_NAME=x VERSION=y   - Promover modelo"
	@echo "  make rollback-model MODEL_NAME=x            - Rollback de modelo"
	@echo ""
	@echo "💡 Exemplos:"
	@echo "  make quick-notebook                                    # Demo rápida"
	@echo "  make train-notebook SYMBOL=ETHUSDT TIMEFRAME=1h       # Produção ETHUSDT"
	@echo "  make validate-notebook                                 # Validação completa"
	@echo "  make train-xgb SYMBOL=ETHUSDT TIMEFRAME=1h            # XGBoost tradicional"
	@echo "  make promote-model MODEL_NAME=crypto_xgb VERSION=1     # MLOps"
	@echo "  make security-audit                                    # Segurança"

status:
	@echo "📊 Status do Projeto - Sistema ML Trading"
	@echo "=========================================="
	@echo "Configurações YAML:"
	@ls -la configs/*.yaml 2>/dev/null | wc -l | xargs -I {} echo "  ✅ Configs: {} arquivos"
	@echo ""
	@echo "Dados:"
	@ls -la data/raw/ 2>/dev/null | wc -l | xargs -I {} echo "  📁 Raw: {} arquivos"
	@ls -la data/processed/ 2>/dev/null | wc -l | xargs -I {} echo "  📁 Processed: {} arquivos"
	@echo ""
	@echo "Modelos e MLflow:"
	@ls -la artifacts/models/ 2>/dev/null | wc -l | xargs -I {} echo "  🤖 Modelos salvos: {} arquivos"
	@ls -la artifacts/mlruns/*/*/ 2>/dev/null | wc -l | xargs -I {} echo "  📊 MLflow runs: {} experimentos"
	@echo ""
	@echo "Segurança:"
	@if [ -f .pre-commit-config.yaml ]; then echo "  ✅ Pre-commit configurado"; else echo "  ❌ Pre-commit não configurado"; fi
	@if [ -f requirements.txt ]; then echo "  ✅ Requirements.txt presente"; else echo "  ❌ Requirements.txt ausente"; fi
	@if [ -f .secrets.baseline ]; then echo "  ✅ Secrets baseline criado"; else echo "  ❌ Secrets baseline ausente"; fi
	@echo ""
	@echo "Scripts:"
	@ls -la scripts/*.py 2>/dev/null | wc -l | xargs -I {} echo "  🔧 Scripts: {} utilitários"

# Comando padrão
.DEFAULT_GOAL := help
