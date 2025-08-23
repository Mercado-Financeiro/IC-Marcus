# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core code — `data/`, `features/`, `models/`, `backtest/`, `dashboard/`, `mlops/`, `trading/`, `utils/`.
- `tests/`: unit, integration, regression, validation, performance suites.
- `configs/`: YAML configs (e.g., `xgb.yaml`, `lstm.yaml`, `multi_horizon.yaml`).
- `artifacts/`: `mlruns`, models, reports; `data/`: `raw/` and `processed/` (DVC tracked).
- `notebooks/` and `scripts/`: experiments and utilities.

## Build, Test, and Development Commands
- Setup: `make setup` (folders), `make deterministic` (reproducibility).
- Train: `make train-xgb SYMBOL=BTCUSDT TIMEFRAME=15m`, `make train-lstm`, `make train-both`, `make quick-test`.
- Pipelines: `make quick-notebook`, `make quick-multi-horizon`, `make quick-lstm-complete`.
- Dashboard & MLflow: `make dashboard`, `make mlflow-ui`.
- Quality & Security: `make pre-commit-run`, `make validate-all`, `make security-audit`.
- Tests: `pytest -q`, `pytest tests/unit -v`, coverage `pytest --cov=src --cov-report=term`.

## Coding Style & Naming Conventions
- Python 3.11; format with `black` (line length 100) and lint with `ruff`.
- Type hints required; `mypy` runs with `disallow_untyped_defs = true`.
- Naming: `snake_case` for files/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Tests: `tests/<suite>/test_*.py` (e.g., `tests/unit/test_features/test_adaptive_labeling.py`).

## Testing Guidelines
- Frameworks: `pytest`, `hypothesis`; markers: `slow`, `integration`, `asyncio`.
- Quick run: `pytest -m "not slow and not integration" -q`.
- Targeted suites: unit (`tests/unit`), integration (`tests/integration`), regression (`tests/regression`).
- Coverage configured in `pyproject.toml`; aim to keep deltas covered for new code.

## Commit & Pull Request Guidelines
- Commits follow Conventional Commits: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
- PRs must: describe changes, link issues, list config impacts, include/adjust tests, and attach screenshots for dashboard/UI when relevant. Reference MLflow run IDs for model changes.
- CI hygiene: run `pre-commit run --all-files`, `pytest`, `ruff`, `black`, and `mypy` locally before opening/merging.

## Security & Configuration Tips
- Never commit secrets; use `.env` from `.env.example`. Run `make detect-secrets` and `make security-audit`.
- Large data is DVC-managed (`data.dvc`); do not commit `data/raw`.
- Prefer CPU determinism for benchmarks; document any GPU flags used.

## Agent Memory (CLAUDE.md‑aligned)
- Log each task in `AI_MEMORY.md`: goal, decisions, files changed, commands run, configs used, artifacts/MLflow run IDs.
- Keep `CODE_MAP.md` updated with new entrypoints and important paths after merges.
- Record leak‑prevention checks (purged/embargo splits), determinism flags, and the chosen evaluation threshold(s).
- Use concise, dated entries; avoid secrets or private data. Follow the tone and structure shown in `CLAUDE.md`.
