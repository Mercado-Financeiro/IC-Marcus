# CI/CD Pipeline Guide

## Overview

This document describes the comprehensive CI/CD pipeline setup for the ML Trading System, including GitHub Actions workflows, pre-commit hooks, dependency management, and MLOps integration.

## Table of Contents

1. [Quick Start](#quick-start)
2. [GitHub Actions Workflows](#github-actions-workflows)
3. [Pre-commit Hooks](#pre-commit-hooks)
4. [Dependency Management](#dependency-management)
5. [Docker Setup](#docker-setup)
6. [MLOps Integration](#mlops-integration)
7. [Security Best Practices](#security-best-practices)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Initial Setup

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Generate dependency lock file
pip install pip-tools
pip-compile --generate-hashes requirements.in

# Run pre-commit on all files
pre-commit run --all-files

# Start Docker services
docker-compose up -d
```

### Running CI Locally

```bash
# Run linting
ruff check src tests
black --check src tests

# Run security checks
bandit -r src
pip-audit -r requirements.txt

# Run tests
pytest tests/unit -v --cov=src

# Check for secrets
detect-secrets scan --baseline .secrets.baseline
```

## GitHub Actions Workflows

### 1. Main CI Pipeline (`.github/workflows/ci.yml`)

**Triggers:**
- Push to main, develop, or feature branches
- Pull requests to main or develop
- Daily schedule at 2 AM UTC

**Jobs:**
- **lint-and-format**: Code quality checks with Ruff, Black, and MyPy
- **security**: Security scanning with Bandit, Safety, pip-audit, and Gitleaks
- **test**: Unit tests with coverage for Python 3.11 and 3.12
- **integration-test**: Integration tests with timeout protection
- **ml-validation**: Model training smoke tests and MLflow tracking
- **docker-build**: Docker image build and test
- **performance-check**: Performance regression detection (PR only)
- **license-check**: Dependency license compliance

### 2. MLOps Pipeline (`.github/workflows/mlops.yml`)

**Triggers:**
- Push/PR to model or feature code
- Manual workflow dispatch with model selection

**Features:**
- Model validation and training
- Data drift detection
- Model registry operations
- Performance reporting
- Automatic model degradation checks

### 3. Dependency Lock (`.github/workflows/dependency-lock.yml`)

**Triggers:**
- Changes to requirements.in or pyproject.toml
- Weekly schedule
- Manual dispatch

**Features:**
- Generates requirements.lock with hashes
- Creates PR with updated dependencies
- Ensures reproducible builds

## Pre-commit Hooks

### Configuration

The `.pre-commit-config.yaml` file includes:

1. **General Fixes**
   - Trailing whitespace removal
   - End-of-file fixing
   - YAML/JSON/TOML validation
   - Large file detection
   - Merge conflict detection

2. **Python Tools**
   - Black (formatting)
   - Ruff (linting)
   - MyPy (type checking)
   - Bandit (security)
   - isort (import sorting)
   - pydocstyle (docstring checking)

3. **Security**
   - detect-secrets (secret scanning)
   - safety (dependency vulnerabilities)

4. **Other Tools**
   - Jupyter notebook cleanup
   - Markdown formatting
   - Dockerfile linting
   - Git commit message linting

### Usage

```bash
# Install hooks
pre-commit install

# Run on staged files
pre-commit run

# Run on all files
pre-commit run --all-files

# Update hook versions
pre-commit autoupdate

# Skip hooks temporarily
git commit --no-verify -m "Emergency fix"
```

## Dependency Management

### Using pip-tools

1. **Add dependencies to `requirements.in`**
   ```
   package>=1.0.0,<2.0.0
   ```

2. **Generate lock file with hashes**
   ```bash
   pip-compile --generate-hashes requirements.in -o requirements.lock
   ```

3. **Install from lock file**
   ```bash
   pip install -r requirements.lock
   ```

### Dependabot Configuration

Located in `.github/dependabot.yml`:

- **Python dependencies**: Weekly updates on Monday
- **GitHub Actions**: Weekly updates
- **Docker**: Weekly updates
- Grouped updates for related packages
- Automatic PR creation with reviewers

## Docker Setup

### Services

The `docker-compose.yml` includes:

1. **Core Services**
   - `mlflow`: MLflow tracking server
   - `dashboard`: Streamlit dashboard
   - `api`: Model serving API
   - `jupyter`: Development notebook server

2. **Data Services**
   - `postgres`: MLflow backend database
   - `redis`: Caching layer

3. **Monitoring**
   - `prometheus`: Metrics collection
   - `grafana`: Visualization dashboards

### Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f [service_name]

# Stop services
docker-compose down

# Clean volumes
docker-compose down -v

# Rebuild images
docker-compose build --no-cache

# Access services
# MLflow: http://localhost:5000
# Dashboard: http://localhost:8501
# API: http://localhost:8000
# Jupyter: http://localhost:8888
# Grafana: http://localhost:3000
```

### Production Dockerfile

The `deployment/Dockerfile` uses:
- Multi-stage build for size optimization
- Non-root user for security
- Health checks for reliability
- Deterministic environment variables

## MLOps Integration

### Model Validation

Every model PR triggers:
1. Data quality checks
2. Feature engineering validation
3. Model training smoke tests
4. Performance regression checks
5. MLflow experiment tracking

### Model Registry

Production models are:
- Versioned with semantic versioning
- Tagged with metadata (commit, dataset hash, config)
- Validated before promotion
- Support rollback capabilities

### Monitoring

- **Data Drift**: Scheduled checks for distribution changes
- **Model Performance**: Continuous metric tracking
- **System Health**: Resource usage and latency monitoring

## Security Best Practices

### Secret Management

1. **Never commit secrets**
   - Use `.env` files (gitignored)
   - Store in GitHub Secrets for CI
   - Use environment variables

2. **Secret Scanning**
   - Pre-commit detect-secrets hook
   - GitHub Gitleaks action
   - Regular baseline updates

### Dependency Security

1. **Vulnerability Scanning**
   - pip-audit in CI
   - Safety checks
   - Dependabot alerts

2. **License Compliance**
   - Automated license checking
   - Fail on copyleft licenses
   - Regular audits

### Container Security

1. **Image Scanning**
   - Use official base images
   - Regular rebuilds
   - Minimal attack surface

2. **Runtime Security**
   - Non-root users
   - Read-only filesystems where possible
   - Network isolation

## Troubleshooting

### Common Issues

#### 1. Pre-commit Hook Failures

```bash
# Fix formatting issues automatically
black src tests
ruff check --fix src tests

# Update hook versions
pre-commit autoupdate
```

#### 2. CI Pipeline Failures

```bash
# Test locally first
make test
make security-audit

# Check specific job logs in GitHub Actions
```

#### 3. Docker Issues

```bash
# Reset Docker environment
docker-compose down -v
docker system prune -a
docker-compose up --build
```

#### 4. Dependency Conflicts

```bash
# Recreate virtual environment
rm -rf venv/
python -m venv venv
pip install -r requirements.lock
```

### CI Environment Variables

Required in GitHub Secrets:
- `MLFLOW_TRACKING_URI`
- `GITHUB_TOKEN` (automatically provided)
- Any API keys for data sources

### Performance Optimization

1. **Cache Usage**
   - pip cache in GitHub Actions
   - Docker layer caching
   - Dependency caching

2. **Parallel Execution**
   - Matrix strategy for multiple Python versions
   - Concurrent job execution
   - pytest-xdist for parallel tests

3. **Conditional Execution**
   - Path filters for relevant changes
   - Skip unchanged components
   - Fast-fail on critical errors

## Maintenance

### Regular Tasks

1. **Weekly**
   - Review Dependabot PRs
   - Check security alerts
   - Update pre-commit hooks

2. **Monthly**
   - Review and update CI workflows
   - Audit dependencies
   - Update Docker images

3. **Quarterly**
   - Security audit
   - Performance baseline update
   - Documentation review

### Monitoring CI/CD Health

1. **Metrics to Track**
   - Build success rate
   - Average build time
   - Test coverage trends
   - Security vulnerability count

2. **Alerts to Configure**
   - Failed builds on main
   - Security vulnerabilities
   - Performance regressions
   - Low test coverage

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pre-commit Documentation](https://pre-commit.com/)
- [pip-tools Documentation](https://pip-tools.readthedocs.io/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)

## Contact

For issues or questions about the CI/CD pipeline, please:
1. Check this documentation
2. Review existing GitHub Issues
3. Contact the DevOps team
4. Create a new issue with the `ci/cd` label