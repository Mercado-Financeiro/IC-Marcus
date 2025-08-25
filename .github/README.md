# GitHub Actions Workflows for LSTM Profit-Oriented Pipeline

This directory contains comprehensive GitHub Actions workflows for the ML Finance Crypto project, with special focus on the LSTM profit-oriented trading pipeline.

## 🔄 Workflows Overview

### Core CI/CD Workflows

#### 1. **CI Pipeline** (`ci.yml`)
Comprehensive continuous integration pipeline that runs on every push and PR.

**Features:**
- **Code Quality**: Ruff linting, Black formatting, MyPy type checking
- **Security Scanning**: Bandit, Safety, Gitleaks with SARIF upload
- **Testing**: Unit tests with coverage, integration tests
- **ML Validation**: Model smoke tests, MLflow tracking, determinism verification
- **Docker**: Container build and test
- **Performance**: Regression detection for PR changes

**Triggers:** Push to main/develop, Pull requests, Daily scheduled runs

#### 2. **Model Validation** (`model_validation.yml`) 
Specialized validation for ML models and performance gates.

**Features:**
- **Determinism Tests**: Ensures reproducible model training
- **Data Schema Validation**: Validates input data consistency
- **Performance Regression**: Compares model metrics against baselines
- **Memory Efficiency**: Monitors resource usage

**Triggers:** Push/PR to model-related paths

#### 3. **MLOps Pipeline** (`mlops.yml`)
End-to-end ML operations for model training and deployment.

**Features:**
- **Multi-Model Training**: XGBoost and LSTM support
- **Feature Engineering**: Automated feature pipeline
- **Data Quality Checks**: Comprehensive data validation
- **Model Registry**: Automated model registration
- **Data Drift Detection**: Monitoring for data distribution changes

**Triggers:** Model/feature changes, Manual dispatch with options

### Specialized Workflows

#### 4. **Smoke Backtest** (`smoke-backtest.yml`) 🔥
**NEW:** Quick validation specifically for the profit-oriented pipeline.

**Features:**
- **Fast Validation**: 5-15 minute pipeline tests
- **Multi-Model Support**: Tests both XGBoost and LSTM
- **Realistic Data**: Generates crypto-like test data
- **Memory Monitoring**: Ensures efficient resource usage
- **PR Comments**: Automated results reporting

**Triggers:** PRs affecting models/backtest/metrics, Manual dispatch

#### 5. **PR Automation** (`pr-automation.yml`) 🤖
**NEW:** Advanced PR interactions with Claude Code integration.

**Features:**
- **Welcome Comments**: Automated PR onboarding
- **Claude Integration**: AI-powered code review and assistance
- **CI Results Summary**: Aggregated status reporting
- **Smart Analysis**: Code quality and complexity analysis

**Triggers:** PR events, Issue comments mentioning `@claude`

#### 6. **Security Scanning** (`security.yml`)
Comprehensive security analysis and monitoring.

**Features:**
- **Python Security**: Bandit, Safety, Semgrep analysis
- **Container Security**: Hadolint, Trivy scanning
- **Secret Detection**: Gitleaks, TruffleHog
- **Dependency Analysis**: OWASP checks, license compliance
- **CodeQL**: GitHub's semantic code analysis
- **ML Security**: Model integrity, data leakage detection

**Triggers:** Push/PR, Weekly scheduled, Manual

#### 7. **Cache Management** (`cache-management.yml`) 📦
**NEW:** Intelligent caching and dependency optimization.

**Features:**
- **Cache Analysis**: Size and usage monitoring
- **Cache Warming**: Pre-builds common dependencies
- **Dependency Management**: Update recommendations and security
- **Cleanup Automation**: Scheduled cache maintenance

**Triggers:** Weekly scheduled, Manual with cleanup options

## 🚀 Quick Start

### Required Secrets

Add these secrets in your repository settings (`Settings > Secrets and variables > Actions`):

```bash
# Required
GITHUB_TOKEN        # Automatically provided by GitHub

# Optional (for enhanced features)
CLAUDE_API_KEY      # For Claude Code integration
MLFLOW_TRACKING_URI # External MLflow server (optional)
BINANCE_API_KEY     # For live data (optional, use testnet)
BINANCE_SECRET_KEY  # For live data (optional, use testnet)
```

### Repository Configuration

1. **Branch Protection Rules**:
   ```
   Branch: main
   Require status checks: ✅
   Required checks:
   - CI Status Check
   - Smoke Test Status
   Require PR reviews: ✅ (1 approver)
   ```

2. **Security Settings**:
   ```
   Enable vulnerability alerts: ✅
   Enable security fixes: ✅
   Enable secret scanning: ✅
   Enable code scanning: ✅
   ```

### First Time Setup

1. **Clone and setup environment**:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -e .[dev]
   ```

2. **Test integration locally**:
   ```bash
   python test_integration_final.py
   ```

3. **Run smoke tests**:
   ```bash
   ./run_profit_pipeline.sh --model xgboost --quick
   ./run_profit_pipeline.sh --model lstm --trials 3
   ```

4. **Create your first PR** to trigger the workflows.

## 🎯 Workflow Triggers and Usage

### For Developers

**When you push code:**
- CI Pipeline runs automatically
- Security scans activate for main branches
- Model validation runs for model changes

**When you create a PR:**
- All CI checks run
- Smoke tests validate model changes
- Welcome comment provides guidance
- Use `@claude review` for AI assistance

**Manual triggers:**
```bash
# Full training with all models
gh workflow run mlops.yml -f model_type=both -f run_full_validation=true

# Quick smoke test
gh workflow run smoke-backtest.yml -f model_type=lstm

# Cache cleanup
gh workflow run cache-management.yml -f cleanup_type=full
```

### For CI/CD Management

**Weekly maintenance:**
- Dependency scanning (automated)
- Cache optimization (automated)
- Security updates (automated)

**Monthly tasks:**
- Review security reports
- Update baseline metrics
- Cleanup old artifacts

## 🔧 Customization

### Environment Variables

Modify these in workflow files for customization:

```yaml
env:
  PYTHON_VERSION: '3.11'          # Python version
  MLFLOW_TRACKING_URI: 'file://./artifacts/mlruns'  # MLflow location
  CACHE_VERSION: 'v2'             # Cache invalidation
  PYTHONHASHSEED: 0               # Deterministic behavior
```

### Smoke Test Configuration

Adjust test parameters in `smoke-backtest.yml`:

```yaml
- name: Run XGBoost smoke test
  run: |
    ./run_profit_pipeline.sh \
      --model xgboost \
      --trials 10 \           # Increase for thorough testing
      --symbol BTCUSDT \
      --timeframe 15m \
      --quick                 # Remove for full test
```

### Claude Integration

Customize Claude responses in `pr-automation.yml`:

```python
# Add custom analysis patterns
if action == 'review':
    # Custom code analysis logic
    pass
```

## 📊 Monitoring and Reporting

### Dashboard Links
- **Actions**: `https://github.com/owner/repo/actions`
- **Security**: `https://github.com/owner/repo/security`
- **Insights**: `https://github.com/owner/repo/pulse`

### Key Metrics to Monitor

1. **CI Success Rate**: Should be > 95%
2. **Test Coverage**: Aim for > 80%
3. **Security Issues**: Zero critical issues
4. **Cache Hit Rate**: Monitor for performance
5. **Model Performance**: Track regression metrics

### Artifact Retention

- **Test Results**: 30 days
- **Security Reports**: 90 days
- **Model Artifacts**: 7 days (smoke tests)
- **Cache Analysis**: 30 days

## 🐛 Troubleshooting

### Common Issues

**1. Smoke tests failing:**
```bash
# Check integration locally first
python test_integration_final.py

# Run with verbose output
./run_profit_pipeline.sh --model lstm --trials 3 --verbose
```

**2. Cache issues:**
```bash
# Clear local cache
rm -rf data/cache/*
rm -rf artifacts/models/*

# Force cache rebuild (manual workflow dispatch)
gh workflow run cache-management.yml -f force_rebuild=true
```

**3. Claude not responding:**
- Check that comment contains exactly `@claude`
- Verify PR is from a branch (not fork)
- Check workflow permissions

**4. Memory issues in CI:**
- Reduce trials in smoke tests
- Check for data leaks in model code
- Monitor cache sizes

### Debug Commands

```bash
# Check workflow status
gh run list --workflow=ci.yml

# Download artifacts
gh run download <run-id>

# View logs
gh run view <run-id> --log
```

## 🔒 Security Best Practices

1. **Secrets Management**:
   - Never hardcode secrets in workflows
   - Use environment-specific secrets
   - Rotate secrets regularly

2. **Permissions**:
   - Minimal required permissions only
   - No write access unless necessary
   - Regular permission audits

3. **Dependencies**:
   - Pin action versions (not @main)
   - Regular security scanning
   - Monitor for vulnerabilities

## 📈 Performance Optimization

### Cache Strategy

```yaml
# Layered caching approach
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

### Parallel Execution

```yaml
strategy:
  matrix:
    model: [xgboost, lstm]
    python-version: ['3.11', '3.12']
  fail-fast: false  # Continue other jobs on failure
```

### Resource Limits

```yaml
timeout-minutes: 30        # Prevent runaway jobs
runs-on: ubuntu-latest     # Cost-effective runners
```

## 🆘 Support

For workflow issues:

1. **Check the logs**: Review failed workflow runs
2. **Local testing**: Reproduce issues locally first
3. **Documentation**: Check this README and workflow comments
4. **Issues**: Create GitHub issue with workflow logs
5. **Claude Help**: Use `@claude explain workflow failure` in PRs

---

**Last Updated**: 2025-01-25
**Workflow Version**: v2.0
**Maintained by**: ML Engineering Team