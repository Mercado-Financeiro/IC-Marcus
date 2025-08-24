# Quick Venv Setup
Write-Host "SETUP RAPIDO DO AMBIENTE PYTHON" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Fechar VS Code
$vscode = Get-Process Code -ErrorAction SilentlyContinue
if ($vscode) {
    Write-Host "Fechando VS Code..." -ForegroundColor Yellow
    Stop-Process -Name Code -Force
    Start-Sleep -Seconds 2
}

# Limpar
Write-Host "Limpando ambiente antigo..." -ForegroundColor Yellow
if (Test-Path .venv) {
    Remove-Item -Recurse -Force .venv
}
pip cache purge 2>$null | Out-Null

# Criar venv
Write-Host "Criando novo ambiente..." -ForegroundColor Yellow
python -m venv .venv

# Ativar
& .\.venv\Scripts\Activate.ps1

# Atualizar pip
Write-Host "Atualizando pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --no-cache-dir -q

# Instalar essenciais
Write-Host "Instalando pacotes (vai demorar alguns minutos)..." -ForegroundColor Cyan

Write-Host "[1/9] NumPy e Pandas" -ForegroundColor Gray
pip install --no-cache-dir numpy pandas -q

Write-Host "[2/9] Scikit-learn" -ForegroundColor Gray
pip install --no-cache-dir scikit-learn -q

Write-Host "[3/9] XGBoost" -ForegroundColor Gray
pip install --no-cache-dir xgboost -q

Write-Host "[4/9] PyTorch CPU" -ForegroundColor Gray
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu -q

Write-Host "[5/9] Optuna e MLflow" -ForegroundColor Gray
pip install --no-cache-dir optuna mlflow -q

Write-Host "[6/9] Validacao" -ForegroundColor Gray
pip install --no-cache-dir pandera pydantic-settings -q

Write-Host "[7/9] Dev tools" -ForegroundColor Gray
pip install --no-cache-dir pytest ruff black mypy -q

Write-Host "[8/9] Jupyter" -ForegroundColor Gray
pip install --no-cache-dir jupyter ipykernel -q

Write-Host "[9/9] Streamlit" -ForegroundColor Gray
pip install --no-cache-dir streamlit matplotlib plotly -q

Write-Host "CONCLUIDO!" -ForegroundColor Green
Write-Host "Abra o VS Code e selecione o interpretador .venv\Scripts\python.exe" -ForegroundColor Yellow
pause