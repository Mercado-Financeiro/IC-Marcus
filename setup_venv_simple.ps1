# Script Simplificado para Setup do Ambiente Virtual
Write-Host "==================================================`n" -ForegroundColor Cyan
Write-Host "        SETUP RÁPIDO DO AMBIENTE PYTHON`n" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Fechar VS Code se estiver aberto
$vscode = Get-Process "Code" -ErrorAction SilentlyContinue
if ($vscode) {
    Write-Host "`n⚠️  VS Code detectado. Fechando para economizar RAM..." -ForegroundColor Yellow
    Stop-Process -Name "Code" -Force
    Start-Sleep -Seconds 3
}

# Limpar cache pip
Write-Host "`nLimpando cache do pip..." -ForegroundColor Yellow
pip cache purge 2>$null | Out-Null

# Remover venv antigo
if (Test-Path ".venv") {
    Write-Host "Removendo ambiente antigo..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".venv"
    Start-Sleep -Seconds 2
}

# Criar novo venv
Write-Host "Criando novo ambiente virtual..." -ForegroundColor Yellow
python -m venv .venv

# Ativar e atualizar pip
Write-Host "Ativando ambiente..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip --no-cache-dir -q

Write-Host "`nInstalando pacotes essenciais..." -ForegroundColor Cyan

# Instalar pacotes básicos primeiro
Write-Host "`n[1/9] NumPy e Pandas..." -ForegroundColor Gray
pip install --no-cache-dir numpy pandas -q

Write-Host "[2/9] Scikit-learn..." -ForegroundColor Gray
pip install --no-cache-dir scikit-learn -q

Write-Host "[3/9] XGBoost..." -ForegroundColor Gray
pip install --no-cache-dir xgboost -q

Write-Host "[4/9] PyTorch CPU (isso demora um pouco)..." -ForegroundColor Gray
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu -q

Write-Host "[5/9] Optuna e MLflow..." -ForegroundColor Gray
pip install --no-cache-dir optuna mlflow -q

Write-Host "[6/9] Validação de dados..." -ForegroundColor Gray
pip install --no-cache-dir pandera pydantic-settings -q

Write-Host "[7/9] Ferramentas de desenvolvimento..." -ForegroundColor Gray
pip install --no-cache-dir pytest ruff black mypy -q

Write-Host "[8/9] Jupyter..." -ForegroundColor Gray
pip install --no-cache-dir jupyter ipykernel -q

Write-Host "[9/9] Streamlit e visualização..." -ForegroundColor Gray
pip install --no-cache-dir streamlit matplotlib plotly -q

Write-Host "`n✅ Ambiente configurado com sucesso!" -ForegroundColor Green
Write-Host "`nPacotes instalados:" -ForegroundColor Yellow
pip list | Select-String "numpy|pandas|torch|xgboost|streamlit"

Write-Host "`n📝 Próximos passos:" -ForegroundColor Cyan
Write-Host "1. Abra o VS Code" -ForegroundColor White
Write-Host "2. Use Ctrl+Shift+P > Python: Select Interpreter" -ForegroundColor White
Write-Host "3. Selecione .\.venv\Scripts\python.exe" -ForegroundColor White

Write-Host "`nPressione Enter para sair..." -ForegroundColor Gray
Read-Host