# Script para instalação em lotes com controle de memória
# Útil quando o script principal falhar em algum pacote específico

param(
    [string]$Batch = "all"
)

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "        INSTALAÇÃO EM LOTES CONTROLADA           " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Verificar se venv existe
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "❌ Ambiente virtual não encontrado!" -ForegroundColor Red
    Write-Host "   Execute primeiro: .\setup_venv_optimized.ps1" -ForegroundColor Yellow
    exit 1
}

# Ativar venv
& ".\.venv\Scripts\Activate.ps1"

# Definir lotes de instalação
$batches = @{
    "core" = @{
        packages = "numpy pandas"
        description = "Bibliotecas numéricas essenciais"
    }
    "ml" = @{
        packages = "scikit-learn xgboost"
        description = "Machine Learning"
    }
    "torch" = @{
        packages = "torch --index-url https://download.pytorch.org/whl/cpu"
        description = "PyTorch (CPU)"
    }
    "opt" = @{
        packages = "optuna mlflow"
        description = "Otimização e MLOps"
    }
    "data" = @{
        packages = "pandera pydantic-settings structlog pyyaml python-dotenv"
        description = "Validação e utilidades"
    }
    "dev" = @{
        packages = "pytest ruff black mypy bandit pre-commit"
        description = "Ferramentas de desenvolvimento"
    }
    "jupyter" = @{
        packages = "jupyter ipykernel jupytext nbconvert"
        description = "Jupyter ecosystem"
    }
    "viz" = @{
        packages = "matplotlib plotly seaborn"
        description = "Visualização"
    }
    "dash" = @{
        packages = "streamlit"
        description = "Dashboard"
    }
    "extra" = @{
        packages = "hypothesis safety pip-audit gitleaks"
        description = "Ferramentas extras de segurança e teste"
    }
}

function Install-Batch {
    param($name, $info)
    
    Write-Host "`n📦 Instalando: $($info.description)" -ForegroundColor Cyan
    Write-Host "   Pacotes: $($info.packages)" -ForegroundColor Gray
    
    # Limpar cache antes
    pip cache purge 2>$null | Out-Null
    
    # Instalar com timeout e sem cache
    $cmd = "pip install --no-cache-dir --timeout 60 $($info.packages)"
    
    try {
        $output = Invoke-Expression $cmd 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✓ Sucesso!" -ForegroundColor Green
            return $true
        } else {
            Write-Host "   ✗ Falhou!" -ForegroundColor Red
            Write-Host "   Erro: $output" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "   ✗ Erro na instalação!" -ForegroundColor Red
        Write-Host "   $_" -ForegroundColor Red
        return $false
    }
    
    # Forçar coleta de lixo
    [System.GC]::Collect()
    Start-Sleep -Seconds 2
}

# Processar instalação
if ($Batch -eq "all") {
    Write-Host "`nInstalando TODOS os lotes..." -ForegroundColor Yellow
    $failed = @()
    
    foreach ($batchName in $batches.Keys) {
        $success = Install-Batch -name $batchName -info $batches[$batchName]
        if (-not $success) {
            $failed += $batchName
        }
    }
    
    if ($failed.Count -gt 0) {
        Write-Host "`n⚠️  Lotes que falharam:" -ForegroundColor Yellow
        foreach ($f in $failed) {
            Write-Host "   - $f" -ForegroundColor Red
        }
        Write-Host "`nPara reinstalar um lote específico:" -ForegroundColor Yellow
        Write-Host ".\install_batch.ps1 -Batch <nome_do_lote>" -ForegroundColor White
    } else {
        Write-Host "`n✅ Todos os lotes instalados com sucesso!" -ForegroundColor Green
    }
    
} elseif ($batches.ContainsKey($Batch)) {
    # Instalar lote específico
    $success = Install-Batch -name $Batch -info $batches[$Batch]
    if (-not $success) {
        Write-Host "`n❌ Falha na instalação do lote '$Batch'" -ForegroundColor Red
        exit 1
    }
} else {
    # Mostrar lotes disponíveis
    Write-Host "`n❌ Lote '$Batch' não encontrado!" -ForegroundColor Red
    Write-Host "`nLotes disponíveis:" -ForegroundColor Yellow
    foreach ($key in $batches.Keys | Sort-Object) {
        Write-Host "   $key - $($batches[$key].description)" -ForegroundColor White
    }
    Write-Host "`nUso:" -ForegroundColor Cyan
    Write-Host "   .\install_batch.ps1 -Batch all      # Instalar todos" -ForegroundColor White
    Write-Host "   .\install_batch.ps1 -Batch core     # Instalar apenas core" -ForegroundColor White
    Write-Host "   .\install_batch.ps1 -Batch torch    # Instalar apenas PyTorch" -ForegroundColor White
}

Write-Host "`nPressione qualquer tecla para sair..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")