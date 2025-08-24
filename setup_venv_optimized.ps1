# Script Otimizado para Setup do Ambiente Virtual
# Minimiza uso de memória e evita travamentos

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   SETUP OTIMIZADO DO AMBIENTE VIRTUAL PYTHON    " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar se VS Code está rodando
$vscodeProcess = Get-Process "Code" -ErrorAction SilentlyContinue
if ($vscodeProcess) {
    Write-Host "⚠️  VS Code está aberto!" -ForegroundColor Yellow
    Write-Host "   É recomendado fechar o VS Code para economizar memória." -ForegroundColor Yellow
    $response = Read-Host "   Deseja continuar mesmo assim? (s/n)"
    if ($response -ne 's' -and $response -ne 'S') {
        Write-Host "Operação cancelada. Feche o VS Code e execute novamente." -ForegroundColor Red
        exit
    }
}

# Limpar cache do pip primeiro
Write-Host "`n[1/6] Limpando cache do pip..." -ForegroundColor Yellow
pip cache purge 2>$null

# Verificar Python instalado
Write-Host "`n[2/6] Verificando Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python encontrado: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python não encontrado no PATH!" -ForegroundColor Red
    Write-Host "  Instale Python 3.11+ primeiro: https://python.org" -ForegroundColor Red
    exit 1
}

# Remover venv antigo se existir
Write-Host "`n[3/6] Preparando ambiente..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "  Removendo ambiente virtual antigo..." -ForegroundColor Gray
    Remove-Item -Recurse -Force ".venv" -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2  # Dar tempo para liberar recursos
}

# Criar novo venv
Write-Host "`n[4/6] Criando novo ambiente virtual..." -ForegroundColor Yellow
python -m venv .venv --clear --upgrade-deps

# Verificar se foi criado corretamente
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "✗ Erro ao criar ambiente virtual!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Ambiente virtual criado com sucesso!" -ForegroundColor Green

# Ativar venv
Write-Host "`n[5/6] Ativando ambiente virtual..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

# Upgrade pip primeiro (importante!)
Write-Host "  Atualizando pip..." -ForegroundColor Gray
python -m pip install --upgrade pip --no-cache-dir --quiet

Write-Host "`n[6/6] Instalação de pacotes em lotes..." -ForegroundColor Yellow
Write-Host "  Esta etapa pode demorar alguns minutos." -ForegroundColor Gray
Write-Host "  Instalando em lotes pequenos para economizar memória..." -ForegroundColor Gray

# Função para instalar com retry
function Install-Package {
    param($packages, $description)
    
    Write-Host "`n  📦 $description" -ForegroundColor Cyan
    $retries = 2
    $success = $false
    
    for ($i = 1; $i -le $retries; $i++) {
        try {
            $cmd = "pip install --no-cache-dir $packages"
            Invoke-Expression $cmd 2>&1 | Out-Null
            $success = $true
            Write-Host "    ✓ Instalado com sucesso" -ForegroundColor Green
            break
        }
        catch {
            if ($i -eq $retries) {
                Write-Host "    ✗ Falha após $retries tentativas" -ForegroundColor Red
                return $false
            }
            Write-Host "    ⚠️  Tentativa $i falhou, tentando novamente..." -ForegroundColor Yellow
            Start-Sleep -Seconds 3
        }
    }
    
    # Pequena pausa entre lotes para liberar memória
    Start-Sleep -Seconds 2
    [System.GC]::Collect()
    
    return $success
}

# Instalar em lotes organizados
$allSuccess = $true

# Lote 1: Essenciais básicos
$allSuccess = $allSuccess -and (Install-Package "numpy pandas" "Bibliotecas numéricas essenciais")

# Lote 2: Machine Learning básico
$allSuccess = $allSuccess -and (Install-Package "scikit-learn xgboost" "Machine Learning básico")

# Lote 3: Deep Learning (CPU version para economizar memória)
Write-Host "`n  📦 PyTorch (versão CPU)" -ForegroundColor Cyan
Write-Host "    Instalando PyTorch CPU (menor que GPU)..." -ForegroundColor Gray
$torchCmd = "pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu"
Invoke-Expression $torchCmd 2>&1 | Out-Null
Write-Host "    ✓ PyTorch instalado" -ForegroundColor Green

# Lote 4: Otimização e tracking
$allSuccess = $allSuccess -and (Install-Package "optuna mlflow" "Otimização e tracking")

# Lote 5: Validação e qualidade
$allSuccess = $allSuccess -and (Install-Package "pandera pydantic-settings" "Validação de dados")

# Lote 6: Desenvolvimento
$allSuccess = $allSuccess -and (Install-Package "pytest ruff black mypy" "Ferramentas de desenvolvimento")

# Lote 7: Jupyter e visualização
$allSuccess = $allSuccess -and (Install-Package "jupyter ipykernel matplotlib plotly" "Jupyter e visualização")

# Lote 8: Dashboard
$allSuccess = $allSuccess -and (Install-Package "streamlit" "Dashboard Streamlit")

# Lote 9: Utilitários
$allSuccess = $allSuccess -and (Install-Package "structlog pyyaml python-dotenv" "Utilitários")

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "                    RESUMO                        " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

if ($allSuccess) {
    Write-Host "✓ Ambiente configurado com sucesso!" -ForegroundColor Green
    
    # Mostrar informações do ambiente
    Write-Host "`nInformações do ambiente:" -ForegroundColor Yellow
    & ".\.venv\Scripts\python.exe" --version
    
    Write-Host "`nPacotes principais instalados:" -ForegroundColor Yellow
    & ".\.venv\Scripts\pip.exe" list | Select-String "numpy|pandas|torch|xgboost|mlflow|streamlit"
    
    Write-Host "`n📝 Próximos passos:" -ForegroundColor Cyan
    Write-Host "1. Feche completamente o VS Code" -ForegroundColor White
    Write-Host "2. Abra o VS Code novamente" -ForegroundColor White
    Write-Host "3. VS Code deve detectar automaticamente o .venv" -ForegroundColor White
    Write-Host "4. Se não detectar, use Ctrl+Shift+P > 'Python: Select Interpreter'" -ForegroundColor White
    Write-Host "   e selecione '.\.venv\Scripts\python.exe'" -ForegroundColor White
    
} else {
    Write-Host "⚠️  Alguns pacotes falharam na instalação" -ForegroundColor Yellow
    Write-Host "Execute 'pip install -r requirements.txt' manualmente" -ForegroundColor Yellow
}

Write-Host "`nPressione qualquer tecla para sair..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")