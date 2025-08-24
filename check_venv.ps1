# Script de verificação de saúde do ambiente virtual
# Executa diagnósticos e corrige problemas comuns

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "     VERIFICAÇÃO DE SAÚDE DO AMBIENTE PYTHON     " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

$issues = @()
$warnings = @()

# 1. Verificar se .venv existe
Write-Host "[1/8] Verificando existência do ambiente virtual..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "  ✓ Diretório .venv encontrado" -ForegroundColor Green
    
    # Verificar estrutura Windows vs Linux
    $isWindows = Test-Path ".venv\Scripts\python.exe"
    $isLinux = Test-Path ".venv\bin\python"
    
    if ($isWindows) {
        Write-Host "  ✓ Estrutura Windows detectada" -ForegroundColor Green
    } elseif ($isLinux) {
        Write-Host "  ⚠️  Estrutura Linux/WSL detectada" -ForegroundColor Yellow
        $warnings += "Ambiente criado no Linux/WSL - pode haver incompatibilidades"
    } else {
        Write-Host "  ✗ Estrutura inválida do venv" -ForegroundColor Red
        $issues += "Estrutura do ambiente virtual corrompida"
    }
} else {
    Write-Host "  ✗ Diretório .venv não encontrado" -ForegroundColor Red
    $issues += "Ambiente virtual não existe - execute setup_venv_optimized.ps1"
}

# 2. Verificar Python no venv
Write-Host "`n[2/8] Verificando interpretador Python..." -ForegroundColor Yellow
if (Test-Path ".venv\Scripts\python.exe") {
    $venvPython = & ".venv\Scripts\python.exe" --version 2>&1
    Write-Host "  ✓ Python encontrado: $venvPython" -ForegroundColor Green
} else {
    Write-Host "  ✗ Python não encontrado no venv" -ForegroundColor Red
    $issues += "Interpretador Python ausente"
}

# 3. Verificar pip
Write-Host "`n[3/8] Verificando pip..." -ForegroundColor Yellow
if (Test-Path ".venv\Scripts\pip.exe") {
    $pipVersion = & ".venv\Scripts\pip.exe" --version 2>&1
    Write-Host "  ✓ Pip encontrado" -ForegroundColor Green
    
    # Verificar se pip está atualizado
    $outdated = & ".venv\Scripts\pip.exe" list --outdated 2>&1 | Select-String "pip"
    if ($outdated) {
        Write-Host "  ⚠️  Pip desatualizado" -ForegroundColor Yellow
        $warnings += "Pip desatualizado - execute: python -m pip install --upgrade pip"
    }
} else {
    Write-Host "  ✗ Pip não encontrado" -ForegroundColor Red
    $issues += "Pip ausente no ambiente"
}

# 4. Verificar pacotes essenciais
Write-Host "`n[4/8] Verificando pacotes essenciais..." -ForegroundColor Yellow
$essentialPackages = @("numpy", "pandas", "scikit-learn", "torch", "xgboost", "mlflow", "streamlit")
$missingPackages = @()

if (Test-Path ".venv\Scripts\pip.exe") {
    $installedPackages = & ".venv\Scripts\pip.exe" list 2>&1
    
    foreach ($package in $essentialPackages) {
        if ($installedPackages -match $package) {
            Write-Host "  ✓ $package instalado" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $package não encontrado" -ForegroundColor Red
            $missingPackages += $package
        }
    }
    
    if ($missingPackages.Count -gt 0) {
        $warnings += "Pacotes essenciais faltando: $($missingPackages -join ', ')"
    }
}

# 5. Verificar configuração VS Code
Write-Host "`n[5/8] Verificando configuração do VS Code..." -ForegroundColor Yellow
if (Test-Path ".vscode\settings.json") {
    $settings = Get-Content ".vscode\settings.json" -Raw
    
    if ($settings -match '\\.venv\\Scripts\\python\.exe') {
        Write-Host "  ✓ Caminho do Python configurado corretamente" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Caminho do Python pode estar incorreto" -ForegroundColor Yellow
        $warnings += "Verificar python.defaultInterpreterPath em settings.json"
    }
    
    if ($settings -match '"python\.terminal\.activateEnvironment":\s*true') {
        Write-Host "  ✓ Ativação automática habilitada" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Ativação automática pode estar desabilitada" -ForegroundColor Yellow
        $warnings += "Ativação automática do venv pode não funcionar"
    }
} else {
    Write-Host "  ⚠️  Arquivo .vscode\settings.json não encontrado" -ForegroundColor Yellow
    $warnings += "Configurações do VS Code ausentes"
}

# 6. Verificar espaço em disco
Write-Host "`n[6/8] Verificando espaço em disco..." -ForegroundColor Yellow
$drive = (Get-Location).Drive
$driveInfo = Get-PSDrive $drive
$freeGB = [math]::Round($driveInfo.Free / 1GB, 2)

if ($freeGB -lt 2) {
    Write-Host "  ✗ Espaço insuficiente: ${freeGB}GB livres" -ForegroundColor Red
    $issues += "Espaço em disco muito baixo (menos de 2GB)"
} elseif ($freeGB -lt 5) {
    Write-Host "  ⚠️  Espaço limitado: ${freeGB}GB livres" -ForegroundColor Yellow
    $warnings += "Espaço em disco baixo (menos de 5GB)"
} else {
    Write-Host "  ✓ Espaço adequado: ${freeGB}GB livres" -ForegroundColor Green
}

# 7. Verificar memória disponível
Write-Host "`n[7/8] Verificando memória RAM..." -ForegroundColor Yellow
$os = Get-CimInstance Win32_OperatingSystem
$totalMemGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
$freeMemGB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
$usedPercent = [math]::Round((($totalMemGB - $freeMemGB) / $totalMemGB) * 100, 1)

Write-Host "  Total: ${totalMemGB}GB | Livre: ${freeMemGB}GB | Uso: ${usedPercent}%" -ForegroundColor Cyan

if ($freeMemGB -lt 2) {
    Write-Host "  ⚠️  Memória disponível baixa" -ForegroundColor Yellow
    $warnings += "Pouca memória RAM disponível - feche aplicações desnecessárias"
} else {
    Write-Host "  ✓ Memória adequada" -ForegroundColor Green
}

# 8. Verificar processos VS Code
Write-Host "`n[8/8] Verificando processos do VS Code..." -ForegroundColor Yellow
$vscodeProcesses = Get-Process "Code" -ErrorAction SilentlyContinue
if ($vscodeProcesses) {
    $totalMemMB = ($vscodeProcesses | Measure-Object WorkingSet -Sum).Sum / 1MB
    Write-Host "  VS Code rodando: $($vscodeProcesses.Count) processos usando $([math]::Round($totalMemMB, 0))MB" -ForegroundColor Cyan
    
    if ($totalMemMB -gt 2000) {
        Write-Host "  ⚠️  VS Code usando muita memória" -ForegroundColor Yellow
        $warnings += "VS Code consumindo mais de 2GB - considere reiniciar"
    }
} else {
    Write-Host "  VS Code não está rodando" -ForegroundColor Gray
}

# RELATÓRIO FINAL
Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "                   RELATÓRIO                      " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

if ($issues.Count -eq 0 -and $warnings.Count -eq 0) {
    Write-Host "`n✅ AMBIENTE SAUDÁVEL!" -ForegroundColor Green
    Write-Host "   Tudo está funcionando corretamente." -ForegroundColor Green
} else {
    if ($issues.Count -gt 0) {
        Write-Host "`n❌ PROBLEMAS CRÍTICOS ENCONTRADOS:" -ForegroundColor Red
        foreach ($issue in $issues) {
            Write-Host "   • $issue" -ForegroundColor Red
        }
    }
    
    if ($warnings.Count -gt 0) {
        Write-Host "`n⚠️  AVISOS:" -ForegroundColor Yellow
        foreach ($warning in $warnings) {
            Write-Host "   • $warning" -ForegroundColor Yellow
        }
    }
    
    Write-Host "`n📝 SOLUÇÕES RECOMENDADAS:" -ForegroundColor Cyan
    if ($issues -match "não existe") {
        Write-Host "   1. Execute: .\setup_venv_optimized.ps1" -ForegroundColor White
    }
    if ($missingPackages.Count -gt 0) {
        Write-Host "   2. Execute: .\install_batch.ps1 -Batch all" -ForegroundColor White
    }
    if ($warnings -match "memória") {
        Write-Host "   3. Feche aplicações desnecessárias" -ForegroundColor White
        Write-Host "   4. Reinicie o VS Code" -ForegroundColor White
    }
}

# Função de reparo rápido
Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "Deseja tentar reparo automático? (s/n): " -NoNewline -ForegroundColor Yellow
$repair = Read-Host

if ($repair -eq 's' -or $repair -eq 'S') {
    Write-Host "`nIniciando reparo automático..." -ForegroundColor Cyan
    
    # Limpar cache pip
    Write-Host "  Limpando cache do pip..." -ForegroundColor Gray
    pip cache purge 2>$null
    
    # Atualizar pip se o venv existe
    if (Test-Path ".venv\Scripts\pip.exe") {
        Write-Host "  Atualizando pip..." -ForegroundColor Gray
        & ".venv\Scripts\python.exe" -m pip install --upgrade pip --no-cache-dir 2>&1 | Out-Null
    }
    
    # Instalar pacotes faltantes
    if ($missingPackages.Count -gt 0 -and (Test-Path ".venv\Scripts\pip.exe")) {
        Write-Host "  Instalando pacotes faltantes..." -ForegroundColor Gray
        foreach ($pkg in $missingPackages) {
            Write-Host "    Instalando $pkg..." -ForegroundColor Gray
            & ".venv\Scripts\pip.exe" install --no-cache-dir $pkg 2>&1 | Out-Null
        }
    }
    
    Write-Host "`n✅ Reparo concluído!" -ForegroundColor Green
    Write-Host "   Execute novamente para verificar o status." -ForegroundColor Yellow
}

Write-Host "`nPressione qualquer tecla para sair..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")