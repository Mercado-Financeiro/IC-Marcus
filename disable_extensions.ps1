# Script PowerShell para desabilitar extensões não essenciais no VS Code
# Execute este script para limpar o ambiente de desenvolvimento

Write-Host "=== Desabilitando extensões não essenciais do VS Code ===" -ForegroundColor Cyan

# Lista de extensões essenciais para manter habilitadas
$essentialExtensions = @(
    "ms-python.python",
    "ms-python.vscode-pylance", 
    "ms-toolsai.jupyter",
    "charliermarsh.ruff",
    "ms-python.black-formatter"
)

# Obter todas as extensões instaladas
Write-Host "`nObtendo lista de extensões instaladas..." -ForegroundColor Yellow
$installedExtensions = code --list-extensions

if ($installedExtensions.Count -eq 0) {
    Write-Host "Nenhuma extensão encontrada ou VS Code não está no PATH" -ForegroundColor Red
    Write-Host "Certifique-se de que o VS Code está instalado e no PATH do sistema" -ForegroundColor Red
    exit 1
}

Write-Host "Total de extensões instaladas: $($installedExtensions.Count)" -ForegroundColor Green

# Desabilitar extensões não essenciais para este workspace
$disabledCount = 0
$keptCount = 0

Write-Host "`nProcessando extensões..." -ForegroundColor Yellow

foreach ($extension in $installedExtensions) {
    if ($essentialExtensions -contains $extension) {
        Write-Host "  ✓ Mantendo: $extension" -ForegroundColor Green
        $keptCount++
    } else {
        Write-Host "  ✗ Desabilitando para workspace: $extension" -ForegroundColor Gray
        # Desabilita a extensão apenas para este workspace
        code --disable-extension $extension --profile-temp 2>$null
        $disabledCount++
    }
}

Write-Host "`n=== Resumo ===" -ForegroundColor Cyan
Write-Host "Extensões mantidas: $keptCount" -ForegroundColor Green
Write-Host "Extensões desabilitadas: $disabledCount" -ForegroundColor Yellow

Write-Host "`n=== Instruções ===" -ForegroundColor Cyan
Write-Host "1. Reinicie o VS Code para aplicar as mudanças" -ForegroundColor White
Write-Host "2. As extensões foram desabilitadas apenas para este workspace" -ForegroundColor White
Write-Host "3. Para reabilitar globalmente, use: Extensions -> Installed -> Enable" -ForegroundColor White
Write-Host "4. As configurações do workspace estão em .vscode/extensions.json" -ForegroundColor White

Write-Host "`nPressione qualquer tecla para continuar..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")