# Script para ativar o ambiente virtual automaticamente
$venvPath = "C:\Projetos\Projeto_IC\venv_windows\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    & $venvPath
    Write-Host "Virtual environment activated!" -ForegroundColor Green
} else {
    Write-Host "Virtual environment not found at: $venvPath" -ForegroundColor Red
}