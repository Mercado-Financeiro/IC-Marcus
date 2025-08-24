# Script para listar todas as extensões instaladas no VS Code
# e gerar um relatório detalhado

Write-Host "=== Analisando extensões do VS Code ===" -ForegroundColor Cyan

# Verificar se o VS Code está instalado
try {
    $vscodeVersion = code --version
    Write-Host "VS Code encontrado!" -ForegroundColor Green
} catch {
    Write-Host "VS Code não encontrado no PATH" -ForegroundColor Red
    exit 1
}

# Obter lista de extensões
Write-Host "`nObtendo lista de extensões instaladas..." -ForegroundColor Yellow
$extensions = code --list-extensions --show-versions

if ($extensions.Count -eq 0) {
    Write-Host "Nenhuma extensão instalada encontrada" -ForegroundColor Yellow
    exit 0
}

# Categorizar extensões
$categories = @{
    "Python" = @("python", "pylance", "jupyter", "black", "ruff", "mypy", "pylint", "flake8")
    "Git" = @("git", "gitlens", "githistory")
    "Docker" = @("docker", "container")
    "Web" = @("html", "css", "javascript", "typescript", "react", "vue", "angular")
    "Database" = @("sql", "mongodb", "postgresql", "mysql")
    "Cloud" = @("azure", "aws", "gcp", "kubernetes")
    "Themes" = @("theme", "icon", "color")
    "Productivity" = @("todo", "bookmark", "snippet", "spell")
}

$categorized = @{}
$uncategorized = @()

foreach ($ext in $extensions) {
    $found = $false
    $extLower = $ext.ToLower()
    
    foreach ($category in $categories.Keys) {
        foreach ($keyword in $categories[$category]) {
            if ($extLower -match $keyword) {
                if (-not $categorized.ContainsKey($category)) {
                    $categorized[$category] = @()
                }
                $categorized[$category] += $ext
                $found = $true
                break
            }
        }
        if ($found) { break }
    }
    
    if (-not $found) {
        $uncategorized += $ext
    }
}

# Gerar relatório
Write-Host "`n=== RELATÓRIO DE EXTENSÕES ===" -ForegroundColor Cyan
Write-Host "Total de extensões instaladas: $($extensions.Count)" -ForegroundColor White

foreach ($category in $categorized.Keys | Sort-Object) {
    Write-Host "`n[$category] ($($categorized[$category].Count) extensões)" -ForegroundColor Yellow
    foreach ($ext in $categorized[$category] | Sort-Object) {
        Write-Host "  - $ext" -ForegroundColor Gray
    }
}

if ($uncategorized.Count -gt 0) {
    Write-Host "`n[Outras] ($($uncategorized.Count) extensões)" -ForegroundColor Yellow
    foreach ($ext in $uncategorized | Sort-Object) {
        Write-Host "  - $ext" -ForegroundColor Gray
    }
}

# Extensões essenciais para o projeto
Write-Host "`n=== EXTENSÕES ESSENCIAIS PARA ESTE PROJETO ===" -ForegroundColor Green
$essential = @(
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",
    "charliermarsh.ruff",
    "ms-python.black-formatter"
)

foreach ($ext in $essential) {
    $installed = $extensions | Where-Object { $_ -like "$ext*" }
    if ($installed) {
        Write-Host "  ✓ $ext (instalada)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $ext (não instalada)" -ForegroundColor Red
    }
}

# Salvar relatório em arquivo
$reportPath = "vscode_extensions_report.txt"
Write-Host "`nSalvando relatório em $reportPath..." -ForegroundColor Yellow

$report = @"
RELATÓRIO DE EXTENSÕES DO VS CODE
Gerado em: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Total de extensões: $($extensions.Count)

TODAS AS EXTENSÕES INSTALADAS:
$($extensions -join "`n")

EXTENSÕES ESSENCIAIS PARA O PROJETO:
$($essential -join "`n")
"@

$report | Out-File -FilePath $reportPath -Encoding UTF8
Write-Host "Relatório salvo com sucesso!" -ForegroundColor Green

Write-Host "`nPressione qualquer tecla para continuar..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")