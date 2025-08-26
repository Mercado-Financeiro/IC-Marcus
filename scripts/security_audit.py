#!/usr/bin/env python3
"""
Script de auditoria de segurança para o projeto ML Trading.
Executa pip-audit, safety e outras verificações de segurança.
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json


def run_command(cmd: str, description: str = ""):
    """Executa comando e retorna resultado."""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"Comando: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd.split(), 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        return result.returncode == 0, result.stdout, result.stderr
        
    except Exception as e:
        print(f"❌ Erro ao executar: {e}")
        return False, "", str(e)


def pip_audit_check():
    """Executa pip-audit para verificar vulnerabilidades."""
    success, stdout, stderr = run_command(
        "pip-audit --desc --format=json",
        "PIP-AUDIT: Verificando vulnerabilidades em dependências"
    )
    
    if not success:
        print("⚠️ pip-audit falhou, tentando formato padrão...")
        success, stdout, stderr = run_command(
            "pip-audit --desc",
            "PIP-AUDIT: Formato padrão"
        )
    
    return success, stdout, stderr


def safety_check():
    """Executa safety para verificar vulnerabilidades."""
    success, stdout, stderr = run_command(
        "safety check --json",
        "SAFETY: Verificando banco de dados de vulnerabilidades"
    )
    
    if not success:
        print("⚠️ Safety falhou, tentando formato padrão...")
        success, stdout, stderr = run_command(
            "safety check",
            "SAFETY: Formato padrão"
        )
    
    return success, stdout, stderr


def bandit_check():
    """Executa bandit para análise de código."""
    if not Path("src").exists():
        print("⚠️ Diretório src/ não encontrado, pulando bandit")
        return True, "", ""
    
    success, stdout, stderr = run_command(
        "bandit -r src/ -f json",
        "BANDIT: Análise de segurança do código fonte"
    )
    
    if not success:
        success, stdout, stderr = run_command(
            "bandit -r src/",
            "BANDIT: Formato padrão"
        )
    
    return success, stdout, stderr


def check_secrets():
    """Verifica se há segredos no código."""
    patterns = [
        "api_key",
        "secret",
        "password", 
        "token",
        "private_key"
    ]
    
    print(f"\n{'='*60}")
    print("🔐 SECRETS: Verificando segredos no código")
    print(f"{'='*60}")
    
    issues = []
    
    for pattern in patterns:
        success, stdout, stderr = run_command(
            f"grep -r -i '{pattern}' src/ --exclude-dir=__pycache__",
            f"Procurando por '{pattern}'"
        )
        
        if stdout.strip():
            issues.append(f"Possível segredo encontrado: {pattern}")
            print(f"⚠️ {pattern} encontrado:")
            print(stdout)
    
    if not issues:
        print("✅ Nenhum segredo óbvio encontrado")
        return True, "No secrets found", ""
    else:
        return False, "\n".join(issues), ""


def generate_report(results: dict):
    """Gera relatório de segurança."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_file = f"artifacts/reports/security_audit_{timestamp}.json"
    
    # Criar diretório se não existe
    Path("artifacts/reports").mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": timestamp,
        "summary": {
            "pip_audit_passed": results.get("pip_audit", [False])[0],
            "safety_passed": results.get("safety", [False])[0], 
            "bandit_passed": results.get("bandit", [False])[0],
            "secrets_passed": results.get("secrets", [False])[0]
        },
        "details": results
    }
    
    # Salvar relatório
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Relatório salvo: {report_file}")
    
    # Resumo
    print(f"\n{'='*60}")
    print("📋 RESUMO DA AUDITORIA DE SEGURANÇA")
    print(f"{'='*60}")
    
    total_checks = len(report["summary"])
    passed_checks = sum(report["summary"].values())
    
    print(f"✅ Verificações passadas: {passed_checks}/{total_checks}")
    
    for check, passed in report["summary"].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    if passed_checks == total_checks:
        print("\n🎉 TODAS as verificações de segurança PASSARAM!")
        return True
    else:
        print(f"\n⚠️ {total_checks - passed_checks} verificações FALHARAM - Revisar!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Auditoria de segurança do projeto ML Trading")
    parser.add_argument("--skip-pip-audit", action="store_true", help="Pular pip-audit")
    parser.add_argument("--skip-safety", action="store_true", help="Pular safety")
    parser.add_argument("--skip-bandit", action="store_true", help="Pular bandit")
    parser.add_argument("--skip-secrets", action="store_true", help="Pular verificação de segredos")
    
    args = parser.parse_args()
    
    print("🛡️ AUDITORIA DE SEGURANÇA - PROJETO ML TRADING")
    print(f"Data/Hora: {datetime.now()}")
    
    results = {}
    
    # Pip-audit
    if not args.skip_pip_audit:
        results["pip_audit"] = pip_audit_check()
    
    # Safety
    if not args.skip_safety:
        results["safety"] = safety_check()
    
    # Bandit
    if not args.skip_bandit:
        results["bandit"] = bandit_check()
    
    # Secrets check
    if not args.skip_secrets:
        results["secrets"] = check_secrets()
    
    # Gerar relatório
    all_passed = generate_report(results)
    
    # Exit code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()