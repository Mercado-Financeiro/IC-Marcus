#!/usr/bin/env python3
"""
Script para detectar segredos e informações sensíveis no código.
Alternativa ao gitleaks focada em projetos Python.
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import json


# Padrões de segredos conhecidos
SECRET_PATTERNS = {
    'aws_access_key': r'AKIA[0-9A-Z]{16}',
    'aws_secret_key': r'[0-9a-zA-Z/+]{40}',
    'github_token': r'ghp_[0-9a-zA-Z]{36}',
    'generic_api_key': r'[aA][pP][iI]_?[kK][eE][yY].*[\'\"]{0,1}[0-9a-zA-Z]{32,45}[\'\"]{0,1}',
    'generic_secret': r'[sS][eE][cC][rR][eE][tT].*[\'\"]{0,1}[0-9a-zA-Z]{32,45}[\'\"]{0,1}',
    'password_in_url': r'[a-zA-Z]{3,10}://[^/\s:@]{3,20}:[^/\s:@]{3,20}@.{1,100}[\"\'\\s]',
    'private_key': r'-----BEGIN.*PRIVATE KEY-----',
    'certificate': r'-----BEGIN CERTIFICATE-----',
    'jwt_token': r'eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*',
    'slack_webhook': r'https://hooks\.slack\.com/services/T[a-zA-Z0-9_]{8}/B[a-zA-Z0-9_]{8}/[a-zA-Z0-9_]{24}',
    'discord_webhook': r'https://discord(app)?\.com/api/webhooks/\d+/[A-Za-z0-9\-_]{68}',
    'stripe_key': r'sk_live_[0-9a-zA-Z]{24}',
    'mailgun_api_key': r'key-[0-9a-zA-Z]{32}',
    'twilio_sid': r'AC[a-zA-Z0-9_\-]{32}',
    'binance_api_key': r'[a-zA-Z0-9]{64}',  # Binance API keys
    'binance_secret_key': r'[a-zA-Z0-9]{64}'  # Binance secret keys
}

# Padrões de falso positivo para ignorar
FALSE_POSITIVES = [
    r'example\.com',
    r'localhost',
    r'127\.0\.0\.1',
    r'0\.0\.0\.0',
    r'test.*key',
    r'dummy.*key',
    r'fake.*key',
    r'mock.*key',
    r'placeholder',
    r'YOUR.*KEY',
    r'REPLACE.*WITH',
    r'<.*>',
    r'\{.*\}',
    r'\$\{.*\}',
    r'None',
    r'null',
    r'undefined'
]

# Extensões de arquivos para verificar
EXTENSIONS = {'.py', '.yaml', '.yml', '.json', '.txt', '.md', '.sh', '.env', '.conf', '.cfg'}

# Diretórios para ignorar
IGNORE_DIRS = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'venv', '.venv', 'artifacts'}


def is_false_positive(match: str) -> bool:
    """Verifica se o match é um falso positivo."""
    for pattern in FALSE_POSITIVES:
        if re.search(pattern, match, re.IGNORECASE):
            return True
    return False


def scan_file(file_path: Path) -> List[Dict]:
    """Escaneia um arquivo em busca de segredos."""
    findings = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        for line_num, line in enumerate(content.split('\n'), 1):
            for secret_type, pattern in SECRET_PATTERNS.items():
                matches = re.finditer(pattern, line)
                
                for match in matches:
                    secret = match.group()
                    
                    # Ignorar falsos positivos
                    if is_false_positive(secret):
                        continue
                    
                    # Ignorar se parece com variável de exemplo
                    if any(word in secret.lower() for word in ['example', 'test', 'dummy', 'fake', 'placeholder']):
                        continue
                    
                    findings.append({
                        'file': str(file_path),
                        'line': line_num,
                        'type': secret_type,
                        'secret': secret[:20] + '...' if len(secret) > 20 else secret,
                        'full_line': line.strip()[:100] + '...' if len(line.strip()) > 100 else line.strip()
                    })
                    
    except Exception as e:
        print(f"⚠️ Erro ao escanear {file_path}: {e}")
    
    return findings


def scan_directory(directory: Path, extensions: set = None) -> List[Dict]:
    """Escaneia um diretório recursivamente."""
    if extensions is None:
        extensions = EXTENSIONS
    
    all_findings = []
    
    for root, dirs, files in os.walk(directory):
        # Remover diretórios ignorados
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            file_path = Path(root) / file
            
            # Verificar extensão
            if file_path.suffix not in extensions:
                continue
            
            # Ignorar arquivos muito grandes (>1MB)
            try:
                if file_path.stat().st_size > 1024 * 1024:
                    continue
            except:
                continue
            
            findings = scan_file(file_path)
            all_findings.extend(findings)
    
    return all_findings


def generate_report(findings: List[Dict], output_file: str = None) -> Dict:
    """Gera relatório dos achados."""
    report = {
        'total_files_scanned': len(set(f['file'] for f in findings)),
        'total_secrets_found': len(findings),
        'secrets_by_type': {},
        'secrets_by_file': {},
        'findings': findings
    }
    
    # Agrupar por tipo
    for finding in findings:
        secret_type = finding['type']
        if secret_type not in report['secrets_by_type']:
            report['secrets_by_type'][secret_type] = 0
        report['secrets_by_type'][secret_type] += 1
    
    # Agrupar por arquivo
    for finding in findings:
        file = finding['file']
        if file not in report['secrets_by_file']:
            report['secrets_by_file'][file] = 0
        report['secrets_by_file'][file] += 1
    
    # Salvar relatório se especificado
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Relatório salvo: {output_file}")
    
    return report


def print_summary(report: Dict):
    """Imprime resumo dos achados."""
    print(f"\n{'='*60}")
    print("🔍 RESUMO DA DETECÇÃO DE SEGREDOS")
    print(f"{'='*60}")
    
    print(f"📁 Arquivos escaneados: {report['total_files_scanned']}")
    print(f"🚨 Segredos encontrados: {report['total_secrets_found']}")
    
    if report['total_secrets_found'] == 0:
        print("✅ Nenhum segredo encontrado!")
        return
    
    print(f"\n📊 Por tipo:")
    for secret_type, count in sorted(report['secrets_by_type'].items()):
        print(f"  {secret_type}: {count}")
    
    print(f"\n📂 Por arquivo:")
    for file, count in sorted(report['secrets_by_file'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {Path(file).name}: {count}")
    
    print(f"\n🔍 Detalhes:")
    for finding in report['findings'][:20]:  # Mostrar apenas os primeiros 20
        print(f"  {finding['file']}:{finding['line']} - {finding['type']}")
        print(f"    {finding['full_line']}")


def main():
    parser = argparse.ArgumentParser(description="Detector de segredos para projetos Python")
    parser.add_argument("path", default=".", nargs="?", help="Caminho para escanear (padrão: .)")
    parser.add_argument("--output", "-o", help="Arquivo de saída para relatório JSON")
    parser.add_argument("--extensions", "-e", nargs="+", 
                       default=['.py', '.yaml', '.yml', '.json', '.env'],
                       help="Extensões de arquivo para verificar")
    parser.add_argument("--quiet", "-q", action="store_true", help="Modo silencioso")
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🔐 DETECTOR DE SEGREDOS")
        print(f"Escaneando: {args.path}")
        print(f"Extensões: {', '.join(args.extensions)}")
    
    # Escanear
    scan_path = Path(args.path)
    extensions = set(args.extensions)
    
    if scan_path.is_file():
        findings = scan_file(scan_path)
    else:
        findings = scan_directory(scan_path, extensions)
    
    # Gerar relatório
    report = generate_report(findings, args.output)
    
    # Mostrar resumo
    if not args.quiet:
        print_summary(report)
    
    # Exit code
    exit_code = 1 if report['total_secrets_found'] > 0 else 0
    return exit_code


if __name__ == "__main__":
    exit(main())