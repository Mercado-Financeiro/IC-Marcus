#!/bin/bash

# Script para ativar automaticamente a venv do Projeto IC
# Adicione este script ao seu .bashrc ou .zshrc

# Função para ativar venv automaticamente
auto_activate_venv() {
    # Verifica se estamos no diretório do Projeto_IC
    if [[ "$PWD" == */Projeto_IC* ]] || [[ "$PWD" == */projeto_ic* ]]; then
        # Procura pela venv no diretório atual ou em diretórios pais
        CURRENT_DIR="$PWD"
        while [[ "$CURRENT_DIR" != "/" ]]; do
            if [[ -d "$CURRENT_DIR/.venv" ]]; then
                if [[ -z "$VIRTUAL_ENV" ]] || [[ "$VIRTUAL_ENV" != "$CURRENT_DIR/.venv" ]]; then
                    source "$CURRENT_DIR/.venv/bin/activate"
                    echo "✅ Virtual environment ativado: $CURRENT_DIR/.venv"
                    echo "📊 Projeto IC - Predição de Criptomoedas"
                    echo "🐍 Python: $(python --version)"
                fi
                return
            fi
            CURRENT_DIR=$(dirname "$CURRENT_DIR")
        done
    fi
}

# Desativa venv quando sair do diretório do projeto
auto_deactivate_venv() {
    if [[ -n "$VIRTUAL_ENV" ]] && [[ "$PWD" != */Projeto_IC* ]] && [[ "$PWD" != */projeto_ic* ]]; then
        deactivate
        echo "❌ Virtual environment desativado"
    fi
}

# Hook para executar ao mudar de diretório
cd() {
    builtin cd "$@"
    auto_deactivate_venv
    auto_activate_venv
}

# Ativa ao iniciar o terminal se já estiver no diretório do projeto
auto_activate_venv
