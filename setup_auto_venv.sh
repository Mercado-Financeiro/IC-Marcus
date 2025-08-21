#!/bin/bash

echo "🔧 Configurando ativação automática da venv..."

# Detecta o shell do usuário
SHELL_CONFIG=""
if [ -n "$ZSH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
    echo "📝 Detectado: ZSH"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
    echo "📝 Detectado: Bash"
else
    echo "❌ Shell não suportado. Configure manualmente."
    exit 1
fi

# Adiciona o script ao arquivo de configuração do shell
MARKER="# Projeto_IC venv auto-activation"
PROJECT_DIR="$(pwd)"

# Remove configuração antiga se existir
sed -i "/$MARKER/,/# End Projeto_IC venv/d" "$SHELL_CONFIG" 2>/dev/null

# Adiciona nova configuração
cat >> "$SHELL_CONFIG" << EOF

$MARKER
# Ativa automaticamente a venv quando entrar no diretório do Projeto_IC
projeto_ic_venv() {
    if [[ "\$PWD" == "$PROJECT_DIR"* ]]; then
        if [[ -z "\$VIRTUAL_ENV" ]] && [[ -d "$PROJECT_DIR/.venv" ]]; then
            source "$PROJECT_DIR/.venv/bin/activate"
            echo "✅ Virtual environment ativado: Projeto IC"
            echo "📊 Predição de Criptomoedas - LSTM & XGBoost"
            echo "🐍 Python: \$(python --version)"
            echo "📁 Diretório: \$PWD"
        fi
    elif [[ -n "\$VIRTUAL_ENV" ]] && [[ "\$VIRTUAL_ENV" == "$PROJECT_DIR/.venv" ]]; then
        deactivate
        echo "❌ Virtual environment desativado (saiu do Projeto_IC)"
    fi
}

# Hook para cd
cd() {
    builtin cd "\$@"
    projeto_ic_venv
}

# Hook para pushd/popd
pushd() {
    builtin pushd "\$@"
    projeto_ic_venv
}

popd() {
    builtin popd "\$@"
    projeto_ic_venv
}

# Ativa ao iniciar o terminal se já estiver no diretório
projeto_ic_venv
# End Projeto_IC venv
EOF

echo "✅ Configuração adicionada ao $SHELL_CONFIG"
echo ""
echo "📌 Para ativar as mudanças, execute:"
echo "   source $SHELL_CONFIG"
echo ""
echo "Ou simplesmente abra um novo terminal!"
echo ""
echo "🎯 Funcionamento:"
echo "   - Ao entrar em $PROJECT_DIR → venv ativa automaticamente"
echo "   - Ao sair do diretório → venv desativa automaticamente"
