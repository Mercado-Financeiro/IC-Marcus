"""
Testes de blindagem para proteger o sistema contra problemas comuns.

Este módulo contém testes críticos de segurança que validam:

1. **Data Leakage Protection** (test_data_leakage_protection.py):
   - Splits temporais sem vazamento de dados futuros
   - Calibração e threshold optimization apenas em validation
   - Feature engineering sem lookahead bias
   - Pipeline completa temporalmente consistente

2. **Model Robustness** (test_model_robustness.py):
   - Tratamento de valores ausentes, infinitos e extremos
   - Datasets pequenos e desbalanceados
   - Features constantes e correlacionadas
   - Estabilidade numérica e reprodutibilidade

3. **Financial Constraints** (test_financial_constraints.py):
   - Custos de transação realistas
   - Execução t+1 sem lookahead bias
   - Restrições de leverage e tamanho mínimo
   - Funding costs e spread bid-ask
   - Ausência de arbitragem óbvia

## Como usar:

```bash
# Executar todos os testes de blindagem
pytest tests/blindagem/ -v

# Executar apenas testes de data leakage
pytest tests/blindagem/test_data_leakage_protection.py -v

# Executar com coverage
pytest tests/blindagem/ --cov=src --cov-report=html
```

## Gates de Segurança:

Estes testes implementam "gates" que devem SEMPRE passar:

- ✅ **Gate Temporal**: Nenhum dado futuro usado em treino/validação
- ✅ **Gate Robustez**: Modelo lida com inputs adversos  
- ✅ **Gate Financeiro**: Custos realistas e restrições respeitadas
- ✅ **Gate Reprodutibilidade**: Resultados consistentes com mesma seed
- ✅ **Gate Performance**: Nenhum "free lunch" ou arbitragem óbvia

Se qualquer teste falhar, o modelo NÃO deve ir para produção.
"""