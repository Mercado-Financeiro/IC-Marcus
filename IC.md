# PREDIÇÃO E OTIMIZAÇÃO NO MERCADO DE CRIPTOMOEDAS COM BASE EM MÉTODOS DE APRENDIZADO DE MÁQUINA E OTIMIZAÇÃO BAYESIANA

**Marcus Vinicius Santos da Silva**
*Orientadora: Maria José Pereira Dantas*
Pontifícia Universidade Católica de Goiás - PUC Goiás
Grupo de Pesquisa: Identificação, Modelagem e Inteligência Artificial para Otimização de Sistemas

---

## RESUMO

Este trabalho de Iniciação Científica propõe o desenvolvimento de um sistema inteligente de negociação automatizada para o mercado de criptomoedas, utilizando técnicas avançadas de Aprendizado de Máquina (Machine Learning - ML). O estudo foca na implementação e comparação de dois modelos principais: Long Short-Term Memory (LSTM) e eXtreme Gradient Boosting (XGBoost), otimizados através da técnica de Otimização Bayesiana. O objetivo é criar um modelo preditivo capaz de antecipar com precisão as oscilações de preço das principais criptomoedas (Bitcoin, Ethereum, BNB, Solana e XRP), possibilitando operações autônomas que maximizem o retorno sobre investimento. A pesquisa busca validar a eficiência desses métodos através de métricas rigorosas de desempenho, incluindo acurácia, precisão, recall, F1-score e ROI, além de integrar o sistema final às plataformas de negociação Binance e KuCoin para execução automática de operações.

**Palavras-chave:** Mercado Financeiro, Machine Learning, Criptomoedas, LSTM, XGBoost, Otimização Bayesiana

---

## 1. INTRODUÇÃO

### 1.1 Contextualização do Problema

O mercado financeiro representa um dos sistemas mais complexos e dinâmicos da economia moderna, caracterizado por sua natureza quase imprevisível devido às múltiplas interações entre mercados e participantes. Esta complexidade é amplificada por não linearidades, descontinuidades e componentes polinomiais de alta frequência, resultantes da interação com diversos fatores como eventos políticos, condições econômicas gerais e expectativas dos investidores (Hadavandi, Shavandi & Ghanbari, 2010).

A dificuldade na previsão de tendências do mercado financeiro decorre principalmente das perspectivas heterogêneas dos participantes e do desafio em identificar características generalizadas e informativas nos dados de preços e volumes (Wang et al., 2018). Apesar do desenvolvimento de técnicas analíticas tradicionais ao longo de décadas, estas ainda apresentam limitações significativas na captura da complexidade inerente aos mercados.

### 1.2 A Revolução do Aprendizado de Máquina em Finanças

A integração de métodos computacionais em finanças, iniciada na década de 1990, catalisou pesquisas significativas sobre a aplicação de Inteligência Artificial (IA) em investimentos no mercado de ações. A automatização do processo de investimento financeiro através de abordagens computacionais oferece vantagens substanciais, incluindo:

- **Mitigação da irracionalidade momentânea:** Redução do impacto da tomada de decisão emocional
- **Identificação de padrões complexos:** Capacidade de detectar e explorar padrões negligenciados por observadores humanos
- **Processamento de grandes volumes de dados:** Análise eficiente de múltiplas variáveis em tempo real

Conforme destacado por Khattak et al. (2023), essas abordagens computacionais revolucionaram a forma como os investimentos são gerenciados, proporcionando uma vantagem competitiva significativa no mercado.

### 1.3 O Mercado de Criptomoedas

As criptomoedas emergem como um meio de troca revolucionário baseado em rede, utilizando algoritmos criptográficos para garantir a segurança das transações. Fundamentadas na tecnologia Blockchain, elas herdam propriedades essenciais como:

- **Descentralização:** Ausência de controle por autoridade central
- **Transparência:** Registro público e verificável de todas as transações
- **Imutabilidade:** Impossibilidade de alteração retroativa dos registros

Ao contrário dos sistemas financeiros tradicionais, as transações de criptomoedas operam de forma independente, sem supervisão de instituições centralizadas (Mohil Maheshkumar Patel et al., 2020). Esta característica única apresenta tanto oportunidades quanto desafios para a aplicação de modelos preditivos.

---

## 2. REVISÃO DA LITERATURA

### 2.1 Estado da Arte em Predição de Criptomoedas

A aplicação de modelos de Machine Learning para prever tendências no mercado de criptomoedas, embora em fase inicial, tem demonstrado resultados promissores. Chen et al. (2020) conduziram um estudo comparativo abrangente entre diferentes modelos de ML na análise do preço do Bitcoin, examinando:

- **Support Vector Machine (SVM):** Acurácia de 54,9%
- **Long Short-Term Memory (LSTM):** Acurácia de 67,2%
- **Random Forest (RF):** Acurácia de 64,8%
- **XGBoost (XGB):** Acurácia de 65,4%

Os resultados demonstram que modelos de ML podem ser ferramentas eficazes para a previsão de preços de criptomoedas, com destaque para a superioridade do LSTM em capturar dependências temporais.

### 2.2 Deep Learning e Redes Neurais

O surgimento do Deep Learning como subcampo do Machine Learning despertou interesse crescente na comunidade financeira, impulsionado principalmente por sua capacidade superior em comparação aos modelos convencionais (Ozbayoglu, Gudelek e Sezer, 2020). As redes neurais profundas têm demonstrado particular eficácia em:

- Captura de padrões não-lineares complexos
- Processamento de séries temporais multivariadas
- Adaptação a mudanças de regime no mercado

---

## 3. OBJETIVOS

### 3.1 Objetivo Geral

Desenvolver um modelo baseado em aprendizado de máquina que possibilite operações autônomas no mercado de criptomoedas, maximizando o lucro através da identificação da melhor estratégia para cada situação de mercado.

### 3.2 Objetivos Específicos

1. **Validação de Modelos:** Testar e validar a eficiência dos métodos LSTM e XGBoost para a predição do mercado de criptomoedas

2. **Métricas de Desempenho:** Estabelecer métricas robustas para avaliar a precisão das previsões e a rentabilidade das operações, incluindo:
   - Indicadores de acurácia
   - Precisão (Precision)
   - Recall
   - F1-score
   - Retorno sobre o investimento (ROI)

3. **Otimização Avançada:** Investigar o comportamento dos modelos LSTM e XGBoost quando otimizados com o algoritmo de Otimização Bayesiana

4. **Integração com Plataformas:** Integrar a ferramenta com plataformas de negociação existentes (Binance e KuCoin) para facilitar a execução automática de operações baseadas em sinais preditivos

---

## 4. METODOLOGIA

### 4.1 Conjunto de Dados

A pesquisa utilizará dados históricos das criptomoedas com maior relevância no mercado atual (CoinMarketCap, abril de 2024):

| Criptomoeda | Símbolo | Critérios de Seleção |
|-------------|---------|---------------------|
| Bitcoin | BTC | Maior capitalização de mercado |
| Ethereum | ETH | Segunda maior capitalização, plataforma de smart contracts |
| BNB | BNB | Token da maior exchange mundial |
| Solana | SOL | Alta velocidade de transação e baixas taxas |
| XRP | XRP | Foco em pagamentos institucionais |

### 4.2 Pré-processamento de Dados

O pré-processamento incluirá:
- Limpeza e tratamento de valores ausentes
- Normalização de features
- Criação de indicadores técnicos
- Divisão temporal dos dados (treino, validação e teste)

### 4.3 Arquitetura dos Modelos

#### 4.3.1 Long Short-Term Memory (LSTM)

O LSTM é uma arquitetura de Rede Neural Recorrente (RNN) especialmente projetada para superar o problema do desaparecimento do gradiente em sequências longas. Conforme Hasim Sak et al. (2014), as LSTMs têm sido usadas com sucesso para:

- Reconhecimento de escrita à mão
- Modelagem de linguagem
- Rotulação fonética de quadros acústicos
- **Previsão de séries temporais financeiras**

A arquitetura LSTM possui três portões (gates) fundamentais:

1. **Forget Gate (ft):** Decide quais informações descartar do estado da célula
2. **Input Gate (it):** Determina quais novos valores serão armazenados
3. **Output Gate (ot):** Controla quais partes do estado da célula serão outputadas

As equações matemáticas que governam o LSTM são:

```
ft = σ(Wf·[ht-1, xt] + bf)
it = σ(Wi·[ht-1, xt] + bi)
C̃t = tanh(WC·[ht-1, xt] + bC)
Ct = ft * Ct-1 + it * C̃t
ot = σ(Wo·[ht-1, xt] + bo)
ht = ot * tanh(Ct)
```

Onde:
- σ representa a função sigmoid
- W são as matrizes de pesos
- b são os vetores de bias
- ht é o hidden state
- Ct é o cell state

#### 4.3.2 eXtreme Gradient Boosting (XGBoost)

O XGBoost, descrito por Chen et al. (2016), é um algoritmo de gradient boosting altamente eficaz que se destaca por:

- **Processamento paralelo:** Utilização eficiente de múltiplas CPUs
- **Regularização:** Prevenção de overfitting através de penalização L1 e L2
- **Tratamento de valores ausentes:** Capacidade nativa de lidar com missing data
- **Flexibilidade:** Suporte a múltiplas funções objetivo

O modelo geral de ensemble de árvores com n exemplos e m features tem uma saída que é a soma das predições de árvores independentes:

```
ŷi = ΣK(k=1) fk(xi), fk ∈ F
```

A função objetivo do XGBoost incorpora tanto o erro de predição quanto a complexidade do modelo:

```
L(φ) = Σi l(ŷi, yi) + Σk Ω(fk)
```

Onde:
- l é uma função de perda diferenciável e convexa
- Ω(f) = γT + ½λΣj wj² é o termo de regularização

### 4.4 Otimização Bayesiana

A Otimização Bayesiana (BO) tem origem no Teorema de Bayes e é caracterizada pelo uso de modelos probabilísticos para orientar a busca por soluções ótimas. Conforme Bobak Shahriari et al. (2016), a BO é particularmente eficiente para:

- Encontrar hiperparâmetros ideais para algoritmos de ML
- Otimizar funções custosas de avaliar
- Trabalhar com espaços de busca de alta dimensionalidade

O algoritmo de Otimização Bayesiana segue os seguintes passos:

1. **Inicialização:** Conjunto inicial de observações D₀ = {(x₁, y₁), ..., (xₙ, yₙ)}
2. **Modelagem:** Treinar um Processo Gaussiano para modelar f(x)
3. **Aquisição:** Maximizar a função de aquisição para encontrar o próximo ponto
4. **Avaliação:** Calcular f(xt) para o ponto selecionado
5. **Atualização:** Incorporar nova observação ao conjunto de dados
6. **Iteração:** Repetir até convergência

---

## 5. IMPLEMENTAÇÃO

### 5.1 Ambiente de Desenvolvimento

A implementação será realizada em Python, utilizando as seguintes bibliotecas principais:

| Biblioteca | Versão | Propósito |
|------------|--------|-----------|
| TensorFlow/Keras | 2.x | Implementação da LSTM |
| XGBoost | 1.x | Implementação do XGBoost |
| Optuna | 3.x | Otimização Bayesiana |
| Pandas | 1.x | Manipulação de dados |
| NumPy | 1.x | Operações numéricas |
| Scikit-learn | 1.x | Métricas e pré-processamento |
| ccxt | 3.x | Integração com exchanges |

### 5.2 Pipeline de Desenvolvimento

```python
# Pseudocódigo do Pipeline Principal

1. COLETA DE DADOS
   - Conectar às APIs das exchanges
   - Baixar dados históricos (OHLCV)
   - Armazenar em formato estruturado

2. PRÉ-PROCESSAMENTO
   - Calcular indicadores técnicos
   - Normalizar features
   - Criar janelas temporais para LSTM

3. TREINAMENTO
   Para cada modelo (LSTM, XGBoost):
      - Dividir dados (70% treino, 15% validação, 15% teste)
      - Aplicar Otimização Bayesiana
      - Treinar com hiperparâmetros otimizados
      - Validar performance

4. AVALIAÇÃO
   - Calcular métricas (acurácia, precisão, recall, F1)
   - Simular trading com dados históricos
   - Calcular ROI e Sharpe Ratio

5. DEPLOYMENT
   - Integrar com APIs Binance/KuCoin
   - Implementar sistema de alertas
   - Monitorar performance em tempo real
```

### 5.3 Estratégias de Trading

O sistema implementará múltiplas estratégias:

1. **Trend Following:** Identificação e acompanhamento de tendências de médio/longo prazo
2. **Mean Reversion:** Exploração de desvios temporários da média
3. **Momentum:** Capitalização em movimentos de forte impulso
4. **Arbitragem:** Exploração de diferenças de preço entre exchanges

---

## 6. RESULTADOS ESPERADOS

### 6.1 Métricas de Performance

Espera-se alcançar os seguintes benchmarks:

| Métrica | Meta Mínima | Meta Ideal |
|---------|-------------|------------|
| Acurácia | 70% | 85% |
| Precisão | 75% | 90% |
| Recall | 70% | 85% |
| F1-Score | 72% | 87% |
| ROI Anual | 20% | 50% |
| Sharpe Ratio | 1.5 | 2.5 |

### 6.2 Contribuições Científicas

1. **Validação Empírica:** Comprovação da eficácia dos modelos LSTM e XGBoost em diferentes condições de mercado

2. **Framework de Otimização:** Desenvolvimento de um framework robusto para otimização de hiperparâmetros específico para trading de criptomoedas

3. **Análise Comparativa:** Estudo detalhado comparando diferentes arquiteturas de ML/DL no contexto de criptomoedas

4. **Sistema Integrado:** Criação de um sistema end-to-end desde a predição até a execução automática de trades

### 6.3 Impacto Prático

- **Para Investidores Individuais:** Democratização do acesso a ferramentas sofisticadas de trading algorítmico
- **Para Instituições:** Framework adaptável para gestão de portfólios de criptoativos
- **Para a Academia:** Contribuição para o corpo de conhecimento em finanças computacionais

---

## 7. CRONOGRAMA DE ATIVIDADES

| Atividade | Início | Fim | Status |
|-----------|--------|-----|--------|
| Revisão Sistemática da Literatura | Set/2024 | Dez/2024 | ✓ Concluído |
| Estudo da Linguagem Python | Set/2024 | Dez/2024 | ✓ Concluído |
| Participação em Reuniões do Grupo | Set/2024 | Fev/2025 | ⚡ Em andamento |
| Coleta de Dados | Out/2024 | Fev/2025 | ⚡ Em andamento |
| Tabulação dos Dados | Out/2024 | Fev/2025 | ⚡ Em andamento |
| Análise dos Dados | Nov/2024 | Fev/2025 | ⚡ Em andamento |
| Modelagem e Implementação LSTM/XGBoost | Jan/2025 | Mar/2025 | ⏳ Planejado |
| Negociação Automática | Mai/2025 | Ago/2025 | ⏳ Planejado |
| Testes e Validação | Jun/2025 | Ago/2025 | ⏳ Planejado |
| Publicação de Resultados | Jul/2025 | Ago/2025 | ⏳ Planejado |

---

## 8. DESAFIOS E LIMITAÇÕES

### 8.1 Desafios Técnicos

1. **Volatilidade Extrema:** O mercado de criptomoedas apresenta volatilidade significativamente maior que mercados tradicionais
2. **Dados Ruidosos:** Presença de manipulação de mercado e wash trading
3. **Custos Computacionais:** Treinamento de modelos deep learning requer recursos significativos
4. **Latência:** Necessidade de execução em tempo real para capitalizar oportunidades

### 8.2 Limitações do Estudo

- **Período Temporal:** Dados limitados devido à juventude do mercado de criptomoedas
- **Regulamentação:** Incerteza regulatória pode impactar a aplicabilidade do sistema
- **Black Swan Events:** Dificuldade em prever eventos extremos e sem precedentes

---

## 9. CONSIDERAÇÕES ÉTICAS

### 9.1 Transparência e Responsabilidade

O desenvolvimento deste sistema considera importantes aspectos éticos:

- **Transparência:** Documentação clara sobre limitações e riscos do sistema
- **Responsabilidade:** Avisos explícitos sobre os riscos de investimento
- **Segurança:** Implementação de medidas robustas de segurança para proteção de fundos
- **Compliance:** Aderência às regulamentações locais e internacionais

### 9.2 Impacto Social

O projeto busca contribuir para:
- Democratização do acesso a ferramentas avançadas de investimento
- Educação financeira através da divulgação científica
- Desenvolvimento do ecossistema de criptomoedas no Brasil

---

## 10. CONCLUSÃO

Este projeto de Iniciação Científica representa uma contribuição significativa para a interseção entre Inteligência Artificial e mercados financeiros, especificamente no contexto emergente das criptomoedas. A combinação de técnicas avançadas de Machine Learning (LSTM e XGBoost) com Otimização Bayesiana promete desenvolver um sistema robusto e eficiente para predição e negociação automatizada.

A pesquisa não apenas validará a aplicabilidade desses métodos em um ambiente de alta volatilidade e complexidade, mas também fornecerá um framework prático e replicável para futuras investigações. A integração com plataformas reais de negociação (Binance e KuCoin) garante que os resultados tenham aplicabilidade imediata no mundo real.

Os resultados esperados incluem não apenas métricas técnicas superiores (acurácia > 70%, ROI > 20%), mas também contribuições acadêmicas significativas através de publicações e apresentações em congressos científicos. O projeto culminará com a apresentação no Congresso de Ciência e Tecnologia da PUC Goiás em 2025, disseminando o conhecimento adquirido para a comunidade acadêmica.

Este trabalho estabelece as bases para futuras pesquisas em:
- Aplicação de técnicas de reinforcement learning para trading
- Análise de sentimento em redes sociais para predição de preços
- Desenvolvimento de portfolios multi-ativos otimizados
- Implementação de técnicas de quantum computing em finanças

A combinação de rigor acadêmico com aplicabilidade prática posiciona este projeto como uma contribuição valiosa tanto para a academia quanto para o mercado, pavimentando o caminho para avanços futuros na área de finanças computacionais e criptoativos.

---

## REFERÊNCIAS

AHMADI, E.; JASEMI, M.; MONPLAISIR, L.; NABAVI, M. A.; MAHMOODI, A.; JAMD, P. A. New efficient hybrid candlestick technical analysis model for stock market timing on the basis of the Support Vector Machine and Heuristic Algorithms of Imperialist Competition and Genetic. **Expert Systems with Applications**, 2017.

ALI, J.; KHAN, R.; AHMAD, N.; MAQSOOD, I. Random Forests and Decision Trees. **International Journal of Computer Science Issues (IJCSI)**, v. 9, 2012.

BOSWELL, D. Introduction to Support Vector Machines. **Technical Report**, 2002.

CHEN, T.; GUESTRIN, C. XGBoost: A Scalable Tree Boosting System. In: **Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining**, p. 785-794, 2016.

CHEN, Z.; LI, C.; SUN, W. Bitcoin price prediction using machine learning: An approach to sample dimension engineering. **Journal of Computational and Applied Mathematics**, v. 365, p. 112395, 2020.

DEISENROTH, M.; FAISAL, A. A.; ONG, C. S. **Mathematics for Machine Learning**. Cambridge University Press, 2020.

GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. **Deep Learning**. MIT Press, 2016.

GURESEN, E.; KAYAKUTLU, G.; DAIM, T. U. Using artificial neural network models in stock market index prediction. **Expert Systems with Applications**, v. 38, n. 8, p. 10389-10397, 2011.

HADAVANDI, E.; SHAVANDI, H.; GHANBARI, A. Integration of genetic fuzzy systems and artificial neural networks for stock price forecasting. **Knowledge-Based Systems**, v. 23, n. 8, p. 800-808, 2010.

KHATTAK, B. H. A.; SHAFI, I.; KHAN, A. S.; FLORES, E. S.; LARA, R. G.; SAMAD, M. A.; ASHRAF, I. A Systematic Survey of AI Models in Financial Market Forecasting for Profitability Analysis. **IEEE Access**, 2023.

LOUPPE, G. Understanding Random Forests: From Theory to Practice. **arXiv preprint arXiv:1407.7502**, 2014.

McNALLY, S.; ROCHE, J.; CATON, S. Predicting the price of Bitcoin using Machine Learning. In: **26th Euromicro International Conference on Parallel, Distributed and Network-based Processing**, p. 339-343, 2018.

MOHRI, M.; ROSTAMIZADEH, A.; TALWALKAR, A. **Foundations of Machine Learning**. MIT Press, 2012.

OZBAYOGLU, A. M.; GUDELEK, M. U.; SEZER, O. B. Deep learning for financial applications: A survey. **Applied Soft Computing**, v. 93, p. 106384, 2020.

PATEL, M. M.; TANWAR, S.; GUPTA, R.; KUMAR, N. A Deep Learning-based Cryptocurrency Price Prediction Scheme for Financial Institutions. **Journal of Information Security and Applications**, v. 55, p. 102583, 2020.

POONGODI, M.; SHARMA, A.; VIJAYAKUMAR, V.; BHARDWAJ, V.; SHARMA, A. P.; IQBAL, R.; KUMAR, R. Prediction of the price of Ethereum blockchain cryptocurrency in an industrial finance system. **Computers & Electrical Engineering**, v. 81, p. 106527, 2020.

SAK, H.; SENIOR, A.; BEAUFAYS, F. Long Short-Term Memory Based Recurrent Neural Network Architectures for Large Vocabulary Speech Recognition. **arXiv preprint arXiv:1402.1128**, 2014.

SHAHRIARI, B.; SWERSKY, K.; WANG, Z.; ADAMS, R.; FREITAS, N. Taking the Human Out of the Loop: A Review of Bayesian Optimization. **Proceedings of the IEEE**, v. 104, n. 1, p. 148-175, 2016.

VAPNIK, V.; GOLOWICH, S. E.; SMOLA, A. Support vector method for function approximation, regression estimation and signal processing. In: **Proceedings of the 9th International Conference on Neural Information Processing Systems (NIPS'96)**, MIT Press, p. 281-287, 1996.

VERSACE, M.; BHATT, R.; HINDS, O.; SHIFFER, M. Predicting the exchange traded fund DIA with a combination of genetic algorithms and neural networks. **Expert Systems with Applications**, v. 27, n. 3, p. 417-425, 2004.

WANG, J.; SUN, T.; LIU, B.; CAO, Y.; WANG, D. Financial Markets Prediction with Deep Learning. In: **Proceedings of the 17th IEEE International Conference on Machine Learning and Applications**, p. 97-104, 2018.

---

## APÊNDICES

### Apêndice A - Glossário de Termos Técnicos

- **LSTM:** Long Short-Term Memory - Arquitetura de rede neural recorrente
- **XGBoost:** eXtreme Gradient Boosting - Algoritmo de ensemble baseado em árvores
- **ROI:** Return on Investment - Retorno sobre o investimento
- **API:** Application Programming Interface - Interface de programação de aplicações
- **OHLCV:** Open, High, Low, Close, Volume - Dados básicos de mercado
- **Sharpe Ratio:** Medida de retorno ajustado ao risco

### Apêndice B - Código de Exemplo

```python
# Exemplo simplificado de implementação LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_lstm_model(input_shape, n_features):
    model = keras.Sequential([
        layers.LSTM(100, return_sequences=True,
                   input_shape=(input_shape, n_features)),
        layers.Dropout(0.2),
        layers.LSTM(100, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(50),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model
```

---

**Nota:** Este artigo representa o trabalho de Iniciação Científica em desenvolvimento na PUC Goiás, sob orientação da Profa. Maria José Pereira Dantas, no âmbito do Grupo de Pesquisa em Identificação, Modelagem e Inteligência Artificial para Otimização de Sistemas.
