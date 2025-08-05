# Crypto Analysis & Forecast Platform

Uma plataforma para análise técnica e previsão de criptomoedas usando modelos de séries temporais SARIMAX e indicadores técnicos avançados. 

⚠️ Apenas para fins educacionais.

## Funcionalidades

### Análise Técnica Completa
- **Indicadores Técnicos**: RSI, MACD, Bandas de Bollinger, Médias Móveis, Estocástico, Williams %R, ATR
- **Análise de Volume**: OBV, Volume Rate of Change, Volume Moving Average
- **Indicadores de Momentum**: ROC, Momentum, Price Velocity
- **Indicadores de Tendência**: ADX, Aroon Oscillator
- **Suporte e Resistência**: Identificação automática de níveis críticos

### Previsão de Preços
- **Modelo SARIMAX**: Previsão baseada em séries temporais com variáveis exógenas
- **Intervalos de Confiança**: Previsões com bandas de incerteza
- **Validação Cruzada**: Backtesting para avaliar precisão do modelo

### Simulação de Investimento
- **Simulação Simples**: Calcular retorno esperado para um investimento
- **Portfolio Diversificado**: Simulação com múltiplas criptomoedas
- **Análise de Cenários**: Projeções otimistas, realistas e pessimistas

### Análise de Risco
- **Métricas de Volatilidade**: Cálculo para períodos de 7, 30, 90 e 365 dias
- **Value at Risk (VaR)**: Estimativa de perdas potenciais
- **Maximum Drawdown**: Análise de perdas máximas consecutivas
- **Sharpe e Sortino Ratios**: Retornos ajustados ao risco
- **Score de Risco**: Classificação de 0-10 com recomendações

## Arquitetura do Projeto

```
crypto-plataform/
│
├── main.py                    # Aplicação principal Streamlit
├── data_collector.py          # Coleta de dados via API CoinGecko
├── technical_indicators.py    # Cálculo de indicadores técnicos
├── forecast_model.py          # Modelos de previsão SARIMAX
├── investment_simulator.py    # Simulação de investimentos
├── risk_analyzer.py           # Análise de risco e volatilidade
├── setup_guide.py             # Setup da aplicação
├── requirements.txt           # Dependências do projeto
└── README.md                  # Documentação
```

## Como Executar

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Executar a Aplicação
```bash
python setup_guide.py
```
ou
```bash
streamlit run main.py
```

### 3. Acessar a Interface
Abra o navegador em: `http://localhost:8501`

## Criptomoedas Suportadas

- Bitcoin (BTC)
- Ethereum (ETH)
- Cardano (ADA)
- XRP (XRP)
- Solana (SOL)
- Polygon (MATIC)
- Chainlink (LINK)
- Avalanche (AVAX)

##  Configurações Disponíveis

### Parâmetros de Análise
- **Dias de Histórico**: 180-365 dias
- **Dias de Previsão**: 7-60 dias
- **Valor de Investimento**: $100-$100,000

### Indicadores Técnicos
- **RSI**: Período padrão de 14 dias
- **MACD**: 12/26/9 (rápida/lenta/sinal)
- **Bandas de Bollinger**: 20 períodos, 2 desvios padrão
- **Médias Móveis**: 7, 21, 50, 200 períodos

## Métricas de Performance

### Métricas do Modelo
- **AIC/BIC**: Critérios de informação para seleção de modelo
- **MAE**: Erro absoluto médio
- **RMSE**: Raiz do erro quadrático médio
- **MAPE**: Erro percentual absoluto médio

### Métricas de Risco
- **Volatilidade**: Anualizada para diferentes períodos
- **CVaR**: Conditional Value at Risk (Expected Shortfall)
- **Sharpe Ratio**: Retorno por unidade de risco

## Funcionalidades Avançadas

### 1. Seleção Automática de Parâmetros
O sistema usa grid search para encontrar os melhores parâmetros SARIMAX automaticamente.

### 2. Variáveis Exógenas
Utiliza indicadores técnicos como variáveis explicativas no modelo SARIMAX.

### 3. Tratamento de Dados
- Limpeza automática de outliers
- Interpolação de valores faltantes
- Validação de qualidade dos dados

### 4. Interface Responsiva
- Gráficos interativos com Plotly
- Métricas em tempo real
- Design moderno e intuitivo

## Limitações e Disclaimers

### Limitações Técnicas
- **Dados**: Dependente da API CoinGecko (limite de requisições, possíveis inconsistência nos dados)
- **Modelo**: SARIMAX assume certas propriedades estatísticas
- **Previsão**: Eficácia diminui com horizonte temporal longo

### Disclaimers Importantes
- **Não é Aconselhamento Financeiro**: Use apenas para fins educacionais
- **Alto Risco**: Criptomoedas são investimentos de alto risco
- **Volatilidade**: Mercado extremamente volátil e imprevisível
- **Pesquisa Própria**: Sempre faça sua própria pesquisa (DYOR)

## Metodologia Científica

### Modelo SARIMAX
- **Componente Sazonal**: Detecta padrões cíclicos
- **Variáveis Exógenas**: Incorpora indicadores técnicos
- **Auto-Regressivo**: Usa valores passados para prever futuros
- **Média Móvel**: Suaviza ruídos na série temporal

### Validação do Modelo
- **Teste de Estacionariedade**: Augmented Dickey-Fuller
- **Backtesting**: Validação em dados históricos
- **Análise de Resíduos**: Verificação de autocorrelação
- **Precisão Direcional**: Capacidade de prever direção do movimento

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, siga estas diretrizes:

1. Fork o repositório
2. Crie uma branch para sua feature
3. Implemente testes para novas funcionalidades
4. Mantenha o código documentado
5. Envie um pull request

## 📄 Licença

Este projeto é open source e está disponível sob a [MIT License](LICENSE).


---

**⚠️ AVISO IMPORTANTE**: Este projeto é apenas para fins educacionais e de pesquisa. Não constitui aconselhamento financeiro.
