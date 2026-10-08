# O que mudou na versão 2.0

Este documento registra a análise do projeto original e as melhorias feitas. Os problemas listados foram **reproduzidos executando o código original**, não apenas lidos.

## Problemas encontrados no projeto original

### Críticos (resultado errado na tela)

1. **A previsão SARIMAX nunca rodava.** Em 100% das execuções o app exibia uma reta de tendência linear (o *fallback*), enquanto o spinner dizia "Gerando previsão com modelo SARIMAX". Duas causas independentes:
   - `fillna(method='forward')` não é um método válido do pandas → exceção ao preparar as variáveis exógenas;
   - `ARIMA(...).fit(disp=False)`: o ARIMA atual do statsmodels não aceita `disp` → `TypeError`. A busca automática de parâmetros engolia o erro com `except: continue` e testava **0** combinações.
   
   Além disso, mesmo que rodasse, as variáveis exógenas vazavam informação (RSI e MACD do mesmo dia do preço) e o "futuro" delas era a média dos últimos 30 dias repetida. A reta de fallback podia ainda prever preço negativo.

2. **O preço atual podia aparecer muito errado.** Quando a validação detectava qualquer problema (por exemplo, 3 dias faltando na resposta da API), `clean_data` substituía dias de alta forte pela mediana e **reconstruía toda a série de preços** a partir dos retornos — com um erro de deslocamento de uma posição. Num histórico com um dia de +40%, o preço "atual" exibido ficou **29% abaixo do real**. Mesmo sem outliers, a reconstrução deslocava o preço.

3. **Cache eterno.** Os dados eram salvos em `cache/bitcoin_365.json` e reutilizados para sempre — o "preço atual" podia ter semanas de atraso sem nenhum aviso.

4. **Interface travável.** Em caso de limite de requisições (HTTP 429), um `while True` com `sleep(60)` podia prender o app indefinidamente.

5. **Resultados sumiam a cada clique.** Tudo era calculado dentro de `if st.sidebar.button(...)`, então qualquer interação apagava a análise, e cada clique refazia todas as chamadas à API.

### Cálculos incorretos

- **RSI** usava `ewm(span=14)` em vez da suavização de Wilder — diferença média de ~6 pontos em relação ao RSI padrão de mercado.
- **ADX** "simplificado" usava o desvio-padrão do preço no lugar do ATR.
- **Sortino** calculava o desvio apenas dos dias negativos (fórmula incorreta) e podia retornar infinito.
- **Kelly** no dimensionamento de posição resultava sempre em 0 (ganho médio = perda média por construção).
- **Simulação de portfólio** usava preço fixo de 100 para todas as moedas (placeholder), tornando o ROI sem sentido.
- Suportes/resistências geravam níveis quase duplicados (o `set` de floats não agrupa valores próximos).
- `get_available_cryptos` pegava as 100 primeiras moedas em ordem alfabética, não as maiores.
- O nível de risco "Muito Alto" não tinha cor associada na interface.

### Estrutura e manutenção

- `requirements.txt` incluía **TensorFlow, Keras, XGBoost e `ta`**, nenhum usado — centenas de MB de instalação a mais, deixando o deploy lento e sujeito a falhas.
- O README anunciava funcionalidades que não existiam na interface (portfólio, backtesting, cenários, testes ADF e análise de resíduos).
- Polygon usava o ID `matic-network`; o token migrou para POL em 2024.
- `setup_guide.py` rodava `pip install` e gerava um `config.py` que nada usava.
- Não havia testes automatizados.

## O que foi feito

### Dados
- Três fontes com troca automática: **CoinGecko → Yahoo Finance → último dado salvo** (com aviso de desatualização).
- Cache em memória e em disco com validade de 1 hora e botão **Atualizar dados**.
- Novas tentativas limitadas (3) que respeitam o `Retry-After`, com espera máxima de 10 s.
- Mensagens de erro em português e acionáveis (moeda não encontrada, limite de requisições, chave inválida).
- Preços nunca são reescalados: dias faltantes recebem o último preço e movimentos extremos são apenas sinalizados.
- Cotação em **USD, BRL ou EUR**; histórico de até 5 anos (via Yahoo); candles OHLC completos quando disponíveis; suporte a qualquer ID da CoinGecko; chave de API opcional via `secrets.toml`.
- Índice Fear & Greed e dados de mercado (ranking, máxima histórica) — opcionais, sem quebrar o app se falharem.

### Análise
- Indicadores revisados (RSI, ATR e ADX de Wilder conferidos contra implementação de referência) e novos: Aroon, Williams %R, ATR %, Fibonacci, suportes/resistências por agrupamento.
- Resumo técnico com 12 regras em dois grupos (osciladores e tendência) e histórico do score.
- **Previsão honesta:** 6 modelos + combinação, intervalos de 80% e 95%, validação *walk-forward* sem vazamento e comparação obrigatória com o passeio aleatório. A interface diz claramente quando nenhum modelo supera a referência.
- Diagnóstico estatístico: ADF, Ljung-Box e ordem ARIMA escolhida por AICc.
- Risco: score explicável (cada fator e seu peso), VaR/CVaR histórico, normal e Cornish-Fisher, EWMA, beta/correlação com o BTC, calculadora de tamanho de posição por ATR.

### Novas funcionalidades
- **Simulador:** Monte Carlo com reamostragem de blocos de retornos reais; "e se eu tivesse investido?" (aporte único vs. DCA com preços reais); cenários da previsão.
- **Estratégias:** backtest de 6 regras técnicas vs. comprar e segurar, sem olhar para o futuro, com custos, operações e mapa de parâmetros que expõe sobreajuste.
- **Portfólio:** correlação, risco × retorno, otimização (mínima variância, máximo Sharpe, paridade de risco, manual) com covariância Ledoit-Wolf, fronteira eficiente e backtest com rebalanceamento.
- Download em CSV (padrão brasileiro: `;` e vírgula decimal).

### Interface
- App multipágina com navegação no topo; barra lateral compartilhada entre as páginas.
- Resultados persistem ao interagir (cálculos em cache); só a página aberta é calculada.
- Identidade visual própria, números e datas no padrão brasileiro, gráficos interativos consistentes, layout que funciona no celular.

### Engenharia
- Código organizado no pacote `crypto_platform/`, com a lógica de cálculo separada da interface (pode ser usada em notebooks).
- **102 testes automatizados** (dados com API simulada, indicadores, ausência de *look-ahead*, previsão sem vazamento, risco, simulação, backtest, portfólio e todas as páginas da interface).
- Testado com pandas 2.2 e 3.0, numpy 2.1 e 2.5, statsmodels 0.14 e 0.15, plotly 6 e 7.
- `ruff` para estilo, GitHub Actions para CI, `requirements.txt` enxuto, devcontainer atualizado.
- Modo demonstração (`CRYPTO_PLATFORM_DEMO=1`) com dados sintéticos para desenvolvimento offline.

## Compatibilidade

- O ponto de entrada continua sendo `main.py` — o deploy no Streamlit Cloud não precisa mudar.
- As classes `CryptoDataCollector`, `TechnicalIndicators`, `CryptoForecaster`, `RiskAnalyzer` e `InvestmentSimulator` continuam existindo (agora em `crypto_platform/`), mas com resultados em estruturas novas.
- `setup_guide.py` foi removido: a instalação é `pip install -r requirements.txt` e a verificação é `pytest`.
