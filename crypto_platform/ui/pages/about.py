"""Metodologia, fontes de dados e limitações."""

from __future__ import annotations

import streamlit as st

from ... import __version__
from ... import forecast_model as fm


def render() -> None:
    st.markdown("## Metodologia")
    st.markdown(
        "Esta plataforma é uma ferramenta educacional para entender o comportamento de criptoativos: tendência, risco e o "
        "quanto (ou quão pouco) dá para prever. Cada número exibido vem de um método descrito abaixo."
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Dados")
        st.markdown(
            "- **CoinGecko**: preço diário, volume e valor de mercado. Na versão gratuita o histórico é limitado a 365 dias.\n"
            "- **Yahoo Finance**: candles diários completos (abertura, máxima, mínima, fechamento) e histórico longo. "
            "Em BRL e EUR o preço em dólar é convertido pelo câmbio diário.\n"
            "- **alternative.me**: índice Fear & Greed.\n"
            "- Os dados ficam em cache por até 1 hora. Se todas as fontes falharem, o app mostra o último dado salvo e avisa.\n"
            "- Dias sem cotação são preenchidos com o último preço; movimentos extremos são sinalizados, nunca alterados."
        )
        st.markdown("#### Indicadores técnicos")
        st.markdown(
            "RSI, ATR e ADX seguem a suavização de Wilder (mesma convenção do TradingView e do TA-Lib). O resumo técnico é a "
            "média de votos de 12 regras, separadas em osciladores (sinais de reversão) e médias/tendência (continuidade). "
            "Suportes e resistências vêm de topos e fundos confirmados, agrupados por proximidade."
        )
        st.markdown("#### Risco")
        st.markdown(
            "Volatilidade anualizada com 365 dias (cripto negocia todos os dias). VaR e CVaR históricos, normal e de "
            "Cornish-Fisher (corrige assimetria e caudas pesadas). Sortino com desvio de perdas calculado sobre todos os dias. "
            "O score de risco combina volatilidade de 30 dias (30%), drawdown máximo (25%), VaR 95% (25%) e Sharpe (20%), "
            "cada um convertido numa nota de 0 a 10 em escala calibrada para criptoativos."
        )
    with c2:
        st.markdown("#### Previsão")
        st.markdown(
            "Todos os modelos trabalham no logaritmo do preço e mostram a **mediana** prevista com intervalos de 80% e 95%. "
            "Antes de exibir qualquer previsão, cada modelo é testado no passado (*walk-forward*): previsões feitas em várias "
            "datas anteriores, sempre treinadas apenas com dados disponíveis naquela data, e comparadas ao que aconteceu. "
            "A referência é o **passeio aleatório** — um modelo que não o supera não acrescenta informação."
        )
        for name, desc in fm.MODEL_DESCRIPTIONS.items():
            st.markdown(f"- **{name}:** {desc}")
        st.markdown("#### Simulações e backtests")
        st.markdown(
            "- Monte Carlo por reamostragem de blocos de 5 dias de retornos reais (preserva caudas e agrupamento de "
            "volatilidade) ou por movimento browniano geométrico.\n"
            "- Backtests sem olhar para o futuro: o sinal de um dia só afeta a posição a partir do dia seguinte, com taxa e "
            "slippage em cada troca.\n"
            "- Otimização de carteira com covariância encolhida (Ledoit-Wolf) e limite de peso por ativo."
        )

    st.markdown("#### Limitações")
    st.markdown(
        "Modelos estatísticos assumem que o futuro se parece com o passado; em cripto, mudanças de regime (regulação, "
        "falências de corretoras, ciclos de liquidez) quebram essa hipótese com frequência. Backtests e otimizações usam o "
        "mesmo período que avaliam e tendem a parecer melhores do que seriam na prática. Impostos não são considerados."
    )

    with st.expander("Configurar chave da CoinGecko (opcional)", icon=":material/key:"):
        st.markdown(
            "Sem chave, a API pública da CoinGecko tem limite baixo de requisições — principalmente em servidores "
            "compartilhados como o Streamlit Community Cloud. Uma chave **Demo** gratuita resolve:\n\n"
            "1. Crie a chave em coingecko.com/pt/api.\n"
            '2. Localmente, crie `.streamlit/secrets.toml` com `COINGECKO_DEMO_API_KEY = "sua-chave"`, ou defina a '
            "variável de ambiente de mesmo nome.\n"
            "3. No Streamlit Cloud, cole a mesma linha em *Settings → Secrets*."
        )

    st.warning(
        "Conteúdo educacional. Não é recomendação de investimento. Criptoativos são altamente voláteis e podem perder a "
        "maior parte do valor; faça sua própria pesquisa e, se necessário, procure um profissional certificado.",
        icon=":material/gavel:",
    )
    st.caption(f"Versão {__version__}")
