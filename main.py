import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Importar módulos do projeto
from data_collector import CryptoDataCollector
from technical_indicators import TechnicalIndicators
from forecast_model import CryptoForecaster
from investment_simulator import InvestmentSimulator
from risk_analyzer import RiskAnalyzer

# Configuração da página
st.set_page_config(
    page_title="Crypto Analysis & Forecast",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .stAlert > div {
        background-color: #1e1e1e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header"> Crypto Analysis & Forecast Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar para configurações
    st.sidebar.header("Configurações")
    
    # Seleção de criptomoeda
    crypto_options = {
        'Bitcoin': 'bitcoin',
        'Ethereum': 'ethereum',
        'Cardano': 'cardano',
        'XRP': 'ripple',
        'Solana': 'solana',
        'Polygon': 'matic-network',
        'Chainlink': 'chainlink',
        'Avalanche': 'avalanche-2'
    }
    
    selected_crypto = st.sidebar.selectbox(
        "Selecione a Criptomoeda:",
        options=list(crypto_options.keys()),
        index=0
    )
    
    # Configurações de análise
    historical_days = st.sidebar.slider(
        "Dias de Histórico:",
        min_value=180,
        max_value=365,
        value=365,
        step=30
    )
    
    forecast_days = st.sidebar.slider(
        "Dias para Previsão:",
        min_value=7,
        max_value=60,
        value=30,
        step=7
    )
    
    # Valor para simulação de investimento
    st.sidebar.subheader("Simulação de Investimento")
    investment_amount = st.sidebar.number_input(
        "Valor do Investimento (USD):",
        min_value=100.0,
        max_value=100000.0,
        value=1000.0,
        step=100.0
    )
    
    # Botão para executar análise
    if st.sidebar.button("Executar Análise", type="primary"):
        with st.spinner(f"Coletando dados de {selected_crypto}..."):
            # Coleta de dados
            collector = CryptoDataCollector()
            crypto_id = crypto_options[selected_crypto]
            data = collector.get_historical_data(crypto_id, historical_days)
            
            if data is not None:
                st.success(f"Dados coletados: {len(data)} dias de histórico")
                
                # Calcular indicadores técnicos
                with st.spinner("Calculando indicadores técnicos..."):
                    indicators = TechnicalIndicators()
                    data_with_indicators = indicators.calculate_all_indicators(data)
                
                # Criar previsão
                with st.spinner("Gerando previsão com modelo SARIMAX..."):
                    forecaster = CryptoForecaster()
                    forecast_result = forecaster.forecast_sarimax(data, forecast_days)
                
                # Análise de risco
                with st.spinner("Analisando riscos..."):
                    risk_analyzer = RiskAnalyzer()
                    risk_metrics = risk_analyzer.calculate_risk_metrics(data)
                
                # Simulação de investimento
                simulator = InvestmentSimulator()
                simulation_result = simulator.simulate_investment(
                    data, forecast_result, investment_amount, forecast_days
                )
                
                # Exibir resultados
                display_results(
                    selected_crypto,
                    data_with_indicators,
                    forecast_result,
                    risk_metrics,
                    simulation_result
                )
            else:
                st.error("Erro ao coletar dados. Tente novamente.")

def display_results(crypto_name, data, forecast_result, risk_metrics, simulation_result):
    """Exibe todos os resultados da análise"""
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data['price'].iloc[-1]
    predicted_price = forecast_result['forecast'].iloc[-1]
    price_change = ((predicted_price - current_price) / current_price) * 100
    
    with col1:
        st.metric(
            "Preço Atual (USD)",
            f"${current_price:.2f}",
            f"{((current_price - data['price'].iloc[-2]) / data['price'].iloc[-2] * 100):.2f}%"
        )
    
    with col2:
        st.metric(
            "Previsão Final (USD)",
            f"${predicted_price:.2f}",
            f"{price_change:.2f}%"
        )
    
    with col3:
        st.metric(
            "Volatilidade (30d)",
            f"{risk_metrics['volatility_30d']:.2f}%"
        )
    
    with col4:
        st.metric(
            "Score de Risco",
            f"{risk_metrics['risk_score']:.1f}/10",
            risk_metrics['risk_level']
        )
    
    # Gráfico principal
    st.subheader("📈 Análise Técnica e Previsão")
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.3, 0.15, 0.25, 0.3]
    )
    
    # Preço e Bandas de Bollinger
    fig.add_trace(
        go.Scatter(x=data.index, y=data['price'], name='Preço', line=dict(color='white')),
        row=1, col=1
    )
    
    if 'bb_upper' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['bb_upper'], name='BB Superior', 
                      line=dict(color='red', dash='dash'), opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['bb_lower'], name='BB Inferior', 
                      line=dict(color='red', dash='dash'), opacity=0.7, 
                      fill='tonexty', fillcolor='rgba(255,0,0,0.1)'),
            row=1, col=1
        )
    
    # Adicionar previsão
    forecast_dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1),
        periods=len(forecast_result['forecast']),
        freq='D'
    )
    
    fig.add_trace(
        go.Scatter(x=forecast_dates, y=forecast_result['forecast'], 
                  name='Previsão', line=dict(color='yellow', width=3)),
        row=1, col=1
    )
    
    # Intervalo de confiança
    fig.add_trace(
        go.Scatter(x=forecast_dates, y=forecast_result['upper_ci'], 
                  name='IC Superior', line=dict(color='yellow', dash='dot'), opacity=0.5),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=forecast_dates, y=forecast_result['lower_ci'], 
                  name='IC Inferior', line=dict(color='yellow', dash='dot'), 
                  opacity=0.5, fill='tonexty', fillcolor='rgba(255,255,0,0.1)'),
        row=1, col=1
    )
    
    # Volume
    fig.add_trace(
        go.Bar(x=data.index, y=data['volume'], name='Volume', marker_color='blue', opacity=0.6),
        row=2, col=1
    )
    
    # RSI
    if 'rsi' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['rsi'], name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    if 'macd' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['macd'], name='MACD', line=dict(color='blue')),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['macd_signal'], name='Signal', line=dict(color='red')),
            row=4, col=1
        )
        fig.add_trace(
            go.Bar(x=data.index, y=data['macd_histogram'], name='Histogram', 
                   marker_color='green', opacity=0.6),
            row=4, col=1
        )
    
    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=True,
        title=f"{crypto_name} - Análise Técnica Completa"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Simulação de investimento
    st.subheader("Simulação de Investimento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Resultados da Simulação")
        
        profit_loss = simulation_result['estimated_profit_loss']
        roi = simulation_result['roi_percentage']
        
        if profit_loss > 0:
            st.success(f"Lucro Estimado: ${profit_loss:.2f}")
            st.success(f"ROI: {roi:.2f}%")
        else:
            st.error(f"Prejuízo Estimado: ${abs(profit_loss):.2f}")
            st.error(f"ROI: {roi:.2f}%")
        
        st.info(f"Investimento Inicial: ${simulation_result['initial_investment']:.2f}")
        st.info(f"Valor Final Estimado: ${simulation_result['final_value']:.2f}")
    
    with col2:
        st.markdown("### Análise de Risco")
        
        risk_level = risk_metrics['risk_level']
        risk_color = {
            'Baixo': 'success',
            'Médio': 'warning',
            'Alto': 'error'
        }.get(risk_level, 'info')
        
        getattr(st, risk_color)(f"Nível de Risco: {risk_level}")
        
        st.write(f"Volatilidade 7d: {risk_metrics['volatility_7d']:.2f}%")
        st.write(f"Volatilidade 30d: {risk_metrics['volatility_30d']:.2f}%")
        st.write(f"Máximo Drawdown: {risk_metrics['max_drawdown']:.2f}%")
        st.write(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
    
    # Indicadores técnicos atuais
    st.subheader("Indicadores Técnicos Atuais")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### RSI")
        current_rsi = data['rsi'].iloc[-1] if 'rsi' in data.columns else 0
        if current_rsi > 70:
            st.error(f"Sobrecomprado: {current_rsi:.1f}")
        elif current_rsi < 30:
            st.success(f"Sobrevendido: {current_rsi:.1f}")
        else:
            st.info(f"Neutro: {current_rsi:.1f}")
    
    with col2:
        st.markdown("#### MACD")
        if 'macd' in data.columns:
            macd_current = data['macd'].iloc[-1]
            signal_current = data['macd_signal'].iloc[-1]
            if macd_current > signal_current:
                st.success("Sinal de Compra")
            else:
                st.error("Sinal de Venda")
    
    with col3:
        st.markdown("#### Bandas de Bollinger")
        if 'bb_upper' in data.columns:
            current_price = data['price'].iloc[-1]
            bb_upper = data['bb_upper'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            
            if current_price > bb_upper:
                st.error("Acima da banda superior")
            elif current_price < bb_lower:
                st.success("Abaixo da banda inferior")
            else:
                st.info("Dentro das bandas")
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    ⚠️ **DISCLAIMER**: Esta análise é apenas para fins educacionais e informativos. 
    Não constitui aconselhamento financeiro. Criptomoedas são investimentos de alto risco. 
    """)

if __name__ == "__main__":
    main()