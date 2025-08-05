import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RiskAnalyzer:
    """Classe para análise de risco de investimentos em criptomoedas"""
    
    def __init__(self):
        pass
    
    def calculate_volatility(self, prices, periods=[7, 30, 90, 365]):
        """
        Calcula volatilidade para diferentes períodos
        
        Args:
            prices (pd.Series): Série de preços
            periods (list): Períodos em dias para cálculo
            
        Returns:
            dict: Volatilidades para cada período
        """
        returns = prices.pct_change().dropna()
        volatilities = {}
        
        for period in periods:
            if len(returns) >= period:
                # Volatilidade rolling para o período
                rolling_vol = returns.rolling(window=period).std()
                current_vol = rolling_vol.iloc[-1] * np.sqrt(365)  # Anualizada
                volatilities[f'volatility_{period}d'] = current_vol * 100  # Em percentual
            else:
                volatilities[f'volatility_{period}d'] = 0
        
        return volatilities
    
    def calculate_var_cvar(self, returns, confidence_levels=[0.95, 0.99]):
        """
        Calcula Value at Risk (VaR) e Conditional VaR (CVaR)
        
        Args:
            returns (pd.Series): Série de retornos
            confidence_levels (list): Níveis de confiança
            
        Returns:
            dict: VaR e CVaR para cada nível de confiança
        """
        risk_metrics = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            
            # VaR histórico
            var = np.percentile(returns, alpha * 100)
            
            # CVaR (Expected Shortfall)
            cvar = returns[returns <= var].mean() if len(returns[returns <= var]) > 0 else 0
            
            conf_pct = int(conf_level * 100)
            risk_metrics[f'var_{conf_pct}'] = var * 100  # Em percentual
            risk_metrics[f'cvar_{conf_pct}'] = cvar * 100  # Em percentual
        
        return risk_metrics
    
    def calculate_maximum_drawdown(self, prices):
        """
        Calcula o Maximum Drawdown
        
        Args:
            prices (pd.Series): Série de preços
            
        Returns:
            dict: Métricas de drawdown
        """
        # Calcular retornos cumulativos
        returns = prices.pct_change().fillna(0)
        cumulative_returns = (1 + returns).cumprod()
        
        # Calcular running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calcular drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Duração do máximo drawdown
        max_dd_start = None
        max_dd_end = None
        
        if max_drawdown < 0:
            max_dd_idx = drawdown.idxmin()
            
            # Encontrar início do drawdown
            before_max_dd = drawdown.loc[:max_dd_idx]
            dd_start_candidates = before_max_dd[before_max_dd == 0]
            if len(dd_start_candidates) > 0:
                max_dd_start = dd_start_candidates.index[-1]
            else:
                max_dd_start = drawdown.index[0]
            
            # Encontrar fim do drawdown (recovery)
            after_max_dd = drawdown.loc[max_dd_idx:]
            recovery_candidates = after_max_dd[after_max_dd >= -0.01]  # -1% tolerance
            if len(recovery_candidates) > 0:
                max_dd_end = recovery_candidates.index[0]
            else:
                max_dd_end = drawdown.index[-1]
        
        # Calcular duração
        if max_dd_start and max_dd_end:
            dd_duration = (max_dd_end - max_dd_start).days
        else:
            dd_duration = 0
        
        return {
            'max_drawdown': max_drawdown * 100,  # Em percentual
            'max_drawdown_duration_days': dd_duration,
            'current_drawdown': drawdown.iloc[-1] * 100,
            'drawdown_series': drawdown * 100
        }
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """
        Calcula Sharpe Ratio
        
        Args:
            returns (pd.Series): Série de retornos
            risk_free_rate (float): Taxa livre de risco anual
            
        Returns:
            float: Sharpe Ratio
        """
        # Retorno médio anualizado
        mean_return = returns.mean() * 365
        
        # Volatilidade anualizada
        std_return = returns.std() * np.sqrt(365)
        
        # Sharpe Ratio
        if std_return > 0:
            sharpe_ratio = (mean_return - risk_free_rate) / std_return
        else:
            sharpe_ratio = 0
        
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, returns, risk_free_rate=0.02):
        """
        Calcula Sortino Ratio (considera apenas downside risk)
        
        Args:
            returns (pd.Series): Série de retornos
            risk_free_rate (float): Taxa livre de risco anual
            
        Returns:
            float: Sortino Ratio
        """
        # Retorno médio anualizado
        mean_return = returns.mean() * 365
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(365)
        else:
            downside_std = 0
        
        # Sortino Ratio
        if downside_std > 0:
            sortino_ratio = (mean_return - risk_free_rate) / downside_std
        else:
            sortino_ratio = float('inf') if mean_return > risk_free_rate else 0
        
        return sortino_ratio
    
    def calculate_beta(self, asset_returns, market_returns):
        """
        Calcula Beta em relação ao mercado
        
        Args:
            asset_returns (pd.Series): Retornos do ativo
            market_returns (pd.Series): Retornos do mercado (ex: Bitcoin)
            
        Returns:
            dict: Beta e correlação
        """
        # Alinhar séries temporais
        aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
        
        if len(aligned_data) < 30:  # Dados insuficientes
            return {'beta': 1.0, 'correlation': 0.0, 'r_squared': 0.0}
        
        asset_ret = aligned_data.iloc[:, 0]
        market_ret = aligned_data.iloc[:, 1]
        
        # Calcular beta usando regressão linear
        covariance = np.cov(asset_ret, market_ret)[0, 1]
        market_variance = np.var(market_ret)
        
        if market_variance > 0:
            beta = covariance / market_variance
        else:
            beta = 1.0
        
        # Correlação
        correlation = np.corrcoef(asset_ret, market_ret)[0, 1]
        
        # R-squared
        r_squared = correlation ** 2
        
        return {
            'beta': beta,
            'correlation': correlation,
            'r_squared': r_squared
        }
    
    def analyze_tail_risk(self, returns):
        """
        Analisa risco de cauda (eventos extremos)
        
        Args:
            returns (pd.Series): Série de retornos
            
        Returns:
            dict: Métricas de risco de cauda
        """
        # Skewness (assimetria)
        skewness = returns.skew()
        
        # Kurtosis (curtose)
        kurtosis = returns.kurtosis()
        
        # Jarque-Bera test (normalidade)
        jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
        
        # Eventos extremos (> 3 desvios padrão)
        std_dev = returns.std()
        extreme_events = returns[abs(returns) > 3 * std_dev]
        
        # Frequency of extreme events
        extreme_frequency = len(extreme_events) / len(returns) * 100
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'is_normal_distribution': jb_pvalue > 0.05,
            'extreme_events_count': len(extreme_events),
            'extreme_events_frequency_pct': extreme_frequency,
            'largest_gain': returns.max() * 100,
            'largest_loss': returns.min() * 100
        }
    
    def calculate_risk_score(self, volatility_30d, max_drawdown, sharpe_ratio, var_95):
        """
        Calcula um score de risco de 0-10
        
        Args:
            volatility_30d (float): Volatilidade 30 dias
            max_drawdown (float): Maximum drawdown
            sharpe_ratio (float): Sharpe ratio
            var_95 (float): VaR 95%
            
        Returns:
            float: Score de risco (0-10)
        """
        # Normalizar cada métrica para escala 0-10
        
        # Volatilidade (>100% = score máximo)
        vol_score = min(volatility_30d / 10, 10)
        
        # Maximum Drawdown (>50% = score máximo)
        dd_score = min(abs(max_drawdown) / 5, 10)
        
        # Sharpe Ratio (invertido: sharpe baixo = risco alto)
        if sharpe_ratio > 0:
            sharpe_score = max(0, 5 - sharpe_ratio * 2)
        else:
            sharpe_score = min(10, 5 + abs(sharpe_ratio) * 2)
        
        # VaR (>10% daily loss = score máximo)
        var_score = min(abs(var_95) / 1, 10)
        
        # Média ponderada
        risk_score = (vol_score * 0.3 + dd_score * 0.3 + sharpe_score * 0.2 + var_score * 0.2)
        
        return min(max(risk_score, 0), 10)
    
    def classify_risk_level(self, risk_score):
        """
        Classifica o nível de risco baseado no score
        
        Args:
            risk_score (float): Score de risco (0-10)
            
        Returns:
            str: Classificação do risco
        """
        if risk_score <= 3:
            return "Baixo"
        elif risk_score <= 6:
            return "Médio"
        elif risk_score <= 8:
            return "Alto"
        else:
            return "Muito Alto"
    
    def calculate_risk_metrics(self, df):
        """
        Calcula todas as métricas de risco
        
        Args:
            df (pd.DataFrame): DataFrame com dados históricos
            
        Returns:
            dict: Todas as métricas de risco
        """
        prices = df['price']
        returns = prices.pct_change().dropna()
        
        # Volatilidades
        volatilities = self.calculate_volatility(prices)
        
        # VaR e CVaR
        var_cvar = self.calculate_var_cvar(returns)
        
        # Maximum Drawdown
        drawdown_metrics = self.calculate_maximum_drawdown(prices)
        
        # Ratios
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        
        # Análise de cauda
        tail_risk = self.analyze_tail_risk(returns)
        
        # Score de risco
        risk_score = self.calculate_risk_score(
            volatilities.get('volatility_30d', 0),
            drawdown_metrics['max_drawdown'],
            sharpe_ratio,
            var_cvar.get('var_95', 0)
        )
        
        # Classificação do risco
        risk_level = self.classify_risk_level(risk_score)
        
        # Combinar todas as métricas
        all_metrics = {
            **volatilities,
            **var_cvar,
            **drawdown_metrics,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'risk_score': risk_score,
            'risk_level': risk_level,
            **tail_risk
        }
        
        return all_metrics
    
    def generate_risk_report(self, risk_metrics):
        """
        Gera relatório de risco em formato texto
        
        Args:
            risk_metrics (dict): Métricas de risco calculadas
            
        Returns:
            str: Relatório de risco formatado
        """
        report = []
        report.append("=== RELATÓRIO DE ANÁLISE DE RISCO ===\n")
        
        # Resumo geral
        report.append(f"📊 SCORE DE RISCO: {risk_metrics['risk_score']:.1f}/10")
        report.append(f"🎯 CLASSIFICAÇÃO: {risk_metrics['risk_level']}\n")
        
        # Volatilidade
        report.append("📈 VOLATILIDADE:")
        report.append(f"  • 7 dias: {risk_metrics.get('volatility_7d', 0):.2f}%")
        report.append(f"  • 30 dias: {risk_metrics.get('volatility_30d', 0):.2f}%")
        report.append(f"  • 90 dias: {risk_metrics.get('volatility_90d', 0):.2f}%\n")
        
        # Drawdown
        report.append("📉 DRAWDOWN:")
        report.append(f"  • Máximo: {risk_metrics['max_drawdown']:.2f}%")
        report.append(f"  • Atual: {risk_metrics['current_drawdown']:.2f}%")
        report.append(f"  • Duração máxima: {risk_metrics['max_drawdown_duration_days']} dias\n")
        
        # Value at Risk
        report.append("⚠️ VALUE AT RISK:")
        report.append(f"  • VaR 95%: {risk_metrics.get('var_95', 0):.2f}%")
        report.append(f"  • CVaR 95%: {risk_metrics.get('cvar_95', 0):.2f}%\n")
        
        # Ratios de performance
        report.append("📊 RATIOS DE PERFORMANCE:")
        report.append(f"  • Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
        report.append(f"  • Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}\n")
        
        # Análise de distribuição
        report.append("📋 ANÁLISE DE DISTRIBUIÇÃO:")
        report.append(f"  • Assimetria: {risk_metrics['skewness']:.2f}")
        report.append(f"  • Curtose: {risk_metrics['kurtosis']:.2f}")
        report.append(f"  • Distribuição normal: {'Sim' if risk_metrics['is_normal_distribution'] else 'Não'}")
        report.append(f"  • Eventos extremos: {risk_metrics['extreme_events_frequency_pct']:.2f}%\n")
        
        # Recomendações
        report.append("💡 RECOMENDAÇÕES:")
        if risk_metrics['risk_level'] == 'Baixo':
            report.append("  • Risco controlado, adequado para investidores conservadores")
            report.append("  • Considere aumentar a exposição gradualmente")
        elif risk_metrics['risk_level'] == 'Médio':
            report.append("  • Risco moderado, adequado para perfil equilibrado")
            report.append("  • Monitore indicadores de volatilidade")
        elif risk_metrics['risk_level'] == 'Alto':
            report.append("  • Alto risco, apenas para investidores experientes")
            report.append("  • Use stop-loss e position sizing conservador")
        else:
            report.append("  • Risco extremo, evite ou use apenas small caps")
            report.append("  • Considere aguardar menor volatilidade")
        
        return "\n".join(report)
    
    def calculate_position_sizing(self, portfolio_value, risk_tolerance, risk_metrics):
        """
        Calcula tamanho de posição baseado no risco
        
        Args:
            portfolio_value (float): Valor total do portfolio
            risk_tolerance (float): Tolerância ao risco (% do portfolio)
            risk_metrics (dict): Métricas de risco
            
        Returns:
            dict: Recomendações de position sizing
        """
        # Kelly Criterion simplificado
        win_rate = 0.5  # Assumir 50% de trades ganhos (conservador)
        avg_win = abs(risk_metrics.get('var_95', 5))  # Usar VaR como proxy
        avg_loss = abs(risk_metrics.get('var_95', 5))
        
        if avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Limitar a 25%
        else:
            kelly_fraction = 0.05
        
        # Position sizing baseado no risco
        risk_score = risk_metrics['risk_score']
        
        # Ajustar baseado no score de risco
        if risk_score <= 3:
            max_position = min(0.2, risk_tolerance)  # Até 20% para baixo risco
        elif risk_score <= 6:
            max_position = min(0.1, risk_tolerance)  # Até 10% para risco médio
        elif risk_score <= 8:
            max_position = min(0.05, risk_tolerance)  # Até 5% para alto risco
        else:
            max_position = min(0.02, risk_tolerance)  # Até 2% para risco extremo
        
        # Combinar Kelly com limite de risco
        recommended_position = min(kelly_fraction, max_position)
        position_value = portfolio_value * recommended_position
        
        return {
            'recommended_position_pct': recommended_position * 100,
            'recommended_position_value': position_value,
            'kelly_fraction': kelly_fraction * 100,
            'risk_adjusted_limit': max_position * 100,
            'stop_loss_pct': min(15, abs(risk_metrics.get('var_95', 5)) * 2),
            'take_profit_pct': abs(risk_metrics.get('var_95', 5)) * 3
        }
    
    def monitor_risk_alerts(self, current_metrics, historical_metrics):
        """
        Monitora alertas de risco baseado em mudanças nas métricas
        
        Args:
            current_metrics (dict): Métricas atuais
            historical_metrics (dict): Métricas históricas para comparação
            
        Returns:
            list: Lista de alertas de risco
        """
        alerts = []
        
        # Alerta de volatilidade crescente
        if (current_metrics.get('volatility_30d', 0) > 
            historical_metrics.get('volatility_30d', 0) * 1.5):
            alerts.append({
                'type': 'volatility_spike',
                'severity': 'high',
                'message': 'Volatilidade aumentou significativamente nos últimos 30 dias'
            })
        
        # Alerta de drawdown prolongado
        if current_metrics.get('current_drawdown', 0) < -20:
            alerts.append({
                'type': 'extended_drawdown',
                'severity': 'medium',
                'message': f"Drawdown atual de {current_metrics.get('current_drawdown', 0):.1f}%"
            })
        
        # Alerta de Sharpe Ratio deteriorando
        if (current_metrics.get('sharpe_ratio', 0) < 0 and 
            historical_metrics.get('sharpe_ratio', 0) > 0):
            alerts.append({
                'type': 'performance_decline',
                'severity': 'medium',
                'message': 'Sharpe Ratio tornou-se negativo'
            })
        
        # Alerta de risco extremo
        if current_metrics.get('risk_score', 0) > 8:
            alerts.append({
                'type': 'extreme_risk',
                'severity': 'critical',
                'message': 'Score de risco atingiu níveis extremos'
            })
        
        # Alerta de eventos de cauda frequentes
        if current_metrics.get('extreme_events_frequency_pct', 0) > 5:
            alerts.append({
                'type': 'tail_events',
                'severity': 'high',
                'message': 'Frequência alta de eventos extremos detectada'
            })
        
        return alerts