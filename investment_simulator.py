import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class InvestmentSimulator:
    """Classe para simulação de investimentos em criptomoedas"""
    
    def __init__(self):
        self.transaction_fee = 0.001  # 0.1% fee padrão
        self.slippage = 0.002  # 0.2% slippage padrão
    
    def simulate_investment(self, historical_data, forecast_result, initial_investment, holding_period):
        """
        Simula um investimento baseado na previsão
        
        Args:
            historical_data (pd.DataFrame): Dados históricos
            forecast_result (dict): Resultado da previsão
            initial_investment (float): Valor inicial do investimento (USD)
            holding_period (int): Período de manutenção em dias
            
        Returns:
            dict: Resultados da simulação
        """
        current_price = historical_data['price'].iloc[-1]
        
        # Calcular quantidade de moedas que pode comprar
        effective_investment = initial_investment * (1 - self.transaction_fee)
        coins_purchased = effective_investment / (current_price * (1 + self.slippage))
        
        # Preço previsto no final do período
        if holding_period <= len(forecast_result['forecast']):
            predicted_final_price = forecast_result['forecast'].iloc[holding_period - 1]
            lower_bound = forecast_result['lower_ci'].iloc[holding_period - 1]
            upper_bound = forecast_result['upper_ci'].iloc[holding_period - 1]
        else:
            predicted_final_price = forecast_result['forecast'].iloc[-1]
            lower_bound = forecast_result['lower_ci'].iloc[-1]
            upper_bound = forecast_result['upper_ci'].iloc[-1]
        
        # Calcular valor final (considerando taxas de venda)
        gross_final_value = coins_purchased * predicted_final_price
        final_value = gross_final_value * (1 - self.transaction_fee) * (1 - self.slippage)
        
        # Calcular cenários otimista e pessimista
        optimistic_value = coins_purchased * upper_bound * (1 - self.transaction_fee) * (1 - self.slippage)
        pessimistic_value = coins_purchased * lower_bound * (1 - self.transaction_fee) * (1 - self.slippage)
        
        # Calcular métricas
        profit_loss = final_value - initial_investment
        roi_percentage = (profit_loss / initial_investment) * 100
        
        optimistic_profit = optimistic_value - initial_investment
        optimistic_roi = (optimistic_profit / initial_investment) * 100
        
        pessimistic_profit = pessimistic_value - initial_investment
        pessimistic_roi = (pessimistic_profit / initial_investment) * 100
        
        # Calcular retorno anualizado
        daily_return = (final_value / initial_investment) ** (1 / holding_period) - 1
        annualized_return = (1 + daily_return) ** 365 - 1
        
        return {
            'initial_investment': initial_investment,
            'current_price': current_price,
            'predicted_final_price': predicted_final_price,
            'coins_purchased': coins_purchased,
            'final_value': final_value,
            'estimated_profit_loss': profit_loss,
            'roi_percentage': roi_percentage,
            'annualized_return': annualized_return * 100,
            'holding_period_days': holding_period,
            'scenarios': {
                'optimistic': {
                    'final_value': optimistic_value,
                    'profit_loss': optimistic_profit,
                    'roi_percentage': optimistic_roi
                },
                'pessimistic': {
                    'final_value': pessimistic_value,
                    'profit_loss': pessimistic_profit,
                    'roi_percentage': pessimistic_roi
                }
            },
            'transaction_costs': {
                'buy_fee': initial_investment * self.transaction_fee,
                'sell_fee': gross_final_value * self.transaction_fee,
                'slippage_cost': (current_price * self.slippage * coins_purchased) + 
                               (predicted_final_price * self.slippage * coins_purchased),
                'total_costs': initial_investment - effective_investment + 
                              gross_final_value - final_value
            }
        }
    
    def simulate_dca_strategy(self, historical_data, forecast_result, total_investment, 
                             frequency_days=7, holding_period=30):
        """
        Simula estratégia de Dollar Cost Averaging (DCA)
        
        Args:
            historical_data (pd.DataFrame): Dados históricos
            forecast_result (dict): Resultado da previsão
            total_investment (float): Investimento total
            frequency_days (int): Frequência de compra em dias
            holding_period (int): Período total em dias
            
        Returns:
            dict: Resultados da simulação DCA
        """
        num_purchases = holding_period // frequency_days
        investment_per_period = total_investment / num_purchases
        
        total_coins = 0
        total_invested = 0
        purchase_history = []
        
        current_price = historical_data['price'].iloc[-1]
        
        # Simular compras periódicas
        for i in range(num_purchases):
            day_index = i * frequency_days
            
            if day_index < len(forecast_result['forecast']):
                # Usar preço previsto para compras futuras
                purchase_price = forecast_result['forecast'].iloc[day_index]
            else:
                # Usar último preço previsto
                purchase_price = forecast_result['forecast'].iloc[-1]
            
            # Aplicar taxas e slippage
            effective_investment = investment_per_period * (1 - self.transaction_fee)
            coins_bought = effective_investment / (purchase_price * (1 + self.slippage))
            
            total_coins += coins_bought
            total_invested += investment_per_period
            
            purchase_history.append({
                'day': day_index,
                'price': purchase_price,
                'amount_invested': investment_per_period,
                'coins_purchased': coins_bought,
                'cumulative_coins': total_coins
            })
        
        # Calcular valor final
        final_price = forecast_result['forecast'].iloc[min(holding_period - 1, len(forecast_result['forecast']) - 1)]
        gross_final_value = total_coins * final_price
        final_value = gross_final_value * (1 - self.transaction_fee) * (1 - self.slippage)
        
        average_purchase_price = total_invested / total_coins if total_coins > 0 else 0
        profit_loss = final_value - total_invested
        roi_percentage = (profit_loss / total_invested) * 100 if total_invested > 0 else 0
        
        return {
            'strategy': 'Dollar Cost Averaging',
            'total_investment': total_invested,
            'final_value': final_value,
            'total_coins': total_coins,
            'average_purchase_price': average_purchase_price,
            'final_price': final_price,
            'profit_loss': profit_loss,
            'roi_percentage': roi_percentage,
            'number_of_purchases': num_purchases,
            'purchase_frequency_days': frequency_days,
            'purchase_history': purchase_history
        }
    
    def simulate_portfolio_allocation(self, allocations, forecast_results, total_investment):
        """
        Simula portfolio com múltiplas criptomoedas
        
        Args:
            allocations (dict): {'crypto_name': percentage} - soma deve ser 100
            forecast_results (dict): {'crypto_name': forecast_result}
            total_investment (float): Investimento total
            
        Returns:
            dict: Resultados da simulação do portfolio
        """
        portfolio_results = {}
        total_final_value = 0
        total_invested = 0
        
        for crypto, allocation_pct in allocations.items():
            if crypto not in forecast_results:
                continue
            
            # Calcular investimento para esta crypto
            crypto_investment = total_investment * (allocation_pct / 100)
            
            # Simular investimento individual (usando dados fictícios para demonstração)
            # Em implementação real, precisaria dos dados históricos de cada crypto
            forecast = forecast_results[crypto]
            
            # Assumir preço atual (seria obtido dos dados históricos reais)
            current_price = 100  # Placeholder - deveria vir dos dados reais
            predicted_price = forecast['forecast'].iloc[-1] if len(forecast['forecast']) > 0 else current_price
            
            # Calcular resultado
            effective_investment = crypto_investment * (1 - self.transaction_fee)
            coins = effective_investment / (current_price * (1 + self.slippage))
            final_value = coins * predicted_price * (1 - self.transaction_fee) * (1 - self.slippage)
            
            portfolio_results[crypto] = {
                'allocation_percentage': allocation_pct,
                'investment_amount': crypto_investment,
                'final_value': final_value,
                'profit_loss': final_value - crypto_investment,
                'roi_percentage': ((final_value - crypto_investment) / crypto_investment) * 100
            }
            
            total_final_value += final_value
            total_invested += crypto_investment
        
        portfolio_profit_loss = total_final_value - total_invested
        portfolio_roi = (portfolio_profit_loss / total_invested) * 100 if total_invested > 0 else 0
        
        return {
            'portfolio_composition': portfolio_results,
            'total_investment': total_invested,
            'total_final_value': total_final_value,
            'portfolio_profit_loss': portfolio_profit_loss,
            'portfolio_roi_percentage': portfolio_roi,
            'best_performer': max(portfolio_results.items(), 
                                key=lambda x: x[1]['roi_percentage'])[0] if portfolio_results else None,
            'worst_performer': min(portfolio_results.items(), 
                                 key=lambda x: x[1]['roi_percentage'])[0] if portfolio_results else None
        }
    
    def calculate_risk_adjusted_returns(self, historical_data, simulation_result):
        """
        Calcula retornos ajustados ao risco
        
        Args:
            historical_data (pd.DataFrame): Dados históricos
            simulation_result (dict): Resultado da simulação
            
        Returns:
            dict: Métricas de risco-retorno
        """
        returns = historical_data['returns'].dropna()
        
        # Calcular métricas de risco
        volatility = returns.std() * np.sqrt(365)  # Volatilidade anualizada
        
        # Sharpe Ratio (assumindo risk-free rate de 2% ao ano)
        risk_free_rate = 0.02
        excess_return = simulation_result['annualized_return'] / 100 - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino Ratio (considera apenas downside risk)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        calmar_ratio = (simulation_result['annualized_return'] / 100) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(returns, 5)
        
        # Expected Shortfall (CVaR)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown * 100,  # Em percentual
            'volatility_annualized': volatility * 100,  # Em percentual
            'var_95': var_95 * 100,  # Em percentual
            'cvar_95': cvar_95 * 100,  # Em percentual
            'risk_score': self.calculate_risk_score(volatility, max_drawdown, sharpe_ratio)
        }
    
    def calculate_risk_score(self, volatility, max_drawdown, sharpe_ratio):
        """
        Calcula um score de risco de 0-10 (0 = muito baixo risco, 10 = muito alto risco)
        
        Args:
            volatility (float): Volatilidade anualizada
            max_drawdown (float): Maximum drawdown
            sharpe_ratio (float): Sharpe ratio
            
        Returns:
            float: Score de risco
        """
        # Normalizar métricas (valores típicos para crypto)
        vol_score = min(volatility * 10, 10)  # Volatilidade > 100% = score máximo
        dd_score = min(abs(max_drawdown) * 10, 10)  # Drawdown > 100% = score máximo
        sharpe_score = max(0, 10 - (sharpe_ratio + 1) * 2)  # Sharpe < -1 = score máximo
        
        # Média ponderada
        risk_score = (vol_score * 0.4 + dd_score * 0.4 + sharpe_score * 0.2)
        
        return min(max(risk_score, 0), 10)
    
    def generate_investment_recommendation(self, simulation_result, risk_metrics):
        """
        Gera recomendação de investimento baseada nos resultados
        
        Args:
            simulation_result (dict): Resultado da simulação
            risk_metrics (dict): Métricas de risco
            
        Returns:
            dict: Recomendação de investimento
        """
        roi = simulation_result['roi_percentage']
        risk_score = risk_metrics['risk_score']
        sharpe_ratio = risk_metrics['sharpe_ratio']
        
        # Determinar recomendação
        if roi > 20 and risk_score < 5 and sharpe_ratio > 1:
            recommendation = "COMPRA FORTE"
            confidence = "Alta"
            reason = "Alto potencial de retorno com risco controlado"
        elif roi > 10 and risk_score < 7:
            recommendation = "COMPRA"
            confidence = "Média"
            reason = "Potencial de retorno positivo com risco moderado"
        elif roi > 0 and risk_score < 8:
            recommendation = "COMPRA FRACA"
            confidence = "Baixa"
            reason = "Pequeno potencial de ganho"
        elif roi < 0 and risk_score > 7:
            recommendation = "VENDA"
            confidence = "Alta"
            reason = "Alto risco com expectativa de perda"
        else:
            recommendation = "NEUTRO"
            confidence = "Baixa"
            reason = "Incerteza nos resultados esperados"
        
        # Calcular tamanho de posição recomendado (% do portfolio)
        if risk_score < 3:
            position_size = min(20, max(5, roi / 2))  # 5-20% para baixo risco
        elif risk_score < 7:
            position_size = min(10, max(2, roi / 3))  # 2-10% para risco médio
        else:
            position_size = min(5, max(1, roi / 5))   # 1-5% para alto risco
        
        return {
            'recommendation': recommendation,
            'confidence_level': confidence,
            'reason': reason,
            'suggested_position_size_pct': max(0, position_size),
            'risk_level': 'Baixo' if risk_score < 4 else 'Médio' if risk_score < 7 else 'Alto',
            'investment_horizon': 'Curto prazo' if roi > 15 else 'Longo prazo',
            'key_risks': self.identify_key_risks(risk_metrics),
            'profit_target': roi * 0.8,  # Target de 80% do ROI previsto
            'stop_loss': -min(15, abs(risk_metrics['var_95']))  # Stop loss baseado no VaR
        }
    
    def identify_key_risks(self, risk_metrics):
        """
        Identifica os principais riscos do investimento
        
        Args:
            risk_metrics (dict): Métricas de risco
            
        Returns:
            list: Lista dos principais riscos
        """
        risks = []
        
        if risk_metrics['volatility_annualized'] > 80:
            risks.append("Alta volatilidade - movimentos de preço extremos")
        
        if risk_metrics['max_drawdown'] < -50:
            risks.append("Possibilidade de grandes perdas consecutivas")
        
        if risk_metrics['sharpe_ratio'] < 0:
            risks.append("Retorno ajustado ao risco negativo")
        
        if abs(risk_metrics['var_95']) > 10:
            risks.append("Alto risco de perdas diárias significativas")
        
        if risk_metrics['risk_score'] > 8:
            risks.append("Investimento de muito alto risco")
        
        if not risks:
            risks.append("Riscos dentro de parâmetros aceitáveis")
        
        return risks