import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools

warnings.filterwarnings('ignore')

class CryptoForecaster:
    """Classe para previsão de preços de criptomoedas usando modelos de séries temporais"""
    
    def __init__(self):
        self.model = None
        self.model_fit = None
        self.best_params = None
        
    def check_stationarity(self, ts, significance_level=0.05):
        """
        Verifica se a série temporal é estacionária usando o teste ADF
        
        Args:
            ts (pd.Series): Série temporal
            significance_level (float): Nível de significância
            
        Returns:
            dict: Resultado do teste de estacionariedade
        """
        try:
            result = adfuller(ts.dropna())
            
            return {
                'is_stationary': result[1] <= significance_level,
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'conclusion': 'Estacionária' if result[1] <= significance_level else 'Não estacionária'
            }
        except Exception as e:
            return {
                'is_stationary': False,
                'error': str(e)
            }
    
    def make_stationary(self, ts):
        """
        Torna a série temporal estacionária através de diferenciação
        
        Args:
            ts (pd.Series): Série temporal original
            
        Returns:
            tuple: (série estacionária, número de diferenciações)
        """
        diff_series = ts.copy()
        n_diff = 0
        max_diff = 3
        
        while n_diff < max_diff:
            stationarity = self.check_stationarity(diff_series)
            
            if stationarity['is_stationary']:
                break
            
            diff_series = diff_series.diff().dropna()
            n_diff += 1
        
        return diff_series, n_diff
    
    def auto_arima_params(self, ts, max_p=3, max_d=2, max_q=3, seasonal=False):
        """
        Encontra os melhores parâmetros ARIMA automaticamente
        
        Args:
            ts (pd.Series): Série temporal
            max_p, max_d, max_q (int): Valores máximos para p, d, q
            seasonal (bool): Se deve considerar sazonalidade
            
        Returns:
            dict: Melhores parâmetros encontrados
        """
        best_aic = float('inf')
        best_params = None
        best_seasonal_params = None
        
        # Parâmetros não sazonais
        p_values = range(0, max_p + 1)
        d_values = range(0, max_d + 1)
        q_values = range(0, max_q + 1)
        
        # Parâmetros sazonais (se aplicável)
        if seasonal:
            seasonal_p = range(0, 2)
            seasonal_d = range(0, 2)
            seasonal_q = range(0, 2)
            m_values = [7, 30]  # Sazonalidade semanal ou mensal
        else:
            seasonal_p = [0]
            seasonal_d = [0]
            seasonal_q = [0]
            m_values = [0]
        
        total_combinations = len(p_values) * len(d_values) * len(q_values)
        if seasonal:
            total_combinations *= len(seasonal_p) * len(seasonal_d) * len(seasonal_q) * len(m_values)
        
        # Limitar combinações para evitar timeout
        max_combinations = 50
        tested_combinations = 0
        
        for p, d, q in itertools.product(p_values, d_values, q_values):
            if tested_combinations >= max_combinations:
                break
                
            for sp, sd, sq, m in itertools.product(seasonal_p, seasonal_d, seasonal_q, m_values):
                if tested_combinations >= max_combinations:
                    break
                
                try:
                    if seasonal and m > 0:
                        model = SARIMAX(ts, 
                                      order=(p, d, q),
                                      seasonal_order=(sp, sd, sq, m),
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
                    else:
                        model = ARIMA(ts, order=(p, d, q))
                    
                    model_fit = model.fit(disp=False)
                    aic = model_fit.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        if seasonal and m > 0:
                            best_seasonal_params = (sp, sd, sq, m)
                        else:
                            best_seasonal_params = None
                    
                    tested_combinations += 1
                    
                except:
                    continue
        
        return {
            'order': best_params if best_params else (1, 1, 1),
            'seasonal_order': best_seasonal_params,
            'aic': best_aic,
            'tested_combinations': tested_combinations
        }
    
    def prepare_exogenous_variables(self, df, forecast_days):
        """
        Prepara variáveis exógenas para o modelo SARIMAX
        
        Args:
            df (pd.DataFrame): DataFrame com dados históricos
            forecast_days (int): Número de dias para previsão
            
        Returns:
            tuple: (variáveis exógenas treino, variáveis exógenas previsão)
        """
        # Usar indicadores técnicos como variáveis exógenas
        exog_columns = []
        
        # Selecionar indicadores mais relevantes
        if 'rsi' in df.columns:
            exog_columns.append('rsi')
        if 'macd' in df.columns:
            exog_columns.append('macd')
        if 'volume' in df.columns:
            exog_columns.append('volume')
        if 'bb_width' in df.columns:
            exog_columns.append('bb_width')
        
        if not exog_columns:
            return None, None
        
        # Preparar dados de treino
        exog_train = df[exog_columns].fillna(method='forward').fillna(method='backward')
        
        # Preparar dados para previsão (usar valores médios recentes)
        recent_values = exog_train.tail(30).mean()
        exog_forecast = pd.DataFrame([recent_values] * forecast_days, 
                                   columns=exog_columns)
        
        return exog_train, exog_forecast
    
    def forecast_sarimax(self, df, forecast_days=30, use_exog=True):
        """
        Realiza previsão usando modelo SARIMAX
        
        Args:
            df (pd.DataFrame): DataFrame com dados históricos
            forecast_days (int): Número de dias para previsão
            use_exog (bool): Se deve usar variáveis exógenas
            
        Returns:
            dict: Resultados da previsão
        """
        try:
            # Preparar dados
            ts = df['price'].copy()
            
            # Verificar se há dados suficientes
            if len(ts) < 50:
                raise ValueError("Dados insuficientes para previsão (mínimo: 50 pontos)")
            
            # Preparar variáveis exógenas
            exog_train, exog_forecast = None, None
            if use_exog:
                exog_train, exog_forecast = self.prepare_exogenous_variables(df, forecast_days)
            
            # Log transform para estabilizar variância
            ts_log = np.log(ts)
            
            # Encontrar melhores parâmetros
            params = self.auto_arima_params(ts_log, max_p=2, max_d=2, max_q=2, seasonal=False)
            
            # Ajustar modelo
            try:
                if exog_train is not None:
                    model = SARIMAX(ts_log,
                                  exog=exog_train,
                                  order=params['order'],
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
                else:
                    model = ARIMA(ts_log, order=params['order'])
                
                self.model_fit = model.fit(disp=False)
                self.best_params = params
                
            except:
                # Fallback para modelo mais simples
                model = ARIMA(ts_log, order=(1, 1, 1))
                self.model_fit = model.fit(disp=False)
                exog_forecast = None
            
            # Fazer previsão
            if exog_forecast is not None:
                try:
                    forecast_result = self.model_fit.forecast(steps=forecast_days, exog=exog_forecast)
                    conf_int = self.model_fit.get_forecast(steps=forecast_days, exog=exog_forecast).conf_int()
                except:
                    forecast_result = self.model_fit.forecast(steps=forecast_days)
                    conf_int = self.model_fit.get_forecast(steps=forecast_days).conf_int()
            else:
                forecast_result = self.model_fit.forecast(steps=forecast_days)
                conf_int = self.model_fit.get_forecast(steps=forecast_days).conf_int()
            
            # Converter de volta da escala log
            forecast_prices = np.exp(forecast_result)
            lower_ci = np.exp(conf_int.iloc[:, 0])
            upper_ci = np.exp(conf_int.iloc[:, 1])
            
            # Validação in-sample
            fitted_values = np.exp(self.model_fit.fittedvalues)
            actual_values = ts[fitted_values.index]
            
            mae = mean_absolute_error(actual_values, fitted_values)
            rmse = np.sqrt(mean_squared_error(actual_values, fitted_values))
            mape = np.mean(np.abs((actual_values - fitted_values) / actual_values)) * 100
            
            return {
                'forecast': pd.Series(forecast_prices.values, 
                                    index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                                                       periods=forecast_days, freq='D')),
                'lower_ci': pd.Series(lower_ci.values,
                                    index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                                                       periods=forecast_days, freq='D')),
                'upper_ci': pd.Series(upper_ci.values,
                                    index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                                                       periods=forecast_days, freq='D')),
                'model_params': params,
                'model_summary': {
                    'aic': self.model_fit.aic,
                    'bic': self.model_fit.bic,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape
                },
                'fitted_values': fitted_values,
                'residuals': self.model_fit.resid
            }
            
        except Exception as e:
            # Fallback para previsão simples baseada em tendência
            return self.simple_trend_forecast(df, forecast_days, error=str(e))
    
    def simple_trend_forecast(self, df, forecast_days, error=None):
        """
        Previsão simples baseada em tendência linear (fallback)
        
        Args:
            df (pd.DataFrame): DataFrame com dados históricos
            forecast_days (int): Número de dias para previsão
            error (str): Mensagem de erro do método principal
            
        Returns:
            dict: Resultados da previsão simples
        """
        ts = df['price'].copy()
        
        # Calcular tendência linear dos últimos 30 dias
        recent_data = ts.tail(30)
        x = np.arange(len(recent_data))
        y = recent_data.values
        
        # Regressão linear simples
        slope, intercept = np.polyfit(x, y, 1)
        
        # Gerar previsão
        forecast_x = np.arange(len(recent_data), len(recent_data) + forecast_days)
        forecast_values = slope * forecast_x + intercept
        
        # Adicionar incerteza baseada na volatilidade histórica
        volatility = ts.pct_change().std()
        uncertainty = forecast_values * volatility * np.sqrt(forecast_x - len(recent_data) + 1)
        
        forecast_dates = pd.date_range(
            start=df.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        return {
            'forecast': pd.Series(forecast_values, index=forecast_dates),
            'lower_ci': pd.Series(forecast_values - 1.96 * uncertainty, index=forecast_dates),
            'upper_ci': pd.Series(forecast_values + 1.96 * uncertainty, index=forecast_dates),
            'model_params': {'method': 'linear_trend', 'slope': slope, 'intercept': intercept},
            'model_summary': {
                'method': 'Simple Linear Trend (Fallback)',
                'error': error,
                'volatility': volatility
            },
            'fitted_values': None,
            'residuals': None
        }
    
    def validate_forecast_accuracy(self, df, test_days=30, forecast_days=7):
        """
        Valida a precisão do modelo usando backtesting
        
        Args:
            df (pd.DataFrame): DataFrame com dados históricos
            test_days (int): Número de dias para teste
            forecast_days (int): Dias de previsão para cada teste
            
        Returns:
            dict: Métricas de validação
        """
        if len(df) < test_days + forecast_days + 50:
            return {'error': 'Dados insuficientes para validação'}
        
        errors = []
        actual_values = []
        predicted_values = []
        
        # Realizar múltiplas previsões
        for i in range(0, test_days, forecast_days):
            # Dividir dados
            train_end = len(df) - test_days + i
            train_data = df.iloc[:train_end]
            
            if len(train_data) < 50:
                continue
            
            # Fazer previsão
            result = self.forecast_sarimax(train_data, forecast_days, use_exog=False)
            
            # Comparar com valores reais
            actual_period = df.iloc[train_end:train_end + forecast_days]['price']
            predicted_period = result['forecast'][:len(actual_period)]
            
            if len(predicted_period) > 0 and len(actual_period) > 0:
                # Calcular erro para o primeiro dia (mais confiável)
                if len(actual_period) > 0 and len(predicted_period) > 0:
                    actual_values.append(actual_period.iloc[0])
                    predicted_values.append(predicted_period.iloc[0])
                    errors.append(abs(actual_period.iloc[0] - predicted_period.iloc[0]) / actual_period.iloc[0])
        
        if len(errors) > 0:
            return {
                'mean_absolute_percentage_error': np.mean(errors) * 100,
                'median_absolute_percentage_error': np.median(errors) * 100,
                'accuracy_score': (1 - np.mean(errors)) * 100,
                'number_of_tests': len(errors),
                'directional_accuracy': self.calculate_directional_accuracy(actual_values, predicted_values)
            }
        else:
            return {'error': 'Não foi possível realizar validação'}
    
    def calculate_directional_accuracy(self, actual, predicted):
        """
        Calcula a precisão direcional (se previu corretamente a direção do movimento)
        
        Args:
            actual (list): Valores reais
            predicted (list): Valores previstos
            
        Returns:
            float: Precisão direcional (%)
        """
        if len(actual) < 2 or len(predicted) < 2:
            return 0
        
        actual_directions = [1 if actual[i] > actual[i-1] else 0 for i in range(1, len(actual))]
        predicted_directions = [1 if predicted[i] > predicted[i-1] else 0 for i in range(1, len(predicted))]
        
        correct_directions = sum(1 for a, p in zip(actual_directions, predicted_directions) if a == p)
        
        return (correct_directions / len(actual_directions)) * 100 if actual_directions else 0