import pandas as pd
import numpy as np
from scipy import stats

class TechnicalIndicators:
    """Classe para cálculo de indicadores técnicos"""
    
    def __init__(self):
        pass
    
    def calculate_rsi(self, prices, period=14):
        """
        Calcula o Relative Strength Index (RSI)
        
        Args:
            prices (pd.Series): Série de preços
            period (int): Período para cálculo (padrão: 14)
            
        Returns:
            pd.Series: Valores do RSI
        """
        delta = prices.diff()
        
        # Separar ganhos e perdas
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calcular médias móveis exponenciais
        avg_gains = gains.ewm(span=period, adjust=False).mean()
        avg_losses = losses.ewm(span=period, adjust=False).mean()
        
        # Calcular RS e RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """
        Calcula o MACD (Moving Average Convergence Divergence)
        
        Args:
            prices (pd.Series): Série de preços
            fast_period (int): Período da EMA rápida
            slow_period (int): Período da EMA lenta
            signal_period (int): Período da linha de sinal
            
        Returns:
            dict: MACD, Signal line e Histogram
        """
        # Calcular EMAs
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        
        # Calcular MACD
        macd = ema_fast - ema_slow
        
        # Calcular linha de sinal
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # Calcular histograma
        histogram = macd - signal
        
        return {
            'macd': macd,
            'macd_signal': signal,
            'macd_histogram': histogram
        }
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """
        Calcula as Bandas de Bollinger
        
        Args:
            prices (pd.Series): Série de preços
            period (int): Período para média móvel
            std_dev (float): Número de desvios padrão
            
        Returns:
            dict: Upper band, Lower band, Middle band (SMA)
        """
        # Calcular média móvel simples
        sma = prices.rolling(window=period).mean()
        
        # Calcular desvio padrão
        rolling_std = prices.rolling(window=period).std()
        
        # Calcular bandas
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        
        return {
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band,
            'bb_width': (upper_band - lower_band) / sma * 100
        }
    
    def calculate_moving_averages(self, prices, periods=[7, 21, 50, 200]):
        """
        Calcula múltiplas médias móveis
        
        Args:
            prices (pd.Series): Série de preços
            periods (list): Lista de períodos para cálculo
            
        Returns:
            dict: Médias móveis para cada período
        """
        moving_averages = {}
        
        for period in periods:
            moving_averages[f'ma_{period}'] = prices.rolling(window=period).mean()
            moving_averages[f'ema_{period}'] = prices.ewm(span=period, adjust=False).mean()
        
        return moving_averages
    
    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """
        Calcula o Oscilador Estocástico
        
        Args:
            high (pd.Series): Preços máximos
            low (pd.Series): Preços mínimos
            close (pd.Series): Preços de fechamento
            k_period (int): Período para %K
            d_period (int): Período para %D
            
        Returns:
            dict: %K e %D
        """
        # Se não temos dados de high/low, usar o preço de fechamento
        if high is None or low is None:
            high = close
            low = close
        
        # Calcular %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # Calcular %D (média móvel de %K)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }
    
    def calculate_williams_r(self, high, low, close, period=14):
        """
        Calcula o Williams %R
        
        Args:
            high (pd.Series): Preços máximos
            low (pd.Series): Preços mínimos
            close (pd.Series): Preços de fechamento
            period (int): Período de cálculo
            
        Returns:
            pd.Series: Williams %R
        """
        if high is None or low is None:
            high = close
            low = close
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    def calculate_atr(self, high, low, close, period=14):
        """
        Calcula o Average True Range (ATR)
        
        Args:
            high (pd.Series): Preços máximos
            low (pd.Series): Preços mínimos
            close (pd.Series): Preços de fechamento
            period (int): Período de cálculo
            
        Returns:
            pd.Series: ATR
        """
        if high is None or low is None:
            # Se não temos dados OHLC, calcular com base no preço
            price_change = close.diff().abs()
            return price_change.rolling(window=period).mean()
        
        # Calcular True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calcular ATR (média móvel do True Range)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_volume_indicators(self, prices, volumes):
        """
        Calcula indicadores baseados em volume
        
        Args:
            prices (pd.Series): Série de preços
            volumes (pd.Series): Série de volumes
            
        Returns:
            dict: Indicadores de volume
        """
        indicators = {}
        
        # On-Balance Volume (OBV)
        price_change = prices.diff()
        obv = np.where(price_change > 0, volumes, 
                      np.where(price_change < 0, -volumes, 0))
        indicators['obv'] = pd.Series(obv, index=prices.index).cumsum()
        
        # Volume Rate of Change
        indicators['volume_roc'] = volumes.pct_change(periods=10) * 100
        
        # Volume Moving Average
        indicators['volume_ma'] = volumes.rolling(window=20).mean()
        
        # Volume Ratio
        indicators['volume_ratio'] = volumes / indicators['volume_ma']
        
        return indicators
    
    def calculate_momentum_indicators(self, prices):
        """
        Calcula indicadores de momentum
        
        Args:
            prices (pd.Series): Série de preços
            
        Returns:
            dict: Indicadores de momentum
        """
        indicators = {}
        
        # Rate of Change (ROC)
        indicators['roc_10'] = prices.pct_change(periods=10) * 100
        indicators['roc_20'] = prices.pct_change(periods=20) * 100
        
        # Momentum
        indicators['momentum_10'] = prices / prices.shift(10) * 100
        indicators['momentum_20'] = prices / prices.shift(20) * 100
        
        # Price Velocity
        indicators['velocity'] = prices.diff()
        indicators['acceleration'] = indicators['velocity'].diff()
        
        return indicators
    
    def calculate_support_resistance(self, prices, window=20, min_touches=2):
        """
        Identifica níveis de suporte e resistência
        
        Args:
            prices (pd.Series): Série de preços
            window (int): Janela para identificação de picos/vales
            min_touches (int): Mínimo de toques para confirmar nível
            
        Returns:
            dict: Níveis de suporte e resistência
        """
        # Identificar máximos e mínimos locais
        highs = prices.rolling(window=window, center=True).max() == prices
        lows = prices.rolling(window=window, center=True).min() == prices
        
        # Obter valores dos picos e vales
        peak_values = prices[highs].dropna()
        valley_values = prices[lows].dropna()
        
        # Agrupar níveis próximos (dentro de 2% de diferença)
        tolerance = 0.02
        
        resistance_levels = []
        support_levels = []
        
        # Processar níveis de resistência
        for level in peak_values:
            similar_levels = peak_values[abs(peak_values - level) / level <= tolerance]
            if len(similar_levels) >= min_touches:
                resistance_levels.append(similar_levels.mean())
        
        # Processar níveis de suporte
        for level in valley_values:
            similar_levels = valley_values[abs(valley_values - level) / level <= tolerance]
            if len(similar_levels) >= min_touches:
                support_levels.append(similar_levels.mean())
        
        return {
            'resistance_levels': list(set(resistance_levels)),
            'support_levels': list(set(support_levels))
        }
    
    def calculate_trend_indicators(self, prices):
        """
        Calcula indicadores de tendência
        
        Args:
            prices (pd.Series): Série de preços
            
        Returns:
            dict: Indicadores de tendência
        """
        indicators = {}
        
        # Parabolic SAR (simplificado)
        high = prices
        low = prices
        
        # ADX (Average Directional Index) - versão simplificada
        price_change = prices.diff()
        up_move = price_change.where(price_change > 0, 0)
        down_move = -price_change.where(price_change < 0, 0)
        
        # Suavizar com EMA
        up_smooth = up_move.ewm(span=14).mean()
        down_smooth = down_move.ewm(span=14).mean()
        
        # Calcular DI+ e DI-
        atr_simple = prices.rolling(window=14).std()  # Aproximação do ATR
        di_plus = 100 * up_smooth / atr_simple
        di_minus = 100 * down_smooth / atr_simple
        
        # Calcular ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.ewm(span=14).mean()
        
        indicators['adx'] = adx
        indicators['di_plus'] = di_plus
        indicators['di_minus'] = di_minus
        
        # Aroon Oscillator
        period = 25
        high_roll = prices.rolling(window=period)
        low_roll = prices.rolling(window=period)
        
        aroon_up = 100 * (period - (high_roll.apply(lambda x: period - 1 - x.argmax()))) / period
        aroon_down = 100 * (period - (low_roll.apply(lambda x: period - 1 - x.argmin()))) / period
        
        indicators['aroon_up'] = aroon_up
        indicators['aroon_down'] = aroon_down
        indicators['aroon_oscillator'] = aroon_up - aroon_down
        
        return indicators
    
    def calculate_all_indicators(self, df):
        """
        Calcula todos os indicadores técnicos
        
        Args:
            df (pd.DataFrame): DataFrame com dados OHLCV
            
        Returns:
            pd.DataFrame: DataFrame com todos os indicadores
        """
        result_df = df.copy()
        
        # Usar preço como proxy para high/low se não existirem
        price = df['price']
        volume = df['volume'] if 'volume' in df.columns else None
        high = df.get('high', price)
        low = df.get('low', price)
        
        # RSI
        result_df['rsi'] = self.calculate_rsi(price)
        
        # MACD
        macd_data = self.calculate_macd(price)
        for key, value in macd_data.items():
            result_df[key] = value
        
        # Bandas de Bollinger
        bb_data = self.calculate_bollinger_bands(price)
        for key, value in bb_data.items():
            result_df[key] = value
        
        # Médias móveis
        ma_data = self.calculate_moving_averages(price)
        for key, value in ma_data.items():
            result_df[key] = value
        
        # Estocástico
        stoch_data = self.calculate_stochastic(high, low, price)
        for key, value in stoch_data.items():
            result_df[key] = value
        
        # Williams %R
        result_df['williams_r'] = self.calculate_williams_r(high, low, price)
        
        # ATR
        result_df['atr'] = self.calculate_atr(high, low, price)
        
        # Indicadores de volume
        if volume is not None:
            volume_data = self.calculate_volume_indicators(price, volume)
            for key, value in volume_data.items():
                result_df[key] = value
        
        # Indicadores de momentum
        momentum_data = self.calculate_momentum_indicators(price)
        for key, value in momentum_data.items():
            result_df[key] = value
        
        # Indicadores de tendência
        trend_data = self.calculate_trend_indicators(price)
        for key, value in trend_data.items():
            result_df[key] = value
        
        # Níveis de suporte e resistência
        sr_data = self.calculate_support_resistance(price)
        
        # Adicionar como metadados (não como colunas pois são listas)
        result_df.attrs['support_levels'] = sr_data['support_levels']
        result_df.attrs['resistance_levels'] = sr_data['resistance_levels']
        
        return result_df
    
    def get_trading_signals(self, df):
        """
        Gera sinais de trading baseado nos indicadores
        
        Args:
            df (pd.DataFrame): DataFrame com indicadores calculados
            
        Returns:
            pd.DataFrame: DataFrame com sinais de compra/venda
        """
        signals = pd.DataFrame(index=df.index)
        
        # Sinal RSI
        signals['rsi_oversold'] = df['rsi'] < 30  # Sinal de compra
        signals['rsi_overbought'] = df['rsi'] > 70  # Sinal de venda
        
        # Sinal MACD
        signals['macd_bullish'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        signals['macd_bearish'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # Sinal Bandas de Bollinger
        signals['bb_oversold'] = df['price'] < df['bb_lower']
        signals['bb_overbought'] = df['price'] > df['bb_upper']
        
        # Sinal de cruzamento de médias móveis
        if 'ma_7' in df.columns and 'ma_21' in df.columns:
            signals['ma_bullish'] = (df['ma_7'] > df['ma_21']) & (df['ma_7'].shift(1) <= df['ma_21'].shift(1))
            signals['ma_bearish'] = (df['ma_7'] < df['ma_21']) & (df['ma_7'].shift(1) >= df['ma_21'].shift(1))
        
        # Sinal combinado
        buy_signals = signals[['rsi_oversold', 'macd_bullish', 'bb_oversold', 'ma_bullish']].sum(axis=1)
        sell_signals = signals[['rsi_overbought', 'macd_bearish', 'bb_overbought', 'ma_bearish']].sum(axis=1)
        
        signals['buy_strength'] = buy_signals
        signals['sell_strength'] = sell_signals
        signals['net_signal'] = buy_signals - sell_signals
        
        return signals