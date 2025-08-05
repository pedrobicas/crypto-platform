import requests
import pandas as pd
import numpy as np
import time
import json
import os

class CryptoDataCollector:
    """Classe para coleta de dados de criptomoedas via API CoinGecko com cache e retry"""

    def __init__(self, cache_dir='cache'):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoAnalyzer/1.0'
        })
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _cache_filepath(self, crypto_id, days):
        return os.path.join(self.cache_dir, f"{crypto_id}_{days}.json")

    def _load_cache(self, crypto_id, days):
        path = self._cache_filepath(crypto_id, days)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def _save_cache(self, crypto_id, days, data):
        path = self._cache_filepath(crypto_id, days)
        with open(path, 'w') as f:
            json.dump(data, f)

    def get_historical_data(self, crypto_id, days=365):
        """
        Coleta dados históricos com cache e retry em caso de 429.
        Retorna DataFrame limpo e validado.
        """
        cached_data = self._load_cache(crypto_id, days)
        if cached_data:
            print(f"Carregando dados do cache para {crypto_id} ({days} dias)")
            data = cached_data
        else:
            print(f"Buscando dados na API CoinGecko para {crypto_id} ({days} dias)")
            url = f"{self.base_url}/coins/{crypto_id}/market_chart"
            params = {'vs_currency': 'usd', 'days': days, 'interval': 'daily'}

            while True:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code == 429:
                    print("Limite de requisições atingido. Aguardando 60 segundos para tentar novamente...")
                    time.sleep(60)
                elif response.status_code == 200:
                    data = response.json()
                    self._save_cache(crypto_id, days, data)
                    break
                else:
                    try:
                        response.raise_for_status()
                    except Exception as e:
                        print(f"Erro na requisição: {e}")
                        return None

        # Processar os dados em DataFrame
        try:
            prices = data['prices']
            volumes = data['total_volumes']
            market_caps = data['market_caps']

            df = pd.DataFrame({
                'timestamp': [item[0] for item in prices],
                'price': [item[1] for item in prices],
                'volume': [item[1] for item in volumes],
                'market_cap': [item[1] for item in market_caps]
            })

            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('date', inplace=True)
            df.drop('timestamp', axis=1, inplace=True)

            # Calcular retornos diários
            df['returns'] = df['price'].pct_change()
            df = df.dropna()

            # Validar e limpar dados
            validation = self.validate_data_quality(df)
            if not validation['valid']:
                print(f"Atenção: Problemas detectados nos dados: {validation['issues']}")
                df = self.clean_data(df)

            return df

        except Exception as e:
            print(f"Erro ao processar dados: {e}")
            return None

    def get_current_price(self, crypto_id):
        try:
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': crypto_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true',
                'include_market_cap': 'true'
            }
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if crypto_id in data:
                return data[crypto_id]
            else:
                return None
        except Exception as e:
            print(f"Erro ao obter preço atual: {e}")
            return None

    def get_available_cryptos(self):
        try:
            url = f"{self.base_url}/coins/list"
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            major_cryptos = []
            for crypto in data[:100]:  # Top 100
                major_cryptos.append({
                    'id': crypto['id'],
                    'symbol': crypto['symbol'].upper(),
                    'name': crypto['name']
                })
            return major_cryptos
        except Exception as e:
            print(f"Erro ao obter lista de criptomoedas: {e}")
            return []

    def validate_data_quality(self, df):
        if df is None or df.empty:
            return {'valid': False, 'issues': ['DataFrame vazio ou None']}
        issues = []
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            issues.append(f"Valores nulos encontrados: {null_counts.to_dict()}")
        if (df['price'] <= 0).any():
            issues.append("Preços negativos ou zero encontrados")
        if (df['volume'] < 0).any():
            issues.append("Volumes negativos encontrados")
        if df.index.duplicated().any():
            issues.append("Datas duplicadas encontradas")
        if not df.index.is_monotonic_increasing:
            issues.append("Dados não estão em ordem cronológica")
        time_diff = df.index.to_series().diff()
        max_gap = time_diff.max()
        if max_gap > pd.Timedelta(days=2):
            issues.append(f"Gap temporal grande detectado: {max_gap}")
        if 'returns' in df.columns:
            extreme_returns = df['returns'].abs() > 0.5
            if extreme_returns.any():
                issues.append(f"Retornos extremos detectados: {extreme_returns.sum()} ocorrências")
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'data_points': len(df),
            'date_range': {
                'start': df.index.min(),
                'end': df.index.max()
            },
            'missing_data_percentage': (null_counts.sum() / (len(df) * len(df.columns))) * 100
        }

    def clean_data(self, df):
        if df is None or df.empty:
            return df
        df_clean = df.copy()
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        df_clean = df_clean.sort_index()
        for col in ['price', 'volume', 'market_cap']:
            if col in df_clean.columns:
                null_pct = df_clean[col].isnull().sum() / len(df_clean)
                if null_pct < 0.05:
                    df_clean[col] = df_clean[col].interpolate(method='linear')
        if 'returns' in df_clean.columns:
            returns_mean = df_clean['returns'].mean()
            returns_std = df_clean['returns'].std()
            outlier_threshold = 5 * returns_std
            outliers = df_clean['returns'].abs() > outlier_threshold
            if outliers.any():
                for idx in df_clean[outliers].index:
                    window_data = df_clean.loc[:idx].tail(8)['returns']
                    if len(window_data) > 1:
                        df_clean.loc[idx, 'returns'] = window_data.median()
        if 'returns' in df_clean.columns:
            df_clean['price'] = df_clean['price'].iloc[0] * (1 + df_clean['returns']).cumprod()
        for col in ['price', 'volume', 'market_cap']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].abs()
        return df_clean
