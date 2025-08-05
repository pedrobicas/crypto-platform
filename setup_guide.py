#!/usr/bin/env python3
"""
Script de configuração e teste do ambiente para o Crypto Analysis & Forecast Platform
Execute este script antes de usar a aplicação principal para verificar se tudo está funcionando.
"""

import sys
import subprocess
import importlib
import requests
from datetime import datetime

def check_python_version():
    """Verifica se a versão do Python é compatível"""
    print("Verificando versão do Python...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERRO: Python 3.8+ é necessário")
        print(f"   Versão atual: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"Python {version.major}.{version.minor}.{version.micro} - OK")
        return True

def install_requirements():
    """Instala as dependências necessárias"""
    print("\nInstalando dependências...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependências instaladas com sucesso")
        return True
    except subprocess.CalledProcessError:
        print("ERRO: Falha ao instalar dependências")
        return False

def check_imports():
    """Verifica se todas as bibliotecas podem ser importadas"""
    print("\nVerificando imports...")
    
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'requests',
        'statsmodels',
        'sklearn',
        'scipy'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"{package} - OK")
        except ImportError:
            print(f"{package} - ERRO")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFalha ao importar: {', '.join(failed_imports)}")
        return False
    else:
        print("\nTodos os imports funcionando corretamente")
        return True

def test_api_connection():
    """Testa conexão com a API do CoinGecko"""
    print("\nTestando conexão com API CoinGecko...")
    
    try:
        response = requests.get("https://api.coingecko.com/api/v3/ping", timeout=10)
        
        if response.status_code == 200:
            print("Conexão com API CoinGecko - OK")
            return True
        else:
            print(f"API retornou status: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Erro de conexão: {e}")
        return False

def test_data_collection():
    """Testa coleta de dados básica"""
    print("\nTestando coleta de dados...")
    
    try:
        from data_collector import CryptoDataCollector
        
        collector = CryptoDataCollector()
        data = collector.get_historical_data('bitcoin', 30)
        
        if data is not None and len(data) > 0:
            print(f"Coletados {len(data)} pontos de dados do Bitcoin")
            return True
        else:
            print("Falha na coleta de dados")
            return False
            
    except Exception as e:
        print(f"Erro na coleta de dados: {e}")
        return False

def test_technical_indicators():
    """Testa cálculo de indicadores técnicos"""
    print("\nTestando indicadores técnicos...")
    
    try:
        from data_collector import CryptoDataCollector
        from technical_indicators import TechnicalIndicators
        
        collector = CryptoDataCollector()
        data = collector.get_historical_data('bitcoin', 50)
        
        if data is None:
            print("Não foi possível obter dados para teste")
            return False
        
        indicators = TechnicalIndicators()
        data_with_indicators = indicators.calculate_all_indicators(data)
        
        if 'rsi' in data_with_indicators.columns and 'macd' in data_with_indicators.columns:
            print("Indicadores técnicos calculados com sucesso")
            return True
        else:
            print("Falha no cálculo de indicadores")
            return False
            
    except Exception as e:
        print(f"Erro no cálculo de indicadores: {e}")
        return False

def test_forecast_model():
    """Testa modelo de previsão"""
    print("\nTestando modelo de previsão...")
    
    try:
        from data_collector import CryptoDataCollector
        from forecast_model import CryptoForecaster
        
        collector = CryptoDataCollector()
        data = collector.get_historical_data('bitcoin', 100)
        
        if data is None:
            print("Não foi possível obter dados para teste")
            return False
        
        forecaster = CryptoForecaster()
        result = forecaster.forecast_sarimax(data, 7)
        
        if 'forecast' in result and len(result['forecast']) > 0:
            print("Modelo de previsão funcionando")
            return True
        else:
            print("Falha no modelo de previsão")
            return False
            
    except Exception as e:
        print(f"Erro no modelo de previsão: {e}")
        return False

def create_sample_config():
    """Cria arquivo de configuração de exemplo"""
    print("\n⚙Criando configuração de exemplo...")
    
    config = """# Configurações do Crypto Analysis Platform

# API Configuration
API_BASE_URL = "https://api.coingecko.com/api/v3"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

# Model Parameters
DEFAULT_FORECAST_DAYS = 30
DEFAULT_HISTORICAL_DAYS = 365
MAX_FORECAST_DAYS = 60
MIN_HISTORICAL_DAYS = 180

# Risk Parameters
RISK_FREE_RATE = 0.02  # 2% annual
DEFAULT_CONFIDENCE_LEVEL = 0.95
TRANSACTION_FEE = 0.001  # 0.1%
SLIPPAGE = 0.002  # 0.2%

# UI Configuration
DEFAULT_THEME = "dark"
CHART_HEIGHT = 800
"""
    
    try:
        with open('config.py', 'w') as f:
            f.write(config)
        print("Arquivo config.py criado")
        return True
    except Exception as e:
        print(f"Erro ao criar configuração: {e}")
        return False

def run_diagnostics():
    """Executa todos os testes de diagnóstico"""
    print("CRYPTO ANALYSIS PLATFORM - DIAGNÓSTICO DO SISTEMA")
    print("=" * 60)
    print(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tests = [
        ("Versão Python", check_python_version),
        ("Instalação de Dependências", install_requirements),
        ("Verificação de Imports", check_imports),
        ("Conexão API", test_api_connection),
        ("Coleta de Dados", test_data_collection),
        ("Indicadores Técnicos", test_technical_indicators),
        ("Modelo de Previsão", test_forecast_model),
        ("Configuração", create_sample_config)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"Erro inesperado em {test_name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"RESUMO: {passed}/{total} testes passaram")
    
    if passed == total:
        print("SUCESSO! Sistema pronto para uso")
        print("\nPara executar a aplicação:")
        print("   streamlit run main.py")
    else:
        print("ATENÇÃO: Alguns testes falharam")
        print("   Verifique os erros acima antes de continuar")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = run_diagnostics()
    
    if success:
        print("\nDeseja executar a aplicação agora? (y/n): ", end="")
        response = input().lower().strip()
        
        if response in ['y', 'yes', 's', 'sim']:
            print("\nIniciando aplicação...")
            try:
                subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"])
            except KeyboardInterrupt:
                print("\nAplicação encerrada pelo usuário")
            except Exception as e:
                print(f"\nErro ao executar aplicação: {e}")
    else:
        sys.exit(1)
