import pandas as pd
import numpy as np


AVAILABLE_ASSETS = {
    "Bitcoin (USD)": "BTC-USD",
    "Ethereum (USD)": "ETH-USD",
    "S&P 500 (Indice US)": "^GSPC",
    "CAC 40 (France)": "^FCHI",
    "Euro / Dollar": "EURUSD=X",
    "Or (Gold)": "GC=F",
    "Apple": "AAPL",
    "NVIDIA": "NVDA",
    "Tesla": "TSLA",
    "TotalEnergies": "TTE.PA",
    "Autre (Saisir manuellement)": "CUSTOM"
}

def calculate_metrics(daily_returns):
    """Compute the Sharpe Ratio and the Max Drawdown"""
    if daily_returns.empty:
        return 0.0, 0.0
    
    # Sharpe Ratio
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    
    if std_return == 0:
        sharpe = 0
    else:
        sharpe = (mean_return / std_return) * np.sqrt(252)
    
    cumul = (1 + daily_returns).cumprod()
    max_ = cumul.cummax()
    drawdown = (cumul - max_) / max_
    max_drawdown = drawdown.min()
    
    return sharpe, max_drawdown

def apply_strategies(df):
    """
    Applique deux stratégies :
    1. Buy & Hold (Performance brute)
    2. Momentum (Croisement Moyennes Mobiles 20j / 50j)
    """
    data = df.copy()
    
    # Calcul des rendements journaliers
    data['Returns'] = data['Close'].pct_change()
    
    # --- STRATEGIE 1 : BUY & HOLD ---
    # On commence avec une base 100
    data['Strat_BuyHold'] = 100 * (1 + data['Returns']).cumprod()
    
    # --- STRATEGIE 2 : MOMENTUM (Simple Moving Average) ---
    data['SMA_Short'] = data['Close'].rolling(window=20).mean()
    data['SMA_Long'] = data['Close'].rolling(window=50).mean()
    
    # Signal : 1 si Court > Long, sinon 0 (on est cash)
    data['Signal'] = 0
    data.loc[data['SMA_Short'] > data['SMA_Long'], 'Signal'] = 1
    
    # Le rendement de la stratégie est le rendement de l'actif * le signal de la VEILLE
    data['Strat_Momentum_Returns'] = data['Returns'] * data['Signal'].shift(1)
    data['Strat_Momentum'] = 100 * (1 + data['Strat_Momentum_Returns']).cumprod()
    
    # Nettoyage des NaN (début de l'historique)
    data.dropna(inplace=True)
    
    # Calcul des métriques
    metrics = {}
    metrics['BuyHold_Sharpe'], metrics['BuyHold_MDD'] = calculate_metrics(data['Returns'])
    metrics['Momentum_Sharpe'], metrics['Momentum_MDD'] = calculate_metrics(data['Strat_Momentum_Returns'])
    
    return data, metrics