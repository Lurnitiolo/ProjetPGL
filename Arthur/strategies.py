import pandas as pd
import numpy as np

AVAILABLE_ASSETS = {
    "Airbus": "AIR.PA",
    "LVMH": "MC.PA",
    "TotalEnergies": "TTE.PA",
    "Sanofi": "SAN.PA",
    "L'Oréal": "OR.PA",
    "Schneider Electric": "SU.PA",
    "Other (Manual Input)": "MANUAL" 
}

def calculate_metrics(df):
    """Fonction utilitaire pour calculer Sharpe, Volatilité et Drawdown"""
    metrics = {}
    
    # 1. Buy & Hold (Référence)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Strat_BuyHold'] = df['Log_Ret'].cumsum().apply(np.exp) * 100
    
    # Rendement Stratégie
    # On multiplie la position d'HIER (shift 1) par le rendement d'AUJOURD'HUI
    df['Strat_Ret'] = df['Position'].shift(1) * df['Log_Ret']
    df['Strat_Active'] = df['Strat_Ret'].cumsum().apply(np.exp) * 100
    
    # Nettoyage
    df.dropna(inplace=True)
    
    # Fonction interne metrics
    def get_stats(returns_col, equity_col):
        if len(returns_col) == 0: return 0, 0, 0
        mean_ret = returns_col.mean() * 252
        vol = returns_col.std() * np.sqrt(252)
        sharpe = mean_ret / vol if vol != 0 else 0
        
        # Drawdown
        running_max = equity_col.cummax()
        drawdown = (equity_col - running_max) / running_max
        mdd = drawdown.min()
        
        return np.exp(returns_col.sum()) - 1, vol, sharpe, mdd

    # Stats B&H
    ret, vol, sharpe, mdd = get_stats(df['Log_Ret'], df['Strat_BuyHold'])
    metrics['BuyHold_Return'] = ret
    metrics['BuyHold_Vol'] = vol
    metrics['BuyHold_Sharpe'] = sharpe
    metrics['BuyHold_MDD'] = mdd
    
    # Stats Stratégie Active
    ret, vol, sharpe, mdd = get_stats(df['Strat_Ret'], df['Strat_Active'])
    metrics['Active_Return'] = ret
    metrics['Active_Vol'] = vol
    metrics['Active_Sharpe'] = sharpe
    metrics['Active_MDD'] = mdd
    
    return df, metrics

def apply_strategies(df, strategy_type, params):
    """
    Routeur de stratégies.
    strategy_type: "Moving Average", "Bollinger Bands", "RSI"
    params: dict contenant les paramètres (ex: {'window': 20})
    """
    df = df.copy()
    
    # --- STRATÉGIE 1 : MOVING AVERAGE CROSSOVER ---
    if strategy_type == "Moving Average":
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        
        df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
        
        # 1 si Court > Long, sinon 0
        df['Position'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)

    # --- STRATÉGIE 2 : BOLLINGER BANDS (Mean Reversion) ---
    elif strategy_type == "Bollinger Bands":
        window = params.get('bb_window', 20)
        std_dev = params.get('bb_std', 2.0)
        
        df['SMA'] = df['Close'].rolling(window=window).mean()
        df['STD'] = df['Close'].rolling(window=window).std()
        df['Upper'] = df['SMA'] + (df['STD'] * std_dev)
        df['Lower'] = df['SMA'] - (df['STD'] * std_dev)
        
        # Logique Mean Reversion :
        # Si prix < Lower Band -> Achat (1)
        # Si prix > Upper Band -> Vente (0) ou Short (-1)
        # Ici on fait simple : Long Only (1 ou 0)
        
        df['Position'] = np.nan # On initialise à NaN pour gérer le maintien de position
        
        # Conditions d'entrée/sortie
        df.loc[df['Close'] < df['Lower'], 'Position'] = 1 # Achat sur excès baissier
        df.loc[df['Close'] > df['Upper'], 'Position'] = 0 # Vente sur excès haussier
        
        # On remplit les trous (Hold la position précédente tant qu'on a pas de signal inverse)
        df['Position'] = df['Position'].ffill().fillna(0)

    # --- STRATÉGIE 3 : RSI (Oscillateur) ---
    elif strategy_type == "RSI":
        window = params.get('rsi_window', 14)
        overbought = params.get('rsi_overbought', 70)
        oversold = params.get('rsi_oversold', 30)
        
        # Calcul RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['Position'] = np.nan
        # Achat si RSI < 30 (Oversold)
        df.loc[df['RSI'] < oversold, 'Position'] = 1
        # Vente si RSI > 70 (Overbought)
        df.loc[df['RSI'] > overbought, 'Position'] = 0
        
        df['Position'] = df['Position'].ffill().fillna(0)

    else:
        # Fallback
        df['Position'] = 0

    return calculate_metrics(df)