import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

AVAILABLE_ASSETS = {
    "Airbus": "AIR.PA",
    "LVMH": "MC.PA",
    "TotalEnergies": "TTE.PA",
    "Sanofi": "SAN.PA",
    "L'Oréal": "OR.PA",
    "Schneider Electric": "SU.PA",
    "Amundi ETF MSCI World": "CW8.PA",
    "Other (Manual Input)": "MANUAL" 
}

def calculate_metrics(df):
    """Calcule les métriques financières avancées"""
    metrics = {}
    
    # 1. Préparation des données
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Stratégie Buy & Hold (Base 100)
    df['Strat_BuyHold'] = df['Log_Ret'].cumsum().apply(np.exp) * 100
    
    # Stratégie Active
    # Si Position = 1 (Hier), on prend le rendement du jour. Si 0, on prend 0.
    df['Strat_Ret'] = df['Position'].shift(1) * df['Log_Ret']
    df['Strat_Active'] = df['Strat_Ret'].cumsum().apply(np.exp) * 100
    
    # Nettoyage
    df.dropna(inplace=True)
    
    # --- FONCTION DE CALCUL ---
    def get_advanced_stats(equity_col):
        # On recalcule les rendements arithmétiques exacts depuis la courbe de capital
        returns_col = equity_col.pct_change().dropna()
        
        if len(returns_col) == 0: 
            return {k: 0 for k in ['Return', 'Vol', 'Sharpe', 'Sortino', 'MDD', 'VaR', 'CVaR', 'Skew', 'Kurt']}
        
        # 1. Rendement & Volatilité (Annualisés)
        total_ret = (equity_col.iloc[-1] / equity_col.iloc[0]) - 1
        vol_ann = returns_col.std() * np.sqrt(252)
        
        # 2. Sharpe (Rf=0)
        mean_ret_ann = returns_col.mean() * 252
        sharpe = mean_ret_ann / vol_ann if vol_ann != 0 else 0
        
        # 3. Sortino
        negative_returns = returns_col[returns_col < 0]
        downside_std = negative_returns.std() * np.sqrt(252)
        sortino = mean_ret_ann / downside_std if downside_std != 0 else 0
        
        # 4. Max Drawdown
        running_max = equity_col.cummax()
        drawdown = (equity_col - running_max) / running_max
        mdd = drawdown.min()
        
        # 5. VaR & CVaR (95%)
        var_95 = np.percentile(returns_col, 5)
        cvar_95 = returns_col[returns_col <= var_95].mean()
        
        # 6. Distribution
        dist_skew = skew(returns_col)
        dist_kurt = kurtosis(returns_col)
        
        return {
            'Return': total_ret,
            'Vol': vol_ann,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'MDD': mdd,
            'VaR': var_95,
            'CVaR': cvar_95,
            'Skew': dist_skew,
            'Kurt': dist_kurt
        }

    # Calculs Buy & Hold
    bh_stats = get_advanced_stats(df['Strat_BuyHold'])
    for k, v in bh_stats.items():
        metrics[f'BuyHold_{k}'] = v

    # Calculs Stratégie Active
    act_stats = get_advanced_stats(df['Strat_Active'])
    for k, v in act_stats.items():
        metrics[f'Active_{k}'] = v
    
    return df, metrics

def apply_strategies(df, strategy_type, params):
    df = df.copy()
    
    # 1. Moving Average
    if strategy_type == "Moving Average":
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
        df['Position'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)

    # 2. Bollinger Bands
    elif strategy_type == "Bollinger Bands":
        window = params.get('bb_window', 20)
        std_dev = params.get('bb_std', 2.0)
        df['SMA'] = df['Close'].rolling(window=window).mean()
        df['STD'] = df['Close'].rolling(window=window).std()
        df['Upper'] = df['SMA'] + (df['STD'] * std_dev)
        df['Lower'] = df['SMA'] - (df['STD'] * std_dev)
        
        df['Position'] = np.nan
        df.loc[df['Close'] < df['Lower'], 'Position'] = 1 
        df.loc[df['Close'] > df['Upper'], 'Position'] = 0 
        df['Position'] = df['Position'].ffill().fillna(0)

    # 3. RSI
    elif strategy_type == "RSI":
        window = params.get('rsi_window', 14)
        overbought = params.get('rsi_overbought', 70)
        oversold = params.get('rsi_oversold', 30)
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['Position'] = np.nan
        df.loc[df['RSI'] < oversold, 'Position'] = 1
        df.loc[df['RSI'] > overbought, 'Position'] = 0
        df['Position'] = df['Position'].ffill().fillna(0)

    else:
        df['Position'] = 0

    return calculate_metrics(df)