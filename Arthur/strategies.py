import pandas as pd
import numpy as np

# Liste des actifs disponibles
AVAILABLE_ASSETS = {
    "Airbus": "AIR.PA",
    "LVMH": "MC.PA",
    "TotalEnergies": "TTE.PA",
    "Sanofi": "SAN.PA",
    "L'Oréal": "OR.PA",
    "Other (Manual Input)": "MANUAL" 
}

def apply_strategies(df, short_window=20, long_window=50):
    """
    Applique les stratégies Buy&Hold et Momentum sur le DataFrame.
    Prend en compte les fenêtres dynamiques pour les moyennes mobiles.
    """
    # On travaille sur une copie pour ne pas modifier l'original
    df = df.copy()
    
    # --- 1. Calcul des rendements (Log Returns) ---
    # Log returns sont additifs, mieux pour les maths financières
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # --- 2. Stratégie 1 : Buy & Hold (Référence) ---
    # On garde simplement les rendements de l'actif
    df['Strat_BuyHold'] = df['Log_Ret'].cumsum().apply(np.exp) * 100
    
    # --- 3. Stratégie 2 : Momentum (Moving Average Crossover) ---
    # Utilisation des paramètres dynamiques (short_window, long_window)
    df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
    
    # Signal : 1 si Court > Long, sinon 0 (On reste cash)
    df['Signal'] = 0
    df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)
    
    # IMPORTANT : On décale le signal de 1 jour. 
    # On ne peut pas trader sur le signal de cloture d'aujourd'hui pour le mouvement d'aujourd'hui.
    df['Position'] = df['Signal'].shift(1)
    
    # Rendement de la stratégie
    df['Strat_Ret'] = df['Position'] * df['Log_Ret']
    df['Strat_Momentum'] = df['Strat_Ret'].cumsum().apply(np.exp) * 100
    
    # Nettoyage des NaN générés par les moyennes mobiles
    df.dropna(inplace=True)
    
    # --- 4. Calcul des Métriques (KPIs) ---
    metrics = {}
    
    # Fonction utilitaire pour Sharpe et Volatilité
    def get_metrics(returns_col):
        # Annualisation (252 jours de trading)
        mean_ret = returns_col.mean() * 252
        vol = returns_col.std() * np.sqrt(252)
        sharpe = mean_ret / vol if vol != 0 else 0
        return mean_ret, vol, sharpe

    # Fonction pour Max Drawdown
    def get_max_drawdown(cumulative_col):
        # cumulative_col est la série de prix/valeur (ex: 100, 102, 98...)
        running_max = cumulative_col.cummax()
        drawdown = (cumulative_col - running_max) / running_max
        return drawdown.min()

    # Calculs Buy & Hold
    bh_ret, bh_vol, bh_sharpe = get_metrics(df['Log_Ret'])
    metrics['BuyHold_Return'] = np.exp(df['Log_Ret'].sum()) - 1 # Rendement total simple
    metrics['BuyHold_Vol'] = bh_vol
    metrics['BuyHold_Sharpe'] = bh_sharpe
    metrics['BuyHold_MDD'] = get_max_drawdown(df['Strat_BuyHold'])

    # Calculs Momentum
    mom_ret, mom_vol, mom_sharpe = get_metrics(df['Strat_Ret'])
    metrics['Momentum_Return'] = np.exp(df['Strat_Ret'].sum()) - 1
    metrics['Momentum_Vol'] = mom_vol
    metrics['Momentum_Sharpe'] = mom_sharpe
    metrics['Momentum_MDD'] = get_max_drawdown(df['Strat_Momentum'])
    
    return df, metrics