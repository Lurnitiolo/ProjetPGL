import pandas as pd
import numpy as np

def apply_strategies(df, ma_window=20, stop_loss=0.1, take_profit=0.2):
    """
    Applique les stratégies avec gestion du risque (Stop Loss / Take Profit).
    """
    if df is None or df.empty:
        return None
    
    data = df.copy()
    data['Returns'] = data['Close'].pct_change()
    
    # --- STRATÉGIE 1 : BUY AND HOLD ---
    data['Strat_BuyHold'] = (1 + data['Returns']).cumprod() * 100

    # --- STRATÉGIE 2 : MOMENTUM ENRICHI (SL / TP / MA) ---
    data['MA'] = data['Close'].rolling(window=ma_window).mean()
    
    # Initialisation des colonnes pour la boucle
    signals = np.zeros(len(data))
    in_position = False
    entry_price = 0.0

    # On parcourt les données pour gérer les sorties SL et TP
    # On commence à l'index ma_window pour avoir une MA valide
    for i in range(ma_window, len(data)):
        current_price = data['Close'].iloc[i]
        ma_prev = data['MA'].iloc[i-1] # MA d'hier pour décider aujourd'hui
        
        if not in_position:
            # CONDITION D'ENTRÉE : Prix > MA
            if current_price > ma_prev:
                in_position = True
                entry_price = current_price
                signals[i] = 1
        else:
            # ON EST EN POSITION : On vérifie les 3 conditions de sortie
            perf_since_entry = (current_price - entry_price) / entry_price
            
            # 1. Sortie Technique (Prix < MA)
            # 2. Sortie Sécurité (Stop Loss atteint)
            # 3. Sortie Profit (Take Profit atteint)
            if current_price < ma_prev or perf_since_entry <= -stop_loss or perf_since_entry >= take_profit:
                in_position = False
                signals[i] = 0
            else:
                signals[i] = 1

    data['Signal'] = signals
    
    # --- CALCUL DES RENDEMENTS (Avec le Shift critique) ---
    data['Strat_Momentum_Returns'] = data['Signal'].shift(1) * data['Returns']
    data['Strat_Momentum'] = (1 + data['Strat_Momentum_Returns'].fillna(0)).cumprod() * 100
    
    data.dropna(inplace=True)
    return data

def calculate_metrics(series):
    """
    Calcule la performance globale et le Max Drawdown.
    """
    if len(series) < 2:
        return 0.0, 0.0

    total_return = (series.iloc[-1] / series.iloc[0]) - 1
    
    # Max Drawdown
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return total_return, max_drawdown