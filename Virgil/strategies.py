import pandas as pd
import numpy as np

def apply_strategies(df, ma_window=20, stop_loss=0.1, take_profit=0.2):
    """
    Applique une stratégie de suivi de tendance avec Moyenne Mobile,
    incluant une gestion stricte du Stop Loss et du Take Profit.
    """
    if df is None or df.empty:
        return None
    
    data = df.copy()
    
    # --- 1. PRÉPARATION DES DONNÉES ---
    data['Returns'] = data['Close'].pct_change().fillna(0)
    data['MA'] = data['Close'].rolling(window=ma_window).mean()
    
    # Initialisation des colonnes de gestion
    signals = np.zeros(len(data))
    exit_types = [""] * len(data) # Contiendra 'Stop Loss', 'Take Profit' ou 'Signal Exit'
    
    in_position = False
    entry_price = 0.0

    # --- 2. LOGIQUE DE LA STRATÉGIE (BOUCLE) ---
    for i in range(ma_window, len(data)):
        current_price = data['Close'].iloc[i]
        ma_prev = data['MA'].iloc[i-1]
        
        if not in_position:
            # Signal d'entrée : le prix repasse au-dessus de la MA
            if current_price > ma_prev:
                in_position = True
                entry_price = current_price
                signals[i] = 1
        else:
            # Calcul de la performance latente du trade
            perf = (current_price - entry_price) / entry_price
            
            # Cas 1 : Sortie par Stop Loss
            if perf <= -stop_loss:
                in_position = False
                exit_types[i] = "Stop Loss"
            
            # Cas 2 : Sortie par Take Profit
            elif perf >= take_profit:
                in_position = False
                exit_types[i] = "Take Profit"
            
            # Cas 3 : Sortie par signal technique (le prix repasse sous la MA)
            elif current_price < ma_prev:
                in_position = False
                exit_types[i] = "Signal Exit"
            
            # Cas 4 : On reste en position
            else:
                signals[i] = 1

    # --- 3. CALCUL DES RÉSULTATS ---
    data['Signal'] = signals
    data['Exit_Type'] = exit_types
    
    # Calcul des rendements de la stratégie
    # On shift(1) car on prend le signal de la bougie précédente pour la bougie actuelle
    data['Strat_Returns'] = data['Signal'].shift(1) * data['Returns']
    
    # Calcul de la courbe de capital (Base 100)
    data['Strat_Momentum'] = (1 + data['Strat_Returns'].fillna(0)).cumprod() * 100
    
    return data

def calculate_metrics(series):
    if series is None or series.empty or series.iloc[0] == 0:
        return 0.0, 0.0
    
    # Rendement total
    total_return = (series.iloc[-1] / series.iloc[0]) - 1
    
    # MDD (Ta logique, simplifiée)
    rolling_max = series.cummax()
    # On s'assure que rolling_max n'est pas 0 pour la division
    drawdown = (series - rolling_max) / rolling_max.replace(0, np.nan)
    max_drawdown = drawdown.min()
    
    return total_return, max_drawdown if not np.isnan(max_drawdown) else 0.0