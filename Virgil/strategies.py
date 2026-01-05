import pandas as pd
import numpy as np

def apply_strategies(df, ma_window=20, stop_loss=0.1, take_profit=0.2):
    if df is None or df.empty: return None
    
    data = df.copy()
    data['Returns'] = data['Close'].pct_change().fillna(0)
    data['MA'] = data['Close'].rolling(window=ma_window).mean()
    
    signals = np.zeros(len(data))
    exit_types = [""] * len(data)
    in_position = False
    entry_price = 0.0

    # On commence strictement après la fenêtre MA
    for i in range(ma_window, len(data)):
        current_price = data['Close'].iloc[i]
        ma_prev = data['MA'].iloc[i-1]
        
        if not in_position:
            if current_price > ma_prev:
                in_position = True
                entry_price = current_price
                signals[i] = 1
        else:
            perf = (current_price - entry_price) / entry_price
            if perf <= -stop_loss:
                in_position = False
                exit_types[i] = "SL"
            elif perf >= take_profit:
                in_position = False
                exit_types[i] = "TP"
            elif current_price < ma_prev:
                in_position = False
                exit_types[i] = "MA"
            else:
                signals[i] = 1

    data['Signal'] = signals
    data['Exit_Type'] = exit_types
    data['Strat_Returns'] = data['Signal'].shift(1) * data['Returns']
    data['Strat_Momentum'] = (1 + data['Strat_Returns'].fillna(0)).cumprod() * 100

    # --- CORRECTION DU "TROU DE 10 JOURS" ---
    # On crée un index quotidien complet (Lundi -> Dimanche)
    full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
    data = data.reindex(full_index)
    
    # On propage la dernière valeur connue (Forward Fill)
    # Ton argent reste le même pendant les week-ends ou les trous
    data['Strat_Momentum'] = data['Strat_Momentum'].ffill()
    data['Strat_Returns'] = data['Strat_Returns'].fillna(0)
    data['Signal'] = data['Signal'].ffill().fillna(0)
    data['Exit_Type'] = data['Exit_Type'].fillna("")
    data['MA'] = data['MA'].ffill()
    data['Close'] = data['Close'].ffill()
    
    return data


# CETTE FONCTION DOIT ÊTRE ICI POUR ÊTRE IMPORTÉE
def calculate_metrics(series):
    """
    Calcule la performance globale et le Max Drawdown.
    """
    if series is None or series.empty or len(series) < 2:
        return 0.0, 0.0

    total_return = (series.iloc[-1] / series.iloc[0]) - 1
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return total_return, max_drawdown