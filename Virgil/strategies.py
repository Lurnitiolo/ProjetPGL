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
    # Rendement réel : on suit le signal de la veille
    data['Strat_Returns'] = data['Signal'].shift(1) * data['Returns']
    data['Strat_Momentum'] = (1 + data['Strat_Returns'].fillna(0)).cumprod() * 100
    
    return data

def calculate_metrics(series):
    if series.empty or len(series) < 2: return 0.0, 0.0
    total_return = (series.iloc[-1] / series.iloc[0]) - 1
    mdd = (series - series.cummax()) / series.cummax()
    return total_return, mdd.min()