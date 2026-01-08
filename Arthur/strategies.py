import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor

AVAILABLE_ASSETS = {
    # --- 🇫🇷 FRANCE (CAC 40) ---
    "LVMH (Luxury)": "MC.PA",
    "TotalEnergies (Energy)": "TTE.PA",
    "Airbus (Industrial)": "AIR.PA",
    "BNP Paribas (Banking)": "BNP.PA",
    "Sanofi (Health/Defensive)": "SAN.PA",
    
    # --- 🇺🇸 US TECH (High Growth/Vol) ---
    "NVIDIA (AI Boom)": "NVDA",
    "Tesla (High Volatility)": "TSLA",
    "Apple (Quality Growth)": "AAPL",
    "Microsoft (Big Tech)": "MSFT",
    
    # --- 🌍 ETFS & INDICES ---
    "S&P 500 (US Market)": "SPY",
    "MSCI World (Global)": "CW8.PA",
    "Nasdaq 100 (Tech Index)": "QQQ",
    "Gold (Commodity)": "GLD",
    
    # --- ₿ CRYPTO (Extreme Risk) ---
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD"
}

def calculate_metrics(df):
    """Calcule les métriques financières avancées + Stats de Trading"""
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
    
    # --- CALCUL DES TRADES (CORRIGÉ) ---
    trades = []
    current_trade_ret = 0
    
    # On itère pour détecter les entrées/sorties
    positions = df['Position'].values
    returns = df['Log_Ret'].values
    
    # On commence à 1 car on regarde i-1
    for i in range(1, len(df)):
        # Si on était en position la veille (donc exposé aujourd'hui)
        if positions[i-1] == 1:
            current_trade_ret += returns[i] # On cumule la perf
            
            # Si on sort aujourd'hui (0) ou que c'est la fin des données
            if positions[i] == 0 or i == len(df)-1:
                trades.append(current_trade_ret)
                current_trade_ret = 0
    
    # Calcul des stats de trading
    nb_trades = len(trades)
    if nb_trades > 0:
        # On considère un trade gagnant si rendement > 0
        winning_trades = sum(1 for t in trades if t > 0)
        win_rate = winning_trades / nb_trades
    else:
        win_rate = 0
        
    # C'EST ICI QUE J'AI CORRIGÉ LE NOM DE LA VARIABLE :
    metrics['Active_Trades'] = nb_trades  
    metrics['Active_WinRate'] = win_rate

    # --- FONCTION DE CALCUL FINANCIER ---
    def get_advanced_stats(equity_col):
        returns_col = equity_col.pct_change().dropna()
        
        if len(returns_col) == 0: 
            return {k: 0 for k in ['Return', 'Vol', 'Sharpe', 'Sortino', 'MDD', 'VaR', 'CVaR']}
        
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
        
        return {
            'Return': total_ret,
            'Vol': vol_ann,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'MDD': mdd,
            'VaR': var_95,
            'CVaR': cvar_95
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
    
    if strategy_type == "Buy & Hold":
        df['Position'] = 1

    elif strategy_type == "MA Crossover":
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
        df['Position'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)

    elif strategy_type == "Momentum":
        window = params.get('mom_window', 20)
        df['ROC'] = df['Close'].pct_change(periods=window)
        df['Position'] = np.where(df['ROC'] > 0, 1, 0)

    elif strategy_type == "Mean Reversion":
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

    else:
        df['Position'] = 0

    return calculate_metrics(df)

def predict_ml_model(df, days_ahead=30, n_lags=5):
    """
    Modèle ML : Random Forest avec Lags dynamiques.
    n_lags : Nombre de jours passés utilisés pour prédire le jour suivant.
    """
    data = df[['Close']].copy()
    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)

    for i in range(1, n_lags + 1):
        data[f'Lag_{i}'] = data['Log_Ret'].shift(i)
    data.dropna(inplace=True)
    
    lag_cols = [f'Lag_{i}' for i in range(1, n_lags + 1)]
    X = data[lag_cols].values
    y = data['Log_Ret'].values
    
    # Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    y_pred_train = model.predict(X)
    std_resid = np.std(y - y_pred_train)
    
    current_feats = data.iloc[-1][lag_cols].values.reshape(1, -1)
    future_log_returns = []
    
    np.random.seed(42)
    
    for _ in range(days_ahead):
        pred_ret = model.predict(current_feats)[0]
        noise = np.random.normal(0, std_resid * 0.5) 
        final_pred = pred_ret + noise
        
        future_log_returns.append(final_pred)
        
        new_feats = np.roll(current_feats, 1)
        new_feats[0, 0] = final_pred
        current_feats = new_feats
        
    last_price = df['Close'].iloc[-1]
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=days_ahead + 1, freq='B')[1:]
    
    cumulative_returns = np.cumsum(future_log_returns)
    predicted_prices = last_price * np.exp(cumulative_returns)
    
    time_sqrt = np.sqrt(np.arange(1, days_ahead + 1))
    volatility = data['Log_Ret'].std()
    
    upper_band = predicted_prices * np.exp(1.96 * volatility * time_sqrt)
    lower_band = predicted_prices * np.exp(-1.96 * volatility * time_sqrt)
    
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Prediction': predicted_prices,
        'Upper': upper_band,
        'Lower': lower_band
    })
    pred_df.set_index('Date', inplace=True)
    
    return pred_df