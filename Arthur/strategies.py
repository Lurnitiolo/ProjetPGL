import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- LISTE D'ACTIFS DIVERSIFIÉE ---
AVAILABLE_ASSETS = {
    # 🇫🇷 FRANCE
    "LVMH (Luxury)": "MC.PA",
    "TotalEnergies (Energy)": "TTE.PA",
    "Airbus (Industrial)": "AIR.PA",
    "BNP Paribas (Banking)": "BNP.PA",
    "Sanofi (Health)": "SAN.PA",
    
    # 🇺🇸 US TECH
    "NVIDIA (AI Boom)": "NVDA",
    "Tesla (High Vol)": "TSLA",
    "Apple (Quality)": "AAPL",
    "Microsoft (Big Tech)": "MSFT",
    
    # 🌍 INDICES & COMMODITIES
    "S&P 500 (US Market)": "SPY",
    "MSCI World (Global)": "CW8.PA",
    "Nasdaq 100 (Tech)": "QQQ",
    "Gold (Safe Haven)": "GLD",
    
    # ₿ CRYPTO
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD"
}

def calculate_metrics(df):
    """Calcule les métriques financières et stats de trading"""
    metrics = {}
    
    # Returns
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Strat_BuyHold'] = df['Log_Ret'].cumsum().apply(np.exp) * 100
    
    # Stratégie
    df['Strat_Ret'] = df['Position'].shift(1) * df['Log_Ret']
    df['Strat_Active'] = df['Strat_Ret'].cumsum().apply(np.exp) * 100
    
    df.dropna(inplace=True)
    
    # Trades Stats
    trades = []
    current_trade_ret = 0
    positions = df['Position'].values
    returns = df['Log_Ret'].values
    
    for i in range(1, len(df)):
        if positions[i-1] == 1:
            current_trade_ret += returns[i]
            if positions[i] == 0 or i == len(df)-1:
                trades.append(current_trade_ret)
                current_trade_ret = 0
    
    nb_trades = len(trades)
    win_rate = sum(1 for t in trades if t > 0) / nb_trades if nb_trades > 0 else 0
        
    metrics['Active_Trades'] = nb_trades  
    metrics['Active_WinRate'] = win_rate

    # Advanced Stats Helper
    def get_advanced_stats(equity_col):
        returns_col = equity_col.pct_change().dropna()
        if len(returns_col) == 0: 
            return {k: 0 for k in ['Return', 'Vol', 'Sharpe', 'MDD', 'VaR', 'CVaR']}
        
        total_ret = (equity_col.iloc[-1] / equity_col.iloc[0]) - 1
        vol_ann = returns_col.std() * np.sqrt(252)
        mean_ret_ann = returns_col.mean() * 252
        sharpe = mean_ret_ann / vol_ann if vol_ann != 0 else 0
        
        running_max = equity_col.cummax()
        mdd = ((equity_col - running_max) / running_max).min()
        
        var_95 = np.percentile(returns_col, 5)
        cvar_95 = returns_col[returns_col <= var_95].mean()
        
        return {'Return': total_ret, 'Vol': vol_ann, 'Sharpe': sharpe, 'MDD': mdd, 'VaR': var_95, 'CVaR': cvar_95}

    # Store Metrics
    bh_stats = get_advanced_stats(df['Strat_BuyHold'])
    for k, v in bh_stats.items(): metrics[f'BuyHold_{k}'] = v

    act_stats = get_advanced_stats(df['Strat_Active'])
    for k, v in act_stats.items(): metrics[f'Active_{k}'] = v
    
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
    Random Forest avec Lags dynamiques et Seed fixe.
    """
    # 1. Préparation
    data = df[['Close']].copy()
    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)
    
    # Feature Engineering Dynamique
    for i in range(1, n_lags + 1):
        data[f'Lag_{i}'] = data['Log_Ret'].shift(i)
    data.dropna(inplace=True)
    
    # 2. Train
    lag_cols = [f'Lag_{i}' for i in range(1, n_lags + 1)]
    X = data[lag_cols].values
    y = data['Log_Ret'].values
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    y_pred_train = model.predict(X)
    std_resid = np.std(y - y_pred_train)
    
    # 3. Prédiction Récursive
    current_feats = data.iloc[-1][lag_cols].values.reshape(1, -1)
    future_log_returns = []
    
    # FIXER LE SEED POUR STABILITÉ GRAPHIQUE
    np.random.seed(42)
    
    for _ in range(days_ahead):
        pred_ret = model.predict(current_feats)[0]
        noise = np.random.normal(0, std_resid * 0.5) 
        final_pred = pred_ret + noise
        
        future_log_returns.append(final_pred)
        
        new_feats = np.roll(current_feats, 1)
        new_feats[0, 0] = final_pred
        current_feats = new_feats
        
    # 4. Reconstruction Prix
    last_price = df['Close'].iloc[-1]
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=days_ahead + 1, freq='B')[1:]
    
    cumulative_returns = np.cumsum(future_log_returns)
    predicted_prices = last_price * np.exp(cumulative_returns)
    
    # 5. Bandes de confiance
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