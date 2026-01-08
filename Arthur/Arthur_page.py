import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_echarts import st_echarts

# ==============================================================================
# 1. GESTION ROBUSTE DES IMPORTS
# ==============================================================================
# On définit d'abord les valeurs par défaut au cas où tout plante
AVAILABLE_ASSETS = {"Airbus": "AIR.PA", "LVMH": "MC.PA", "TotalEnergies": "TTE.PA", "Apple": "AAPL"}

def load_stock_data(*args, **kwargs): return None
def apply_strategies(*args, **kwargs): return pd.DataFrame(), {}
def predict_ml_model(*args, **kwargs): return pd.DataFrame()

# On essaie d'importer les modules correctement
try:
    # Tentative 1 : Import relatif (si lancé depuis un module parent comme app.py)
    from .data_loader import load_stock_data
    from .strategies import AVAILABLE_ASSETS, apply_strategies, predict_ml_model
except ImportError:
    try:
        # Tentative 2 : Import absolu (si lancé directement dans le dossier)
        from data_loader import load_stock_data
        from strategies import AVAILABLE_ASSETS, apply_strategies, predict_ml_model
    except ImportError:
        # Si tout échoue, on garde les fonctions vides définies au début et on affiche une alerte
        st.warning("⚠️ Impossible de charger 'strategies.py' ou 'data_loader.py'. Mode dégradé activé.")

# --- CONFIG PAGE ---
# Note: set_page_config doit être la première commande Streamlit
# Si app.py l'a déjà fait, cette ligne sera ignorée (ce qui est bien)
try:
    st.set_page_config(page_title="Quant Dashboard", page_icon="📊", layout="wide")
except:
    pass

# ==============================================================================
# HELPER : LOGOS & URLS
# ==============================================================================
def get_logo_url(ticker):
    """Récupère l'URL du logo en fonction du Ticker"""
    domains = {
        "MC.PA": "lvmh.com", "TTE.PA": "totalenergies.com", "AIR.PA": "airbus.com",
        "BNP.PA": "group.bnpparibas", "SAN.PA": "sanofi.com",
        "AAPL": "apple.com", "MSFT": "microsoft.com", "TSLA": "tesla.com", "NVDA": "nvidia.com",
        "SPY": "ssga.com", "CW8.PA": "amundi.com", "QQQ": "invesco.com", "GLD": "spdrgoldshares.com"
    }
    
    if "-USD" in ticker:
        symbol = ticker.split("-")[0].lower()
        return f"https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/{symbol}.png"
    
    if ticker in ["GC=F", "GLD"] and ticker not in domains:
        return "https://cdn-icons-png.flaticon.com/512/272/272530.png"
    
    domain = domains.get(ticker)
    if domain:
        return f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
    
    return "https://cdn-icons-png.flaticon.com/512/4256/4256900.png"

# ==============================================================================
# HELPER : GROS CHIFFRES UNIFIÉS
# ==============================================================================
def show_big_number(label, value, delta=None, fmt="{:.2%}", color_cond="neutral"):
    val_str = fmt.format(value)
    color_str = "" 
    if color_cond == "green_if_pos":
        color_str = ":green" if value > 0 else ":red"
    elif color_cond == "red_if_neg":
        color_str = ":red"
    elif color_cond == "green_bool": 
        color_str = ":green" if value > 0.5 else ":red"
    elif color_cond == "always_blue":
        color_str = ":blue"
    
    st.markdown(f"**{label}**")
    if color_str: st.markdown(f"## {color_str}[{val_str}]")
    else: st.markdown(f"## {val_str}")
        
    if delta:
        d_color = ""
        if "vs" in delta: 
            if "+" in delta.split(" ")[0]: d_color = ":green"
            elif "-" in delta.split(" ")[0]: d_color = ":red"
        if d_color: st.markdown(f"{d_color}[{delta}]")
        else: st.caption(delta)

# ==============================================================================
# HELPER : STYLING TABLEAU
# ==============================================================================
def style_dataframe(df):
    format_dict = {
        'Open': '{:.2f} €', 'High': '{:.2f} €', 'Low': '{:.2f} €', 'Close': '{:.2f} €', 
        'Volume': '{:,.0f}', 'Log_Ret': '{:.2%}',
        'SMA_Short': '{:.2f}', 'SMA_Long': '{:.2f}', 'ROC': '{:.2%}',
        'Upper': '{:.2f}', 'Lower': '{:.2f}', 'SMA': '{:.2f}', 'STD': '{:.2f}',
        'Strat_Active': '{:.2f}', 'Sim_Active': '{:.2f}', 'Position': '{:.0f}'
    }
    styler = df.style.format(format_dict, na_rep="-")
    if 'Log_Ret' in df.columns:
        def color_coding(val):
            if isinstance(val, (int, float)):
                if val > 0: return 'color: #3fb950'
                elif val < 0: return 'color: #f85149'
            return ''
        styler.map(color_coding, subset=['Log_Ret'])
    return styler

# ==============================================================================
# RENDERING PRINCIPAL
# ==============================================================================

def render_dashboard(df_strat, metrics, ticker, asset_name, strat_name, params, pred_days, model_lags):
    
    # 1. HEADER
    st.markdown("### 🔎 Single Asset Analysis")
    with st.container(border=True):
        col_logo, col_name, col_price, col_void = st.columns([0.4, 1.5, 1, 1.5])
        
        last_price = float(df_strat['Close'].iloc[-1])
        prev_price = float(df_strat['Close'].iloc[-2])
        var_abs = last_price - prev_price
        var_pct = (last_price / prev_price) - 1
        
        with col_logo: st.image(get_logo_url(ticker), width=70)
        with col_name:
            st.markdown(f"## **{asset_name}**")
            st.caption(f"Ticker: **{ticker}**")
        with col_price:
            st.metric("Market Price", f"{last_price:.2f} €", f"{var_abs:+.2f} ({var_pct:+.2%})")

    # 2. ASSET OVERVIEW
    st.markdown("#### 📈 Asset Overview")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1: show_big_number("Total Return", metrics['BuyHold_Return'], color_cond="green_if_pos")
        with c2: show_big_number("Volatility", metrics['BuyHold_Vol'], color_cond="neutral")
        with c3: show_big_number("Max Drawdown", metrics['BuyHold_MDD'], color_cond="red_if_neg")
        with c4: show_big_number("Sharpe Ratio", metrics['BuyHold_Sharpe'], fmt="{:.2f}", color_cond="always_blue")

    # 3. RISK ANALYSIS
    st.markdown("#### 📉 Risk Analysis")
    returns_series = df_strat['Close'].pct_change().dropna()
    var_95 = float(np.percentile(returns_series, 5))
    es_95 = float(returns_series[returns_series <= var_95].mean())
    
    with st.container(border=True):
        r1, r2, r3 = st.columns(3)
        with r1: show_big_number("VaR (95%)", var_95, color_cond="neutral") 
        with r2: show_big_number("Expected Shortfall (95%)", es_95, color_cond="neutral")
        with r3: show_big_number("Worst Day", float(returns_series.min()), color_cond="red_if_neg")
        st.divider()
        
        chron_data = [{"value": [d.strftime('%Y-%m-%d'), float(v)], "itemStyle": {"color": "#da3633" if v <= var_95 else "#238636" if v >= 0 else "#4B4B4B"}} for d, v in returns_series.items()]
        sorted_ret = returns_series.sort_values().tolist()
        sorted_data = [{"value": [i, float(v)], "itemStyle": {"color": "#da3633" if v <= var_95 else "#238636" if v >= 0 else "#4B4B4B"}} for i, v in enumerate(sorted_ret)]

        risk_option = {
            "backgroundColor": "#0e1117", "tooltip": {"trigger": "axis"},
            "legend": {"show": True, "data": ["Daily Returns", "Sorted Distribution"], "top": 0, "textStyle": {"color": "#8b949e"}, "selectedMode": "single"},
            "grid": {"left": "3%", "right": "3%", "bottom": "10%", "top": "15%"},
            "xAxis": [{"type": "category", "data": returns_series.index.strftime('%Y-%m-%d').tolist(), "show": True}, {"type": "category", "show": False}],
            "yAxis": {"type": "value", "scale": True, "splitLine": {"show": True, "lineStyle": {"color": "#30363d", "type": "dashed"}}},
            "series": [
                {"name": "Daily Returns", "type": "bar", "data": chron_data, "markLine": {"symbol": "none", "data": [{"yAxis": var_95}], "lineStyle": {"color": "#e6edf3", "type": "dashed"}}},
                {"name": "Sorted Distribution", "type": "bar", "xAxisIndex": 1, "data": sorted_data, "markLine": {"symbol": "none", "data": [{"yAxis": var_95}], "lineStyle": {"color": "#e6edf3", "type": "dashed"}}}
            ]
        }
        st_echarts(options=risk_option, height="350px")

    # 4. STRATEGY SIMULATION (TRADE MARKERS OPACITY)
    # ==========================================================================
    st.markdown("#### ⚙️ Strategy Simulation")
    param_txt = " • ".join([f"**{k}**: {v}" for k,v in params.items()])
    st.info(f"**Strategy Running**: {strat_name} — Settings: {param_txt}")

    with st.container(border=True):
        col_graph_title, col_graph_toggle = st.columns([3, 1])
        with col_graph_title: st.write("")
        with col_graph_toggle: show_trades = st.toggle("Show Trade Markers", value=True)

        dates = df_strat.index.strftime('%Y-%m-%d').tolist()
        data_k = [[round(float(r['Open']),2), round(float(r['Close']),2), round(float(r['Low']),2), round(float(r['High']),2)] for _, r in df_strat.iterrows()]
        line_col = 'Sim_Active' if 'Sim_Active' in df_strat.columns else 'Strat_Active'
        data_line = [round(float(x), 2) for x in df_strat[line_col]]

        series_list = [
            {"name": "Asset", "type": "candlestick", "data": data_k, "itemStyle": {"color": "#238636", "color0": "#da3633", "borderColor": "#238636", "borderColor0": "#da3633"}},
            {"name": "Strategy", "type": "line", "data": data_line, "smooth": True, "showSymbol": False, "lineStyle": {"color": "#58a6ff", "width": 2}}
        ]

        # --- LOGIQUE MARQUEURS AVEC OPACITÉ ---
        if show_trades:
            df_strat['Signal'] = df_strat['Position'].diff().fillna(0)
            buys = df_strat[df_strat['Signal'] > 0]
            sells = df_strat[df_strat['Signal'] < 0]
            
            buy_data = []
            for d, r in buys.iterrows():
                d_str = d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)
                buy_data.append([d_str, float(r['Low']) * 0.98])

            sell_data = []
            for d, r in sells.iterrows():
                d_str = d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)
                sell_data.append([d_str, float(r['High']) * 1.02])

            if buy_data:
                series_list.append({
                    "name": "Buy Signal", "type": "scatter", "symbol": "triangle", "symbolSize": 15,
                    # VERT avec Opacité 0.7
                    "itemStyle": {"color": "#00ff00", "opacity": 0.7}, 
                    "data": buy_data, "z": 10
                })
            if sell_data:
                series_list.append({
                    "name": "Sell Signal", "type": "scatter", "symbol": "triangle", "symbolRotate": 180, "symbolSize": 15,
                    # ROUGE avec Opacité 0.7
                    "itemStyle": {"color": "#ff0000", "opacity": 0.7}, 
                    "data": sell_data, "z": 10
                })

        option = {
            "backgroundColor": "#0e1117", "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
            "legend": {"show": True, "data": ["Asset", "Strategy", "Buy Signal", "Sell Signal"], "top": "0%", "textStyle": {"color": "#e6edf3"}},
            "grid": {"left": "3%", "right": "3%", "bottom": "15%", "top": "15%"},
            "xAxis": {"type": "category", "data": dates, "scale": True},
            "yAxis": {"scale": True, "splitLine": {"show": True, "lineStyle": {"color": "#30363d", "type": "dashed"}}},
            "dataZoom": [{"type": "inside"}, {"show": True, "type": "slider", "top": "90%"}],
            "series": series_list
        }
        st_echarts(options=option, height="450px")

    # 5. PERFORMANCE
    st.markdown("#### ⚖️ Performance & Execution")
    col_comp, col_exec = st.columns([1.5, 1])
    with col_comp:
        with st.container(border=True):
            st.markdown("##### 📊 Strategy vs Benchmark")
            m1, m2 = st.columns(2); m3, m4 = st.columns(2)
            s_ret = metrics['Active_Return']; d_ret = s_ret - metrics['BuyHold_Return']
            with m1: show_big_number("Total Return", s_ret, f"{d_ret:+.2%} vs Asset", color_cond="green_if_pos")
            s_sh = metrics['Active_Sharpe']; d_sh = s_sh - metrics['BuyHold_Sharpe']
            with m2: show_big_number("Sharpe Ratio", s_sh, f"{d_sh:+.2f} vs Asset", fmt="{:.2f}", color_cond="always_blue")
            s_mdd = metrics['Active_MDD']; d_mdd = s_mdd - metrics['BuyHold_MDD']
            with m3: show_big_number("Max Drawdown", s_mdd, f"{d_mdd:+.2%} vs Asset", color_cond="red_if_neg")
            s_vol = metrics['Active_Vol']; d_vol = s_vol - metrics['BuyHold_Vol']
            with m4: show_big_number("Volatility", s_vol, f"{d_vol:+.2%} vs Asset", color_cond="neutral")
    with col_exec:
        with st.container(border=True):
            st.markdown("##### ⏱️ Trade Execution")
            k1, k2 = st.columns(2); k3, k4 = st.columns(2)
            nb_trades = int(metrics.get('Active_Trades', 0))
            win_rate = float(metrics.get('Active_WinRate', 0))
            avg_ret = float((metrics['Active_Return'] / nb_trades)) if nb_trades > 0 else 0
            with k1: show_big_number("Trades", nb_trades, fmt="{}", color_cond="neutral")
            with k2: show_big_number("Win Rate", win_rate, color_cond="green_bool")
            with k3: show_big_number("Avg Ret/Trade", avg_ret, color_cond="green_if_pos")
            with k4: st.write("")

    # 6. FORECASTING (AVEC BACKGROUND MODIFIÉ)
    # ==========================================================================
    st.markdown(f"#### 🧠 Price Forecasting ({pred_days} days | Random Forest)")
    pred_df = predict_ml_model(df_strat, days_ahead=pred_days, n_lags=model_lags)
    
    hist_subset = df_strat
    hist_dates = hist_subset.index.strftime('%Y-%m-%d').tolist()
    pred_dates = pred_df.index.strftime('%Y-%m-%d').tolist()
    all_dates = hist_dates + pred_dates
    
    last_val = float(df_strat['Close'].iloc[-1])
    padding = [None] * (len(hist_dates) - 1)
    
    pred_vals = [float(x) for x in pred_df['Prediction'].values]
    upper_vals = [float(x) for x in pred_df['Upper'].values]
    lower_vals = [float(x) for x in pred_df['Lower'].values]
    
    pred_line = [last_val] + pred_vals
    upper_line = [last_val] + upper_vals
    lower_line = [last_val] + lower_vals
    
    hist_k_data = [[round(float(r['Open']),2), round(float(r['Close']),2), round(float(r['Low']),2), round(float(r['High']),2)] for _, r in hist_subset.iterrows()]
    
    # --- ZONE DE FORECAST (BACKGROUND) ---
    # On définit une zone qui couvre toute la partie "Prédiction"
    # Couleur : Bleu très léger transparent pour indiquer la "Forecast Zone"
    mark_area_data = [
        [
            {"xAxis": hist_dates[-1]}, # Start: Dernier jour historique
            {"xAxis": pred_dates[-1]}  # End: Fin prédiction
        ]
    ]

    s_hist = {"name": "Historical", "type": "candlestick", "data": hist_k_data, "itemStyle": {"color": "#238636", "color0": "#da3633", "borderColor": "#238636", "borderColor0": "#da3633"}}
    
    s_pred = {
        "name": "Random Forest Forecast", 
        "type": "line", 
        "data": padding + pred_line, 
        "showSymbol": False, 
        "lineStyle": {"color": "#58a6ff", "width": 3},
        # CHANGEMENT DU BACKGROUND ICI via MarkArea
        "markArea": {
            "silent": True,
            "itemStyle": {"color": "rgba(58, 166, 255, 0.08)"}, # Fond bleuté léger
            "data": mark_area_data
        }
    }
    
    s_upper = {"name": "Bull Case (95%)", "type": "line", "data": padding + upper_line, "showSymbol": False, "lineStyle": {"color": "#3fb950", "type": "dashed", "width": 1}}
    s_lower = {"name": "Bear Case (5%)", "type": "line", "data": padding + lower_line, "showSymbol": False, "lineStyle": {"color": "#f85149", "type": "dashed", "width": 1}}

    forecast_option = {
        "backgroundColor": "#0e1117", "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "legend": {"show": True, "top": "0%", "textStyle": {"color": "#8b949e"}},
        "grid": {"left": "3%", "right": "3%", "bottom": "15%", "top": "15%"},
        "xAxis": {"type": "category", "data": all_dates, "boundaryGap": True},
        "yAxis": {"type": "value", "scale": True, "splitLine": {"show": True, "lineStyle": {"color": "#30363d", "type": "dashed"}}},
        "series": [s_hist, s_pred, s_upper, s_lower],
        "dataZoom": [{"type": "inside"}, {"show": True, "type": "slider", "top": "90%"}]
    }

    with st.container(border=True):
        st_echarts(options=forecast_option, height="400px")
        c1, c2, c3 = st.columns(3)
        final_pred = float(pred_df['Prediction'].iloc[-1])
        final_ret = (final_pred - last_val) / last_val
        with c1: st.metric("Predicted Price", f"{final_pred:.2f} €", f"{final_ret:+.2%}")
        with c2: st.caption(f"📉 **Lower Bound:** {float(pred_df['Lower'].iloc[-1]):.2f} €")
        with c3: st.caption(f"📈 **Upper Bound:** {float(pred_df['Upper'].iloc[-1]):.2f} €")
        
    # --- 7. DATASET DISPLAY (AJOUT DEMANDÉ) ---
    st.write("---")
    st.markdown("#### 📋 Detailed Data View")
    
    with st.container(border=True):
        # On trie pour voir les données les plus récentes en premier
        df_display = df_strat.sort_index(ascending=False)
        
        # On applique le style (couleurs) et on affiche
        st.dataframe(
            style_dataframe(df_display),
            use_container_width=True,
            height=400 # Hauteur fixe pour scroller dedans
        )

# ==============================================================================
# MAIN LOGIC
# ==============================================================================

def quant_a_ui():
    with st.sidebar:
        # Titre simple, natif, sans fond bizarre
        st.header("⚙️ Config")
        
        # 1. ASSET
        # On utilise st.caption ou du gras pour que ça ne fasse pas un "bloc"
        st.write("---")
        st.markdown("**1️⃣ Asset Selection**")
        
        options_list = list(AVAILABLE_ASSETS.keys())
        if not options_list: options_list = ["Error"]
            
        selected_name = st.selectbox("Ticker", options=options_list, label_visibility="collapsed")
        
        if not selected_name or selected_name == "Error":
            st.error("Stop")
            st.stop()
            
        ticker = AVAILABLE_ASSETS[selected_name]

        # 2. STRATEGY
        st.write("---")
        st.markdown("**2️⃣ Strategy Logic**")
        
        strategy_type = st.selectbox("Method", ["Buy & Hold", "MA Crossover", "Momentum", "Mean Reversion"], label_visibility="collapsed")
        
        params = {}
        if strategy_type == "MA Crossover":
            st.caption("Moving Average Settings")
            params['short_window'] = st.slider("Fast MA", 5, 100, 20)
            params['long_window'] = st.slider("Slow MA", 20, 300, 50)
            if params['short_window'] >= params['long_window']:
                st.warning("⚠️ Fast < Slow required")
                
        elif strategy_type == "Momentum":
            params['mom_window'] = st.slider("Lookback (Days)", 10, 252, 20)
            
        elif strategy_type == "Mean Reversion":
            params['bb_window'] = st.slider("BB Period", 10, 100, 20)
            params['bb_std'] = st.slider("Std Dev", 1.0, 4.0, 2.0)

        # 3. TIMEFRAME
        st.write("---")
        st.markdown("**3️⃣ Timeframe**")
        
        c1, c2 = st.columns(2)
        with c1: 
            period = st.selectbox("Period", ["1y", "2y", "5y", "max"], index=1, label_visibility="collapsed")
        with c2: 
            interval = st.selectbox("Interval", ["1d", "1wk"], index=0, label_visibility="collapsed")

        # 4. FORECAST
        st.write("---")
        st.markdown("**🔮 Forecast Settings**")
        
        pred_days = st.slider("Horizon (Days)", 10, 90, 30)
        model_lags = st.slider("Context (Lags)", 5, 60, 20)
        
        st.write("") # Petit espace
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            st.session_state.run_algo = True

    # --- EXECUTION ---
    if st.session_state.get('run_algo', False):
        df = load_stock_data(ticker, period=period, interval=interval)
        if df is not None and not df.empty:
            df_strat, metrics = apply_strategies(df, strategy_type, params)
            
            first_idx = df_strat['Position'].first_valid_index()
            if first_idx:
                base_val = float(df_strat.loc[first_idx, 'Strat_Active'])
                start_price = float(df_strat.loc[first_idx, 'Close'])
                df_strat['Sim_Active'] = (df_strat['Strat_Active'] / base_val) * start_price
                
                # Affiche le dashboard
                render_dashboard(df_strat, metrics, ticker, selected_name, strategy_type, params, pred_days, model_lags)
            else:
                st.warning("Strategy inactive (no signals).")
        else:
            st.error("No data available.")
    else:
        st.info("👈 Select an asset and click Run Analysis.")