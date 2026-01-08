import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_echarts import st_echarts

# --- IMPORTS LOCAUX ---
try:
    from .data_loader import load_stock_data
    from .strategies import AVAILABLE_ASSETS, apply_strategies
except ImportError:
    st.warning("⚠️ Modules locaux introuvables.")
    AVAILABLE_ASSETS = {"Airbus": "AIR.PA", "LVMH": "MC.PA"}
    def load_stock_data(*args, **kwargs): return None
    def apply_strategies(*args, **kwargs): return pd.DataFrame(), {}

# --- CONFIG PAGE ---
st.set_page_config(page_title="Quant Dashboard", page_icon="💠", layout="wide")

# ==============================================================================
# HELPER : GROS CHIFFRES UNIFIÉS
# ==============================================================================
def show_big_number(label, value, delta=None, fmt="{:.2%}", color_cond="neutral"):
    val_str = fmt.format(value)
    
    # Gestion Couleur Valeur Principale
    color_str = "" 
    if color_cond == "green_if_pos":
        color_str = ":green" if value > 0 else ":red"
    elif color_cond == "red_if_neg":
        color_str = ":red"
    elif color_cond == "green_bool": 
        color_str = ":green" if value > 0.5 else ":red"
    elif color_cond == "always_blue":
        color_str = ":blue"
    
    # Affichage
    st.markdown(f"**{label}**")
    if color_str:
        st.markdown(f"## {color_str}[{val_str}]")
    else:
        st.markdown(f"## {val_str}")
        
    if delta:
        d_color = ""
        if "vs" in delta: 
            if "+" in delta.split(" ")[0]: d_color = ":green"
            elif "-" in delta.split(" ")[0]: d_color = ":red"
        
        if d_color:
            st.markdown(f"{d_color}[{delta}]")
        else:
            st.caption(delta)


# ==============================================================================
# RENDERING PRINCIPAL
# ==============================================================================
def render_dashboard_final_v6(df_strat, metrics, ticker, asset_name, strat_name, params):
    
    # --------------------------------------------------------------------------
    # 1. TITRE GLOBAL
    # --------------------------------------------------------------------------
    st.markdown("### 💠 Single Asset Analysis")

    # --------------------------------------------------------------------------
    # 2. ASSET INFOS
    # --------------------------------------------------------------------------
    with st.container(border=True):
        col_name, col_price, col_void = st.columns([1.5, 1, 2])
        
        last_price = df_strat['Close'].iloc[-1]
        prev_price = df_strat['Close'].iloc[-2]
        var_abs = last_price - prev_price
        var_pct = (last_price / prev_price) - 1

        with col_name:
            st.title(asset_name)
            st.caption(f"Ticker Symbol: **{ticker}**")
        with col_price:
            st.metric("Market Price", f"{last_price:.2f} €", f"{var_abs:+.2f} ({var_pct:+.2%})")

    # --------------------------------------------------------------------------
    # 3. ASSET OVERVIEW
    # --------------------------------------------------------------------------
    st.markdown("#### 🏦 Asset Overview")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1: show_big_number("Total Return", metrics['BuyHold_Return'], color_cond="green_if_pos")
        with c2: show_big_number("Volatility", metrics['BuyHold_Vol'], color_cond="neutral")
        with c3: show_big_number("Max Drawdown", metrics['BuyHold_MDD'], color_cond="red_if_neg")
        with c4: show_big_number("Sharpe Ratio", metrics['BuyHold_Sharpe'], fmt="{:.2f}", color_cond="always_blue")

    # --------------------------------------------------------------------------
    # 4. RISK ANALYSIS (TOGGLE + LEGENDE)
    # --------------------------------------------------------------------------
    st.markdown("#### ⚠️ Risk Analysis")

    returns_series = df_strat['Close'].pct_change().dropna()
    confidence_level = 0.95
    var_95 = np.percentile(returns_series, (1 - confidence_level) * 100)
    es_95 = returns_series[returns_series <= var_95].mean()
    
    with st.container(border=True):
        # A. Métriques
        r1, r2, r3 = st.columns(3)
        with r1: show_big_number("VaR (95%)", var_95, color_cond="neutral") 
        with r2: show_big_number("Expected Shortfall (95%)", es_95, color_cond="neutral")
        with r3: show_big_number("Worst Day", returns_series.min(), color_cond="red_if_neg")
            
        st.divider()
        
        # B. Plot Risk
        chron_data = []
        for date, val in returns_series.items():
            if val <= var_95: color = "#da3633" # Rouge (Breach)
            elif val >= 0: color = "#238636" # Vert (Positif)
            else: color = "#30363d" # Gris (Négatif normal)
            chron_data.append({"value": [date.strftime('%Y-%m-%d'), round(float(val), 4)], "itemStyle": {"color": color}})

        sorted_ret = returns_series.sort_values().tolist()
        sorted_data = []
        for i, val in enumerate(sorted_ret):
            if val <= var_95: color = "#da3633"
            elif val >= 0: color = "#238636"
            else: color = "#30363d"
            sorted_data.append({"value": [i, round(float(val), 4)], "itemStyle": {"color": color}})

        risk_option = {
            "backgroundColor": "#0e1117",
            "tooltip": {"trigger": "axis"},
            "legend": {
                "show": True,
                "data": ["Daily Returns", "Returns Distribution"], 
                "top": "0%",
                "textStyle": {"color": "#8b949e"},
                "selected": {"Daily Returns": True, "Returns Distribution": False}
            },
            "grid": {"left": "3%", "right": "3%", "bottom": "10%", "top": "15%"},
            "xAxis": [
                {"type": "category", "data": returns_series.index.strftime('%Y-%m-%d').tolist(), "gridIndex": 0, "show": True},
                {"type": "category", "show": False, "gridIndex": 0}
            ],
            "yAxis": {"type": "value", "splitLine": {"show": True, "lineStyle": {"color": "#30363d", "type": "dashed"}}},
            "series": [
                {
                    "name": "Daily Returns",
                    "type": "bar",
                    "xAxisIndex": 0,
                    "data": chron_data,
                    "barWidth": "60%",
                    "markLine": {
                        "symbol": "none",
                        "data": [{"yAxis": round(float(var_95), 4)}],
                        "lineStyle": {"color": "#e6edf3", "type": "dashed", "width": 2},
                        "label": {"formatter": f"VaR: {var_95:.2%}", "color": "#e6edf3"}
                    }
                },
                {
                    "name": "Returns Distribution",
                    "type": "bar",
                    "xAxisIndex": 1,
                    "data": sorted_data,
                    "barWidth": "100%",
                    "markLine": {
                        "symbol": "none",
                        "data": [{"yAxis": round(float(var_95), 4)}],
                        "lineStyle": {"color": "#e6edf3", "type": "dashed", "width": 2},
                         "label": {"formatter": f"VaR: {var_95:.2%}", "color": "#e6edf3"}
                    }
                }
            ]
        }
        st_echarts(options=risk_option, height="350px")

    # --------------------------------------------------------------------------
    # 5. STRATEGY SIMULATION
    # --------------------------------------------------------------------------
    st.markdown("#### 📡 Strategy Simulation")
    param_txt = "  •  ".join([f"**{k}**: {v}" for k,v in params.items()])
    st.info(f"**Strategy Running**: {strat_name}  —  Settings: {param_txt}")

    with st.container(border=True):
        dates = df_strat.index.strftime('%Y-%m-%d').tolist()
        data_k = [[round(float(r['Open']),2), round(float(r['Close']),2), round(float(r['Low']),2), round(float(r['High']),2)] for _, r in df_strat.iterrows()]
        line_col = 'Sim_Active' if 'Sim_Active' in df_strat.columns else 'Strat_Active'
        data_line = [round(float(x), 2) for x in df_strat[line_col]]

        option = {
            "backgroundColor": "#0e1117",
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
            "legend": {
                "show": True,
                "data": ["Asset", "Strategy"],
                "top": "0%",
                "right": "2%",
                "textStyle": {"color": "#e6edf3"}
            },
            "grid": {"left": "3%", "right": "3%", "bottom": "15%", "top": "15%"},
            "xAxis": {"type": "category", "data": dates, "scale": True},
            "yAxis": {"scale": True, "splitLine": {"show": True, "lineStyle": {"color": "#30363d", "type": "dashed"}}},
            "dataZoom": [{"type": "inside"}, {"show": True, "type": "slider", "top": "90%"}],
            "series": [
                {"name": "Asset", "type": "candlestick", "data": data_k, "itemStyle": {"color": "#238636", "color0": "#da3633", "borderColor": "#238636", "borderColor0": "#da3633"}},
                {"name": "Strategy", "type": "line", "data": data_line, "smooth": True, "showSymbol": False, "lineStyle": {"color": "#58a6ff", "width": 2}}
            ]
        }
        st_echarts(options=option, height="450px")

    # --------------------------------------------------------------------------
    # 6. PERFORMANCE & EXECUTION
    # --------------------------------------------------------------------------
    st.markdown("#### ⚔️ Performance & Execution")
    col_comp, col_exec = st.columns([1.5, 1])
    
    with col_comp:
        with st.container(border=True):
            st.markdown("##### 📊 Strategy vs Benchmark")
            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)
            
            s_ret = metrics['Active_Return']
            d_ret = s_ret - metrics['BuyHold_Return']
            with m1: show_big_number("Total Return", s_ret, f"{d_ret:+.2%} vs Asset", color_cond="green_if_pos")
            
            s_sh = metrics['Active_Sharpe']
            d_sh = s_sh - metrics['BuyHold_Sharpe']
            with m2: show_big_number("Sharpe Ratio", s_sh, f"{d_sh:+.2f} vs Asset", fmt="{:.2f}", color_cond="always_blue")
            
            s_mdd = metrics['Active_MDD']
            d_mdd = s_mdd - metrics['BuyHold_MDD']
            with m3: show_big_number("Max Drawdown", s_mdd, f"{d_mdd:+.2%} vs Asset", color_cond="red_if_neg")
            
            s_vol = metrics['Active_Vol']
            d_vol = s_vol - metrics['BuyHold_Vol']
            with m4: show_big_number("Volatility", s_vol, f"{d_vol:+.2%} vs Asset", color_cond="neutral")

    with col_exec:
        with st.container(border=True):
            st.markdown("##### ⚡ Trade Execution")
            k1, k2 = st.columns(2)
            k3, k4 = st.columns(2)
            
            nb_trades = metrics.get('Active_Trades', 0)
            win_rate = metrics.get('Active_WinRate', 0)
            avg_ret = (metrics['Active_Return'] / nb_trades) if nb_trades > 0 else 0
            
            with k1: show_big_number("Trades", nb_trades, fmt="{}", color_cond="neutral")
            with k2: show_big_number("Win Rate", win_rate, color_cond="green_bool")
            with k3: show_big_number("Avg Ret/Trade", avg_ret, color_cond="green_if_pos")
            with k4: st.write("")


# ==============================================================================
# MAIN LOGIC (SIDEBAR & ERROR HANDLING)
# ==============================================================================
def quant_a_ui():
    with st.sidebar:
        st.markdown("## ⚙️ **Dashboard Config**")
        st.divider()
        
        # --- ASSET ---
        st.markdown("### 1️⃣ Asset Selection")
        selected_name = st.selectbox("Ticker Symbol", options=list(AVAILABLE_ASSETS.keys()), label_visibility="collapsed")
        ticker = AVAILABLE_ASSETS[selected_name]

        st.divider()

        # --- STRATEGY ---
        st.markdown("### 2️⃣ Strategy Logic")
        strategy_type = st.selectbox("Method", ["Buy & Hold", "MA Crossover", "Momentum", "Mean Reversion"], label_visibility="collapsed")
        
        params = {}
        if strategy_type == "MA Crossover":
            st.caption("Moving Average Periods")
            params['short_window'] = st.slider("⚡ Fast MA (Days)", 5, 100, 20)
            params['long_window'] = st.slider("🐢 Slow MA (Days)", 20, 300, 50)
            if params['short_window'] >= params['long_window']: st.warning("⚠️ Fast MA should be < Slow MA")
        
        elif strategy_type == "Momentum":
            st.caption("Trend Following")
            params['mom_window'] = st.slider("📅 Lookback Period (Days)", 10, 252, 20)
        
        elif strategy_type == "Mean Reversion":
            st.caption("Bollinger Bands Settings")
            params['bb_window'] = st.slider("⏳ BB Period", 10, 100, 20)
            params['bb_std'] = st.slider("sigma Standard Deviation", 1.0, 4.0, 2.0, step=0.1)

        st.divider()

        # --- DATA ---
        st.markdown("### 3️⃣ Timeframe")
        c1, c2 = st.columns(2)
        with c1: 
            st.caption("History")
            period = st.selectbox("History", ["1y", "2y", "5y", "max"], index=1, label_visibility="collapsed")
        with c2: 
            st.caption("Step")
            interval = st.selectbox("Step", ["1d", "1wk", "1mo"], index=0, label_visibility="collapsed")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- BUTTON ---
        if st.button("📈 Launch Analytics", use_container_width=True, type="primary"):
            st.session_state.show_analysis = True

    # --- EXECUTION & GESTION ERREURS ---
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False

    if st.session_state.show_analysis:
        with st.spinner('Calculating Alpha... 🧠'):
            df = load_stock_data(ticker, period=period, interval=interval)
            
            if df is not None and not df.empty:
                # --- SAFETY CHECK ANTI-CRASH ---
                # On vérifie si on a assez de bougies par rapport aux paramètres
                min_lookback = 0
                if strategy_type == "MA Crossover": min_lookback = params['long_window']
                elif strategy_type == "Momentum": min_lookback = params['mom_window']
                elif strategy_type == "Mean Reversion": min_lookback = params['bb_window']
                
                # Si l'historique est plus court que la période nécessaire pour l'indicateur
                if len(df) < min_lookback:
                    st.error(f"⚠️ Not enough data! (Need {min_lookback} periods, got {len(df)})")
                    st.info("💡 **Fix:** Increase 'History' (e.g. 'max') or choose a smaller 'Step' (e.g. '1d').")
                    st.stop() # Arrêt propre du script
                # -------------------------------

                df_strat, metrics = apply_strategies(df, strategy_type, params)
                
                # Vérification que la stratégie a bien généré des positions (sinon crash au rebase)
                first_valid = df_strat['Position'].first_valid_index()
                if first_valid is None:
                    st.warning("⚠️ Strategy logic yielded no trades with current settings.")
                    st.stop()
                
                # SIMULATION (REBASE SUR CLOSE)
                strat_start = df_strat.loc[first_valid, 'Close']
                base_idx_val = df_strat.loc[first_valid, 'Strat_Active']
                df_strat['Sim_Active'] = (df_strat['Strat_Active'] / base_idx_val) * strat_start
                
                # RENDER
                render_dashboard_final_v6(df_strat, metrics, ticker, selected_name, strategy_type, params)

                # RAW DATA
                with st.expander("🗃️ View Raw Data"):
                    st.dataframe(df_strat.sort_index(ascending=False), use_container_width=True)

            else:
                st.error("❌ No data found.")
    else:
        st.info("👈 Please select an asset and click Launch Analytics.")