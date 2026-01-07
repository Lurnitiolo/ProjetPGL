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
# HELPER : GROS CHIFFRES NATIFS (CLEAN)
# ==============================================================================
def show_big_number(label, value, fmt="{:.2%}", color_cond="neutral"):
    val_str = fmt.format(value)
    color = "" 
    
    if color_cond == "green_if_pos":
        if value > 0: color = ":green"
        elif value < 0: color = ":red"
    elif color_cond == "red_if_neg":
        color = ":red"
    elif color_cond == "green_bool": 
        color = ":green" if value > 0.5 else ":red"
    elif color_cond == "always_blue":
        color = ":blue"
        
    st.markdown(f"**{label}**")
    if color:
        st.markdown(f"## {color}[{val_str}]")
    else:
        st.markdown(f"## {val_str}")

# ==============================================================================
# RENDERING PRINCIPAL
# ==============================================================================
def render_dashboard_risk_v2(df_strat, metrics, ticker, asset_name, strat_name, params):
    
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
    # 4. RISK ANALYSIS (AVEC TOGGLE ET COULEURS)
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
        
        # B. Préparation des Données pour le Plot
        
        # Série 1 : Chronologique (Dates)
        chron_data = []
        for date, val in returns_series.items():
            # Couleur : Vert si > 0, Rouge si < 0
            color = "#238636" if val >= 0 else "#da3633"
            chron_data.append({
                "value": [date.strftime('%Y-%m-%d'), round(float(val), 4)],
                "itemStyle": {"color": color}
            })

        # Série 2 : Triée (S-Curve)
        # On trie les valeurs
        sorted_ret = returns_series.sort_values().tolist()
        sorted_data = []
        for i, val in enumerate(sorted_ret):
            color = "#238636" if val >= 0 else "#da3633"
            # L'axe X est juste un index (0, 1, 2, ...)
            sorted_data.append({
                "value": [i, round(float(val), 4)],
                "itemStyle": {"color": color}
            })

        # C. Configuration ECharts Double Axe
        risk_option = {
            "backgroundColor": "#0e1117",
            "tooltip": {"trigger": "axis"},
            "legend": {
                "show": True,
                "data": ["Chronological", "Sorted Distribution"],
                "top": "0%",
                "textStyle": {"color": "#e6edf3"},
                # Par défaut, on sélectionne 'Chronological' et on désélectionne 'Sorted'
                # L'utilisateur clique pour inverser
                "selected": {
                    "Chronological": True,
                    "Sorted Distribution": False
                }
            },
            "grid": {"left": "3%", "right": "3%", "bottom": "10%", "top": "15%"},
            
            # DEUX AXES X SUPERPOSÉS
            "xAxis": [
                {
                    "type": "category", 
                    "data": returns_series.index.strftime('%Y-%m-%d').tolist(), 
                    "gridIndex": 0,
                    "show": True # Axe des dates visible par défaut
                },
                {
                    "type": "category",
                    # Pas de labels pour le trié, juste l'échelle
                    "show": False, 
                    "gridIndex": 0
                }
            ],
            "yAxis": {
                "type": "value", 
                "splitLine": {"show": True, "lineStyle": {"color": "#30363d", "type": "dashed"}},
            },
            "series": [
                {
                    "name": "Chronological",
                    "type": "bar",
                    "xAxisIndex": 0, # Utilise l'axe des dates
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
                    "name": "Sorted Distribution",
                    "type": "bar", # Bar chart rend mieux qu'une ligne pour voir la densité couleur
                    "xAxisIndex": 1, # Utilise l'axe caché (index)
                    "data": sorted_data,
                    "barWidth": "100%", # Collé pour faire un effet de surface
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
            "grid": {"left": "3%", "right": "3%", "bottom": "15%"},
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
            m1.metric("Total Return (Strategy)", f"{s_ret:+.2%}", f"{d_ret:+.2%} vs Asset")
            
            s_sh = metrics['Active_Sharpe']
            d_sh = s_sh - metrics['BuyHold_Sharpe']
            m2.metric("Sharpe Ratio", f"{s_sh:.2f}", f"{d_sh:+.2f} vs Asset")
            
            s_mdd = metrics['Active_MDD']
            d_mdd = s_mdd - metrics['BuyHold_MDD']
            m3.metric("Max Drawdown", f"{s_mdd:.2%}", f"{d_mdd:+.2%} vs Asset")
            
            s_vol = metrics['Active_Vol']
            d_vol = s_vol - metrics['BuyHold_Vol']
            m4.metric("Volatility", f"{s_vol:.2%}", f"{d_vol:+.2%} vs Asset", delta_color="inverse")

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
# MAIN LOGIC
# ==============================================================================
def quant_a_ui():
    with st.sidebar:
        st.header("🎛️ Settings")
        
        st.caption("ASSET")
        selected_name = st.selectbox("Select Asset", options=list(AVAILABLE_ASSETS.keys()))
        ticker = AVAILABLE_ASSETS[selected_name]

        st.markdown("---")
        st.caption("STRATEGY")
        strategy_type = st.selectbox("Type", ["Buy & Hold", "MA Crossover", "Momentum", "Mean Reversion"], index=1)
        
        params = {}
        if strategy_type == "MA Crossover":
            c1, c2 = st.columns(2)
            with c1: params['short_window'] = st.number_input("Fast MA", 5, 100, 20)
            with c2: params['long_window'] = st.number_input("Slow MA", 50, 300, 50)
        elif strategy_type == "Momentum":
            params['mom_window'] = st.number_input("Lookback", 10, 252, 20)
        elif strategy_type == "Mean Reversion":
            c1, c2 = st.columns(2)
            with c1: params['bb_window'] = st.number_input("BB Period", 10, 100, 20)
            with c2: params['bb_std'] = st.number_input("BB Std", 1.0, 4.0, 2.0)

        st.markdown("---")
        st.caption("DATA")
        c1, c2 = st.columns(2)
        with c1: period = st.selectbox("Period", ["1y", "2y", "5y", "max"], index=1)
        with c2: interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 RUN ANALYSIS", use_container_width=True):
            st.session_state.show_analysis = True

    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False

    if st.session_state.show_analysis:
        with st.spinner('Computing strategy...'):
            df = load_stock_data(ticker, period=period, interval=interval)
            
            if df is not None and not df.empty:
                df_strat, metrics = apply_strategies(df, strategy_type, params)
                
                # REBASE
                first_valid = df_strat['Position'].first_valid_index() or df_strat.index[0]
                start_price = df_strat.loc[first_valid, 'Close']
                
                st.sidebar.markdown("---")
                offset = st.sidebar.slider("Rebase Offset (€)", -50.0, 50.0, 0.0)
                
                strat_start = start_price + offset
                base_idx_val = df_strat.loc[first_valid, 'Strat_Active']
                df_strat['Sim_Active'] = (df_strat['Strat_Active'] / base_idx_val) * strat_start
                
                # RENDER
                render_dashboard_risk_v2(df_strat, metrics, ticker, selected_name, strategy_type, params)

                # RAW DATA
                with st.expander("🗃️ View Raw Data"):
                    st.dataframe(df_strat.sort_index(ascending=False), use_container_width=True)

            else:
                st.error("❌ No data found.")
    else:
        st.info("👈 Please initialize the model from the sidebar.")