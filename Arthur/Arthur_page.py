import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_echarts import st_echarts

# --- IMPORTS LOCAUX ---
try:
    from .data_loader import load_stock_data
    from .strategies import AVAILABLE_ASSETS, apply_strategies, predict_ml_model
except ImportError:
    st.warning("⚠️ Modules locaux introuvables.")
    AVAILABLE_ASSETS = {"Airbus": "AIR.PA", "LVMH": "MC.PA"}
    def load_stock_data(*args, **kwargs): return None
    def apply_strategies(*args, **kwargs): return pd.DataFrame(), {}
    def predict_ml_model(*args, **kwargs): return pd.DataFrame()

# --- CONFIG PAGE ---
st.set_page_config(page_title="Quant Dashboard", page_icon="📊", layout="wide")

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
def render_dashboard_final_v16(df_strat, metrics, ticker, asset_name, strat_name, params, pred_days, model_lags):
    
    # 1. HEADER
    st.markdown("### 🔎 Single Asset Analysis") # Loupe pour l'analyse
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

    # 2. ASSET OVERVIEW
    st.markdown("#### 📈 Asset Overview") # Graphique montant simple
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1: show_big_number("Total Return", metrics['BuyHold_Return'], color_cond="green_if_pos")
        with c2: show_big_number("Volatility", metrics['BuyHold_Vol'], color_cond="neutral")
        with c3: show_big_number("Max Drawdown", metrics['BuyHold_MDD'], color_cond="red_if_neg")
        with c4: show_big_number("Sharpe Ratio", metrics['BuyHold_Sharpe'], fmt="{:.2f}", color_cond="always_blue")

    # 3. RISK ANALYSIS
    st.markdown("#### 📉 Risk Analysis") # Graphique descendant pour le risque
    returns_series = df_strat['Close'].pct_change().dropna()
    confidence_level = 0.95
    var_95 = np.percentile(returns_series, (1 - confidence_level) * 100)
    es_95 = returns_series[returns_series <= var_95].mean()
    
    with st.container(border=True):
        r1, r2, r3 = st.columns(3)
        with r1: show_big_number("VaR (95%)", var_95, color_cond="neutral") 
        with r2: show_big_number("Expected Shortfall (95%)", es_95, color_cond="neutral")
        with r3: show_big_number("Worst Day", returns_series.min(), color_cond="red_if_neg")
        st.divider()
        
        chron_data = []
        for date, val in returns_series.items():
            color = "#da3633" if val <= var_95 else "#238636" if val >= 0 else "#57524e"
            chron_data.append({"value": [date.strftime('%Y-%m-%d'), val], "itemStyle": {"color": color}})

        sorted_ret = returns_series.sort_values().tolist()
        sorted_data = []
        for i, val in enumerate(sorted_ret):
            color = "#da3633" if val <= var_95 else "#238636" if val >= 0 else "#57524e"
            sorted_data.append({"value": [i, val], "itemStyle": {"color": color}})

        risk_option = {
            "backgroundColor": "#0e1117", "tooltip": {"trigger": "axis"},
            "legend": {"show": True, "data": ["Daily Returns", "Sorted Distribution"], "top": "0%", "textStyle": {"color": "#8b949e"}, "selectedMode": "single", "selected": {"Daily Returns": True, "Sorted Distribution": False}},
            "grid": {"left": "3%", "right": "3%", "bottom": "10%", "top": "15%"},
            "xAxis": [{"type": "category", "data": returns_series.index.strftime('%Y-%m-%d').tolist(), "gridIndex": 0, "show": True}, {"type": "category", "show": False, "gridIndex": 0}],
            "yAxis": {"type": "value", "scale": True, "splitLine": {"show": True, "lineStyle": {"color": "#30363d", "type": "dashed"}}},
            "series": [
                {"name": "Daily Returns", "type": "bar", "xAxisIndex": 0, "data": chron_data, "barWidth": "60%", "markLine": {"symbol": "none", "data": [{"yAxis": var_95}], "lineStyle": {"color": "#e6edf3", "type": "dashed", "width": 2}, "label": {"formatter": f"VaR: {var_95:.2%}", "color": "#e6edf3"}}},
                {"name": "Sorted Distribution", "type": "bar", "xAxisIndex": 1, "data": sorted_data, "barWidth": "100%", "markLine": {"symbol": "none", "data": [{"yAxis": var_95}], "lineStyle": {"color": "#e6edf3", "type": "dashed", "width": 2}, "label": {"formatter": f"VaR: {var_95:.2%}", "color": "#e6edf3"}}}
            ]
        }
        st_echarts(options=risk_option, height="350px")

    # 4. STRATEGY SIMULATION
    st.markdown("#### ⚙️ Strategy Simulation") # Engrenage pour la mécanique
    param_txt = "  •  ".join([f"**{k}**: {v}" for k,v in params.items()])
    st.info(f"**Strategy Running**: {strat_name}  —  Settings: {param_txt}")

    with st.container(border=True):
        dates = df_strat.index.strftime('%Y-%m-%d').tolist()
        data_k = [[round(float(r['Open']),2), round(float(r['Close']),2), round(float(r['Low']),2), round(float(r['High']),2)] for _, r in df_strat.iterrows()]
        line_col = 'Sim_Active' if 'Sim_Active' in df_strat.columns else 'Strat_Active'
        data_line = [round(float(x), 2) for x in df_strat[line_col]]

        option = {
            "backgroundColor": "#0e1117", "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
            "legend": {"show": True, "data": ["Asset", "Strategy"], "top": "0%", "right": "2%", "textStyle": {"color": "#e6edf3"}},
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

    # 5. PERFORMANCE
    st.markdown("#### ⚖️ Performance & Execution") # Balance pour comparer
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
            st.markdown("##### ⏱️ Trade Execution") # Chrono pour le timing
            k1, k2 = st.columns(2); k3, k4 = st.columns(2)
            nb_trades = metrics.get('Active_Trades', 0); win_rate = metrics.get('Active_WinRate', 0)
            avg_ret = (metrics['Active_Return'] / nb_trades) if nb_trades > 0 else 0
            with k1: show_big_number("Trades", nb_trades, fmt="{}", color_cond="neutral")
            with k2: show_big_number("Win Rate", win_rate, color_cond="green_bool")
            with k3: show_big_number("Avg Ret/Trade", avg_ret, color_cond="green_if_pos")
            with k4: st.write("")

    # 6. FORECASTING
    st.markdown(f"#### 🧠 Price Forecasting ({pred_days} days | Random Forest)") # Cerveau pour l'IA
    pred_df = predict_ml_model(df_strat, days_ahead=pred_days, n_lags=model_lags)
    
    hist_subset = df_strat
    hist_dates = hist_subset.index.strftime('%Y-%m-%d').tolist()
    pred_dates = pred_df.index.strftime('%Y-%m-%d').tolist()
    
    hist_k_data = [[round(float(r['Open']),2), round(float(r['Close']),2), round(float(r['Low']),2), round(float(r['High']),2)] for _, r in hist_subset.iterrows()]
    last_val = hist_subset['Close'].iloc[-1]
    
    pred_line = [last_val] + [round(x, 2) for x in pred_df['Prediction'].values]
    upper_line = [last_val] + [round(x, 2) for x in pred_df['Upper'].values]
    lower_line = [last_val] + [round(x, 2) for x in pred_df['Lower'].values]
    
    all_dates = hist_dates + pred_dates
    padding = [None] * (len(hist_dates) - 1)
    
    s_hist = {"name": "Historical", "type": "candlestick", "data": hist_k_data, "itemStyle": {"color": "#238636", "color0": "#da3633", "borderColor": "#238636", "borderColor0": "#da3633"}}
    s_pred = {"name": "Random Forest Forecast", "type": "line", "data": padding + pred_line, "showSymbol": False, "lineStyle": {"color": "#58a6ff", "width": 3}, "markArea": {"silent": True, "itemStyle": {"color": "rgba(88, 166, 255, 0.1)"}, "data": [[{"xAxis": hist_dates[-1]}, {"xAxis": pred_dates[-1]}]]}}
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
        final_pred = pred_df['Prediction'].iloc[-1]
        final_ret = (final_pred - last_price) / last_price
        with c1: st.metric("Predicted Price", f"{final_pred:.2f} €", f"{final_ret:+.2%}")
        with c2: st.caption(f"📉 **Lower Bound:** {pred_df['Lower'].iloc[-1]:.2f} €")
        with c3: st.caption(f"📈 **Upper Bound:** {pred_df['Upper'].iloc[-1]:.2f} €")

# ==============================================================================
# MAIN LOGIC
# ==============================================================================
def quant_a_ui():
    with st.sidebar:
        st.markdown("## 🎛️ **Dashboard Config**") # Boutons de contrôle
        st.divider()
        st.markdown("### 1️⃣ Asset Selection")
        selected_name = st.selectbox("Ticker Symbol", options=list(AVAILABLE_ASSETS.keys()), label_visibility="collapsed")
        ticker = AVAILABLE_ASSETS[selected_name]
        st.divider()
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
        st.markdown("### 3️⃣ Timeframe")
        c1, c2 = st.columns(2)
        with c1: period = st.selectbox("History", ["1y", "2y", "5y", "max"], index=1, label_visibility="collapsed")
        with c2: interval = st.selectbox("Step", ["1d", "1wk", "1mo"], index=0, label_visibility="collapsed")
        st.divider()
        st.markdown("### 🔮 Forecast")
        pred_days = st.slider("Horizon (Days)", 5, 90, 30)
        model_lags = st.slider("Model Context (Past Days)", 5, 60, 20)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶️ Start Analysis", use_container_width=True, type="primary"): # Bouton Play simple
            st.session_state.show_analysis = True

    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False

    if st.session_state.show_analysis:
        with st.spinner('Calculating Alpha... 🧠'):
            df = load_stock_data(ticker, period=period, interval=interval)
            if df is not None and not df.empty:
                min_lookback = 0
                if strategy_type == "MA Crossover": min_lookback = params['long_window']
                elif strategy_type == "Momentum": min_lookback = params['mom_window']
                elif strategy_type == "Mean Reversion": min_lookback = params['bb_window']
                if len(df) < min_lookback:
                    st.error(f"⚠️ Not enough data! (Need {min_lookback} periods, got {len(df)})")
                    st.stop()

                df_strat, metrics = apply_strategies(df, strategy_type, params)
                first_valid = df_strat['Position'].first_valid_index()
                if first_valid is None:
                    st.warning("⚠️ Strategy logic yielded no trades with current settings.")
                    st.stop()
                
                strat_start = df_strat.loc[first_valid, 'Close']
                base_idx_val = df_strat.loc[first_valid, 'Strat_Active']
                df_strat['Sim_Active'] = (df_strat['Strat_Active'] / base_idx_val) * strat_start
                
                render_dashboard_final_v16(df_strat, metrics, ticker, selected_name, strategy_type, params, pred_days, model_lags)

                with st.expander("🗃️ View Full Dataset"):
                    st.dataframe(style_dataframe(df_strat.sort_index(ascending=False)), use_container_width=True)
            else:
                st.error("❌ No data found.")
    else:
        st.info("👈 Please select an asset and click Start Analysis.")