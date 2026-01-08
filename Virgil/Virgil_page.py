import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
from .data_loader import load_stock_data, get_logo_url
from .strategies import apply_strategies, calculate_metrics
from .render_efficiency import min_max_scale, render_portfolio_simulation, apply_preset_callback

def quant_b_ui():
    st.header("📈 Portfolio Analysis")

    # --- 1. INITIALISATION DE LA MÉMOIRE ---
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = {}
    if 'tickers_analyzed' not in st.session_state:
        st.session_state.tickers_analyzed = []
    if 'asset_settings' not in st.session_state:
        st.session_state.asset_settings = {}

    with st.sidebar:
        st.title("⚙️ Global Settings")
        cap_init = st.number_input("Capital Initial ($)", value=1000, key="global_cap")

        tickers_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC-USD", "EURUSD=X", "GC=F", "MC.PA", "TTE.PA", "AIR.PA", "BNP.PA", "SAN.PA", "NVDA", "SPY", "CW8.PA", "QQQ", "GLD", "ETH-USD"]
        selected_tickers = st.multiselect("Portfolio's Assets", options=tickers_list, default=["AAPL", "MSFT", "BTC-USD"], key="global_tickers")
        
        st.divider()

        if st.session_state.portfolio_data:
            st.markdown("### ⚖️ Portfolio Presets")
            with st.container(border=True):
                data_dict = st.session_state.portfolio_data
                tickers = st.session_state.tickers_analyzed
                
                stats = {}
                for t in tickers:
                    p_ret, p_mdd = calculate_metrics(data_dict[t]['Strat_Momentum'])
                    vol = data_dict[t]['Strat_Returns'].std() * np.sqrt(252)
                    
                    stats[t] = {
                        'ret': p_ret, 
                        'mdd': abs(p_mdd) if abs(p_mdd) > 0.01 else 0.01,
                        'vol': vol if vol > 0.001 else 0.01, 
                        'sharpe': p_ret / vol if vol > 0.001 else 0
                    }

                presets = {}
                eq_v = round(100.0 / len(tickers), 2)
                
                presets['eq'] = {t: eq_v for t in tickers}
                
                total_inv_vol = sum(1/stats[t]['vol'] for t in tickers)
                presets['rp'] = {t: round(((1/stats[t]['vol'])/total_inv_vol)*100, 2) for t in tickers}
                
                total_inv_mdd = sum(1/stats[t]['mdd'] for t in tickers)
                presets['md'] = {t: round(((1/stats[t]['mdd'])/total_inv_mdd)*100, 2) for t in tickers}
                
                total_pos_ret = sum(max(0, stats[t]['ret']) for t in tickers)
                presets['tp'] = {t: (round((max(0, stats[t]['ret'])/total_pos_ret)*100, 2) if total_pos_ret > 0 else eq_v) for t in tickers}
                
                total_sh = sum(max(0, stats[t]['sharpe']) for t in tickers)
                presets['sr'] = {t: (round((max(0, stats[t]['sharpe'])/total_sh)*100, 2) if total_sh > 0 else eq_v) for t in tickers}

                st.button("🟰 Equal Weights", on_click=apply_preset_callback, args=(presets['eq'],), use_container_width=True, key="side_btn_eq")
                st.button("🛡️ Risk Parity", on_click=apply_preset_callback, args=(presets['rp'],), use_container_width=True, key="side_btn_rp")
                st.button("📉 Min Drawdown", on_click=apply_preset_callback, args=(presets['md'],), use_container_width=True, key="side_btn_md")
                st.button("🚀 Top Performance", on_click=apply_preset_callback, args=(presets['tp'],), use_container_width=True, key="side_btn_tp")
                st.button("💎 Sharpe Ratio", on_click=apply_preset_callback, args=(presets['sr'],), use_container_width=True, key="side_btn_sr")
            
            st.write("") 


        if st.button(f"🔄 Load & Reset (1/{len(selected_tickers)})", use_container_width=True, type="primary", key="main_launch_btn"):
            if len(selected_tickers) < 3:
                st.error("Error: Choose at least 3 assets.")
            else:
                with st.spinner("Analysis and rebalancing in progress..."):
                    new_data = {}
                    eq_val = round(100.0 / len(selected_tickers), 2)
                    
                    st.session_state.portfolio_weights = {t: eq_val for t in selected_tickers}
                    
                    for t in selected_tickers:
                        st.session_state[f"slider_weight_{t}"] = float(eq_val)
                        st.session_state[f"num_weight_{t}"] = float(eq_val)
                        
                        if t not in st.session_state.asset_settings:
                            st.session_state.asset_settings[t] = {'ma': 20, 'sl': 10.0, 'tp': 30.0}
                        
                        df_raw = load_stock_data(t)
                        s = st.session_state.asset_settings[t]
                        new_data[t] = apply_strategies(df_raw, s['ma'], s['sl']/100, s['tp']/100)
                    
                    st.session_state.portfolio_data = new_data
                    st.session_state.tickers_analyzed = selected_tickers
                    
                    st.rerun()




    if st.session_state.portfolio_data:
        data_dict = st.session_state.portfolio_data
        tickers = st.session_state.tickers_analyzed

        st.subheader("🔍 Configuration & Details by Asset")
        
        selected_t = st.pills("Asset to modify:", options=tickers, default=tickers[0])

        if selected_t:
            if selected_t not in st.session_state.asset_settings:
                st.session_state.asset_settings[selected_t] = {'ma': 20, 'sl': 10.0, 'tp': 30.0}

            with st.container(border=True):
                col_settings, col_chart = st.columns([1.2, 2.3])
                current_settings = st.session_state.asset_settings[selected_t]

                with col_settings:
                    st.markdown('<p class="ma-header"> MA</p>', unsafe_allow_html=True)
                    ma = st.number_input("Périodes", 5, 100, value=int(current_settings['ma']), key=f"ma_{selected_t}")
                    sl = st.slider("SL (%)", 0.0, 50.0, value=float(current_settings['sl']), step=0.6, key=f"sl_{selected_t}")
                    tp = st.slider("TP (%)", 0.0, 200.0, value=float(current_settings['tp']), step=0.6, key=f"tp_{selected_t}")

                    if st.button(f"Apply to {selected_t}", use_container_width=True, type="primary"):
                        df_raw = load_stock_data(selected_t)
                        st.session_state.portfolio_data[selected_t] = apply_strategies(df_raw, ma, sl/100, tp/100)
                        st.rerun()

                with col_chart:
                    df_t = data_dict[selected_t]
                    dates = df_t.index.strftime('%Y-%m-%d').tolist()

                    price_rebased = (df_t['Close'] / df_t['Close'].iloc[0] * 100).round(2).replace({np.nan: None}).tolist()
                    
                    ma_series = df_t['Close'].rolling(int(ma)).mean()
                    ma_rebased = (ma_series / df_t['Close'].iloc[0] * 100).round(2).replace({np.nan: None}).tolist()
                    
                    strat_perf = df_t['Strat_Momentum'].round(2).replace({np.nan: None}).tolist()

                    chart_option = {
                        "backgroundColor": "transparent",
                        "tooltip": {"trigger": "axis", "backgroundColor": "#111", "textStyle": {"color": "#fff"}},
                        "legend": {"data": ["Rebased Price", f"MA {ma}", "Strategy"], "bottom": 0, "textStyle": {"color": "#8b949e"}},
                        "grid": {"left": "2%", "right": "2%", "bottom": "15%", "top": "5%", "containLabel": True},
                        "xAxis": {"type": "category", "data": dates, "axisLine": {"lineStyle": {"color": "#30363d"}}},
                        "yAxis": {"type": "value", "scale": True, "splitLine": {"lineStyle": {"color": "#30363d", "type": "dashed"}}},
                        "series": [
                            {
                                "name": "Rebased Price", "type": "line", "data": price_rebased,
                                "smooth": True, "symbol": "none", "lineStyle": {"color": "#8b949e", "width": 1, "opacity": 0.5}
                            },
                            {
                                "name": f"MA {ma}", "type": "line", "data": ma_rebased,
                                "smooth": True, "symbol": "none", "lineStyle": {"color": "#ff9800", "width": 1.5, "type": "dotted"}
                            },
                            {
                                "name": "Strategy", "type": "line", "data": strat_perf,
                                "smooth": True, "symbol": "none", "lineStyle": {"color": "#00d1ff", "width": 3},
                                "areaStyle": {"color": {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1, 
                                            "colorStops": [{"offset": 0, "color": "rgba(0, 209, 255, 0.2)"}, {"offset": 1, "color": "transparent"}]}}
                            }
                        ]
                    }
                    st_echarts(options=chart_option, height="350px", key=f"echart_focus_{selected_t}")

            price_series = df_t['Close']
            bh_cum_return = (price_series / price_series.iloc[0]) 
            bh_ret, bh_mdd = calculate_metrics(bh_cum_return)
            bh_vol = price_series.pct_change().std() * np.sqrt(252)

            strat_ret, strat_mdd = calculate_metrics(df_t['Strat_Momentum'])
            strat_vol = df_t['Strat_Returns'].std() * np.sqrt(252)

            st.markdown(f"**Performance of the strategy vs Buy & Hold ({selected_t})**")

            with st.container(border=True): 
                m1, m2, m3 = st.columns(3)

                with m1:
                    diff_ret = strat_ret - bh_ret
                    st.metric(
                        "Performance", 
                        f"{strat_ret:.2%}", 
                        delta=f"{diff_ret:+.2%}", 
                        help="Comparison vs Buy & Hold"
                    )

                with m2:
                    diff_mdd = strat_mdd - bh_mdd
                    st.metric(
                        "Max Drawdown", 
                        f"{strat_mdd:.2%}", 
                        delta=f"{diff_mdd:+.2%}", 
                        delta_color="inverse"
                    )

                with m3:
                    diff_vol = strat_vol - bh_vol
                    st.metric(
                        "Volatility", 
                        f"{strat_vol:.2%}", 
                        delta=f"{diff_vol:+.2%}", 
                        delta_color="inverse"
                    )

        render_portfolio_simulation(tickers, data_dict, cap_init)