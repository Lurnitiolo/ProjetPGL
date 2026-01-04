import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from .data_loader import load_stock_data
from .strategies import AVAILABLE_ASSETS, apply_strategies

def quant_a_ui():
    # --- 0. STATE MANAGEMENT ---
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False

    def clear_analysis():
        st.session_state.show_analysis = False

    # --- 1. HEADER & INPUTS ---
    st.markdown("<h2 style='text-align: center;'>Quant Module A : Single Asset Analysis</h2>", unsafe_allow_html=True)
    
    _, col_center, _ = st.columns([1, 2, 1])
    
    with col_center:
        with st.container(border=True):
            st.markdown("#### ⚙️ Configuration")
            
            # --- SELECTION ACTIF ---
            col_asset, col_period = st.columns(2)
            with col_asset:
                selected_name = st.selectbox("Select Asset", options=list(AVAILABLE_ASSETS.keys()))
                if selected_name == "Other (Manual Input)":
                    ticker = st.text_input("Yahoo Symbol", value="AIR.PA")
                else:
                    ticker = AVAILABLE_ASSETS[selected_name]
            with col_period:
                period = st.selectbox("History", ["1y", "2y", "5y", "max"], index=1)

            st.markdown("---")
            
            # --- SELECTION STRATÉGIE ---
            strategy_type = st.selectbox(
                "Select Strategy", 
                ["Moving Average", "Bollinger Bands", "RSI"],
                index=0
            )
            
            # --- PARAMÈTRES DYNAMIQUES ---
            params = {}
            col_p1, col_p2 = st.columns(2)
            
            if strategy_type == "Moving Average":
                with col_p1:
                    params['short_window'] = st.number_input("Short MA", 5, 100, 20)
                with col_p2:
                    params['long_window'] = st.number_input("Long MA", 50, 300, 50)
                desc_strat = f"Cross SMA {params['short_window']} / {params['long_window']}"
                    
            elif strategy_type == "Bollinger Bands":
                with col_p1:
                    params['bb_window'] = st.number_input("Period (20)", 10, 100, 20)
                with col_p2:
                    params['bb_std'] = st.number_input("Std Dev (2.0)", 1.0, 4.0, 2.0, step=0.1)
                desc_strat = f"Mean Reversion BB ({params['bb_window']}, {params['bb_std']}σ)"
                    
            elif strategy_type == "RSI":
                with col_p1:
                    params['rsi_window'] = st.number_input("RSI Period", 5, 30, 14)
                with col_p2:
                    rsi_bounds = st.slider("Oversold / Overbought", 10, 90, (30, 70))
                    params['rsi_oversold'] = rsi_bounds[0]
                    params['rsi_overbought'] = rsi_bounds[1]
                desc_strat = f"RSI Oscillator ({params['rsi_window']}) [{params['rsi_oversold']}-{params['rsi_overbought']}]"

            # Buttons
            col_b1, col_b2 = st.columns([2, 1])
            with col_b1:
                if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                    st.session_state.show_analysis = True
            with col_b2:
                if st.button("❌ Clear", on_click=clear_analysis, use_container_width=True):
                    pass 

    # --- 2. RESULTS DASHBOARD ---
    if st.session_state.show_analysis:
        st.divider()
        
        with st.spinner('Crunching numbers...'):
            df = load_stock_data(ticker, period=period, interval="1d")
            
            if df is not None and not df.empty:
                df_strat, metrics = apply_strategies(df, strategy_type, params)
                
                # --- SÉCURITÉ ---
                valid_data = df_strat.dropna()
                if valid_data.empty:
                    st.error(f"⚠️ Not enough data points. Try increasing History.")
                    st.stop()
                
                # LIMITES GLOBALES
                min_global_date = valid_data.index.min().date()
                max_global_date = valid_data.index.max().date()
                min_global_ts = valid_data.index.min()
                max_global_ts = valid_data.index.max()

                # --- HEADER ---
                last_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2]
                daily_return = (last_price - prev_price) / prev_price
                
                with st.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Asset", ticker)
                    c2.metric("Last Price", f"{last_price:.2f} €", f"{daily_return:.2%}")
                    c3.metric("Sharpe (B&H)", f"{metrics['BuyHold_Sharpe']:.2f}")
                    c4.metric("Sharpe (Strat)", f"{metrics['Active_Sharpe']:.2f}", 
                    delta=f"{metrics['Active_Sharpe'] - metrics['BuyHold_Sharpe']:.2f}")

                st.markdown("---")

                # ==============================================================================
                # 🔥 CONTROLEUR GLOBAL DE ZOOM (S'applique aux Tab 1 ET Tab 2)
                # ==============================================================================
                col_search, _ = st.columns([1, 4])
                with col_search:
                    zoom_dates = st.date_input(
                        "🔍 Global Zoom Range",
                        value=(min_global_date, max_global_date),
                        min_value=min_global_date,
                        max_value=max_global_date,
                        format="DD/MM/YYYY"
                    )
                
                # Logique de zoom appliquée partout
                if isinstance(zoom_dates, tuple) and len(zoom_dates) == 2:
                    zoom_start, zoom_end = zoom_dates
                else:
                    zoom_start, zoom_end = min_global_date, max_global_date


                # ==============================================================================
                # TABS
                # ==============================================================================
                tab1, tab2, tab3 = st.tabs(["📈 Price Chart", "📊 Risk & Metrics", "📋 Data"])
                
                # --- TAB 1: PRIX ---
                with tab1:
                    st.caption(f"Strategy: {desc_strat}")
                    
                    fig = go.Figure()

                    # Trace 1 : Buy & Hold
                    fig.add_trace(go.Scatter(
                        x=df_strat.index, y=df_strat['Strat_BuyHold'],
                        mode='lines', name='Buy & Hold',
                        line=dict(color='#7f8c8d', width=2), opacity=0.8
                    ))

                    # Trace 2 : Stratégie Active
                    fig.add_trace(go.Scatter(
                        x=df_strat.index, y=df_strat['Strat_Active'],
                        mode='lines', name=f'{strategy_type} Strat',
                        line=dict(color='#2ecc71', width=3)
                    ))

                    fig.update_layout(
                        template="plotly_dark",
                        hovermode="x unified",
                        margin=dict(l=0, r=0, t=10, b=0), # Marges réduites
                        height=500,
                        dragmode='pan', 
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(
                            showgrid=False, type="date",
                            range=[zoom_start, zoom_end],    # ZOOM GLOBAL
                            minallowed=min_global_ts, maxallowed=max_global_ts, # MURS
                            rangeslider=dict(visible=True, thickness=0.04, bgcolor='rgba(100,100,100,0.1)'),
                            rangeselector=dict(visible=False)
                        ),
                        yaxis=dict(title="Base 100", showgrid=True, gridcolor='rgba(128,128,128,0.2)', fixedrange=True)
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
                    
                
                # --- TAB 2: RISQUE & MÉTRIQUES ---
                with tab2:
                    
                    # 1. TABLEAUX DE METRIQUES STYLISÉS
                    st.subheader("Performance & Risk Profile")
                    
                    col_m1, col_m2 = st.columns(2)
                    
                    with col_m1:
                        st.markdown("**1️⃣ General Performance**")
                        metrics_perf = {
                            "Metric": ["Total Return", "Annual Volatility", "Sharpe Ratio", "Sortino Ratio"],
                            "Buy & Hold": [
                                f"{metrics['BuyHold_Return']:.2%}", f"{metrics['BuyHold_Vol']:.2%}",
                                f"{metrics['BuyHold_Sharpe']:.2f}", f"{metrics['BuyHold_Sortino']:.2f}"
                            ],
                            "Strategy": [
                                f"{metrics['Active_Return']:.2%}", f"{metrics['Active_Vol']:.2%}",
                                f"{metrics['Active_Sharpe']:.2f}", f"{metrics['Active_Sortino']:.2f}"
                            ]
                        }
                        st.dataframe(pd.DataFrame(metrics_perf), use_container_width=True, hide_index=True)

                    with col_m2:
                        st.markdown("**2️⃣ Tail Risk (Extreme Events)**")
                        metrics_risk = {
                            "Metric": ["Max Drawdown", "VaR 95% (Daily)", "CVaR 95% (Daily)", "Skewness", "Kurtosis"],
                            "Buy & Hold": [
                                f"{metrics['BuyHold_MDD']:.2%}", f"{metrics['BuyHold_VaR']:.2%}",
                                f"{metrics['BuyHold_CVaR']:.2%}", f"{metrics['BuyHold_Skew']:.2f}", f"{metrics['BuyHold_Kurt']:.2f}"
                            ],
                            "Strategy": [
                                f"{metrics['Active_MDD']:.2%}", f"{metrics['Active_VaR']:.2%}",
                                f"{metrics['Active_CVaR']:.2%}", f"{metrics['Active_Skew']:.2f}", f"{metrics['Active_Kurt']:.2f}"
                            ]
                        }
                        st.dataframe(pd.DataFrame(metrics_risk), use_container_width=True, hide_index=True)

                    st.markdown("---")
                    
                    # 2. GRAPHIQUE VISUEL DU RISQUE (Risk Monitor)
                    st.subheader("📉 Risk Monitor: VaR Breaches")
                    
                    returns_series = df_strat['Strat_Ret']
                    var_level = metrics['Active_VaR']
                    cvar_level = metrics['Active_CVaR']
                    
                    colors = []
                    for r in returns_series:
                        if r >= 0: colors.append("rgba(46, 204, 113, 0.6)")  # Vert
                        elif r > var_level: colors.append("rgba(149, 165, 166, 0.6)") # Gris
                        else: colors.append("rgba(231, 76, 60, 1.0)")   # Rouge (Crash)

                    fig_risk = go.Figure()

                    fig_risk.add_trace(go.Bar(
                        x=df_strat.index, y=returns_series, marker_color=colors, name="Daily Returns"
                    ))

                    fig_risk.add_hline(y=var_level, line_dash="solid", line_color="orange", line_width=2,
                    annotation_text="VaR 95%", annotation_position="bottom right")
                    
                    fig_risk.add_hline(y=cvar_level, line_dash="dot", line_color="red", line_width=2,
                    annotation_text="CVaR", annotation_position="bottom right")

                    # --- APPLICATION DES MÊMES OPTIONS QUE TAB 1 ---
                    fig_risk.update_layout(
                        template="plotly_dark",
                        height=450,
                        title=dict(text="Daily Returns Distribution & Breaches", font=dict(size=14)),
                        yaxis_title="Daily Return",
                        showlegend=False,
                        margin=dict(l=0, r=0, t=40, b=0),
                        dragmode='pan', # Pan activé
                        
                        # AXE X IDENTIQUE AU TAB 1
                        xaxis=dict(
                            showgrid=False, type="date",
                            range=[zoom_start, zoom_end],    # ZOOM SYNCHRONISÉ
                            minallowed=min_global_ts, maxallowed=max_global_ts, # MURS
                            rangeslider=dict(visible=True, thickness=0.04, bgcolor='rgba(100,100,100,0.1)'), # SCROLLBAR
                            rangeselector=dict(visible=False)
                        ),
                        yaxis=dict(
                            showgrid=True, gridcolor='rgba(128,128,128,0.1)', tickformat=".1%", fixedrange=True
                        )
                    )

                    st.plotly_chart(fig_risk, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
                    
                    st.caption("🔴 **Red Bars** indicate days where losses exceeded the 95% Value at Risk threshold.")

                # --- TAB 3: DONNÉES ---
                with tab3:
                    st.dataframe(df_strat.tail(100), use_container_width=True)
            else:
                st.error("Error: No data found.")