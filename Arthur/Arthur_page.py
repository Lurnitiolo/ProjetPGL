import streamlit as st
import pandas as pd
import numpy as np
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
                
                valid_data = df_strat.dropna()
                if valid_data.empty:
                    st.error(f"⚠️ Not enough data points. Try increasing History.")
                    st.stop()
                
                # --- PRE-CALCULS ---
                returns_series = df_strat['Strat_Active'].pct_change()

                # Limites
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

                # --- GLOBAL ZOOM ---
                col_search, _ = st.columns([1, 4])
                with col_search:
                    zoom_dates = st.date_input(
                        "🔍 Global Zoom Range",
                        value=(min_global_date, max_global_date),
                        min_value=min_global_date,
                        max_value=max_global_date,
                        format="DD/MM/YYYY"
                    )
                
                if isinstance(zoom_dates, tuple) and len(zoom_dates) == 2:
                    zoom_start, zoom_end = zoom_dates
                else:
                    zoom_start, zoom_end = min_global_date, max_global_date

                # ==============================================================================
                # TABS
                # ==============================================================================
                tab1, tab2, tab3 = st.tabs(["💰 Portfolio Simulator", "📊 Risk & Metrics", "📋 Raw Data"])
                
                # --- TAB 1: SIMULATION ---
                with tab1:
                    # 1. INPUT DE CAPITAL
                    col_sim1, col_sim2 = st.columns([1, 3])
                    with col_sim1:
                        initial_capital = st.number_input(
                            "💵 Initial Capital (€)", 
                            min_value=100, 
                            value=10000, 
                            step=1000,
                            help="How much money do you start with?"
                        )
                    
                    # 2. CALCULS DE LA SIMULATION (Mise à l'échelle)
                    # On convertit la Base 100 en Capital Réel
                    # Formule : (Equity_Base_100 / 100) * Capital_Initial
                    df_strat['Sim_Equity'] = (df_strat['Strat_Active'] / 100) * initial_capital
                    df_strat['Sim_BH'] = (df_strat['Strat_BuyHold'] / 100) * initial_capital
                    
                    # Allocation Réelle
                    df_strat['Sim_Invested'] = df_strat['Sim_Equity'] * df_strat['Position']
                    df_strat['Sim_Cash'] = df_strat['Sim_Equity'] * (1 - df_strat['Position'])
                    
                    # Résultats Finaux
                    final_res = df_strat['Sim_Equity'].iloc[-1]
                    final_perf = (final_res - initial_capital)
                    is_profit = final_perf >= 0
                    color_perf = "green" if is_profit else "red"
                    
                    with col_sim2:
                        st.markdown(f"""
                        <div style="padding-top: 25px; font-size: 20px;">
                            Final Value: <b>{final_res:,.2f} €</b> 
                            <span style='color:{color_perf}; margin-left: 10px;'>
                                ({'+' if is_profit else ''}{final_perf:,.2f} €)
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

                    # GRAPH 1 : EVOLUTION DU PORTEFEUILLE (ARGENT REEL)
                    st.markdown("**1️⃣ Wealth Evolution**")
                    fig = go.Figure()
                    
                    # Benchmark (Buy & Hold) en Gris
                    fig.add_trace(go.Scatter(
                        x=df_strat.index, y=df_strat['Sim_BH'],
                        mode='lines', name='Buy & Hold',
                        line=dict(color='#95a5a6', width=2, dash='dot'), opacity=0.7,
                        hovertemplate='%{y:.2f} €'
                    ))
                    
                    # Stratégie en Vert (Ligne principale)
                    fig.add_trace(go.Scatter(
                        x=df_strat.index, y=df_strat['Sim_Equity'],
                        mode='lines', name=f'{strategy_type} Strategy',
                        line=dict(color='#2ecc71', width=3),
                        hovertemplate='%{y:.2f} €'
                    ))

                    fig.update_layout(
                        template="plotly_dark", hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0), height=400, dragmode='pan', 
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(showgrid=False, type="date", range=[zoom_start, zoom_end], minallowed=min_global_ts, maxallowed=max_global_ts, rangeslider=dict(visible=False)),
                        yaxis=dict(title="Portfolio Value (€)", showgrid=True, gridcolor='rgba(128,128,128,0.2)', fixedrange=True)
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})

                    # GRAPH 2 : COMPOSITION (ARGENT REEL)
                    st.markdown("**2️⃣ Wallet Allocation (Cash vs Asset)**")
                    fig_alloc = go.Figure()
                    
                    # Zone Cash (Gris)
                    fig_alloc.add_trace(go.Scatter(
                        x=df_strat.index, y=df_strat['Sim_Cash'],
                        mode='lines', name='Cash (Safe)',
                        stackgroup='one', 
                        line=dict(width=0), 
                        fillcolor='rgba(128, 128, 128, 0.5)',
                        hovertemplate='%{y:.2f} €'
                    ))
                    
                    # Zone Investie (Vert)
                    fig_alloc.add_trace(go.Scatter(
                        x=df_strat.index, y=df_strat['Sim_Invested'],
                        mode='lines', name='Invested in Market',
                        stackgroup='one', 
                        line=dict(width=0), 
                        fillcolor='rgba(46, 204, 113, 0.6)',
                        hovertemplate='%{y:.2f} €'
                    ))

                    fig_alloc.update_layout(
                        template="plotly_dark", hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0), height=250, dragmode='pan',
                        showlegend=False,
                        xaxis=dict(showgrid=False, type="date", range=[zoom_start, zoom_end], minallowed=min_global_ts, maxallowed=max_global_ts, rangeslider=dict(visible=True, thickness=0.08, bgcolor='rgba(100,100,100,0.1)'), rangeselector=dict(visible=False)),
                        yaxis=dict(title="Amount (€)", showgrid=True, gridcolor='rgba(128,128,128,0.2)', fixedrange=True)
                    )
                    st.plotly_chart(fig_alloc, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})

                # --- TAB 2: RISQUE & SCORECARD ---
                with tab2:
                    st.subheader("🛡️ Risk Analysis & Scorecard")
                    
                    # 1. PnL Chart
                    st.markdown("**1️⃣ Daily PnL (Simulated Portfolio)**")
                    pnl_colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in returns_series]
                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Bar(x=df_strat.index, y=returns_series, marker_color=pnl_colors, name="Daily Return (%)"))
                    fig_pnl.update_layout(
                        template="plotly_dark", height=300, margin=dict(l=0, r=0, t=10, b=0), dragmode='pan',
                        xaxis=dict(showgrid=False, type="date", range=[zoom_start, zoom_end], minallowed=min_global_ts, maxallowed=max_global_ts, rangeslider=dict(visible=False)),
                        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)', tickformat=".1%", fixedrange=True)
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})

                    st.markdown("---")
                    
                    # 2. Key Metrics
                    col_risk1, col_risk2, col_risk3 = st.columns(3)
                    with col_risk1: st.metric("VaR 95%", f"{metrics['Active_VaR']:.2%}", help="Max daily loss (95% confidence)")
                    with col_risk2: st.metric("CVaR", f"{metrics['Active_CVaR']:.2%}", help="Expected loss if VaR is breached")
                    with col_risk3: st.metric("Max Drawdown", f"{metrics['Active_MDD']:.2%}", help="Peak to valley loss")

                    st.markdown("---")
                    
                    # 3. Tail Risk Monitor
                    st.markdown("**2️⃣ Tail Risk Monitor (VaR Breaches)**")
                    var_level = metrics['Active_VaR']
                    cvar_level = metrics['Active_CVaR']
                    risk_colors = ["rgba(231, 76, 60, 1.0)" if (pd.notna(r) and r < var_level) else "rgba(149, 165, 166, 0.3)" for r in returns_series]
                    fig_risk = go.Figure()
                    fig_risk.add_trace(go.Bar(x=df_strat.index, y=returns_series, marker_color=risk_colors, name="Simulated Returns"))
                    fig_risk.add_hline(y=var_level, line_dash="solid", line_color="orange", annotation_text="VaR")
                    fig_risk.add_hline(y=cvar_level, line_dash="dot", line_color="red", annotation_text="CVaR")
                    fig_risk.update_layout(
                        template="plotly_dark", height=350, margin=dict(l=0, r=0, t=10, b=0), dragmode='pan',
                        xaxis=dict(showgrid=False, type="date", range=[zoom_start, zoom_end], minallowed=min_global_ts, maxallowed=max_global_ts, rangeslider=dict(visible=True, thickness=0.04, bgcolor='rgba(100,100,100,0.1)')),
                        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)', tickformat=".1%", fixedrange=True)
                    )
                    st.plotly_chart(fig_risk, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
                
                    st.markdown("---")

                    # 4. TABLEAU STYLISÉ
                    st.markdown("**3️⃣ Advanced Metrics Scorecard**")
                    
                    score_data = {
                        "Your Strategy": [metrics['Active_Return'], metrics['Active_Vol'], metrics['Active_Sharpe'], metrics['Active_Sortino'], metrics['Active_MDD'], metrics['Active_VaR']],
                        "Buy & Hold": [metrics['BuyHold_Return'], metrics['BuyHold_Vol'], metrics['BuyHold_Sharpe'], metrics['BuyHold_Sortino'], metrics['BuyHold_MDD'], metrics['BuyHold_VaR']]
                    }
                    df_score = pd.DataFrame(score_data, index=["Total Return", "Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "VaR 95%"])

                    def highlight_winner(row):
                        styles = ['color: white', 'color: white']
                        v1, v2 = row['Your Strategy'], row['Buy & Hold']
                        idx = row.name
                        
                        high_is_good = ["Total Return", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "VaR 95%"] 
                        low_is_good = ["Volatility"]

                        if idx in high_is_good:
                            if v1 > v2: styles = ['color: #2ecc71; font-weight: bold', 'color: #e74c3c'] 
                            elif v1 < v2: styles = ['color: #e74c3c', 'color: #2ecc71; font-weight: bold']
                        elif idx in low_is_good:
                            if v1 < v2: styles = ['color: #2ecc71; font-weight: bold', 'color: #e74c3c'] 
                            elif v1 > v2: styles = ['color: #e74c3c', 'color: #2ecc71; font-weight: bold']
                        return styles

                    idx = pd.IndexSlice
                    
                    styler = df_score.style.apply(highlight_winner, axis=1)\
                        .format("{:.2%}", subset=idx[["Total Return", "Volatility", "Max Drawdown", "VaR 95%"], :])\
                        .format("{:.2f}", subset=idx[["Sharpe Ratio", "Sortino Ratio"], :])\
                        .set_properties(**{'font-size': '16px', 'text-align': 'center', 'height': '40px'})

                    st.dataframe(styler, use_container_width=True, column_config={"_index": st.column_config.Column("Metric", width="medium")})

                # --- TAB 3: RAW DATA (NO SIGNAL) ---
                with tab3:
                    st.subheader("📋 Data Explorer")
                    st.caption("Detailed view: Asset Context -> Portfolio State -> Performance.")

                    # 1. Selection des colonnes techniques
                    tech_cols = []
                    if strategy_type == "Moving Average":
                        tech_cols = ['SMA_Short', 'SMA_Long']
                    elif strategy_type == "Bollinger Bands":
                        tech_cols = ['Upper', 'Lower']
                    elif strategy_type == "RSI":
                        tech_cols = ['RSI']

                    # 2. Préparation
                    df_display = df_strat.tail(500).copy()
                    
                    # --- MISE A JOUR : ON RECALCULE AVEC LE CAPITAL CHOISI (Pour le tab 3 aussi) ---
                    df_display['Sim_Equity'] = (df_display['Strat_Active'] / 100) * initial_capital
                    df_display['Sim_Invested'] = df_display['Sim_Equity'] * df_display['Position']
                    df_display['Sim_Cash'] = df_display['Sim_Equity'] * (1 - df_display['Position'])
                    df_display['Total_Ret'] = (df_display['Strat_Active'] / 100) - 1

                    # 3. Renommage
                    rename_map = {
                        'Close': 'Close Price',       
                        'Sim_Cash': 'Cash',         
                        'Sim_Invested': 'Invested',    
                        'Sim_Equity': 'Total Equity',
                        'Strat_Ret': 'Daily PnL',
                        'Total_Ret': 'Total Return'
                    }
                    df_display.rename(columns=rename_map, inplace=True)

                    # 4. ORDRE DES COLONNES
                    main_cols_order = [
                        'Close Price',                
                        'Cash', 'Invested', 'Total Equity', 
                        'Daily PnL', 'Total Return'   
                    ]
                    
                    final_cols = [c for c in main_cols_order if c in df_display.columns] + tech_cols
                    df_display = df_display[final_cols]

                    # 5. STYLING
                    def color_pnl(val):
                        if pd.isna(val): return ''
                        color = '#2ecc71' if val > 0 else '#e74c3c' if val < 0 else ''
                        return f'color: {color}'

                    styler = df_display.style\
                        .map(color_pnl, subset=['Daily PnL', 'Total Return'])\
                        .format("{:.2f} €", subset=['Close Price', 'Cash', 'Invested', 'Total Equity'])\
                        .format("{:.2%}", subset=['Daily PnL', 'Total Return'])

                    if tech_cols:
                        styler = styler.format("{:.2f}", subset=tech_cols)

                    # 6. AFFICHAGE
                    st.dataframe(
                        styler,
                        use_container_width=True,
                        column_config={
                            "Close Price": st.column_config.NumberColumn("Close Price", format="%.2f €"),
                            "Daily PnL": st.column_config.NumberColumn("Daily PnL", format="%.2f %%"),
                            "Total Return": st.column_config.NumberColumn("Total Return", format="%.2f %%")
                        },
                        height=600
                    )

            else:
                st.error("Error: No data found.")