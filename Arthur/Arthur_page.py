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
                    # Slider double pour les bornes
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
                # APPEL DE LA FONCTION AVEC LA NOUVELLE SIGNATURE
                df_strat, metrics = apply_strategies(df, strategy_type, params)
                
                # --- SÉCURITÉ ---
                valid_data = df_strat.dropna()
                if valid_data.empty:
                    st.error(f"⚠️ Not enough data points to calculate indicators. Try increasing History.")
                    st.stop()
                
                # LIMITES GLOBALES
                min_global_date = valid_data.index.min().date()
                max_global_date = valid_data.index.max().date()
                min_global_ts = valid_data.index.min()
                max_global_ts = valid_data.index.max()

                # A. HEADER
                last_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2]
                daily_return = (last_price - prev_price) / prev_price
                
                with st.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Asset", ticker)
                    c2.metric("Last Price", f"{last_price:.2f} €", f"{daily_return:.2%}")
                    
                    # Mise à jour des clés (Active_... au lieu de Momentum_...)
                    c3.metric("Sharpe (B&H)", f"{metrics['BuyHold_Sharpe']:.2f}")
                    c4.metric("Sharpe (Strat)", f"{metrics['Active_Sharpe']:.2f}", 
                    delta=f"{metrics['Active_Sharpe'] - metrics['BuyHold_Sharpe']:.2f}")

                # B. TABS
                tab1, tab2, tab3 = st.tabs(["📈 Chart", "📊 Detailed Metrics", "📋 Data"])
                
                with tab1:
                    st.subheader("Performance Comparison (Base 100)")
                    
                    # --- SÉLECTEUR DE ZOOM COMPACT ---
                    col_search, _ = st.columns([1, 4])
                    with col_search:
                        zoom_dates = st.date_input(
                            "🔍 Zoom Range",
                            value=(min_global_date, max_global_date),
                            min_value=min_global_date,
                            max_value=max_global_date,
                            format="DD/MM/YYYY"
                        )
                    
                    if isinstance(zoom_dates, tuple) and len(zoom_dates) == 2:
                        zoom_start, zoom_end = zoom_dates
                    else:
                        zoom_start, zoom_end = min_global_date, max_global_date

                    # --- CRÉATION DU GRAPHIQUE ---
                    fig = go.Figure()

                    # Trace 1 : Buy & Hold
                    fig.add_trace(go.Scatter(
                        x=df_strat.index, y=df_strat['Strat_BuyHold'],
                        mode='lines', name='Buy & Hold',
                        line=dict(color='#7f8c8d', width=2), opacity=0.8
                    ))

                    # Trace 2 : Stratégie Active (Nom dynamique)
                    # Note : La colonne s'appelle maintenant 'Strat_Active' dans strategies.py
                    fig.add_trace(go.Scatter(
                        x=df_strat.index, y=df_strat['Strat_Active'],
                        mode='lines', name=f'{strategy_type} Strat',
                        line=dict(color='#2ecc71', width=3)
                    ))

                    fig.update_layout(
                        template="plotly_dark",
                        hovermode="x unified",
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=500,
                        dragmode='pan', 
                        
                        legend=dict(
                            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                        ),
                        
                        xaxis=dict(
                            showgrid=False,
                            type="date",
                            range=[zoom_start, zoom_end],
                            minallowed=min_global_ts,
                            maxallowed=max_global_ts,
                            rangeslider=dict(
                                visible=True, 
                                thickness=0.04, 
                                bgcolor='rgba(100,100,100,0.1)' 
                            ),
                            rangeselector=dict(visible=False)
                        ),
                        
                        yaxis=dict(
                            title="Value (Base 100)", 
                            showgrid=True, 
                            gridcolor='rgba(128,128,128,0.2)',
                            fixedrange=True 
                        )
                    )

                    config = {
                        'scrollZoom': True,
                        'displayModeBar': False,
                        'displaylogo': False,
                    }

                    st.plotly_chart(fig, use_container_width=True, config=config)
                    st.caption("💡 Navigation: Search dates above to jump, or use mouse/touch to explore the full history.")
                    
                    st.info(f"Active Strategy: {desc_strat}")
                    
                with tab2:
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.info("Buy & Hold")
                        st.write(f"**Return:** {metrics['BuyHold_Return']:.2%}")
                        st.write(f"**Max Drawdown:** {metrics['BuyHold_MDD']:.2%}")
                        st.write(f"**Volatility:** {metrics['BuyHold_Vol']:.2%}")
                    with col_m2:
                        st.success(f"{strategy_type} Strategy")
                        # Mise à jour des clés ici aussi
                        st.write(f"**Return:** {metrics['Active_Return']:.2%}")
                        st.write(f"**Max Drawdown:** {metrics['Active_MDD']:.2%}")
                        st.write(f"**Volatility:** {metrics['Active_Vol']:.2%}")

                with tab3:
                    st.dataframe(df_strat.tail(100), use_container_width=True)
            else:
                st.error("Error: No data found.")