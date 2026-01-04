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
            
            # Asset Selection
            selected_name = st.selectbox("Select Asset", options=list(AVAILABLE_ASSETS.keys()))
            if selected_name == "Other (Manual Input)":
                ticker = st.text_input("Yahoo Symbol", value="AIR.PA")
            else:
                ticker = AVAILABLE_ASSETS[selected_name]

            # Time Params
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                period = st.selectbox("History", ["1y", "2y", "5y", "max"], index=1)
            with col_t2:
                st.text_input("Interval", value="Daily (1d)", disabled=True)

            # Strategy Params
            st.markdown("---")
            st.caption("Momentum Strategy Parameters")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                short_window = st.number_input("Short MA", min_value=5, max_value=50, value=20)
            with col_p2:
                long_window = st.number_input("Long MA", min_value=50, max_value=200, value=50)

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
                df_strat, metrics = apply_strategies(df, short_window, long_window)
                
                # --- SÉCURITÉ ---
                valid_data = df_strat.dropna()
                if valid_data.empty:
                    st.error(f"⚠️ Not enough data points. Try increasing History or decreasing MA lengths.")
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
                    c3.metric("Sharpe (B&H)", f"{metrics['BuyHold_Sharpe']:.2f}")
                    c4.metric("Sharpe (Strat)", f"{metrics['Momentum_Sharpe']:.2f}", 
                              delta=f"{metrics['Momentum_Sharpe'] - metrics['BuyHold_Sharpe']:.2f}")

                # B. TABS
                tab1, tab2, tab3 = st.tabs(["📈 Chart", "📊 Detailed Metrics", "📋 Data"])
                
                with tab1:
                    st.subheader("Performance Comparison (Base 100)")
                    
                    # --- SÉLECTEUR DE ZOOM COMPACT ---
                    # CHANGEMENT ICI : [1, 4] force la colonne à être petite (20% de largeur)
                    col_search, _ = st.columns([1, 4])
                    
                    with col_search:
                        zoom_dates = st.date_input(
                            "🔍 Zoom Range", # Label court
                            value=(min_global_date, max_global_date),
                            min_value=min_global_date,
                            max_value=max_global_date,
                            format="DD/MM/YYYY"
                        )
                    
                    # --- LOGIQUE DE ZOOM ---
                    if isinstance(zoom_dates, tuple) and len(zoom_dates) == 2:
                        zoom_start, zoom_end = zoom_dates
                    else:
                        zoom_start, zoom_end = min_global_date, max_global_date

                    # --- CRÉATION DU GRAPHIQUE ---
                    fig = go.Figure()

                    # Trace 1
                    fig.add_trace(go.Scatter(
                        x=df_strat.index, y=df_strat['Strat_BuyHold'],
                        mode='lines', name='Buy & Hold',
                        line=dict(color='#7f8c8d', width=2), opacity=0.8
                    ))

                    # Trace 2
                    fig.add_trace(go.Scatter(
                        x=df_strat.index, y=df_strat['Strat_Momentum'],
                        mode='lines', name='Momentum Strat',
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
                            
                            # Range dynamique
                            range=[zoom_start, zoom_end],
                            minallowed=min_global_ts,
                            maxallowed=max_global_ts,
                            
                            # Scrollbar fine
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
                    
                    st.info(f"Strategy: Cross Short MA ({short_window}) / Long MA ({long_window})")
                    
                with tab2:
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.info("Buy & Hold")
                        st.write(f"**Return:** {metrics['BuyHold_Return']:.2%}")
                        st.write(f"**Max Drawdown:** {metrics['BuyHold_MDD']:.2%}")
                        st.write(f"**Volatility:** {metrics['BuyHold_Vol']:.2%}")
                    with col_m2:
                        st.success("Momentum Strategy")
                        st.write(f"**Return:** {metrics['Momentum_Return']:.2%}")
                        st.write(f"**Max Drawdown:** {metrics['Momentum_MDD']:.2%}")
                        st.write(f"**Volatility:** {metrics['Momentum_Vol']:.2%}")

                with tab3:
                    st.dataframe(df_strat.tail(100), use_container_width=True)
            else:
                st.error("Error: No data found.")