import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from .data_loader import load_stock_data, get_logo_url
from .strategies import apply_strategies, calculate_metrics
from .render_efficiency import min_max_scale, render_portfolio_simulation

def quant_b_ui():
    st.header("Analyse de Portefeuille & Diversification (Quant B)")

    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = {}

    with st.sidebar:
        st.title("⚙️ Paramètres")
        tickers_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC-USD", "EURUSD=X", "GC=F"]
        selected_tickers = st.multiselect("Actifs", options=tickers_list, default=["AAPL", "MSFT", "BTC-USD"], key="sel_tickers_main")
        
        cap_init = st.number_input("Capital Initial ($)", value=1000, key="input_cap_init")
        ma_window = st.number_input("Fenêtre MA", 5, 100, 20, key="input_ma_val")
        sl_val = st.slider("Stop Loss (%)", 0, 25, 10, key="slider_sl") / 100
        tp_val = st.slider("Take Profit (%)", 0, 100, 30, key="slider_tp") / 100
        
        if st.button("🚀 Lancer l'analyse", use_container_width=True, key="btn_run_analysis"):
            # Reset des sliders de poids du portefeuille si besoin
            keys_to_reset = [k for k in st.session_state.keys() if k.startswith('w_') or k.startswith('slide_')]
            for k in keys_to_reset:
                del st.session_state[k]
            
            with st.spinner("Analyse en cours..."):
                # On stocke les résultats dans le session_state
                results = {t: apply_strategies(load_stock_data(t), ma_window, sl_val, tp_val) for t in selected_tickers}
                st.session_state.portfolio_data = results
                st.session_state.tickers_analyzed = selected_tickers
                st.session_state.ma_used = ma_window
            st.rerun()

    if st.session_state.portfolio_data:
        data_dict = st.session_state.portfolio_data
        tickers = st.session_state.tickers_analyzed
        tabs = st.tabs(tickers + ["📊 Portefeuille Global"])

        # --- Dans Virgil_page.py, remplacez le début du bloc "for i, ticker in enumerate(tickers):" ---

        for i, ticker in enumerate(tickers):
            with tabs[i]:
                df_t = data_dict[ticker].copy()
                
                # --- NOUVEAU : Header collé avec Logo 70px ---
                logo_url = get_logo_url(ticker)
                st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <img src="{logo_url if logo_url else ''}" style="width: 70px; margin-right: 15px;">
                        <h2 style="margin: 0;">Analyse {ticker}</h2>
                    </div>
                """, unsafe_allow_html=True)

                # --- Préparation Graphique (Sans le coloriage vert/rouge) ---
                if 'Exit_Type' not in df_t.columns:
                    df_t['Exit_Type'] = None
                    df_t.loc[df_t['Strat_Returns'] <= -sl_val * 0.95, 'Exit_Type'] = 'Stop Loss'
                    df_t.loc[df_t['Strat_Returns'] >= tp_val * 0.95, 'Exit_Type'] = 'Take Profit'

                s_min, s_max = df_t['Strat_Momentum'].min(), df_t['Strat_Momentum'].max()
                y_strat = ((df_t['Strat_Momentum'] - s_min) / (s_max - s_min)) * 100 if s_max > s_min else 50

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_t.index, y=min_max_scale(df_t['Close'])*100, name="Prix", line=dict(color='white', width=1), opacity=0.3))
                fig.add_trace(go.Scatter(x=df_t.index, y=y_strat, name="Stratégie", line=dict(color='#00d1ff', width=2)))
                fig.add_trace(go.Scatter(x=df_t.index, y=min_max_scale(df_t['MA'])*100, name ="Moyenne Mobile", line=dict(color='orange', width=2), opacity=1))

                # Points Stop Loss / Take Profit
                sl_mask = df_t['Exit_Type'] == 'Stop Loss'
                tp_mask = df_t['Exit_Type'] == 'Take Profit'
                if sl_mask.any():
                    fig.add_trace(go.Scatter(x=df_t.index[sl_mask], y=y_strat[sl_mask], mode='markers', name='Stop Loss', marker=dict(color='#ff4b4b', size=10, symbol='x')))
                if tp_mask.any():
                    fig.add_trace(go.Scatter(x=df_t.index[tp_mask], y=y_strat[tp_mask], mode='markers', name='Take Profit', marker=dict(color='#00ff88', size=12, symbol='star')))

                # Mise en page avec Titre dynamique
                fig.update_layout(
                    title=f"Performance et Signaux : {ticker}",
                    title_x=0.5,
                    height=450, 
                    template="plotly_dark", 
                    hovermode="x unified",
                    margin=dict(l=10, r=10, t=60, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{ticker}")

                # --- Métriques ---
                ret, mdd = calculate_metrics(df_t['Strat_Momentum'])
                val_fin = cap_init * (df_t['Strat_Momentum'].iloc[-1] / df_t['Strat_Momentum'].iloc[0])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Performance", f"{ret:.2%}")
                c2.metric("Max Drawdown", f"{mdd:.2%}")
                c3.metric("Valeur Finale", f"{val_fin:,.2f} $")

        # Dernier onglet : Portefeuille Global
        with tabs[-1]:
            render_portfolio_simulation(tickers, data_dict, cap_init)
            
    else:
        st.write("---")
        st.info("👈 Sélectionnez vos actifs et cliquez sur 'Lancer l'analyse'.")