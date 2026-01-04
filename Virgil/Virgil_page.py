import streamlit as st
import plotly.graph_objects as go
from .data_loader import load_stock_data
from .strategies import apply_strategies, calculate_metrics

def quant_b_ui():
    st.header("Analyse de Portefeuille & Diversification (Quant B)")    
    with st.sidebar:
        st.subheader("Composition du Portefeuille")
        
        tickers_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC-USD", "EURUSD=X", "GC=F"]
        selected_tickers = st.multiselect(
            "Sélectionnez les actifs (Min 3)", 
            options=tickers_list, 
            default=["AAPL", "MSFT", "BTC-USD"]
        )
        window = st.number_input(
            "Fenêtre de la Moyenne Mobile", min_value=5, max_value=100, value=20, step=1
        )
    
    if len(selected_tickers) < 3:
        st.warning("⚠️ Veuillez sélectionner au moins 3 actifs pour une analyse de diversification optimale.")
        return 

    if st.button("Lancer l'analyse"):
        for ticker in selected_tickers:
            df = load_stock_data(ticker)
            if df is not None and not df.empty:
                window = 20
                df_strat = apply_strategies(df, ma_window=window)
                
                ret, mdd = calculate_metrics(df_strat['Strat_Momentum'])
                
                m1, m2 = st.columns(2)
                m1.metric(f"Performance Stratégie ({ticker})", f"{ret:.2%}")
                m2.metric(f"Max Drawdown ({ticker})", f"{mdd:.2%}")

                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df_strat.index, y=df_strat['Close'], 
                    mode='lines', name=f'Prix Actif {ticker}',
                    line=dict(color='blue', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df_strat.index, y=df_strat['MA'], 
                    mode='lines', name=f'Moyenne Mobile {window}',
                    line=dict(color='orange', width=1, dash='dot')
                ))

                fig.add_trace(go.Scatter(
                    x=df_strat.index, y=df_strat['Strat_Momentum'],
                    mode='lines', name='Valeur Portefeuille (Base 100)',
                    line=dict(color='green', width=2)
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"Analyse terminée avec succès pour {ticker}.")
            else:
                    st.error(f"Erreur : Impossible de récupérer les données pour l'actif sélectionné ({ticker}). Vérifie le symbole ou la disponibilité des données.")
                    st.error(f"Erreur : Impossible de récupérer les données pour {ticker}. Vérifie le ticker.")