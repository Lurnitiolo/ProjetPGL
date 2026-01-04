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
        ma_window = st.number_input(
            "Fenêtre de la Moyenne Mobile", min_value=5, max_value=100, value=20, step=1
        )
    
    if len(selected_tickers) < 3:
        st.warning("⚠️ Veuillez sélectionner au moins 3 actifs pour une analyse de diversification optimale.")
        return 

    if len(selected_tickers) < 3:
        st.warning("⚠️ Veuillez sélectionner au moins 3 actifs pour une analyse de diversification optimale.")
        return # Arrête l'exécution si condition non remplie

    if st.button("Lancer l'analyse"):
        # Création des onglets dynamiquement selon les tickers choisis
        tabs = st.tabs(selected_tickers)
        
        for i, ticker in enumerate(selected_tickers):
            with tabs[i]: # On affiche le contenu dans l'onglet correspondant
                st.subheader(f"Analyse détaillée : {ticker}")
                
                with st.spinner(f"Chargement de {ticker}..."):
                    df = load_stock_data(ticker)
                
                if df is not None and not df.empty:
                    # Utilisation de la variable ma_window choisie par l'utilisateur
                    df_strat = apply_strategies(df, ma_window=ma_window)
                    
                    ret, mdd = calculate_metrics(df_strat['Strat_Momentum'])
                    
                    # Affichage des métriques dans des colonnes
                    col1, col2 = st.columns(2)
                    col1.metric("Performance Stratégie", f"{ret:.2%}")
                    col2.metric("Max Drawdown", f"{mdd:.2%}", delta_color="inverse")

                    # Graphique interactif
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['Close'], name='Prix Actif', line=dict(color='rgba(0,0,255,0.4)')))
                    fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['MA'], name=f'MA {ma_window}', line=dict(dash='dot')))
                    fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['Strat_Momentum'], name='Stratégie', line=dict(color='green', width=2)))
                    
                    fig.update_layout(title=f"Évolution {ticker}", height=400, margin=dict(l=0, r=0, b=0, t=40))
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"Impossible de récupérer les données pour {ticker}.")