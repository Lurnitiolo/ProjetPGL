import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from .data_loader import load_stock_data
from .strategies import apply_strategies, calculate_metrics

def quant_b_ui():
    st.header("Analyse de Portefeuille & Diversification (Quant B)")    

    # --- 1. INITIALISATION DE LA MÉMOIRE (Session State) ---
    # On crée un dictionnaire pour stocker les DataFrames de chaque actif
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = {} # Stocke les DF complets
    if 'tickers_analyzed' not in st.session_state:
        st.session_state.tickers_analyzed = []
    if 'ma_used' not in st.session_state:
        st.session_state.ma_used = 20

    with st.sidebar:
        st.subheader("Configuration")
        tickers_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC-USD", "EURUSD=X", "GC=F"]
        selected_tickers = st.multiselect(
            "Sélectionnez les actifs", 
            options=tickers_list, 
            default=["AAPL", "MSFT", "BTC-USD"]
        )
        ma_window = st.number_input("Fenêtre Moyenne Mobile", 5, 100, 20)
        
        # Bouton pour déclencher le téléchargement (lourd)
        if st.button("🚀 Lancer l'analyse"):
            with st.spinner("Téléchargement et calculs en cours..."):
                results = {}
                for ticker in selected_tickers:
                    df = load_stock_data(ticker)
                    if df is not None and not df.empty:
                        # On stocke le DF transformé par la stratégie
                        results[ticker] = apply_strategies(df, ma_window=ma_window)
                
                # Sauvegarde en mémoire
                st.session_state.portfolio_data = results
                st.session_state.tickers_analyzed = selected_tickers
                st.session_state.ma_used = ma_window

    # --- 2. AFFICHAGE DES ONGLETS (Si des données sont en mémoire) ---
    if st.session_state.portfolio_data:
        tickers = st.session_state.tickers_analyzed
        data_dict = st.session_state.portfolio_data
        
        # Création dynamique des noms d'onglets
        tab_list = tickers + ["📊 Portefeuille Global"]
        tabs = st.tabs(tab_list)

        # BOUCLE POUR LES ONGLETS INDIVIDUELS
        for i, ticker in enumerate(tickers):
            with tabs[i]:
                st.subheader(f"Analyse {ticker}")
                df_ticker = data_dict[ticker]
                
                # Métriques
                ret, mdd = calculate_metrics(df_ticker['Strat_Momentum'])
                c1, c2 = st.columns(2)
                c1.metric("Performance (Stratégie)", f"{ret:.2%}")
                c2.metric("Max Drawdown", f"{mdd:.2%}")

                # Graphique détaillé
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_ticker.index, y=df_ticker['Close'], name="Prix", line=dict(color='gray', width=1), opacity=0.5))
                fig.add_trace(go.Scatter(x=df_ticker.index, y=df_ticker['MA'], name=f"MA {st.session_state.ma_used}", line=dict(dash='dot')))
                fig.add_trace(go.Scatter(x=df_ticker.index, y=df_ticker['Strat_Momentum'], name="Stratégie", line=dict(color='blue', width=2)))
                
                fig.update_layout(height=400, hovermode="x unified", title=f"Détails {ticker}")
                st.plotly_chart(fig, use_container_width=True)

        # --- 3. ONGLET PORTEFEUILLE GLOBAL (INTERACTIF) ---
        with tabs[-1]:
            st.subheader("Simulateur de Portefeuille Dynamique")
            st.info("Ajustez les curseurs ci-dessous pour voir l'impact sur le portefeuille global.")

            # Création des Sliders pour les poids
            weights = {}
            weight_cols = st.columns(len(tickers))
            for j, t in enumerate(tickers):
                # On utilise une clé unique pour chaque slider pour éviter les conflits
                weights[t] = weight_cols[j].slider(f"{t} %", 0, 100, 33, key=f"weight_{t}")

            total_w = sum(weights.values())

            if total_w > 0:
                # Création d'un DataFrame commun pour le calcul
                # On aligne les dates (dropna) pour avoir un historique propre
                df_global = pd.DataFrame({t: data_dict[t]['Strat_Momentum'] for t in tickers}).dropna()
                
                # Calcul de la valeur pondérée du portefeuille
                df_global['Portfolio_Value'] = 0
                for t in tickers:
                    norm_weight = weights[t] / total_w
                    df_global['Portfolio_Value'] += df_global[t] * norm_weight

                # Métriques Globales
                p_ret, p_mdd = calculate_metrics(df_global['Portfolio_Value'])
                m1, m2 = st.columns(2)
                m1.metric("Rendement Global", f"{p_ret:.2%}")
                m2.metric("Risque Global (MDD)", f"{p_mdd:.2%}")

                # Graphique de comparaison
                fig_glob = go.Figure()
                # La ligne du portefeuille
                fig_glob.add_trace(go.Scatter(x=df_global.index, y=df_global['Portfolio_Value'], 
                                              name="MON PORTEFEUILLE", line=dict(color='gold', width=4)))
                
                # Les actifs en arrière-plan
                for t in tickers:
                    fig_glob.add_trace(go.Scatter(x=df_global.index, y=df_global[t], 
                                                  name=f"Contrib: {t}", line=dict(width=1), opacity=0.3))

                fig_glob.update_layout(height=500, title="Performance du Panier vs Actifs", template="plotly_dark")
                st.plotly_chart(fig_glob, use_container_width=True)
            else:
                st.warning("Veuillez allouer un pourcentage à au moins un actif.")
    else:
        # Message d'accueil si aucune donnée n'est chargée
        st.write("---")
        st.info("👈 Sélectionnez vos actifs dans la barre latérale et cliquez sur 'Lancer l'analyse'.")