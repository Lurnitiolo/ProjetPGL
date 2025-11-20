import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# On suppose que load_stock_data peut gérer une boucle ou on l'appelle plusieurs fois
from .data_loader import load_stock_data
# Pour Quant B, les stratégies sont souvent des allocations (ex: 60/40, Equi-pondéré)
from .strategies import calculate_metrics 

def quant_b_ui():
    st.header("Analyse de Portefeuille & Diversification (Quant B)")

    # --- 1. Barre latérale pour la configuration du Portefeuille ---
    with st.sidebar:
        st.subheader("Composition du Portefeuille")
        
        # [cite_start]Sélection multiple (Obligatoire pour Quant B : min 3 actifs) [cite: 41]
        tickers_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC-USD", "EURUSD=X", "GC=F"]
        selected_tickers = st.multiselect(
            "Sélectionnez les actifs (Min 3)", 
            options=tickers_list, 
            default=["AAPL", "MSFT", "BTC-USD"]
        )
        
        period = st.selectbox("Période d'analyse", ["1y", "2y", "5y", "ytd"], index=1)
        
        st.markdown("---")
        st.subheader("Simulation Stratégie")
        # Pour simplifier, on part sur une allocation équipondérée par défaut
        allocation_type = st.radio("Type d'allocation", ["Équipondéré (Equal Weight)", "Personnalisé (À venir)"])

        run_btn = st.button("Lancer la Simulation", type="primary")

    # --- 2. Logique Principale ---
    if run_btn:
        if len(selected_tickers) < 2:
            st.error("Veuillez sélectionner au moins 2 actifs pour une analyse de portefeuille.")
        else:
            with st.spinner('Récupération des données et calculs...'):
                
                # Dictionnaire pour stocker les DataFrames
                data_dict = {}
                valid_tickers = []

                # Boucle de récupération des données
                for t in selected_tickers:
                    df_temp = load_stock_data(t, period=period)
                    if df_temp is not None and not df_temp.empty:
                        data_dict[t] = df_temp['Close'] # On garde seulement le prix de clôture
                        valid_tickers.append(t)
                
                if not data_dict:
                    st.error("Aucune donnée récupérée.")
                    return

                # Création d'un DataFrame global (Dates alignées)
                df_portfolio = pd.DataFrame(data_dict).dropna()
                
                # --- 3. Calculs Multivariés ---
                
                # A. Performance Normalisée (Base 100 au début)
                # [cite_start]Cela permet de comparer des actifs aux prix très différents (ex: BTC vs EUR/USD) [cite: 44]
                df_normalized = df_portfolio / df_portfolio.iloc[0] * 100
                
                # [cite_start]B. Calcul du Portefeuille (Ici Équipondéré) [cite: 42]
                # Somme des lignes divisée par le nombre d'actifs
                df_normalized['PORTFOLIO_TOTAL'] = df_normalized.mean(axis=1)
                
                # C. Rendements quotidiens pour la volatilité et corrélation
                daily_returns = df_portfolio.pct_change().dropna()
                
                # [cite_start]D. Matrice de Corrélation [cite: 42]
                corr_matrix = daily_returns.corr()

                # --- 4. Affichage des Résultats ---

                # A. Métriques Globales du Portefeuille
                cum_ret = df_normalized['PORTFOLIO_TOTAL'].iloc[-1] - 100
                volatility = daily_returns.std().mean() * np.sqrt(252) # Volatilité moyenne annualisée
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Rendement Portefeuille", f"{cum_ret:.2f}%", delta_color="normal")
                col2.metric("Volatilité Moyenne", f"{volatility:.2%}", delta_color="inverse")
                col3.metric("Nb Actifs", len(valid_tickers))

                # B. Graphique 1 : Comparaison Performance (Base 100)
                st.subheader("Performance Comparée (Base 100)")
                fig_perf = go.Figure()
                
                # Tracer chaque actif en gris clair ou couleur légère
                for ticker in valid_tickers:
                    fig_perf.add_trace(go.Scatter(
                        x=df_normalized.index, y=df_normalized[ticker],
                        mode='lines', name=ticker,
                        line=dict(width=1), opacity=0.6
                    ))
                
                # Tracer le Portefeuille en GROS et visible
                fig_perf.add_trace(go.Scatter(
                    x=df_normalized.index, y=df_normalized['PORTFOLIO_TOTAL'],
                    mode='lines', name='PORTEFEUILLE GLOBAL',
                    line=dict(color='white', width=4) # Blanc ou couleur vive selon votre thème
                ))
                fig_perf.update_layout(hovermode="x unified", template="plotly_dark")
                st.plotly_chart(fig_perf, use_container_width=True)

                # [cite_start]C. Graphique 2 : Matrice de Corrélation (Heatmap) [cite: 42]
                # C'est LE graphique indispensable pour le Quant B
                st.subheader("Matrice de Corrélation des Actifs")
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu', # Rouge (corrélé) à Bleu (inverse)
                    zmin=-1, zmax=1,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate="%{text}",
                    showscale=True
                ))
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)