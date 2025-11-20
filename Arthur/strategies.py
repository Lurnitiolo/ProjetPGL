import streamlit as st
import plotly.graph_objects as go
from .data_loader import load_stock_data
from .strategies import apply_strategies, calculate_metrics

def quant_a_ui():
    st.header("Analyse Univariée (Quant A)")
    
    # --- 1. Barre latérale pour les contrôles ---
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker (Yahoo Finance)", value="BTC-USD")
    with col2:
        window = st.slider("Fenêtre Moyenne Mobile", min_value=5, max_value=200, value=20)

    if st.button("Lancer l'analyse"):
        # --- 2. Récupération et Calculs ---
        with st.spinner('Chargement des données...'):
            df = load_stock_data(ticker, period="2y")
            
            if df is not None:
                df_strat = apply_strategies(df, ma_window=window)
                
                # --- 3. Affichage des Métriques ---
                ret, mdd = calculate_metrics(df_strat['Strat_Momentum'])
                
                # On utilise des colonnes pour faire joli
                m1, m2 = st.columns(2)
                m1.metric("Performance Stratégie", f"{ret:.2%}")
                m2.metric("Max Drawdown", f"{mdd:.2%}")

                # --- 4. Graphique Interactif (Plotly) ---
                fig = go.Figure()
                
                # Courbe Prix Réel
                fig.add_trace(go.Scatter(
                    x=df_strat.index, y=df_strat['Close'], 
                    mode='lines', name='Prix Actif',
                    line=dict(color='blue', width=1)
                ))
                
                # Courbe Moyenne Mobile
                fig.add_trace(go.Scatter(
                    x=df_strat.index, y=df_strat['MA'], 
                    mode='lines', name=f'Moyenne Mobile {window}',
                    line=dict(color='orange', width=1, dash='dot')
                ))
                
                # Pour l'instant on affiche juste le prix pour voir si ça marche
                # On ajoutera la courbe de richesse (portefeuille) juste après
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tableau des dernières données
                st.subheader("Dernières données")
                st.dataframe(df_strat.tail())
                
            else:
                st.error("Erreur : Impossible de récupérer les données pour ce ticker.")