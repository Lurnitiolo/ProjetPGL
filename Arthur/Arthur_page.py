import streamlit as st
import plotly.graph_objects as go

# Le point . devant signifie "cherche dans le dossier où je suis"
from .data_loader import load_stock_data
from .strategies import apply_strategies, calculate_metrics

def quant_a_ui():
    st.header("Analyse Univariée (Arthur)")
    
    # --- 1. Barre latérale pour les contrôles ---
    col1, col2 = st.columns(2)
    with col1:
        # Par défaut Bitcoin, mais tu peux changer
        ticker = st.text_input("Ticker (Yahoo Finance)", value="BTC-USD")
    with col2:
        window = st.slider("Fenêtre Moyenne Mobile", min_value=5, max_value=200, value=20)

    if st.button("Lancer l'analyse"):
        # --- 2. Récupération et Calculs ---
        with st.spinner(f'Chargement des données pour {ticker}...'):
            # On charge 2 ans d'historique
            df = load_stock_data(ticker, period="2y")
            
            if df is not None:
                # On applique tes stratégies
                df_strat = apply_strategies(df, ma_window=window)
                
                # --- 3. Affichage des Métriques ---
                # On récupère la performance de la stratégie Momentum
                ret, mdd = calculate_metrics(df_strat['Strat_Momentum'])
                
                m1, m2 = st.columns(2)
                m1.metric("Performance Stratégie", f"{ret:.2%}")
                m2.metric("Max Drawdown", f"{mdd:.2%}")

                # --- 4. Graphique Interactif ---
                fig = go.Figure()
                
                # Courbe 1 : Prix de l'actif (ex: Bitcoin)
                fig.add_trace(go.Scatter(
                    x=df_strat.index, y=df_strat['Close'], 
                    mode='lines', name='Prix Actif',
                    line=dict(color='blue', width=1)
                ))
                
                # Courbe 2 : La Moyenne Mobile
                fig.add_trace(go.Scatter(
                    x=df_strat.index, y=df_strat['MA'], 
                    mode='lines', name=f'Moyenne Mobile {window}',
                    line=dict(color='orange', width=1, dash='dot')
                ))
                
                # Courbe 3 : Ton Portefeuille (Stratégie)
                # On l'affiche sur un axe secondaire ou juste pour comparer la tendance
                # Pour l'instant, on la met simplement
                fig.add_trace(go.Scatter(
                    x=df_strat.index, y=df_strat['Strat_Momentum'],
                    mode='lines', name='Valeur Portefeuille (Base 100)',
                    line=dict(color='green', width=2)
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("Analyse terminée avec succès.")
                
            else:
                st.error("Erreur : Impossible de récupérer les données. Vérifie le ticker.")