import streamlit as st
# Note: On utilise des imports relatifs qui fonctionneront quand lancé via app.py
from .data_loader import load_stock_data
from .strategies import AVAILABLE_ASSETS, apply_strategies

def quant_a_ui():
    st.header("Module Quant A : Analyse Univariée & Backtesting")
    
    # --- Configuration ---
    with st.expander("Paramètres de l'actif", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_name = st.selectbox("Choisir un actif", options=list(AVAILABLE_ASSETS.keys()))
            if selected_name == "Autre (Saisir manuellement)":
                ticker = st.text_input("Symbole Yahoo", value="AIR.PA")
            else:
                ticker = AVAILABLE_ASSETS[selected_name]
        with col2:
            # On force daily pour que les moyennes mobiles aient du sens
            st.info("Intervalle fixé à '1d' pour le backtest")
            interval = "1d" 
        with col3:
            period = st.selectbox("Historique", ["1y", "2y", "5y", "max"], index=1)

    if st.button(f"Lancer l'analyse sur {selected_name}"):
        with st.spinner(f'Calculs en cours pour {ticker}...'):
            
            # 1. Chargement
            df = load_stock_data(ticker, period=period, interval=interval)
            
            if df is not None and not df.empty:
                # 2. Application des stratégies
                df_strat, metrics = apply_strategies(df)
                
                # 3. Affichage des Métriques (KPIs)
                st.write("### 📊 Performance & Risque")
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                
                kpi1.metric("Sharpe (Buy&Hold)", f"{metrics['BuyHold_Sharpe']:.2f}")
                kpi2.metric("Max Drawdown (B&H)", f"{metrics['BuyHold_MDD']:.2%}")
                kpi3.metric("Sharpe (Momentum)", f"{metrics['Momentum_Sharpe']:.2f}")
                kpi4.metric("Max Drawdown (Mom.)", f"{metrics['Momentum_MDD']:.2%}")
                
                # 4. Graphique Comparatif
                st.write("### 📈 Comparaison des Stratégies (Base 100)")
                # On affiche uniquement les colonnes de valeur
                chart_data = df_strat[['Strat_BuyHold', 'Strat_Momentum']]
                st.line_chart(chart_data)
                
                # 5. Données détaillées
                with st.expander("Voir les données brutes"):
                    st.dataframe(df_strat.tail())
                
            else:
                st.error("Erreur de récupération des données.")