import streamlit as st
from .data_loader import load_stock_data
# On importe la liste des actifs qu'on a définie dans strategies.py
from .strategies import AVAILABLE_ASSETS

def quant_a_ui():
    st.header("Module Quant A : Analyse Univariée")
    
    # --- 1. Zone de Configuration ---
    with st.expander("Paramètres de l'actif", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # MENU DÉROULANT (Selectbox)
            # On affiche les noms lisibles (clés du dictionnaire)
            selected_name = st.selectbox(
                "Choisir un actif", 
                options=list(AVAILABLE_ASSETS.keys()),
                index=0
            )
            
            # LOGIQUE : Si "Autre", on affiche un champ texte, sinon on prend le code
            if selected_name == "Autre (Saisir manuellement)":
                ticker = st.text_input("Symbole Yahoo (ex: AIR.PA)", value="AIR.PA")
            else:
                # On récupère le code associé au nom (ex: "Bitcoin" -> "BTC-USD")
                ticker = AVAILABLE_ASSETS[selected_name]
        
        with col2:
            interval = st.selectbox("Intervalle", ["1d", "1wk", "1mo"], index=0)
            
        with col3:
            period = st.selectbox("Historique", ["1y", "2y", "5y", "max"], index=0)

    # --- 2. Action ---
    if st.button(f"Analyser {selected_name}"):
        
        # Petit message de chargement
        with st.spinner(f'Récupération des données pour {ticker}...'):
            
            # On appelle ton data_loader
            df = load_stock_data(ticker, period=period, interval=interval)
            
            # --- 3. Résultat ---
            if df is not None:
                st.success(f"Données chargées pour {ticker} ({len(df)} lignes)")
                
                # On affiche juste le tableau comme demandé
                st.write("### Données brutes")
                st.dataframe(df.tail())
                
                # Graphique simple de vérification
                st.line_chart(df['Close'])
                
            else:
                st.error(f"Erreur : Impossible de trouver l'actif '{ticker}'.")