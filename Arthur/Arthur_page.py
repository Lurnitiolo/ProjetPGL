import streamlit as st

def quant_a_ui():
    """
    Interface de présentation du module Quant A.
    Pour l'instant, aucun calcul ni chargement de données n'est effectué.
    """
    st.header("Module Quant A : Analyse Univariée")
    
    st.markdown("""
    ### Bienvenue sur l'espace d'analyse Single Asset
    
    Ce module est conçu pour permettre aux gestionnaires de portefeuille d'analyser 
    la performance d'un actif financier spécifique en temps réel.
    
    **Fonctionnalités à venir :**
    - 📈 **Visualisation** : Affichage du prix en temps réel et graphiques interactifs.
    - 🛠 **Backtesting** : Test de stratégies (Moyenne Mobile, Buy & Hold).
    - 📊 **Métriques** : Calcul automatique du Max Drawdown et du Ratio de Sharpe.
    
    ---
    *Sélectionnez un actif et configurez les paramètres ci-dessous pour commencer l'analyse.*
    """)

    # Zone vide pour la future interface
    st.info("L'interface de configuration et les graphiques s'afficheront ici prochainement.")