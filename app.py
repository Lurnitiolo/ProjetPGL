import streamlit as st
# On importe la fonction d'affichage de ton module
from quant_a.quant_a_page import quant_a_ui

# Configuration de la page
st.set_page_config(page_title="Finance Dashboard", layout="wide")

st.title("Plateforme de Gestion Quantitative")

# Menu de navigation sur le côté
sidebar_option = st.sidebar.radio("Navigation", ["Accueil", "Quant A (Single Asset)", "Quant B (Portfolio)"])

if sidebar_option == "Accueil":
    st.write("Bienvenue sur la plateforme. Choisissez un module dans le menu à gauche.")
    
elif sidebar_option == "Quant A (Single Asset)":
    # C'est ici qu'on appelle TON code
    quant_a_ui()
    
elif sidebar_option == "Quant B (Portfolio)":
    st.info("Module en construction par le binôme...")