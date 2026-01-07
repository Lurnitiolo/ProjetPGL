import streamlit as st

st.set_page_config(page_title="Finance Dashboard", layout="wide")

# --- IMPORT CORRIGÉ ---
from Arthur.Arthur_page import quant_a_ui

# --- NAVIGATION ---
sidebar_option = st.sidebar.radio(
    "Navigation", 
    ["Accueil", "Quant A (Arthur)", "Quant B (Virgil)"]
)

# --- LOGIQUE D'AFFICHAGE ---
if sidebar_option == "Accueil":
    st.write("### Bienvenue sur le Dashboard Finance")
    st.write("Sélectionnez un module dans le menu à gauche pour commencer.")
    st.info("Projet réalisé par Arthur & Virgil")

elif sidebar_option == "Quant A (Arthur)":
    quant_a_ui()

elif sidebar_option == "Quant B (Virgil)":
    st.warning("⚠️ Module Portfolio (Virgil) en cours de construction...")