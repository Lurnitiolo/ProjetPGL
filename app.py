import streamlit as st

# --- IMPORT CORRIGÉ ---
# On importe la fonction 'quant_a_ui' depuis le fichier 'Arthur/Arthur_page.py'
from Arthur.Arthur_page import quant_a_ui
from Virgil.Virgil_page import quant_b_ui, min_max_scale

# Configuration de la page (doit être la première commande Streamlit)
st.set_page_config(page_title="Finance Dashboard", layout="wide")

st.title("Plateforme de Gestion Quantitative")

# --- NAVIGATION ---
# J'ai mis les noms clairs pour que tu t'y retrouves
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
    # On appelle la fonction principale de ton module
    quant_a_ui()

elif sidebar_option == "Quant B (Virgil)":
    quant_b_ui()
    