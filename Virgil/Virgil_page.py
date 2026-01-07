import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from .data_loader import load_stock_data, get_logo_url
from .strategies import apply_strategies, calculate_metrics
from .render_efficiency import min_max_scale, render_portfolio_simulation

def quant_b_ui():
    st.header("📈 Analyse de Portefeuille (Quant B)")

    st.markdown("""
    <style>
            /* 1. On restaure les labels des sliders pour qu'ils soient visibles */
            div[data-testid="stSlider"] label {
                display: block !important;
                visibility: visible !important;
                font-size: 14px !important;
                font-weight: 600 !important;
                color: white !important;
                margin-bottom: 5px !important;
            }

            /* 2. On restaure les labels des number_input */
            div[data-testid="stNumberInput"] label {
                display: block !important;
                visibility: visible !important;
                font-size: 14px !important;
                color: white !important;
            }

            /* 3. OPTIONNEL : Si tu veux cacher UNIQUEMENT les labels dans la section simulation 
            (pour garder les lignes fines), on utilise un sélecteur plus précis sinon laisse tel quel */
        </style>
                

    """, unsafe_allow_html=True)

    # --- 1. INITIALISATION DE LA MÉMOIRE ---
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = {}
    if 'tickers_analyzed' not in st.session_state:
        st.session_state.tickers_analyzed = []
    if 'asset_settings' not in st.session_state:
        st.session_state.asset_settings = {}
    if 'portfolio_weights' not in st.session_state:
        st.session_state.portfolio_weights = {}

    # --- 2. CONFIGURATION (SIDEBAR) ---
    with st.sidebar:
        st.title("⚙️ Global")
        cap_init = st.number_input("Capital Initial ($)", value=1000)

        tickers_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC-USD", "EURUSD=X", "GC=F"]
        selected_tickers = st.multiselect("Actifs du panier", options=tickers_list, default=["AAPL", "MSFT", "BTC-USD"])
        
        if st.button("🔄 Charger les données", use_container_width=True):
            with st.spinner("Chargement..."):
                eq_val = round(100.0 / len(selected_tickers), 2) if selected_tickers else 0
                st.session_state.portfolio_weights = {t: eq_val for t in selected_tickers}
                data_dict = {}
                for t in selected_tickers:
                    st.session_state[f"slider_weight_{t}"] = eq_val
                    st.session_state[f"num_weight_{t}"] = eq_val
                    df_raw = load_stock_data(t)
                    
                    
                    # Initialisation des paramètres par défaut si l'actif est nouveau
                    if t not in st.session_state.asset_settings:
                        st.session_state.asset_settings[t] = {
                            'ma': 20, 
                            'sl': 10.0, 
                            'tp': 30.0  
                        }
                    
                    s = st.session_state.asset_settings[t]
                    data_dict[t] = apply_strategies(df_raw, s['ma'], s['sl']/100, s['tp']/100)
                
                st.session_state.portfolio_data = data_dict
                st.session_state.tickers_analyzed = selected_tickers
            st.rerun()

    # --- 3. AFFICHAGE PRINCIPAL (BIEN INDENTÉ DANS LA FONCTION) ---
    if st.session_state.portfolio_data:
        data_dict = st.session_state.portfolio_data
        tickers = st.session_state.tickers_analyzed

        # ==========================================
        # SECTION A : FOCUS INDIVIDUEL (EN HAUT)
        # ==========================================
        st.subheader("🔍 Configuration & Détails par Actif")
        
        selected_t = st.pills("Actif à modifier :", options=tickers, default=tickers[0])

        if selected_t:
            # Sécurité au cas où l'actif n'est pas dans settings
            if selected_t not in st.session_state.asset_settings:
                st.session_state.asset_settings[selected_t] = {'ma': 20, 'sl': 10.0, 'tp': 30.0}

            with st.container(border=True):
                col_settings, col_chart = st.columns([1, 2.5])
                
                current_settings = st.session_state.asset_settings[selected_t]

                with col_settings:
                    st.write(f"**Réglages {selected_t}**")
                    
                    ma = st.number_input("MA (Périodes)", 5, 100, value=int(current_settings['ma']), key=f"ma_{selected_t}")
                    sl = st.slider("SL (%)", 0.0, 50.0, value=float(current_settings['sl']), step=0.1, key=f"sl_{selected_t}")
                    tp = st.slider("TP (%)", 0.0, 200.0, value=float(current_settings['tp']), step=0.1, key=f"tp_{selected_t}")
                                        
                    # Sauvegarde auto de la position des sliders
                    st.session_state.asset_settings[selected_t] = {'ma': ma, 'sl': sl, 'tp': tp}

                    # Bouton Appliquer pour cet actif précis
                    if st.button(f"Appliquer à {selected_t}", use_container_width=True, type="primary"):
                        df_raw = load_stock_data(selected_t)
                        st.session_state.portfolio_data[selected_t] = apply_strategies(df_raw, ma, sl/100, tp/100)
                        st.rerun()

                with col_chart:
                    df_t = data_dict[selected_t]
                    logo_url = get_logo_url(selected_t)
                    st.markdown(f"""<div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <img src="{logo_url if logo_url else ''}" style="width: 40px; margin-right: 15px;">
                        <h4 style="margin: 0;">Analyse : {selected_t}</h4>
                    </div>""", unsafe_allow_html=True)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_t.index, y=min_max_scale(df_t['Close'])*100, name="Prix", line=dict(color='gray', width=1), opacity=0.5))
                    fig.add_trace(go.Scatter(x=df_t.index, y=df_t['Strat_Momentum'], name="Stratégie", line=dict(color='#00d1ff', width=2)))
                    fig.update_layout(height=300, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{selected_t}")

                # Métriques individuelles
                st.markdown("---")
                ret, mdd = calculate_metrics(df_t['Strat_Momentum'])
                m1, m2, m3 = st.columns(3)
                m1.metric("Performance", f"{ret:.2%}")
                m2.metric("Max Drawdown", f"{mdd:.2%}")
                m3.metric("Volatilité", f"{df_t['Strat_Returns'].std() * np.sqrt(252):.2%}")

        st.divider()

        # ==========================================
        # SECTION B : PORTEFEUILLE GLOBAL (EN BAS)
        # ==========================================
        st.subheader("📊 Performance Globale du Portefeuille")
        # On passe bien cap_init défini dans la sidebar
        render_portfolio_simulation(tickers, data_dict, cap_init)