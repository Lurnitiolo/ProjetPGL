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
            /* 1. Réduire l'espace global entre les widgets dans cette colonne */
            [data-testid="column"] [data-testid="stVerticalBlock"] > div {
                margin-top: 20px !important;
                margin-bottom: 10px !important;
            }

            /* 2. Rapprocher les labels (titres) de leurs champs respectifs */
            label[data-testid="stWidgetLabel"] {
                margin-bottom: 15px !important;
                font-size: 0.85rem !important;
                font-weight: 600 !important;
            }

            /* 3. Ajustement spécifique pour remonter le champ MA */
            div[data-testid="stNumberInput"] {
                margin-top: 15px !important;
            }

            /* 4. Ajustement spécifique pour les Sliders SL et TP */
            div[data-testid="stSlider"] {
                margin-top: -5px !important;
                padding-bottom: 0px !important;
            }
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

        tickers_list = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC-USD", "EURUSD=X", "GC=F", 
            "MC.PA", "TTE.PA", "AIR.PA", "BNP.PA", "SAN.PA", "NVDA", "SPY", 
            "CW8.PA", "QQQ", "GLD", "ETH-USD"
        ]
        selected_tickers = st.multiselect("Actifs du panier", options=tickers_list, default=["AAPL", "MSFT", "BTC-USD"])
        
        # AJOUT : Message d'avertissement visuel si < 3
        if len(selected_tickers) < 3:
            st.warning("⚠️ Sélectionnez au moins 3 actifs.")

        if st.button("🔄 Charger les données", use_container_width=True):
            # VERIFICATION de la condition au clic
            if len(selected_tickers) < 3:
                st.error("Erreur : Vous devez choisir au moins 3 actifs pour simuler un portefeuille.")
            else:
                with st.spinner("Chargement..."):
                    eq_val = round(100.0 / len(selected_tickers), 2)
                    st.session_state.portfolio_weights = {t: eq_val for t in selected_tickers}

                    for t in selected_tickers:
                        st.session_state[f"slider_weight_{t}"] = eq_val
                        st.session_state[f"num_weight_{t}"] = eq_val

                    data_dict = {}
                    for t in selected_tickers:
                        df_raw = load_stock_data(t)
                        
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
                    # --- CSS POUR LES TITRES ET ESPACEMENT ---
                    st.markdown("""
                        <style>
                            /* Descend les labels des widgets (SL et TP) */
                            [data-testid="stWidgetLabel"] p {
                                padding-top: 10px !important;
                                margin-bottom: -5px !important;
                                font-size: 0.85rem !important;
                            }
                            
                            /* Style pour le titre MA personnalisé */
                            .ma-header {
                                margin-bottom: -15px !important;
                                padding-top: 5px;
                                font-size: 0.9rem;
                            }

                            /* Espace le bloc MA pour qu'il soit bien visible */
                            div[data-testid="stNumberInput"] {
                                margin-top: 0px !important;
                            }
                        </style>
                    """, unsafe_allow_html=True)

                    # --- TITRE ET INPUT MA ---
                    st.markdown('<p class="ma-header"> MA</p>', unsafe_allow_html=True)
                    ma = st.number_input("Périodes (Fenêtre)", 5, 100, value=int(current_settings['ma']), key=f"ma_{selected_t}")
                    
                    # --- SLIDERS ---
                    sl = st.slider("SL (%)", 0.0, 50.0, value=float(current_settings['sl']), step=0.6, key=f"sl_{selected_t}")
                    tp = st.slider("TP (%)", 0.0, 200.0, value=float(current_settings['tp']), step=0.6, key=f"tp_{selected_t}")
                                                
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
                    fig.add_trace(go.Scatter(x=df_t.index, y=min_max_scale(df_t['Close'].rolling(int(ma)).mean())*100, name=f"MA {ma}", line=dict(color='#ff9800', width=1.5, dash='dot')))
                    fig.add_trace(go.Scatter(x=df_t.index, y=df_t['Strat_Momentum'], name="Stratégie", line=dict(color='#00d1ff', width=2)))
                    fig.update_layout(height=300, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{selected_t}")

                price_series = df_t['Close']
                bh_cum_return = (price_series / price_series.iloc[0]) # Série base 1.0
                bh_ret, bh_mdd = calculate_metrics(bh_cum_return)
                bh_vol = price_series.pct_change().std() * np.sqrt(252)

                # --- 2. CALCUL DES MÉTRIQUES DE LA STRATÉGIE ---
                strat_ret, strat_mdd = calculate_metrics(df_t['Strat_Momentum'])
                strat_vol = df_t['Strat_Returns'].std() * np.sqrt(252)

                # --- 3. AFFICHAGE AVEC BULLES (DELTAS) ---
                st.markdown(f"**Performance de la stratégie vs Buy & Hold ({selected_t})**")
                m1, m2, m3 = st.columns(3)

                with m1:
                    # Delta vert si la stratégie rapporte plus que l'actif seul
                    diff_ret = strat_ret - bh_ret
                    st.metric(
                        "Performance", 
                        f"{strat_ret:.2%}", 
                        delta=f"{diff_ret:+.2%}",
                        help="Comparaison entre votre stratégie et le simple fait de détenir l'actif (Buy & Hold)."
                    )

                with m2:
                    # Delta vert (inverse) si la stratégie réduit la perte maximale (MDD)
                    # Plus le MDD est proche de 0, mieux c'est.
                    diff_mdd = strat_mdd - bh_mdd
                    st.metric(
                        "Max Drawdown", 
                        f"{strat_mdd:.2%}", 
                        delta=f"{diff_mdd:+.2%}", 
                        delta_color="inverse",
                        help="La bulle est verte si votre Stop Loss réduit effectivement la chute maximale."
                    )

                with m3:
                    # Delta vert (inverse) si la stratégie réduit la volatilité
                    diff_vol = strat_vol - bh_vol
                    st.metric(
                        "Volatilité", 
                        f"{strat_vol:.2%}", 
                        delta=f"{diff_vol:+.2%}", 
                        delta_color="inverse",
                        help="La bulle est verte si vos réglages lissent les mouvements de prix."
                    )


        # ==========================================
        # SECTION B : PORTEFEUILLE GLOBAL (EN BAS)
        # ==========================================

        # On passe bien cap_init défini dans la sidebar
        render_portfolio_simulation(tickers, data_dict, cap_init)