import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from .data_loader import get_logo_url
from .strategies import calculate_metrics

def min_max_scale(series):
    if series.max() == series.min(): return series * 0
    return (series - series.min()) / (series.max() - series.min())


def apply_preset_callback(new_weights):
    # On met à jour le dictionnaire central utilisé par les widgets
    if 'portfolio_weights' not in st.session_state:
        st.session_state.portfolio_weights = {}
    
    for t, val in new_weights.items():
        st.session_state.portfolio_weights[t] = float(val)
        # On met aussi à jour les clés directes des widgets pour forcer l'affichage
        st.session_state[f"slider_weight_{t}"] = float(val)
        st.session_state[f"num_weight_{t}"] = float(val)

@st.fragment
def render_portfolio_simulation(tickers, data_dict, cap_init):

    # 1. Initialisation de la mémoire des poids
    if 'portfolio_weights' not in st.session_state:
        eq_val = round(100.0 / len(tickers), 2)
        st.session_state.portfolio_weights = {t: eq_val for t in tickers}
    
    # --- 1. PRÉ-CALCUL DES STATS ---
    stats = {}
    for t in tickers:
        p_ret, p_mdd = calculate_metrics(data_dict[t]['Strat_Momentum'])
        vol = data_dict[t]['Strat_Returns'].std() * np.sqrt(252)
        stats[t] = {
            'ret': p_ret, 
            'mdd': abs(p_mdd) if abs(p_mdd) > 0.01 else 0.01,
            'vol': vol if vol > 0 else 0.01,
            'sharpe': p_ret / vol if vol > 0 else 0
        }

    # --- PRÉPARATION DES DONNÉES DE PRESETS ---
    # On calcule les poids ici pour pouvoir les passer aux callbacks des boutons (haut et bas)
    presets = {}
    
    # Equal Weight
    eq_val = round(100.0 / len(tickers), 2)
    presets['eq'] = {t: eq_val for t in tickers}
    
    # Risk Parity
    total_inv_vol = sum(1/s['vol'] for s in stats.values())
    presets['rp'] = {t: round(((1/stats[t]['vol']) / total_inv_vol) * 100, 2) for t in tickers}
    
    # Min Drawdown
    total_inv_mdd = sum(1/s['mdd'] for s in stats.values())
    presets['md'] = {t: round(((1/stats[t]['mdd']) / total_inv_mdd) * 100, 2) for t in tickers}
    
    # Top Perf
    pos_rets = {t: max(0, stats[t]['ret']) for t in tickers}
    total_pos = sum(pos_rets.values())
    presets['tp'] = {t: (round((pos_rets[t] / total_pos) * 100, 2) if total_pos > 0 else eq_val) for t in tickers}
    
    # Sharpe
    pos_sharpe = {t: max(0, stats[t]['sharpe']) for t in tickers}
    total_sharpe = sum(pos_sharpe.values())
    presets['sr'] = {t: (round((pos_sharpe[t] / total_sharpe) * 100, 2) if total_sharpe > 0 else eq_val) for t in tickers}

    # --- BOUTONS DU HAUT ---
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    c1.button("⚖️ Equal Weight", on_click=apply_preset_callback, args=(presets['eq'],), key="btn_eq_top")
    c2.button("🛡️ Risk Parity", on_click=apply_preset_callback, args=(presets['rp'],), key="btn_rp_top")
    c3.button("📉 Min Drawdown", on_click=apply_preset_callback, args=(presets['md'],), key="btn_md_top")
    c4.button("🚀 Top Perf", on_click=apply_preset_callback, args=(presets['tp'],), key="btn_tp_top")
    c5.button("💎 Sharpe Ratio", on_click=apply_preset_callback, args=(presets['sr'],), key="btn_sr_top")

    st.divider()

    
    # --- 2. RÉPARTITION DU CAPITAL (UI AUTO-SCALÉE) ---
    # On ajuste le ratio global : 1.5 pour les réglages, 1 pour la roue (pie chart)
    col_inputs, col_visual = st.columns([1.5, 1])
    weights = {}
    
    with col_inputs:
        # --- STYLE CSS AVANCÉ (Look Léché & Pro) ---
        st.markdown("""
            <style>
                /* Container principal de la ligne d'actif */
                .asset-row {
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 8px;
                    padding: 6px 12px;
                    margin-bottom: 8px;
                    border: 1px solid rgba(255, 255, 255, 0.05);
                    transition: all 0.2s ease;
                    display: flex;
                    align-items: center;
                }
                .asset-row:hover {
                    background: rgba(255, 255, 255, 0.06);
                    border-color: rgba(0, 209, 255, 0.3);
                }

                /* Typographie et Logos */
                .asset-info {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    min-width: 120px;
                }
                .ticker-name {
                    font-weight: 700;
                    font-family: 'Inter', sans-serif;
                    
                    font-size: 14px;
                    letter-spacing: 0.5px;
                }
                .responsive-logo {
                    width: 64px;
                    height: 64px;
                    object-fit: contain;
                    filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.5));
                }

                /* Customisation Sliders Streamlit */
                div[data-testid="stSlider"] {
                    padding-top: 0px !important;
                    padding-bottom: 0px !important;
                    margin-top: -15px !important; /* Alignement optique avec le texte */
                }
                
                /* Customisation Input Numérique (Pill style) */
                div[data-testid="stNumberInput"] {
                    background: rgba(0, 0, 0, 0.2);
                    border-radius: 6px;
                }


                /* Nettoyage des labels */
                [data-testid='stSlider'] label, [data-testid='stNumberInput'] label {
                    display: none !important;
                }
                
                /* Header de section */
                .section-header {
                    color: #808495;
                    text-transform: uppercase;
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 1.2px;
                    margin-bottom: 15px;
                    display: block;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("<span class='section-header'>Configuration & Allocation</span>", unsafe_allow_html=True)
        
        # --- PARAMÈTRES DE SIMULATION (Design épuré) ---
        with st.container(border=True):
            c_p1, c_p2 = st.columns([2, 1])
            with c_p1:
                rebalance_freq = st.selectbox(
                    "Fréquence de rééquilibrage",
                    ["Quotidien", "Hebdomadaire", "Mensuel", "Annuel", "Aucun (Buy & Hold)"],
                    index=2, help="Fréquence à laquelle le portefeuille revient aux poids cibles."
                )
            with c_p2:
                fees_bps = st.number_input("Frais (bps)", 0, 100, 10, step=5) / 10000

        st.write("") # Spacer

        # --- BOUCLE DES ACTIFS (Look Terminal) ---
        for t in tickers:
            if t not in st.session_state.portfolio_weights:
                st.session_state.portfolio_weights[t] = round(100.0 / len(tickers), 2)
            
            current_w = float(st.session_state.portfolio_weights[t])
            logo_url = get_logo_url(t)

            # Container de ligne
            with st.container():
                # Utilisation de colonnes sans gap excessif via le CSS injecté
                r0, r1, r2 = st.columns([1.5, 3, 0.8])
                
                with r0:
                    st.markdown(f"""
                        <div class="asset-info">
                            <img src="{logo_url if logo_url else ''}" class="responsive-logo">
                            <span class="ticker-name">{t}</span>
                        </div>
                    """, unsafe_allow_html=True)

                with r1:
                    st.slider(
                        f"{t}", 0.0, 100.0, 
                        value=current_w,
                        key=f"slider_weight_{t}",
                        on_change=lambda ticker=t: st.session_state.portfolio_weights.update(
                            {ticker: st.session_state[f"slider_weight_{ticker}"]}
                        )
                    )

                with r2:
                    weights[t] = st.number_input(
                        f"", 0.0, 100.0, 
                        value=current_w,
                        key=f"num_weight_{t}",
                        step=1.0,
                        on_change=lambda ticker=t: st.session_state.portfolio_weights.update(
                            {ticker: st.session_state[f"num_weight_{ticker}"]}
                        )
                    )

    total_w = sum(weights.values())

    if total_w > 0:
        # --- 3. CALCULS DE LA SIMULATION ---
        df_rets = pd.DataFrame({t: data_dict[t]['Strat_Returns'] for t in tickers}).dropna()
        target_weights = np.array([weights[t] / total_w for t in tickers])

        n_days = len(df_rets)
        portfolio_values = np.zeros(n_days)
        current_val = 1.0 
        current_weights = target_weights.copy()
        current_val *= (1 - fees_bps) 

        for i in range(n_days):
            date = df_rets.index[i]
            daily_rets = df_rets.iloc[i].values
            
            do_rebalance = False
            if rebalance_freq == "Quotidien": do_rebalance = True
            elif rebalance_freq == "Hebdomadaire" and date.weekday() == 0: do_rebalance = True
            elif rebalance_freq == "Mensuel" and i > 0 and date.month != df_rets.index[i-1].month: do_rebalance = True
            elif rebalance_freq == "Annuel" and i > 0 and date.year != df_rets.index[i-1].year: do_rebalance = True

            if do_rebalance and rebalance_freq != "Aucun (Buy & Hold)":
                turnover = np.sum(np.abs(current_weights - target_weights))
                current_val *= (1 - (turnover * fees_bps))
                current_weights = target_weights.copy()
            
            current_val *= (1 + np.sum(current_weights * daily_rets))
            portfolio_values[i] = current_val
            
            if rebalance_freq != "Quotidien":
                drift = current_weights * (1 + daily_rets)
                current_weights = drift / np.sum(drift) if np.sum(drift) != 0 else target_weights

        # --- 4. CALCUL DES VARIABLES DE METRIQUES ---
        port_series = pd.Series(portfolio_values, index=df_rets.index)
        port_return, port_mdd = calculate_metrics(port_series)
        port_vol = port_series.pct_change().std() * np.sqrt(252)

        # Préparation du DataFrame pour le graphique cumulé
        df_plot_cum = pd.DataFrame(index=df_rets.index)
        for t in tickers:
            df_plot_cum[t] = (1 + df_rets[t]).cumprod() * 100
        df_plot_cum['Portfolio_Value'] = portfolio_values * 100

        # --- 5. CRÉATION DES FIGURES ---
        
        # Figure Pie (Roue)
        fig_pie = go.Figure(data=[go.Pie(labels=list(weights.keys()), values=list(weights.values()), hole=.5)])
        fig_pie.update_layout(template="plotly_dark", autosize=True, showlegend=False, margin=dict(t=20, b=20, l=10, r=10))

        # Figure Performance Globale
        fig_glob = go.Figure()
        for t in tickers:
            if weights[t] > 0:
                fig_glob.add_trace(go.Scatter(x=df_plot_cum.index, y=df_plot_cum[t], name=f"Contrib: {t}", line=dict(width=1, dash='dot'), opacity=0.4))
        
        fig_glob.add_trace(go.Scatter(x=df_plot_cum.index, y=df_plot_cum['Portfolio_Value'], name="MON PORTEFEUILLE", line=dict(color='gold', width=4)))
        fig_glob.update_layout(height=450, title="Performance du Panier vs Actifs (Base 100)", template="plotly_dark", hovermode="x unified")
        # --- 6. AFFICHAGE (UI) ---

        # A. On affiche la roue dans la colonne de droite (déjà définie en haut du script)
        with col_visual:
            st.plotly_chart(fig_pie, use_container_width=True, key="portfolio_pie_chart_unique")

        # B. On affiche les métriques globales sur toute la largeur
        st.write("**Statistiques Globales du Portefeuille (Net de frais)**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rendement Net", f"{port_return:.2%}")
        m2.metric("Risque (MDD)", f"{port_mdd:.2%}")
        m3.metric("Volatilité Ann.", f"{port_vol:.2%}")
        m4.metric("Sharpe Ratio", f"{(port_return/port_vol):.2f}" if port_vol > 0 else "0.00")

        # C. On affiche le grand graphique de performance
        st.divider()
        st.plotly_chart(fig_glob, use_container_width=True, key="portfolio_perf_main_unique")

        # D. Matrice et Heatmap (En colonnes 50/50)
        st.divider()
        c_mat, c_comp = st.columns([1, 1])
        
        with c_mat:
            st.write("**Matrice de Corrélation**")
            corr_matrix = df_rets.corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
                colorscale='RdBu', zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2), texttemplate="%{text}"
            ))
            fig_corr.update_layout(height=450, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(scaleanchor="y"), yaxis=dict(scaleanchor="x"))
            st.plotly_chart(fig_corr, use_container_width=True, key="portfolio_corr_matrix_unique")
            
        with c_comp:
            st.write("**Risque vs Rendement**")
            # Données pour la heatmap Risk/Return
            comparison_data = []
            for t in tickers:
                ret_t, _ = calculate_metrics(data_dict[t]['Strat_Momentum'])
                vol_t = data_dict[t]['Strat_Returns'].std() * np.sqrt(252)
                comparison_data.append({'Name': t, 'Return': ret_t * 100, 'Vol': vol_t * 100, 'Type': 'Asset'})
            comparison_data.append({'Name': 'PORTFOLIO', 'Return': port_return * 100, 'Vol': port_vol * 100, 'Type': 'Portfolio'})
            df_rr = pd.DataFrame(comparison_data)

            # --- TON CODE HEATMAP OPTIMISÉ ---
            v_max, r_max = df_rr['Vol'].max() * 1.3, df_rr['Return'].max() * 1.3
            r_min = df_rr['Return'].min() * 1.3 if df_rr['Return'].min() < 0 else -r_max * 0.2
            v_mid, r_mid = v_max / 2, (r_max + r_min) / 2
            v_s, r_s = np.linspace(0, v_max, 30), np.linspace(r_min, r_max, 30)
            z_rr = [[(r - v) for v in v_s] for r in r_s]

            fig_rr = go.Figure()
            fig_rr.add_trace(go.Heatmap(z=z_rr, x=v_s, y=r_s, colorscale=[[0, 'rgba(231, 76, 60, 0.8)'], [0.5, 'rgba(255, 251, 0, 0.4)'], [1, 'rgba(46, 204, 113, 0.8)']], showscale=False, hoverinfo='skip'))
            fig_rr.add_shape(type="line", x0=0, y0=r_mid, x1=v_max, y1=r_mid, line=dict(color="black", width=2), layer="above")
            fig_rr.add_shape(type="line", x0=v_mid, y0=r_min, x1=v_mid, y1=r_max, line=dict(color="black", width=2), layer="above")
            
            assets_rr = df_rr[df_rr['Type'] == 'Asset']
            fig_rr.add_trace(go.Scatter(x=assets_rr['Vol'], y=assets_rr['Return'], mode='markers+text', text=assets_rr['Name'], textposition="top center", marker=dict(size=10, color='white', line=dict(width=1, color='black'))))
            port_rr_pt = df_rr[df_rr['Type'] == 'Portfolio']
            fig_rr.add_trace(go.Scatter(x=port_rr_pt['Vol'], y=port_rr_pt['Return'], mode='markers+text', text=['<b>PORTFOLIO</b>'], textposition="bottom center", marker=dict(size=20, color='gold', symbol='star', line=dict(width=1, color='black'))))
            
            fig_rr.update_layout(height=450, template="plotly_white", margin=dict(t=10, b=40, l=40, r=10), showlegend=False, xaxis=dict(ticksuffix="%"), yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_rr, use_container_width=True, key="portfolio_risk_return_unique")

        # --- 7. BOUTONS DE PRESETS DU BAS ---
            st.write("**Appliquer un Preset d'Optimisation rapide :**")
            b_c1, b_c2, b_c3, b_c4, b_c5 = st.columns(5)
            b_c1.button("⚖️ EQ", on_click=apply_preset_callback, args=(presets['eq'],), key="preset_eq_bot")
            b_c2.button("🛡️ RP", on_click=apply_preset_callback, args=(presets['rp'],), key="preset_rp_bot")
            b_c3.button("📉 MD", on_click=apply_preset_callback, args=(presets['md'],), key="preset_md_bot")
            b_c4.button("🚀 TP", on_click=apply_preset_callback, args=(presets['tp'],), key="preset_tp_bot")
            b_c5.button("💎 SR", on_click=apply_preset_callback, args=(presets['sr'],), key="preset_sr_bot")

    else:
        st.warning("Allouez du capital pour activer la simulation.")