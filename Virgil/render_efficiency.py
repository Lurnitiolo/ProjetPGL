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
    """Met à jour le session_state AVANT le rendu des widgets"""
    for t, val in new_weights.items():
        st.session_state[f"w_{t}"] = val
        st.session_state[f"slide_{t}"] = val

@st.fragment
def render_portfolio_simulation(tickers, data_dict, cap_init):
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
    
    # --- 2. RÉPARTITION DU CAPITAL (UI CORRIGÉE) ---
    col_inputs, col_visual = st.columns([1.3, 1])
    weights = {}
    
    with col_inputs:
        st.write("**Répartition du capital**")
        
        # Injection CSS pour remonter les sliders et ajuster l'alignement
        st.markdown("""
            <style>
                div[data-testid="stSlider"] { margin-top: 11px; }
                div[data-testid="stSlider"] label { display: none; }
            </style>
        """, unsafe_allow_html=True)

        for t in tickers:
            if f"w_{t}" not in st.session_state: st.session_state[f"w_{t}"] = round(100.0/len(tickers), 2)
            if f"slide_{t}" not in st.session_state: st.session_state[f"slide_{t}"] = st.session_state[f"w_{t}"]

            def sync_to_num(ticker=t): st.session_state[f"w_{ticker}"] = st.session_state[f"slide_{ticker}"]
            def sync_to_slide(ticker=t): st.session_state[f"slide_{ticker}"] = st.session_state[f"w_{ticker}"]

            r0, r1, r2 = st.columns([0.8, 3, 1.2])
            
            with r0:
                logo_url = get_logo_url(t)
                st.markdown(f"""
                    <div style="display: flex; align-items: center; height: 100%;">
                        <img src="{logo_url if logo_url else ''}" style="width: 60px; height: auto; margin-right: 10px;">
                        <b style="font-size: 14px;">{t}</b>
                    </div>
                """, unsafe_allow_html=True)
            
            with r1:
                st.slider(f"Slider {t}", 0.0, 100.0, key=f"slide_{t}", on_change=sync_to_num, step=0.1)

            with r2:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                weights[t] = st.number_input(f"v_{t}", label_visibility="collapsed", key=f"w_{t}", 
                                             on_change=sync_to_slide, step=0.01, min_value=0.0)

    total_w = sum(weights.values())

    if total_w > 0:
        # --- 3. CALCULS ---
        # --- NOUVEAU BLOC DANS render_efficiency.py ---

# 1. Sélecteur de fréquence (juste avant les calculs)
        rebalance_freq = st.selectbox(
            "Fréquence de rééquilibrage du portefeuille",
            ["Quotidien", "Hebdomadaire", "Mensuel", "Annuel", "Aucun (Buy & Hold)"],
            index=2
        )

        # 2. Logique de calcul itérative
        df_rets = pd.DataFrame({t: data_dict[t]['Strat_Returns'] for t in tickers}).dropna()
        target_weights = np.array([weights[t] / total_w for t in tickers])

        n_days = len(df_rets)
        portfolio_values = np.zeros(n_days)
        current_value = 1.0 
        current_weights = target_weights.copy()

        # Identification des dates de rééquilibrage
        rebalance_dates = []
        if rebalance_freq == "Hebdomadaire": rebalance_dates = df_rets.index[df_rets.index.weekday == 0]
        elif rebalance_freq == "Mensuel": rebalance_dates = df_rets.index[df_rets.index.is_month_start]
        elif rebalance_freq == "Annuel": rebalance_dates = df_rets.index[df_rets.index.is_year_start]

        for i in range(n_days):
            date = df_rets.index[i]
            daily_rets = df_rets.iloc[i].values
            
            # Appliquer le rééquilibrage si nécessaire
            if rebalance_freq not in ["Quotidien", "Aucun (Buy & Hold)"] and date in rebalance_dates:
                current_weights = target_weights.copy()
            
            # Calcul de la valeur du jour
            day_return = np.sum(current_weights * daily_rets)
            current_value *= (1 + day_return)
            portfolio_values[i] = current_value
            
            # Mise à jour des poids pour le lendemain (dérive du marché)
            if rebalance_freq != "Quotidien":
                asset_values = current_weights * (1 + daily_rets)
                # On évite la division par zéro si tout s'effondre
                sum_vals = np.sum(asset_values)
                current_weights = asset_values / sum_vals if sum_vals != 0 else current_weights
            else:
                current_weights = target_weights.copy()

        # Reconstruction de df_global pour la suite du code
        df_global = pd.DataFrame(index=df_rets.index)
        for t in tickers:
            # On remet en base 1 pour l'affichage comparatif
            df_global[t] = (1 + df_rets[t]).cumprod()

        df_global['Portfolio_Value'] = portfolio_values

        # --- 4. GRAPHIQUE PERFORMANCE ---
        st.divider()
        fig_glob = go.Figure()
        for t in tickers:
            if weights[t] > 0:
                fig_glob.add_trace(go.Scatter(x=df_global.index, y=df_global[t], name=f"Contrib: {t}", line=dict(width=3, dash='dot'), opacity=0.5))
        
        fig_glob.add_trace(go.Scatter(x=df_global.index, y=df_global['Portfolio_Value'], name="MON PORTEFEUILLE", line=dict(color='gold', width=4)))
        fig_glob.update_layout(height=450, title="Performance du Panier vs Actifs", template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig_glob, use_container_width=True, key="portfolio_perf_main")

        # Métriques colonnes
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rendement", f"{port_return:.2%}")
        m2.metric("Risque (MDD)", f"{port_mdd:.2%}")
        m3.metric("Volatilité Ann.", f"{port_vol:.2%}")
        m4.metric("Sharpe Ratio", f"{(port_return/port_vol):.2f}" if port_vol > 0 else "0.00")

        st.divider()
        c_mat, c_comp = st.columns([1, 1])

        with c_mat:
            st.write("**Matrice de Corrélation (Carrée)**")
            corr_matrix = df_rets.corr()
            fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index, colorscale='RdBu', zmin=-1, zmax=1, text=np.round(corr_matrix.values, 2), texttemplate="%{text}"))
            fig_corr.update_layout(height=450, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(scaleanchor="y"), yaxis=dict(scaleanchor="x"))
            st.plotly_chart(fig_corr, use_container_width=True, key="portfolio_heatmap_square")

        with c_comp:
            st.write("**Comparaison Risque vs Rendement**")
            comparison_data = []
            for t in tickers:
                ret_t, _ = calculate_metrics(data_dict[t]['Strat_Momentum'])
                vol_t = data_dict[t]['Strat_Returns'].std() * np.sqrt(252)
                comparison_data.append({'Name': t, 'Return': ret_t * 100, 'Vol': vol_t * 100, 'Type': 'Asset'})
            comparison_data.append({'Name': 'PORTFOLIO', 'Return': port_return * 100, 'Vol': port_vol * 100, 'Type': 'Portfolio'})
            df_plot = pd.DataFrame(comparison_data)

            # --- RESTAURATION HEATMAP RISQUE/RENDEMENT ORIGINALE ---
            v_max, r_max = df_plot['Vol'].max() * 1.3, df_plot['Return'].max() * 1.3
            r_min = df_plot['Return'].min() * 1.3 if df_plot['Return'].min() < 0 else -r_max * 0.2
            v_mid, r_mid = v_max / 2, (r_max + r_min) / 2

            fig_risk_ret = go.Figure()
            v_space, r_space = np.linspace(0, v_max, 25), np.linspace(r_min, r_max, 25)
            z = [[(r - v) for v in v_space] for r in r_space]

            fig_risk_ret.add_trace(go.Heatmap(
                z=z, x=v_space, y=r_space,
                colorscale=[[0, 'rgba(231, 76, 60, 1)'], [0.5, 'rgba(255, 251, 0, 0.6)'], [1, 'rgba(46, 204, 113, 1)']],
                showscale=False, hoverinfo='skip'
            ))

            fig_risk_ret.add_shape(type="line", x0=0, y0=r_mid, x1=v_max, y1=r_mid, line=dict(color="black", width=2), layer="below")
            fig_risk_ret.add_shape(type="line", x0=v_mid, y0=r_min, x1=v_mid, y1=r_max, line=dict(color="black", width=2), layer="below")

            fig_risk_ret.add_annotation(xref="paper", yref="paper", x=-0.08, y=0.5, text="<b>RENDEMENT →</b>", showarrow=False, textangle=-90, font=dict(color="black"))
            fig_risk_ret.add_annotation(xref="paper", yref="paper", x=0.5, y=-0.12, text="<b>RISQUE (Volatilité) →</b>", showarrow=False, font=dict(color="black"))

            # Points Actifs (Blancs)
            assets = df_plot[df_plot['Type'] == 'Asset']
            fig_risk_ret.add_trace(go.Scatter(x=assets['Vol'], y=assets['Return'], mode='markers+text', text=assets['Name'], textposition="top center",
                                              marker=dict(size=12, color='white', line=dict(width=1.5, color='black'))))

            # Point Portfolio (Étoile Or)
            port_pt = df_plot[df_plot['Type'] == 'Portfolio']
            fig_risk_ret.add_trace(go.Scatter(x=port_pt['Vol'], y=port_pt['Return'], mode='markers+text', text=['PORTFOLIO'], textposition="bottom center",
                                              marker=dict(size=24, color='gold', symbol='star', line=dict(width=2, color='black'))))

            fig_risk_ret.add_annotation(x=v_mid, y=r_max, ax=0, ay=25, xref="x", yref="y", showarrow=True, arrowhead=2, arrowcolor="black", arrowwidth=2)
            fig_risk_ret.add_annotation(x=v_max, y=r_mid, ax=-25, ay=0, xref="x", yref="y", showarrow=True, arrowhead=2, arrowcolor="black", arrowwidth=2)

            fig_risk_ret.update_layout(
                height=500, template="plotly_white",
                xaxis=dict(range=[0, v_max], showgrid=False, ticksuffix="%", color="black"),
                yaxis=dict(range=[r_min, r_max], showgrid=False, ticksuffix="%", color="black"),
                margin=dict(t=30, b=60, l=70, r=40), showlegend=False
            )
            st.plotly_chart(fig_risk_ret, use_container_width=True, key="risk_ret_staticc", config={'displayModeBar': False})

            st.write("**Presets rapides (Bas) :**")
            b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1, 1])
            b1.button("⚖️ Equal Weight", on_click=apply_preset_callback, args=(presets['eq'],), key="btn_eq_bot")
            b2.button("🛡️ Risk Parity", on_click=apply_preset_callback, args=(presets['rp'],), key="btn_rp_bot")
            b3.button("📉 Min Drawdown", on_click=apply_preset_callback, args=(presets['md'],), key="btn_md_bot")
            b4.button("🚀 Top Perf", on_click=apply_preset_callback, args=(presets['tp'],), key="btn_tp_bot")
            b5.button("💎 Sharpe Ratio", on_click=apply_preset_callback, args=(presets['sr'],), key="btn_sr_bot")
