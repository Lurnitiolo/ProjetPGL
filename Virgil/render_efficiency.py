import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from .data_loader import get_logo_url
from .strategies import calculate_metrics

def min_max_scale(series):
    if series.max() == series.min(): return series * 0
    return (series - series.min()) / (series.max() - series.min())

@st.fragment
def render_portfolio_simulation(tickers, data_dict, cap_init):
    # --- 1. BOUTONS DE PRESETS ---
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    
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

    # Logique des boutons
    if c1.button("⚖️ Equal Weight"):
        val = round(100.0 / len(tickers), 2)
        for t in tickers:
            st.session_state[f"w_{t}"] = val
            st.session_state[f"slide_{t}"] = val
        st.rerun(scope="fragment")

    if c2.button("🛡️ Risk Parity"):
        total_inv_vol = sum(1/s['vol'] for s in stats.values())
        for t in tickers:
            val = round(((1/stats[t]['vol']) / total_inv_vol) * 100, 2)
            st.session_state[f"w_{t}"] = val
            st.session_state[f"slide_{t}"] = val
        st.rerun(scope="fragment")

    if c3.button("📉 Min Drawdown"):
        total_inv_mdd = sum(1/s['mdd'] for s in stats.values())
        for t in tickers:
            val = round(((1/stats[t]['mdd']) / total_inv_mdd) * 100, 2)
            st.session_state[f"w_{t}"] = val
            st.session_state[f"slide_{t}"] = val
        st.rerun(scope="fragment")

    if c4.button("🚀 Top Perf"):
        pos_rets = {t: max(0, stats[t]['ret']) for t in tickers}
        total_pos = sum(pos_rets.values())
        for t in tickers:
            val = round((pos_rets[t] / total_pos) * 100, 2) if total_pos > 0 else round(100/len(tickers), 2)
            st.session_state[f"w_{t}"] = val
            st.session_state[f"slide_{t}"] = val
        st.rerun(scope="fragment")

    if c5.button("💎 Sharpe Ratio"):
        pos_sharpe = {t: max(0, stats[t]['sharpe']) for t in tickers}
        total_sharpe = sum(pos_sharpe.values())
        for t in tickers:
            val = round((pos_sharpe[t] / total_sharpe) * 100, 2) if total_sharpe > 0 else round(100/len(tickers), 2)
            st.session_state[f"w_{t}"] = val
            st.session_state[f"slide_{t}"] = val
        st.rerun(scope="fragment")

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
        df_global = pd.DataFrame({t: data_dict[t]['Strat_Momentum'] for t in tickers}).dropna()
        w_arr = np.array([weights[t] / total_w for t in tickers])
        df_global['Portfolio_Value'] = df_global.dot(w_arr)
        
        port_return, port_mdd = calculate_metrics(df_global['Portfolio_Value'])
        df_rets = pd.DataFrame({t: data_dict[t]['Strat_Returns'] for t in tickers}).dropna()
        port_vol = df_rets.dot(w_arr).std() * np.sqrt(252)

        with col_visual:
            fig_pie = go.Figure(data=[go.Pie(labels=list(weights.keys()), values=list(weights.values()), hole=.5)])
            fig_pie.update_layout(template="plotly_dark", height=350, showlegend=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True, key="portfolio_pie_chart")

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
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
                colorscale='RdBu', zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2), texttemplate="%{text}"
            ))
            fig_corr.update_layout(
                height=450, template="plotly_dark",
                xaxis=dict(constrain="domain", fixedrange=True, showgrid=False, zeroline=False),
                yaxis=dict(scaleanchor="x", scaleratio=1, constrain="domain", fixedrange=True, showgrid=False, zeroline=False),
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_corr, use_container_width=True, key="portfolio_heatmap_square", config={'displayModeBar': False})

        with c_comp:
            st.write("**Comparaison Risque vs Rendement**")
            comparison_data = []
            for t in tickers:
                ret_t, _ = calculate_metrics(data_dict[t]['Strat_Momentum'])
                vol_t = data_dict[t]['Strat_Returns'].std() * np.sqrt(252)
                comparison_data.append({'Name': t, 'Return': ret_t * 100, 'Vol': vol_t * 100, 'Type': 'Asset'})
            
            comparison_data.append({'Name': 'PORTFOLIO', 'Return': port_return * 100, 'Vol': port_vol * 100, 'Type': 'Portfolio'})
            df_plot = pd.DataFrame(comparison_data)

            v_max = df_plot['Vol'].max() * 1.3
            r_max = df_plot['Return'].max() * 1.3
            r_min = df_plot['Return'].min() * 1.3 if df_plot['Return'].min() < 0 else -r_max * 0.2
            v_mid, r_mid = v_max / 2, (r_max + r_min) / 2

            
            fig_risk_ret = go.Figure()

            v_space = np.linspace(0, v_max, 25)
            r_space = np.linspace(r_min, r_max, 25)
            z = [[(r - v) for v in v_space] for r in r_space]

            fig_risk_ret.add_trace(go.Heatmap(
                z=z, x=v_space, y=r_space,
                colorscale=[[0, 'rgba(231, 76, 60, 1)'], [0.5, 'rgba(255, 251, 0, 0.6)'], [1, 'rgba(46, 204, 113, 1)']],
                showscale=False, hoverinfo='skip'
            ))

            
            # --- LA CROIX CENTRALE (+) ---
            fig_risk_ret.add_shape(type="line", x0=0, y0=r_mid, x1=v_max, y1=r_mid, line=dict(color="black", width=2), layer="below")
            fig_risk_ret.add_shape(type="line", x0=v_mid, y0=r_min, x1=v_mid, y1=r_max, line=dict(color="black", width=2), layer="below")
            

            # --- FLÈCHES ET LÉGENDES ---
            fig_risk_ret.add_annotation(xref="paper", yref="paper", x=-0.08, y=0.5, text="<b>RENDEMENT →</b>", showarrow=False, textangle=-90, font=dict(color="black"))
            fig_risk_ret.add_annotation(xref="paper", yref="paper", x=0.5, y=-0.12, text="<b>RISQUE (Volatilité) →</b>", showarrow=False, font=dict(color="black"))

            # Points
            assets = df_plot[df_plot['Type'] == 'Asset']
            fig_risk_ret.add_trace(go.Scatter(x=assets['Vol'], y=assets['Return'], mode='markers+text', text=assets['Name'], textposition="top center",
                                              marker=dict(size=12, color='white', line=dict(width=1.5, color='black'))))

            port_pt = df_plot[df_plot['Type'] == 'Portfolio']
            fig_risk_ret.add_trace(go.Scatter(x=port_pt['Vol'], y=port_pt['Return'], mode='markers+text', text=['PORTFOLIO'], textposition="bottom center",
                                              marker=dict(size=24, color='gold', symbol='star', line=dict(width=2, color='black'))))

            # Flèches aux bouts de la croix
            fig_risk_ret.add_annotation(x=v_mid, y=r_max, ax=0, ay=25, xref="x", yref="y", showarrow=True, arrowhead=2, arrowcolor="black", arrowwidth=2)
            fig_risk_ret.add_annotation(x=v_max, y=r_mid, ax=-25, ay=0, xref="x", yref="y", showarrow=True, arrowhead=2, arrowcolor="black", arrowwidth=2)

            fig_risk_ret.update_layout(
                height=500, template="plotly_white",
                xaxis=dict(range=[0, v_max], fixedrange=True, showgrid=False, showticklabels=True, ticksuffix="%", zeroline=False, color="black"),
                yaxis=dict(range=[r_min, r_max], fixedrange=True, showgrid=False, showticklabels=True, ticksuffix="%", zeroline=False, color="black"),
                margin=dict(t=30, b=60, l=70, r=40), showlegend=False
            )

            st.plotly_chart(fig_risk_ret, use_container_width=True, key="risk_ret_static", config={'displayModeBar': False})