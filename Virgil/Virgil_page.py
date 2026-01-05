import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from .data_loader import load_stock_data, get_logo_url
from .strategies import apply_strategies, calculate_metrics

def min_max_scale(series):
    if series.max() == series.min(): return series * 0
    return (series - series.min()) / (series.max() - series.min())

def quant_b_ui():
    st.header("Analyse de Portefeuille & Diversification (Quant B)")

    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = {}

    with st.sidebar:
        st.title("⚙️ Paramètres")
        tickers_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC-USD", "EURUSD=X", "GC=F"]
        selected_tickers = st.multiselect("Actifs", options=tickers_list, default=["AAPL", "MSFT", "BTC-USD"])
        
        cap_init = st.number_input("Capital Initial ($)", value=1000)
        ma_window = st.number_input("Fenêtre MA", 5, 100, 20)
        sl_val = st.slider("Stop Loss (%)", 0, 25, 10) / 100
        tp_val = st.slider("Take Profit (%)", 0, 100, 30) / 100
        
        if st.button("🚀 Lancer l'analyse", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                results = {t: apply_strategies(load_stock_data(t), ma_window, sl_val, tp_val) for t in selected_tickers}
                st.session_state.portfolio_data = results
                st.session_state.tickers_analyzed = selected_tickers
                st.session_state.ma_used = ma_window

    if st.session_state.portfolio_data:
        data_dict = st.session_state.portfolio_data
        tickers = st.session_state.tickers_analyzed
        tabs = st.tabs(tickers + ["📊 Portefeuille Global"])

        # --- ONGLETS INDIVIDUELS ---
        for i, ticker in enumerate(tickers):
            with tabs[i]:
                l_col, r_col = st.columns([1, 10])
                logo = get_logo_url(ticker)
                if logo: l_col.image(logo, width=50)
                r_col.subheader(f"Analyse {ticker}")
                
                df_t = data_dict[ticker]
                y_strat = ((df_t['Strat_Momentum'] - df_t['Strat_Momentum'].min()) / 
                           (df_t['Strat_Momentum'].max() - df_t['Strat_Momentum'].min())) * 100

                fig = go.Figure()
                # Coloration des zones
                df_t['ret_sign'] = np.sign(df_t['Strat_Returns'])
                df_t['grp'] = (df_t['ret_sign'] != df_t['ret_sign'].shift()).cumsum()
                for _, period in df_t[df_t['ret_sign'] != 0].groupby('grp'):
                    fig.add_vrect(x0=df_t.index[max(0, df_t.index.get_loc(period.index[0])-1)], 
                                  x1=period.index[-1], line_width=0, layer="below",
                                  fillcolor="rgba(46, 204, 113, 0.15)" if period['ret_sign'].iloc[0] > 0 else "rgba(231, 76, 60, 0.2)")

                fig.add_trace(go.Scatter(x=df_t.index, y=y_strat, name="Stratégie", line=dict(color='blue', width=2.5)))
                fig.update_layout(height=450, template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

        # --- ONGLET GLOBAL ---
        with tabs[-1]:
            st.subheader("🛠️ Optimisation de l'Allocation")
            c1, c2, _ = st.columns([1, 1, 2])
            if c1.button("⚖️ Equal Weight"):
                for t in tickers: st.session_state[f"w_{t}"] = round(100.0 / len(tickers), 2)
            if c2.button("🧹 Tout à Zéro"):
                for t in tickers: st.session_state[f"w_{t}"] = 0.0

            st.divider()
            col_in, col_pie = st.columns([1.3, 1])
            weights = {}
            with col_in:
                for t in tickers:
                    if f"w_{t}" not in st.session_state: st.session_state[f"w_{t}"] = round(100.0/len(tickers), 2)
                    r0, r1, r2 = st.columns([0.6, 3, 1.2])
                    logo = get_logo_url(t)
                    if logo: r0.image(logo, width=35)
                    s_val = r1.slider(f"{t}", 0.0, 100.0, key=f"slide_{t}", value=st.session_state[f"w_{t}"], step=0.1)
                    weights[t] = r2.number_input("Precise", label_visibility="collapsed", min_value=0.0, max_value=100.0, key=f"w_{t}", format="%.2f")

            total_w = sum(weights.values())
            with col_pie:
                if total_w > 0:
                    fig_pie = go.Figure(data=[go.Pie(labels=list(weights.keys()), values=list(weights.values()), hole=.5,
                        marker=dict(colors=['#00f2fe', '#4facfe', '#7f00ff', '#e100ff', '#ff0844', '#f77062'], line=dict(color='white', width=2)),
                        textinfo='percent',
                        hovertemplate="<b>%{label}</b><br>📊 Part : %{percent}<br>💰 Poids : %{value:.2f}%<extra></extra>")])
                    fig_pie.update_layout(template="plotly_dark", height=400, showlegend=True,
                        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
                        hoverlabel=dict(bgcolor="white", font_size=35, font_family="Arial", font_color="black", bordercolor="black"))
                    st.plotly_chart(fig_pie, use_container_width=True)

            # --- 3. CALCULS ET GRAPHIQUE DE PERFORMANCE ---
            if total_w > 0:
                st.divider()
                # On prépare le DataFrame avec tous les actifs
                df_global = pd.DataFrame({t: data_dict[t]['Strat_Momentum'] for t in tickers}).dropna()
                
                # CALCUL DE LA VALEUR DU PORTEFEUILLE (Nom exact : Portfolio_Value)
                df_global['Portfolio_Value'] = sum(df_global[t] * (weights[t] / total_w) for t in tickers)

                # Calcul des métriques
                p_ret, p_mdd = calculate_metrics(df_global['Portfolio_Value'])
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Rendement Global", f"{p_ret:.2%}")
                m2.metric("Risque (MDD)", f"{p_mdd:.2%}")
                m3.metric("Total Alloué", f"{total_w:.2f}%", 
                          delta=f"{total_w-100:.2f}%" if abs(total_w-100) > 0.1 else None, 
                          delta_color="inverse")

                # CONSTRUCTION DU GRAPHIQUE
                fig_glob = go.Figure()

                # 1. Ajout des courbes individuelles (en pointillé léger)
                for t in tickers:
                    if weights[t] > 0: # On n'affiche que ceux qui contribuent
                        fig_glob.add_trace(go.Scatter(
                            x=df_global.index, 
                            y=df_global[t], 
                            name=f"Contrib: {t}", 
                            line=dict(width=2, dash='dot'), 
                            opacity=0.7
                        ))

                # 2. Ajout de la courbe Or du portefeuille (au premier plan)
                fig_glob.add_trace(go.Scatter(
                    x=df_global.index, 
                    y=df_global['Portfolio_Value'], 
                    name="MON PORTEFEUILLE", 
                    line=dict(color='gold', width=4)
                ))

                fig_glob.update_layout(
                    height=550, 
                    title="Performance du Panier Dynamique vs Actifs", 
                    template="plotly_dark", 
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_glob, use_container_width=True)
    else:
        st.write("---")
        st.info("👈 Sélectionnez vos actifs et paramètres dans la barre latérale et cliquez sur 'Lancer l'analyse'.")