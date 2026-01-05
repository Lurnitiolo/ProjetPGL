import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from .data_loader import load_stock_data
from .strategies import apply_strategies, calculate_metrics

def min_max_scale(series):
    if series.max() == series.min(): return series * 0
    return (series - series.min()) / (series.max() - series.min())

# ... (imports et min_max_scale identiques)

# ... (imports identiques)

def quant_b_ui():
    st.header("Analyse de Portefeuille & Diversification (Quant B)")

    # INITIALISATION
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = {}

    with st.sidebar:
        st.title("⚙️ Paramètres")
        # On utilise une clé unique pour forcer le rafraîchissement
        tickers_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC-USD", "EURUSD=X", "GC=F"]
        selected_tickers = st.multiselect("Actifs", options=tickers_list, default=["AAPL", "MSFT", "BTC-USD"])
        
        capital_init = st.number_input("Capital Initial ($)", value=1000)
        ma_window = st.number_input("Fenêtre MA", 5, 100, 20)
        stop_loss = st.slider("Stop Loss (%)", 0, 25, 10) / 100
        take_profit = st.slider("Take Profit (%)", 0, 100, 30) / 100
        
        show_price = st.checkbox("Afficher le Prix", value=True)
        show_ma = st.checkbox("Afficher la MA", value=True)
        
        if st.button("🚀 Lancer l'analyse", use_container_width=True):
            # Vider la mémoire pour forcer le recalcul avec les nouveaux paramètres
            st.session_state.portfolio_data = {}
            with st.spinner("Analyse précise..."):
                results = {t: apply_strategies(load_stock_data(t), ma_window, stop_loss, take_profit) for t in selected_tickers}
                st.session_state.portfolio_data = results
                st.session_state.tickers_analyzed = selected_tickers
                st.session_state.ma_used = ma_window

    if st.session_state.portfolio_data:
        data_dict = st.session_state.portfolio_data
        tickers = st.session_state.tickers_analyzed
        tabs = st.tabs(tickers + ["📊 Portefeuille Global"])

        for i, ticker in enumerate(tickers):
            with tabs[i]:
                df_t = data_dict[ticker]
                
                # --- ÉCHELLE ---
                s_min, s_max = df_t['Strat_Momentum'].min(), df_t['Strat_Momentum'].max()
                def scale_strat(val): return ((val - s_min) / (s_max - s_min)) * 100 if s_max > s_min else 50
                y_strat = scale_strat(df_t['Strat_Momentum'])

                fig = go.Figure()

                # --- COLORATION PRÉCISE (BOUGIE PAR BOUGIE) ---
                # On parcourt les indices réels pour ne colorer que là où l'argent bouge
                # Cela règle le problème des écarts inégaux (week-ends, etc.)
                active_indices = np.where(df_t['Strat_Returns'] != 0)[0]
                
                for idx in active_indices:
                    # On colorie l'intervalle entre la bougie précédente et l'actuelle
                    # C'est là que le profit/perte s'est réalisé
                    start = df_t.index[idx-1]
                    end = df_t.index[idx]
                    
                    # Vert si gain sur la bougie, Rouge si perte
                    color = "rgba(0, 255, 0, 0.2)" if df_t['Strat_Returns'].iloc[idx] > 0 else "rgba(255, 0, 0, 0.2)"
                    
                    fig.add_vrect(x0=start, x1=end, fillcolor=color, layer="below", line_width=0)

                # --- TRACÉS ---
                if show_price: fig.add_trace(go.Scatter(x=df_t.index, y=min_max_scale(df_t['Close'])*100, name="Prix", line=dict(color='gray'), opacity=0.3))
                if show_ma: fig.add_trace(go.Scatter(x=df_t.index, y=min_max_scale(df_t['MA'])*100, name=f"MA {st.session_state.ma_used}", line=dict(dash='dot', color='orange')))
                
                # Connectgaps=False force la ligne à être plate si une donnée manque
                fig.add_trace(go.Scatter(x=df_t.index, y=y_strat, name="Stratégie", line=dict(color='blue', width=2.5)))

                fig.update_layout(height=450, template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                # Métriques
                ret, mdd = calculate_metrics(df_t['Strat_Momentum'])
                c1, c2, c3 = st.columns(3)
                c1.metric("Performance", f"{ret:.2%}")
                c2.metric("Max Drawdown", f"{mdd:.2%}")
                c3.metric("Valeur Finale", f"{capital_init*(1+ret):.2f} $")
                
                
                
                
        with tabs[-1]:
            st.subheader("Simulateur de Portefeuille Dynamique")
            st.info("Ajustez les curseurs ci-dessous pour voir l'impact sur le portefeuille global.")

            weights = {}
            weight_cols = st.columns(len(tickers))
            for j, t in enumerate(tickers):
                weights[t] = weight_cols[j].slider(f"{t} %", 0, 100, 33, key=f"weight_{t}")

            total_w = sum(weights.values())

            if total_w > 0:
                df_global = pd.DataFrame({t: data_dict[t]['Strat_Momentum'] for t in tickers}).dropna()
                df_global['Portfolio_Value'] = 0
                for t in tickers:
                    norm_weight = weights[t] / total_w
                    df_global['Portfolio_Value'] += df_global[t] * norm_weight

                p_ret, p_mdd = calculate_metrics(df_global['Portfolio_Value'])
                m1, m2 = st.columns(2)
                m1.metric("Rendement Global", f"{p_ret:.2%}")
                m2.metric("Risque Global (MDD)", f"{p_mdd:.2%}")

                fig_glob = go.Figure()
                fig_glob.add_trace(go.Scatter(x=df_global.index, y=df_global['Portfolio_Value'], 
                                              name="MON PORTEFEUILLE", line=dict(color='gold', width=4)))
                
                for t in tickers:
                    fig_glob.add_trace(go.Scatter(x=df_global.index, y=df_global[t], 
                                                  name=f"Contrib: {t}", line=dict(width=1), opacity=0.3))

                fig_glob.update_layout(height=500, title="Performance du Panier vs Actifs", template="plotly_dark")
                st.plotly_chart(fig_glob, use_container_width=True)
            else:
                st.warning("Veuillez allouer un pourcentage à au moins un actif.")
    else:
        st.write("---")
        st.info("👈 Sélectionnez vos actifs et paramètres dans la barre latérale et cliquez sur 'Lancer l'analyse'.")