import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from .data_loader import load_stock_data
from .strategies import apply_strategies, calculate_metrics

def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min())

def quant_b_ui():
    st.header("Analyse de Portefeuille & Diversification (Quant B)")    

    # --- 1. INITIALISATION DE LA MÉMOIRE ---
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = {} 
    if 'tickers_analyzed' not in st.session_state:
        st.session_state.tickers_analyzed = []
    if 'ma_used' not in st.session_state:
        st.session_state.ma_used = 20

    # --- 2. BARRE LATÉRALE ENRICHIE ---
    with st.sidebar:
        st.title("⚙️ Paramètres")
        
        st.subheader("Actifs")
        tickers_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC-USD", "EURUSD=X", "GC=F"]
        selected_tickers = st.multiselect(
            "Sélectionnez les actifs", 
            options=tickers_list, 
            default=["AAPL", "MSFT", "BTC-USD"]
        )

        st.divider()
        
        st.subheader("Gestion du Capital")
        capital_init = st.number_input("Capital Initial ($)", value=1000, step=100)
        
        st.divider()

        st.subheader("Stratégie MA")
        ma_window = st.number_input("Fenêtre Moyenne Mobile", 5, 100, 20)
        
        st.subheader("Sécurité & Risque")
        stop_loss = st.slider("Stop Loss (%)", 0, 25, 10)
        take_profit = st.slider("Take Profit (%)", 0, 100, 30)

        st.divider()

        st.subheader("Options d'Affichage")
        show_price = st.checkbox("Afficher le Prix", value=True)
        show_ma = st.checkbox("Afficher la MA", value=True)
        
        st.divider()
        
        # Bouton pour déclencher le téléchargement
        launch_btn = st.button("🚀 Lancer l'analyse", use_container_width=True)
        
        if launch_btn:
            with st.spinner("Téléchargement et calculs en cours..."):
                results = {}
                for ticker in selected_tickers:
                    df = load_stock_data(ticker)
                    if df is not None and not df.empty:
                        # On stocke le DF transformé par la stratégie
                        results[ticker] = apply_strategies(df, ma_window=ma_window)
                
                # Sauvegarde en mémoire
                st.session_state.portfolio_data = results
                st.session_state.tickers_analyzed = selected_tickers
                st.session_state.ma_used = ma_window

    # --- 3. AFFICHAGE DES ONGLETS ---
    if st.session_state.portfolio_data:
        tickers = st.session_state.tickers_analyzed
        data_dict = st.session_state.portfolio_data
        
        tab_list = tickers + ["📊 Portefeuille Global"]
        tabs = st.tabs(tab_list)

        for i, ticker in enumerate(tickers):
            with tabs[i]:
                st.subheader(f"Analyse {ticker}")
                df_ticker = data_dict[ticker]

                # Préparation des données pour le survol (Base 100)
                hover_price = (df_ticker['Close'] / df_ticker['Close'].iloc[0]) * 100
                hover_strat = (df_ticker['Strat_Momentum'] / df_ticker['Strat_Momentum'].iloc[0]) * 100
                hover_ma = (df_ticker['MA'] / df_ticker['Close'].iloc[0]) * 100

                # Scaling pour le graphique
                y_price = min_max_scale(df_ticker['Close']) * 100
                y_strat = min_max_scale(df_ticker['Strat_Momentum']) * 100
                y_ma = min_max_scale(df_ticker['MA']) * 100

                # Métriques
                ret, mdd = calculate_metrics(df_ticker['Strat_Momentum'])
                valeur_finale = capital_init * (df_ticker['Strat_Momentum'].iloc[-1] / df_ticker['Strat_Momentum'].iloc[0])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Performance", f"{ret:.2%}")
                c2.metric("Max Drawdown", f"{mdd:.2%}")
                c3.metric("Valeur Finale", f"{valeur_finale:.2f} $")

                # Création du graphique
                fig = go.Figure()

                if show_price:
                    fig.add_trace(go.Scatter(
                        x=df_ticker.index, y=y_price,
                        name="Prix",
                        line=dict(color='gray', width=1),
                        opacity=0.5,
                        customdata=hover_price,
                        hovertemplate="Prix (Indice): %{customdata:.2f}<extra></extra>"
                    ))

                if show_ma:
                    fig.add_trace(go.Scatter(
                        x=df_ticker.index, y=y_ma,
                        name=f"MA {st.session_state.ma_used}",
                        line=dict(dash='dot', color='orange'),
                        customdata=hover_ma,
                        hovertemplate="MA (Indice): %{customdata:.2f}<extra></extra>"
                    ))

                fig.add_trace(go.Scatter(
                    x=df_ticker.index, y=y_strat,
                    name="Stratégie",
                    line=dict(color='blue', width=2),
                    customdata=hover_strat,
                    hovertemplate="<b>Stratégie (Indice): %{customdata:.2f}</b><extra></extra>"
                ))

                fig.update_layout(
                    height=400,
                    hovermode="x unified",
                    yaxis=dict(title="Tendance Indexée", showticklabels=True),
                    xaxis=dict(title="Date"),
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)

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