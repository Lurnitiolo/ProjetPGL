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
            keys_to_reset = [k for k in st.session_state.keys() if k.startswith('w_') or k.startswith('slide_')]
            for k in keys_to_reset:
                del st.session_state[k]
            
            st.session_state.portfolio_data = {}
            
            with st.spinner("Analyse précise..."):
                results = {t: apply_strategies(load_stock_data(t), ma_window, sl_val, tp_val) for t in selected_tickers}
                st.session_state.portfolio_data = results
                st.session_state.tickers_analyzed = selected_tickers
                st.session_state.ma_used = ma_window
            st.rerun()

    if st.session_state.portfolio_data:
        data_dict = st.session_state.portfolio_data
        tickers = st.session_state.tickers_analyzed
        tabs = st.tabs(tickers + ["📊 Portefeuille Global"])


        for i, ticker in enumerate(tickers):
            with tabs[i]:
                l_col, r_col = st.columns([1, 10])
                with l_col:
                    logo = get_logo_url(ticker)
                    if logo: st.image(logo, width=50)
                with r_col:
                    st.subheader(f"Analyse {ticker}")
                
                df_t = data_dict[ticker].copy()
                
                if 'Exit_Type' not in df_t.columns:
                    df_t['Exit_Type'] = None
                    df_t.loc[df_t['Strat_Returns'] <= -sl_val * 0.95, 'Exit_Type'] = 'Stop Loss'
                    df_t.loc[df_t['Strat_Returns'] >= tp_val * 0.95, 'Exit_Type'] = 'Take Profit'

                s_min, s_max = df_t['Strat_Momentum'].min(), df_t['Strat_Momentum'].max()
                y_strat = ((df_t['Strat_Momentum'] - s_min) / (s_max - s_min)) * 100 if s_max > s_min else 50

                fig = go.Figure()

                df_t['ret_sign'] = np.sign(df_t['Strat_Returns'])
                df_t['grp'] = (df_t['ret_sign'] != df_t['ret_sign'].shift()).cumsum()
                
                for _, period in df_t[df_t['ret_sign'] != 0].groupby('grp'):
                    idx_start_raw = df_t.index.get_loc(period.index[0])
                    t_start = df_t.index[max(0, idx_start_raw - 1)]
                    color = "rgba(46, 204, 113, 0.3)" if period['ret_sign'].iloc[0] > 0 else "rgba(231, 76, 60, 0.3)"
                    fig.add_vrect(x0=t_start, x1=period.index[-1], line_width=0, layer="below", fillcolor=color)

                fig.add_trace(go.Scatter(x=df_t.index, y=min_max_scale(df_t['Close'])*100, 
                                         name="Prix (Scaled)", line=dict(color='black'), opacity=0.7))
                
                fig.add_trace(go.Scatter(x=df_t.index, y=min_max_scale(df_t['MA'])*100, 
                                         name=f"MA {st.session_state.ma_used}", line=dict(dash='dot', color='orange'), opacity=0.9))
                
                fig.add_trace(go.Scatter(x=df_t.index, y=y_strat, name="Stratégie", line=dict(color='blue', width=2), opacity=0.8))


                sl_mask = df_t['Exit_Type'] == 'Stop Loss'
                tp_mask = df_t['Exit_Type'] == 'Take Profit'

                if sl_mask.any():
                    fig.add_trace(go.Scatter(
                        x=df_t.index[sl_mask], 
                        y=y_strat[sl_mask], 
                        mode='markers', 
                        name='Stop Loss',
                        marker=dict(color='red', size=12, symbol='x', line=dict(width=2, color='white'))
                    ))

                if tp_mask.any():
                    fig.add_trace(go.Scatter(
                        x=df_t.index[tp_mask], 
                        y=y_strat[tp_mask],
                        mode='markers', 
                        name='Take Profit',
                        marker=dict(color='springgreen', size=15, symbol='x', line=dict(width=1, color='white'))
                    ))

                fig.update_layout(
                    height=450, 
                    template="plotly_dark", 
                    hovermode="x unified",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

                ret, mdd = calculate_metrics(df_t['Strat_Momentum'])
                c1, c2, c3 = st.columns(3)
                c1.metric("Performance", f"{ret:.2%}")
                c2.metric("Max Drawdown", f"{mdd:.2%}")
                
                val_fin = cap_init * (df_t['Strat_Momentum'].iloc[-1] / df_t['Strat_Momentum'].iloc[0])
                c3.metric("Valeur Finale", f"{val_fin:.2f} $")
                
                st.caption(f"Configuration active : SL {sl_val*100:.1f}% | TP {tp_val*100:.1f}% | MA {st.session_state.ma_used}")

        with tabs[-1]:
            st.subheader("🛠️ Optimisation de l'Allocation")
            c1, c2, _ = st.columns([1, 1, 2])
            if c1.button("⚖️ Equal Weight"):
                for t in tickers: 
                    st.session_state[f"w_{t}"] = round(100.0 / len(tickers), 2)
                    st.session_state[f"slide_{t}"] = round(100.0 / len(tickers), 2)
                st.rerun()
            
                st.rerun()

            st.divider()
            
            col_inputs, col_visual = st.columns([1.2, 1])
            weights = {}

            with col_inputs:
                st.write("**Répartition du capital**")
                
                for t in tickers:
                    if f"w_{t}" not in st.session_state:
                        st.session_state[f"w_{t}"] = round(100.0 / len(tickers), 2)
                    if f"slide_{t}" not in st.session_state:
                        st.session_state[f"slide_{t}"] = st.session_state[f"w_{t}"]

                    def sync_slider(ticker=t):
                        st.session_state[f"w_{ticker}"] = st.session_state[f"slide_{ticker}"]
                    
                    def sync_num(ticker=t):
                        st.session_state[f"slide_{ticker}"] = st.session_state[f"w_{ticker}"]

                    r0, r1, r2 = st.columns([0.6, 3, 1.2])
                    
                    with r0:
                        logo_url = get_logo_url(t)
                        if logo_url: st.image(logo_url, width=35)
                    
                    with r1:
                        st.slider(f"{t}", 0.0, 100.0, key=f"slide_{t}", on_change=sync_slider, step=0.1)
                    
                    with r2:
                        weights[t] = st.number_input("Precise", label_visibility="collapsed", min_value=0.0, max_value=100.0, 
                                                     key=f"w_{t}", on_change=sync_num, step=0.01, format="%.2f")

            total_w = sum(weights.values())

            with col_visual:
                if total_w > 0:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(weights.keys()), 
                        values=list(weights.values()), 
                        hole=.5,
                        marker=dict(colors=['#00f2fe', '#4facfe', '#7f00ff', '#e100ff', '#ff0844', '#f77062'], line=dict(color='white', width=2)),
                        textinfo='percent',
                        hovertemplate="<b>%{label}</b><br>📊 Part : %{percent}<br>💰 Poids : %{value:.2f}%<extra></extra>"
                    )])
                    fig_pie.update_layout(
                        template="plotly_dark", height=400, showlegend=True,
                        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
                        hoverlabel=dict(bgcolor="white", font_size=35, font_family="Arial", font_color="black", bordercolor="black")
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

            if total_w > 0:
                st.divider()
                df_global = pd.DataFrame({t: data_dict[t]['Strat_Momentum'] for t in tickers}).dropna()
                df_global['Portfolio_Value'] = sum(df_global[t] * (weights[t] / total_w) for t in tickers)

                p_ret, p_mdd = calculate_metrics(df_global['Portfolio_Value'])
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Rendement Global", f"{p_ret:.2%}")
                m2.metric("Risque (MDD)", f"{p_mdd:.2%}")
                m3.metric("Total Alloué", f"{total_w:.2f}%", 
                          delta=f"{total_w-100:.2f}%" if abs(total_w-100) > 0.1 else None, 
                          delta_color="inverse")

                fig_glob = go.Figure()
                for t in tickers:
                    if weights[t] > 0:
                        fig_glob.add_trace(go.Scatter(
                            x=df_global.index, y=df_global[t], 
                            name=f"Contrib: {t}", line=dict(width=2, dash='dot'), opacity=0.7
                        ))

                fig_glob.add_trace(go.Scatter(
                    x=df_global.index, y=df_global['Portfolio_Value'], 
                    name="MON PORTEFEUILLE", line=dict(color='gold', width=4)
                ))

                fig_glob.update_layout(
                    height=550, title="Performance du Panier Dynamique vs Actifs", 
                    template="plotly_dark", hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_glob, use_container_width=True)
                
                
                # --- AJOUT DES MÉTRIQUES AVANCÉES ---
            st.divider()
            st.subheader("📊 Métriques de Risque & Diversification")

            # 1. Préparation des données de rendements
            df_rets = pd.DataFrame({t: data_dict[t]['Strat_Returns'] for t in tickers}).dropna()
            w_arr = np.array([weights[t] / total_w for t in tickers])
            
            # Rendement et Volatilité du Portefeuille
            port_daily_rets = df_rets.dot(w_arr)
            port_vol = port_daily_rets.std() * np.sqrt(252)
            port_return, _ = calculate_metrics(df_global['Portfolio_Value'])

            # 2. Calcul de l'effet de diversification
            # (Moyenne pondérée des vols - Volatilité réelle du portefeuille)
            indiv_vols = df_rets.std() * np.sqrt(252)
            weighted_vol_avg = np.sum(indiv_vols * w_arr)
            diversification_benefit = weighted_vol_avg - port_vol

            # Affichage des indicateurs
            m1, m2, m3 = st.columns(3)
            m1.metric("Volatilité Annuelle", f"{port_vol:.2%}")
            m2.metric("Bénéfice Diversification", f"+{diversification_benefit:.2%}", 
                      help="Réduction du risque obtenue grâce à la faible corrélation entre les actifs.")
            m3.metric("Sharpe Ratio", f"{(port_return / port_vol):.2f}" if port_vol > 0 else "0.00")

            # 3. Matrice de Corrélation
            st.write("**Matrice de Corrélation des Stratégies**")
            corr_matrix = df_rets.corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                hoverongaps=False
            ))
            fig_corr.update_layout(height=400, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.write("---")
        st.info("👈 Sélectionnez vos actifs et paramètres dans la barre latérale et cliquez sur 'Lancer l'analyse'.")