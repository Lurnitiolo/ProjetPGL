import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from .data_loader import load_stock_data, get_logo_url
from .strategies import apply_strategies, calculate_metrics

def min_max_scale(series):
    if series.max() == series.min(): return series * 0
    return (series - series.min()) / (series.max() - series.min())

@st.fragment
def render_portfolio_simulation(tickers, data_dict, cap_init):
    # --- 1. BOUTONS DE PRESETS (DYNAMISÉS) ---
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    
    # Calcul préalable des métriques pour les presets
    stats = {}
    for t in tickers:
        p_ret, p_mdd = calculate_metrics(data_dict[t]['Strat_Momentum'])
        vol = data_dict[t]['Strat_Returns'].std() * np.sqrt(252)
        stats[t] = {
            'ret': p_ret, 
            'mdd': abs(p_mdd) if abs(p_mdd) > 0.01 else 0.01, # Éviter div par zéro
            'vol': vol if vol > 0 else 0.01,
            'sharpe': p_ret / vol if vol > 0 else 0
        }

    # -- Bouton 1: Poids Égaux
    if c1.button("⚖️ Equal Weight"):
        val = round(100.0 / len(tickers), 2)
        for t in tickers:
            st.session_state[f"w_{t}"] = val
            st.session_state[f"slide_{t}"] = val
        st.rerun(scope="fragment")

    # -- Bouton 2: Risk Parity (Volatilité)
    if c2.button("🛡️ Risk Parity"):
        total_inv_vol = sum(1/s['vol'] for s in stats.values())
        for t in tickers:
            val = round(((1/stats[t]['vol']) / total_inv_vol) * 100, 2)
            st.session_state[f"w_{t}"] = val
            st.session_state[f"slide_{t}"] = val
        st.rerun(scope="fragment")

    # -- Bouton 3: MDD Parity (Protection Downside)
    if c3.button("📉 Min Drawdown"):
        # Alloue plus de poids aux actifs qui chutent le moins
        total_inv_mdd = sum(1/s['mdd'] for s in stats.values())
        for t in tickers:
            val = round(((1/stats[t]['mdd']) / total_inv_mdd) * 100, 2)
            st.session_state[f"w_{t}"] = val
            st.session_state[f"slide_{t}"] = val
        st.rerun(scope="fragment")

    # -- Bouton 4: Max Momentum (Top Performer)
    if c4.button("🚀 Top Perf"):
        # On ne garde que les performances positives pour la pondération
        pos_rets = {t: max(0, stats[t]['ret']) for t in tickers}
        total_pos = sum(pos_rets.values())
        
        if total_pos > 0:
            for t in tickers:
                val = round((pos_rets[t] / total_pos) * 100, 2)
                st.session_state[f"w_{t}"] = val
                st.session_state[f"slide_{t}"] = val
        else:
            # Si tout est négatif, on remet en Equal Weight par défaut
            for t in tickers: st.session_state[f"w_{t}"] = round(100/len(tickers), 2)
            
        st.rerun(scope="fragment")

    # -- Bouton 5: Max Sharpe (Efficacité)
    if c5.button("💎 Sharpe Ratio"):
        # On pondère par le ratio de Sharpe positif
        pos_sharpe = {t: max(0, stats[t]['sharpe']) for t in tickers}
        total_sharpe = sum(pos_sharpe.values())
        
        if total_sharpe > 0:
            for t in tickers:
                val = round((pos_sharpe[t] / total_sharpe) * 100, 2)
                st.session_state[f"w_{t}"] = val
                st.session_state[f"slide_{t}"] = val
        else:
            # Fallback si aucun Sharpe n'est positif
            for t in tickers: st.session_state[f"w_{t}"] = round(100/len(tickers), 2)
            
        st.rerun(scope="fragment")  

    st.divider()
    col_inputs, col_visual = st.columns([1.2, 1])
    weights = {}
    
    with col_inputs:
        st.write("**Répartition du capital**")
        for t in tickers:
            # Initialisation de sécurité pour éviter le KeyError
            if f"w_{t}" not in st.session_state: st.session_state[f"w_{t}"] = 100.0/len(tickers)
            if f"slide_{t}" not in st.session_state: st.session_state[f"slide_{t}"] = st.session_state[f"w_{t}"]

            # Callbacks pour lier Slider <-> Number Input
            def sync_to_num(ticker=t): st.session_state[f"w_{ticker}"] = st.session_state[f"slide_{ticker}"]
            def sync_to_slide(ticker=t): st.session_state[f"slide_{ticker}"] = st.session_state[f"w_{ticker}"]

            r0, r1, r2 = st.columns([0.6, 3, 1.2])
            with r0:
                logo = get_logo_url(t)
                if logo: st.image(logo, width=35)
            
            # Slider utilise slide_t et met à jour w_t
            r1.slider(f"{t}", 0.0, 100.0, key=f"slide_{t}", on_change=sync_to_num, step=0.1)
            # Number input utilise w_t et met à jour slide_t
            weights[t] = r2.number_input(f"v_{t}", label_visibility="collapsed", key=f"w_{t}", on_change=sync_to_slide, step=0.01)

    total_w = sum(weights.values())

    if total_w > 0:
        # --- 2. CALCULS ET GRAPHIQUES ---
        df_global = pd.DataFrame({t: data_dict[t]['Strat_Momentum'] for t in tickers}).dropna()
        w_arr = np.array([weights[t] / total_w for t in tickers])
        df_global['Portfolio_Value'] = df_global.dot(w_arr)

        with col_visual:
            fig_pie = go.Figure(data=[go.Pie(labels=list(weights.keys()), values=list(weights.values()), hole=.5)])
            fig_pie.update_layout(template="plotly_dark", height=350, showlegend=False,
                                  hoverlabel=dict(bgcolor="white", font_color="black", font_size=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()
        
        # --- 3. GRAPHIQUE AVEC TOUTES LES COURBES (Restauré) ---
        fig_glob = go.Figure()
        # On remet les courbes individuelles en arrière-plan
        for t in tickers:
            if weights[t] > 0:
                fig_glob.add_trace(go.Scatter(x=df_global.index, y=df_global[t], 
                                              name=f"Contrib: {t}", line=dict(width=1, dash='dot'), opacity=0.5))
        
        # Courbe principale du portefeuille
        fig_glob.add_trace(go.Scatter(x=df_global.index, y=df_global['Portfolio_Value'], 
                                      name="MON PORTEFEUILLE", line=dict(color='gold', width=4)))
        
        fig_glob.update_layout(height=450, title="Performance du Panier vs Actifs", template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig_glob, use_container_width=True)

        # --- 4. MÉTRIQUES (Vol, Sharpe, Corrélation) ---
        df_rets = pd.DataFrame({t: data_dict[t]['Strat_Returns'] for t in tickers}).dropna()
        port_daily_rets = df_rets.dot(w_arr)
        port_vol = port_daily_rets.std() * np.sqrt(252)
        port_ret, port_mdd = calculate_metrics(df_global['Portfolio_Value'])
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rendement", f"{port_ret:.2%}")
        m2.metric("Risque (MDD)", f"{port_mdd:.2%}")
        m3.metric("Volatilité Ann.", f"{port_vol:.2%}")
        m4.metric("Sharpe Ratio", f"{(port_ret/port_vol):.2f}" if port_vol > 0 else "0.00")

        st.write("**Matrice de Corrélation**")
        corr = df_rets.corr()
        fig_corr = go.Figure(data=go.Heatmap(z=corr, x=corr.columns, y=corr.columns, colorscale='RdBu', zmin=-1, zmax=1))
        fig_corr.update_layout(height=350, template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        
        
        
        
        
        
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
            render_portfolio_simulation(tickers, data_dict, cap_init)
            
            
            
    else:
        st.write("---")
        st.info("👈 Sélectionnez vos actifs et paramètres dans la barre latérale et cliquez sur 'Lancer l'analyse'.")