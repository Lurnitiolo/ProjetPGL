import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from .data_loader import get_logo_url
from .strategies import calculate_metrics
from streamlit_echarts import st_echarts

def min_max_scale(series):
    if series.max() == series.min(): return series * 0
    return (series - series.min()) / (series.max() - series.min())


def show_big_number(label, value, delta=None, fmt="{:.2%}", color_cond="neutral"):
    val_str = fmt.format(value)
    color_str = "" 
    if color_cond == "green_if_pos": color_str = ":green" if value > 0 else ":red"
    elif color_cond == "red_if_neg": color_str = ":red"
    elif color_cond == "always_blue": color_str = ":blue"
    
    st.markdown(f"**{label}**")
    if color_str: st.markdown(f"### {color_str}[{val_str}]")
    else: st.markdown(f"### {val_str}")
        
    if delta:
        d_color = ":green" if "+" in delta else ":red" if "-" in delta else ""
        if d_color: st.markdown(f"{d_color}[{delta}]")
        else: st.caption(delta)

def apply_preset_callback(new_weights):
    # On met à jour le dictionnaire central utilisé par les widgets
    if 'portfolio_weights' not in st.session_state:
        st.session_state.portfolio_weights = {}
    
    for t, val in new_weights.items():
        st.session_state.portfolio_weights[t] = float(val)
        # On met aussi à jour les clés directes des widgets pour forcer l'affichage
        st.session_state[f"slider_weight_{t}"] = float(val)
        st.session_state[f"num_weight_{t}"] = float(val)
def update_slider(t):
    st.session_state[f"slider_weight_{t}"] = st.session_state[f"num_weight_{t}"]
    st.session_state.portfolio_weights[t] = st.session_state[f"num_weight_{t}"]

def update_num(t):
    st.session_state[f"num_weight_{t}"] = st.session_state[f"slider_weight_{t}"]
    st.session_state.portfolio_weights[t] = st.session_state[f"slider_weight_{t}"]




@st.fragment
def render_portfolio_simulation(tickers, data_dict, cap_init):
    # Initialisation des poids
    if 'portfolio_weights' not in st.session_state:
        eq_val = round(100.0 / len(tickers), 2)
        st.session_state.portfolio_weights = {t: eq_val for t in tickers}
    
    # --- PRÉ-CALCUL DES STATS ACTIFS ---
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

    # --- PRÉPARATION DES PRESETS ---
    eq_val = round(100.0 / len(tickers), 2)
    presets = {
        'eq': {t: eq_val for t in tickers},
        'rp': {t: round(((1/stats[t]['vol']) / sum(1/s['vol'] for s in stats.values())) * 100, 2) for t in tickers},
        'md': {t: round(((1/stats[t]['mdd']) / sum(1/s['mdd'] for s in stats.values())) * 100, 2) for t in tickers},
        'tp': {t: round((max(0, stats[t]['ret']) / sum(max(0, stats[s]['ret']) for s in tickers) * 100), 2) if sum(max(0, stats[s]['ret']) for s in tickers) > 0 else eq_val for t in tickers},
        'sr': {t: round((max(0, stats[t]['sharpe']) / sum(max(0, stats[s]['sharpe']) for s in tickers) * 100), 2) if sum(max(0, stats[s]['sharpe']) for s in tickers) > 0 else eq_val for t in tickers}
    }

    st.markdown("### 🔎 Portfolio Optimization & Simulation")

    # --- BOUTONS PRESETS (Encadrés) ---
    with st.container(border=True):
        st.caption("Modèles d'allocation rapide")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.button("⚖️ Equal Weight", on_click=apply_preset_callback, args=(presets['eq'],), use_container_width=True)
        c2.button("🛡️ Risk Parity", on_click=apply_preset_callback, args=(presets['rp'],), use_container_width=True)
        c3.button("📉 Min Drawdown", on_click=apply_preset_callback, args=(presets['md'],), use_container_width=True)
        c4.button("🚀 Top Perf", on_click=apply_preset_callback, args=(presets['tp'],), use_container_width=True)
        c5.button("💎 Sharpe Ratio", on_click=apply_preset_callback, args=(presets['sr'],), use_container_width=True)

    col_inputs, col_visual = st.columns([1, 1], vertical_alignment="top")    
    
    with col_inputs:
        # Style CSS pour le look Terminal/Léché
        st.markdown("""
            <style>
                div[data-testid="stNumberInput"] label { display: none !important; }
                [data-testid="column"]:nth-child(2) { overflow: visible !important; }
                [data-testid="column"]:nth-child(2) > div { position: sticky !important; top: 80px !important; z-index: 99; }
                .responsive-logo { width: 32px; height: 32px; object-fit: contain; border-radius: 4px; }
                .ticker-name { font-weight: 700; color: #e6edf3; margin-left: 8px; }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("#### ⚙️ Configuration & Allocation")
        
        # PARAMÈTRES SIMULATION (Encadrés)
        with st.container(border=True):
            c_p1, c_p2 = st.columns([2, 1])
            with c_p1:
                st.markdown("**🔄 Rééquilibrage**")
                rebalance_freq = st.selectbox("Freq", ["Quotidien", "Hebdomadaire", "Mensuel", "Annuel", "Aucun (Buy & Hold)"], index=2, label_visibility="collapsed")
            with c_p2:
                st.markdown("**💸 Frais (bps)**")
                fees_bps = st.number_input("Fees", 0, 100, 10, step=5, label_visibility="collapsed") / 10000

        st.write("") 

        # BOUCLE DES ACTIFS
        for t in tickers:
            if f"slider_weight_{t}" not in st.session_state:
                val = st.session_state.portfolio_weights.get(t, eq_val)
                st.session_state[f"slider_weight_{t}"] = float(val)
                st.session_state[f"num_weight_{t}"] = float(val)

            r0, r1, r2 = st.columns([1.5, 5, 1.2])
            with r0:
                st.markdown(f"<div style='display:flex;align-items:center;'><img src='{get_logo_url(t)}' class='responsive-logo'><span class='ticker-name'>{t}</span></div>", unsafe_allow_html=True)
            with r1:
                st.slider(f"S_{t}", 0.0, 100.0, key=f"slider_weight_{t}", on_change=update_num, args=(t,), label_visibility="collapsed")
            with r2:
                st.number_input(f"N_{t}", 0.0, 100.0, key=f"num_weight_{t}", step=1.0, on_change=update_slider, args=(t,), label_visibility="collapsed")

    weights = {t: st.session_state.get(f"num_weight_{t}", 0.0) for t in tickers}
    total_w = sum(weights.values())

    # --- CALCULS SIMULATION ---
    if total_w > 0:
        df_rets = pd.DataFrame({t: data_dict[t]['Strat_Returns'] for t in tickers}).dropna()
        target_weights = np.array([weights[t] / total_w for t in tickers])
        n_days = len(df_rets)
        portfolio_values = np.zeros(n_days)
        current_val = 1.0 
        current_weights = target_weights.copy()
        current_val *= (1 - fees_bps) # Frais initiaux

        for i in range(n_days):
            date = df_rets.index[i]
            daily_rets = df_rets.iloc[i].values
            do_rebal = (rebalance_freq == "Quotidien") or \
                       (rebalance_freq == "Hebdomadaire" and date.weekday() == 0) or \
                       (rebalance_freq == "Mensuel" and i > 0 and date.month != df_rets.index[i-1].month) or \
                       (rebalance_freq == "Annuel" and i > 0 and date.year != df_rets.index[i-1].year)

            if do_rebal and rebalance_freq != "Aucun (Buy & Hold)":
                turnover = np.sum(np.abs(current_weights - target_weights))
                current_val *= (1 - (turnover * fees_bps))
                current_weights = target_weights.copy()
            
            current_val *= (1 + np.sum(current_weights * daily_rets))
            portfolio_values[i] = current_val
            if rebalance_freq != "Quotidien":
                drift = current_weights * (1 + daily_rets)
                current_weights = drift / np.sum(drift) if np.sum(drift) != 0 else target_weights

        port_series = pd.Series(portfolio_values, index=df_rets.index)
        port_return, port_mdd = calculate_metrics(port_series)
        port_vol = port_series.pct_change().std() * np.sqrt(252)
        
        # Benchmark 1/N
        bench_rets = df_rets.mean(axis=1)
        bench_cum = (1 + bench_rets).cumprod()
        bench_return, bench_mdd = calculate_metrics(bench_cum)
        bench_vol = bench_rets.std() * np.sqrt(252)

        # --- VISUALISATION DROITE (Sticky Pie) ---
        with col_visual:
            with st.container(border=True):
                st.markdown("<p style='text-align:center;color:#808495;margin-bottom:-10px;'>ALLOCATION CIBLE</p>", unsafe_allow_html=True)
                fig_pie = go.Figure(data=[go.Pie(labels=list(weights.keys()), values=list(weights.values()), hole=.6, textinfo='percent')])
                fig_pie.update_layout(template="plotly_dark", height=380, margin=dict(t=30, b=0, l=0, r=0), showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

        # [cite_start]--- SCORECARD (Style Arthur Big Numbers) [cite: 5, 27, 28, 29] ---
        st.markdown("#### 📊 Scorecard (Vs Benchmark 1/N)")
        with st.container(border=True):
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1: show_big_number("Return Net", port_return, f"{port_return-bench_return:+.2%} vs 1/N", color_cond="green_if_pos")
            with m2: show_big_number("Risk (MDD)", port_mdd, f"{port_mdd-bench_mdd:+.2%} vs 1/N", color_cond="red_if_neg")
            with m3: show_big_number("Volatility", port_vol, f"{port_vol-bench_vol:+.2%} vs 1/N", color_cond="neutral")
            with m4:
                p_sh = port_return / port_vol if port_vol > 0 else 0
                b_sh = bench_return / bench_vol if bench_vol > 0 else 0
                show_big_number("Sharpe", p_sh, f"{p_sh-b_sh:+.2f} vs 1/N", fmt="{:.2f}", color_cond="always_blue")
            with m5:
                w_vol_assets = sum((weights[t]/total_w) * stats[t]['vol'] for t in tickers)
                div_ratio = w_vol_assets / port_vol if port_vol > 0 else 1.0
                show_big_number("Diversif.", div_ratio, fmt="{:.2f}x", color_cond="green_bool")

        # [cite_start]--- ECHARTS INTERACTIF (Séries individuelles en pointillés togglables) [cite: 13, 14, 19, 20] ---
        st.markdown("#### 📉 Simulation Interactive")
        with st.container(border=True):
            dates = df_rets.index.strftime('%Y-%m-%d').tolist()
            legend_data = ["PORTFOLIO", "Benchmark 1/N"] + [f"{t} (Strat)" for t in tickers]
            
            # Initialisation des séries avec l'Asset Performance
            all_series = []
            for t in tickers:
                asset_cum = (1 + df_rets[t]).cumprod() * 100
                all_series.append({
                    "name": f"{t} (Strat)", "type": "line", "data": asset_cum.round(2).tolist(),
                    "smooth": True, "symbol": "none", "lineStyle": {"width": 2, "type": "dashed", "opacity": 0.8}
                })

            # Ajout des séries principales
            all_series.extend([
                {"name": "PORTFOLIO", "type": "line", "data": (port_series * 100).round(2).tolist(), "smooth": True, "symbol": "none", "lineStyle": {"width": 4, "color": "#FFD700"}},
                {"name": "Benchmark 1/N", "type": "line", "data": (bench_cum * 100).round(2).tolist(), "smooth": True, "symbol": "none", "lineStyle": {"width": 3.5, "color": "#8b949e", "type": "dotted"}}
            ])

            option = {
                "backgroundColor": "transparent",
                "tooltip": {"trigger": "axis", "backgroundColor": "#111", "textStyle": {"color": "#fff"}, "axisPointer": {"type": "cross"}},
                "legend": {"data": legend_data, "textStyle": {"color": "#8b949e", "fontSize": 10}, "type": "scroll", "bottom": 0},
                "grid": {"left": "3%", "right": "3%", "bottom": "15%", "top": "5%", "containLabel": True},
                "xAxis": {"type": "category", "data": dates, "axisLine": {"lineStyle": {"color": "#30363d"}}},
                "yAxis": {"scale": True, "splitLine": {"lineStyle": {"color": "#30363d", "type": "dashed"}}},
                "series": all_series
            }
            st_echarts(options=option, height="500px")

        # [cite_start]--- ANALYSE DE RISQUE (Heatmaps Plotly [cite: 25]) ---
        c_mat, c_comp = st.columns([1, 1])
        with c_mat:
            with st.container(border=True):
                st.markdown("**Matrice de Corrélation**")
                corr = df_rets.corr()
                fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='RdBu', zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}"))
                fig_corr.update_layout(height=400, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_corr, use_container_width=True, key="p_corr_heatmap")
            
        with c_comp:
            with st.container(border=True):
                st.markdown("**Risk / Return Map (Heatmap Background)**")
                df_rr = pd.DataFrame([{'Name': t, 'Return': stats[t]['ret']*100, 'Vol': stats[t]['vol']*100, 'Type': 'Asset'} for t in tickers] + [{'Name': 'PORTFOLIO', 'Return': port_return*100, 'Vol': port_vol*100, 'Type': 'Portfolio'}])
                
                # --- CALCUL LOGIQUE HEATMAP ---
                v_max, r_max = df_rr['Vol'].max() * 1.3, df_rr['Return'].max() * 1.3
                r_min = df_rr['Return'].min() * 1.3 if df_rr['Return'].min() < 0 else -r_max * 0.2
                v_s, r_s = np.linspace(0, v_max, 30), np.linspace(r_min, r_max, 30)
                z_rr = [[(r - v) for v in v_s] for r in r_s]

                fig_rr = go.Figure()
                fig_rr.add_trace(go.Heatmap(z=z_rr, x=v_s, y=r_s, colorscale=[[0, 'rgba(231, 76, 60, 1)'], [0.5, 'rgba(255, 190, 0, 1)'], [1, 'rgba(46, 204, 113, 1)']], showscale=False, hoverinfo='skip'))
                
                # Modif Assets : Position "top center" et typo Noire
                fig_rr.add_trace(go.Scatter(
                    x=df_rr[df_rr['Type']=='Asset']['Vol'], 
                    y=df_rr[df_rr['Type']=='Asset']['Return'], 
                    mode='markers+text', 
                    text=df_rr['Name'], 
                    textposition="top center",
                    textfont=dict(color='black', size=11),
                    marker=dict(size=10, color='white', line=dict(width=1, color='black'))
                ))
                
                # Modif Portfolio : Position "bottom center" et typo Noire Bold
                fig_rr.add_trace(go.Scatter(
                    x=df_rr[df_rr['Type']=='Portfolio']['Vol'], 
                    y=df_rr[df_rr['Type']=='Portfolio']['Return'], 
                    mode='markers+text', 
                    text=['<b>PORTFOLIO</b>'], 
                    textposition="bottom center",
                    textfont=dict(color='black', size=12),
                    marker=dict(size=20, color='gold', symbol='star', line=dict(width=1, color='black'))
                ))
                
                fig_rr.update_layout(height=400, template="plotly_white", margin=dict(t=10, b=40, l=40, r=10), showlegend=False, xaxis=dict(ticksuffix="%"), yaxis=dict(ticksuffix="%"))
                st.plotly_chart(fig_rr, use_container_width=True, key="p_rr_optimized_heatmap")

                




    else:
        st.info("👈 Configurez l'allocation pour simuler le panier.")
