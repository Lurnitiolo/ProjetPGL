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

    

    col_inputs, col_visual = st.columns([1, 1], vertical_alignment="top")    

    with col_inputs:
    # --- 1. STYLE CSS (Ajustements précis pour alignement 40px) ---
        st.markdown("""
            <style>
                /* Cache les labels */
                div[data-testid="stNumberInput"] label { display: none !important; }
                
                /* AJUSTEMENT DU CHIFFRE AU-DESSUS DE LA BOULE */
                div[data-testid="stThumbValue"] {
                    transform: translateY(18px) !important; /* Ajusté pour coller à la boule */
                    font-size: 0.8rem !important;
                    font-weight: 600 !important;
                    color: #FFD700 !important;
                }

                /* ALIGNEMENT PILE-POIL : On remonte le bloc slider et l'input */
                div[data-testid="stSlider"] {
                    margin-top: 15px !important; /* Remonte le rail pour l'aligner au centre du logo */
                }
                div[data-testid="stNumberInput"] {
                    margin-top: -5px !important; /* Aligne l'input numérique sur le rail */
                }

                /* Conteneur logo/texte */
                .asset-cell {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    height: 40px; /* Force la hauteur pour correspondre au logo */
                }
                .logo-40 {
                    width: 40px; 
                    height: 40px; 
                    object-fit: contain; 
                    border-radius: 6px;
                    background: rgba(255,255,255,0.03);
                }
                .ticker-name-v18 {
                    font-weight: 700;
                    font-size: 1rem;
                    color: #ffffff;
                }
                
                /* Scrollbar fine */
                [data-testid="stElementContainer"] div:has(> .asset-cell) {
                    scrollbar-width: thin;
                    scrollbar-color: #30363d transparent;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("#### ⚙️ Configuration & Allocation")
        
        # PARAMÈTRES FIXES (Rééquilibrage & Frais)
        with st.container(border=True):
            cp1, cp2 = st.columns([2, 1], vertical_alignment="center")
            with cp1:
                st.markdown("**🔄 Rééquilibrage**")
                rebalance_freq = st.selectbox("Freq", ["Quotidien", "Hebdomadaire", "Mensuel", "Annuel", "Aucun"], index=2, key="rebal_fix_v2", label_visibility="collapsed")
            with cp2:
                st.markdown("**💸 Frais (bps)**")
                fees_bps = st.number_input("Fees", 0, 100, 10, key="fees_fix_v2", label_visibility="collapsed") / 10000

        st.write("") 

        with st.container(border=True):
            st.markdown("**🧺 Allocation du Capital**")

            with st.container(height=215, border=False):
                for t in tickers:
                    if f"slider_weight_{t}" not in st.session_state:
                        val = st.session_state.portfolio_weights.get(t, 100.0/len(tickers))
                        st.session_state[f"slider_weight_{t}"] = float(val)
                        st.session_state[f"num_weight_{t}"] = float(val)

                    # Alignement central et colonnes
                    r0, r1, r2 = st.columns([2.3, 5, 1.2], vertical_alignment="center")
                    
                    with r0:
                        st.markdown(f"""
                            <div class="asset-cell">
                                <img src="{get_logo_url(t)}" class="logo-40">
                                <span class="ticker-name-v18">{t}</span>
                            </div>
                        """, unsafe_allow_html=True)

                    with r1:
                        st.slider(
                            f"S_{t}", 0.0, 100.0, 
                            key=f"slider_weight_{t}", 
                            on_change=update_num, args=(t,), 
                            label_visibility="collapsed"
                        )

                    with r2:
                        st.number_input(
                            f"N_{t}", 0.0, 100.0, 
                            key=f"num_weight_{t}", 
                            step=1.0, 
                            on_change=update_slider, args=(t,), 
                            label_visibility="collapsed"
                        )

        # Calcul des poids totaux (en dehors du cadre visuel)
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

        with col_visual:
            pie_height = 445 
            
            with st.container(border=True):
                st.markdown(f"""
                    <p style='text-align:center; color:#808495; font-size:0.8rem; margin-bottom:5px; font-weight:700;'>
                        ALLOCATION CIBLE
                    </p>
                """, unsafe_allow_html=True)
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(weights.keys()), 
                    values=list(weights.values()), 
                    hole=.6, 
                    textinfo='percent',
                    marker=dict(line=dict(color='#0e1117', width=2)) # Séparation propre des parts
                )])
                
                fig_pie.update_layout(
                    template="plotly_dark", 
                    height=pie_height,  # S'adapte à ta variable
                    margin=dict(t=0, b=0, l=0, r=0), 
                    showlegend=False, 
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

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

        st.markdown("#### 📉 Simulation Interactive")
        with st.container(border=True):
            dates = df_rets.index.strftime('%Y-%m-%d').tolist()
            legend_data = ["PORTFOLIO", "Benchmark 1/N"] + [f"{t} (Strat)" for t in tickers]
            
            legend_selected = {name: False for name in legend_data}
            legend_selected["PORTFOLIO"] = True
            legend_selected["Benchmark 1/N"] = True

            all_series = []
            for t in tickers:
                # Performance cumulative rebasée à 100 pour la comparaison
                asset_cum = (1 + df_rets[t]).cumprod() * 100
                all_series.append({
                    "name": f"{t} (Strat)", 
                    "type": "line", 
                    "data": asset_cum.round(2).tolist(),
                    "smooth": True, 
                    "symbol": "none", 
                    "lineStyle": {"width": 2, "type": "dashed", "opacity": 0.3}
                })

            # Ajout des séries principales
            # Le PORTFOLIO est mis en avant avec une ligne dorée plus épaisse
            all_series.extend([
                {
                    "name": "PORTFOLIO", 
                    "type": "line", 
                    "data": (port_series * 100).round(2).tolist(), 
                    "smooth": True, 
                    "symbol": "none", 
                    "lineStyle": {"width": 4, "color": "#FFD700"}
                },
                {
                    "name": "Benchmark 1/N", 
                    "type": "line", 
                    "data": (bench_cum * 100).round(2).tolist(), 
                    "smooth": True, 
                    "symbol": "none", 
                    "lineStyle": {"width": 3.5, "color": "#8b949e", "type": "dotted"}
                }
            ])

            option = {
                "backgroundColor": "transparent",
                "tooltip": {
                    "trigger": "axis", 
                    "backgroundColor": "#111", 
                    "textStyle": {"color": "#fff"}, 
                    "axisPointer": {"type": "cross"}
                },
                "legend": {
                    "data": legend_data, 
                    "selected": legend_selected, # Applique le filtre de base
                    "textStyle": {"color": "#8b949e", "fontSize": 10}, 
                    "type": "scroll", 
                    "bottom": 0
                },
                "grid": {"left": "3%", "right": "3%", "bottom": "15%", "top": "5%", "containLabel": True},
                "xAxis": {
                    "type": "category", 
                    "data": dates, 
                    "axisLine": {"lineStyle": {"color": "#30363d"}}
                },
                "yAxis": {
                    "scale": True, 
                    "splitLine": {"lineStyle": {"color": "#30363d", "type": "dashed"}}
                },
                "series": all_series
            }

            st_echarts(options=option, height="500px", key="p_main_simulation_echart")

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

                # --- NOUVELLE SECTION : GESTION DU RISQUE AVANCÉE ---
        st.divider()
        st.markdown("### 🛡️ Risk Management & Diversification Analysis")
        
        c_risk_1, c_risk_2 = st.columns([1.5, 1])
        
        with c_risk_1:
            with st.container(border=True):
                st.markdown("**🌊 Résilience face aux baisses (Drawdown)**")
                st.caption("Profondeur des chutes : comparez la robustesse du panier vs ses composants.")
                
                # 1. Calcul des Drawdowns
                def get_dd(series):
                    return (series / series.cummax() - 1)

                dd_port = get_dd(port_series)
                dd_bench = get_dd(bench_cum)
                

                asset_colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#8957e5", "#f0883e", "#1f6feb"]
                
                dd_series = []
                legend_selected = {"MON PORTEFEUILLE": True, "Benchmark 1/N": True}
                
                # 3. Boucle avec index pour assigner une couleur unique à chaque actif
                for i, t in enumerate(tickers):
                    if weights[t] > 0:
                        # On utilise le modulo (%) pour boucler sur la palette si on a plus d'actifs que de couleurs
                        current_color = asset_colors[i % len(asset_colors)]
                        
                        dd_asset = get_dd((1 + df_rets[t]).cumprod())
                        name = f"DD {t}"
                        legend_selected[name] = False  # Masqué par défaut pour la clarté
                        
                        dd_series.append({
                            "name": name, 
                            "type": "line", 
                            "data": (dd_asset * 100).round(2).tolist(),
                            "smooth": True, 
                            "symbol": "none",
                            "areaStyle": {"opacity": 0.08, "color": current_color},
                            "lineStyle": {
                                "width": 1, 
                                "type": "dashed", 
                                "opacity": 0.4, 
                                "color": current_color
                            }
                        })

                dd_series.append({
                    "name": "Benchmark 1/N", "type": "line", "data": (dd_bench * 100).round(2).tolist(),
                    "smooth": True, "symbol": "none", "lineStyle": {"width": 1.5, "type": "dotted", "color": "#8b949e"}
                })
                dd_series.append({
                    "name": "MON PORTEFEUILLE", "type": "line", "data": (dd_port * 100).round(2).tolist(),
                    "smooth": True, "symbol": "none", "areaStyle": {"opacity": 0.15, "color": "#FFD700"},
                    "lineStyle": {"width": 4, "color": "#FFD700"}
                })

                dd_option = {
                    "backgroundColor": "transparent",
                    "tooltip": {
                        "trigger": "axis", 
                        "backgroundColor": "#111", 
                        "textStyle": {"color": "#fff"},
                        "formatter": "{b}<br/>{a}: <b>{c}%</b>"
                    },
                    "legend": {
                        "type": "scroll", 
                        "bottom": 0, 
                        "selected": legend_selected, 
                        "textStyle": {"color": "#8b949e", "fontSize": 10}
                    },
                    "grid": {"left": "3%", "right": "3%", "top": "10%", "bottom": "20%", "containLabel": True},
                    "xAxis": {
                        "type": "category", "data": dates, 
                        "axisLine": {"lineStyle": {"color": "#30363d"}}
                    },
                    "yAxis": {
                        "type": "value", "max": 0, "splitNumber": 3,
                        "axisLabel": {"formatter": "{value}%", "color": "#8b949e"},
                        "splitLine": {"lineStyle": {"color": "#30363d", "type": "dashed"}}
                    },
                    "series": dd_series
                }
                st_echarts(options=dd_option, height="350px", key="p_drawdown_colored_v18")

        with c_risk_2:
            with st.container(border=True):
                st.markdown("**🎯 Risk Contribution**")
                st.caption("Quel actif génère la volatilité du portefeuille ?")
                
                # Calcul simplifié de la contribution au risque (Marginal Risk Contribution)
                cov_matrix = df_rets.cov() * 252
                port_variance = np.dot(target_weights.T, np.dot(cov_matrix, target_weights))
                marginal_risk = np.dot(cov_matrix, target_weights) / np.sqrt(port_variance)
                risk_contribution = target_weights * marginal_risk
                risk_pct = risk_contribution / risk_contribution.sum()

                fig_risk_dec = go.Figure(data=[go.Bar(
                    x=list(tickers), 
                    y=risk_pct * 100,
                    marker_color='rgba(88, 166, 255, 0.6)',
                    text=np.round(risk_pct * 100, 1),
                    texttemplate="%{text}%",
                    textposition="outside"
                )])
                fig_risk_dec.update_layout(
                    template="plotly_dark", height=320, margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(showgrid=False, title="Risk Contribution %"),
                    xaxis=dict(categoryorder='total descending')
                )
                st.plotly_chart(fig_risk_dec, use_container_width=True, key="p_risk_contrib_bar")

        # --- VaR & CVaR du PORTFEUILLE (Style V18) ---
        with st.container(border=True):
            st.markdown("**🛡️ Portfolio VaR Analysis**")
            r_returns = port_series.pct_change().dropna()
            conf = 0.95
            var_95 = np.percentile(r_returns, (1 - conf) * 100)
            es_95 = r_returns[r_returns <= var_95].mean()
            
            v1, v2, v3 = st.columns(3)
            with v1: show_big_number("Daily VaR (95%)", var_95, "Perte max probable / jour", color_cond="red_if_neg")
            with v2: show_big_number("CVaR (Expected Shortfall)", es_95, "Perte moyenne en cas de krach", color_cond="red_if_neg")
            with v3: show_big_number("Worst Day", r_returns.min(), "Pire journée historique", color_cond="red_if_neg")




    else:
        st.info("👈 Configurez l'allocation pour simuler le panier.")
