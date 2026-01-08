import streamlit as st
import pandas as pd
import os
from Arthur import Arthur_page
from Arthur.daily_report import generate_report
from Virgil import Virgil_page

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="PGL Project",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# 2. CSS STYLING (CORRIGÉ POUR SIDEBAR)
# ==============================================================================
st.markdown("""
    <style>
        /* A. HEADER & SIDEBAR TOGGLE (CORRIGÉ) */
        /* Force le bouton > à rester visible même si le header est caché */
        [data-testid="stSidebarCollapsedControl"] {
            visibility: visible !important;
            display: block !important;
            color: #e6edf3 !important; 
        }
        .block-container {padding-top: 3rem;}

        /* B. NAVIGATION (RADIO -> TABS) */
        div.stRadio > div[role="radiogroup"] {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 25px;
            border-bottom: 1px solid #30363d;
            padding-bottom: 10px;
        }
        div.stRadio > div[role="radiogroup"] > label > div:first-child {
            display: none;
        }
        div.stRadio > div[role="radiogroup"] > label {
            background-color: #0e1117;
            padding: 10px 20px;
            border-radius: 8px;
            border: 1px solid #30363d;
            cursor: pointer;
            font-weight: 600;
            color: #8b949e;
            transition: all 0.2s;
        }
        div.stRadio > div[role="radiogroup"] > label:hover {
            color: #e6edf3;
            border-color: #8b949e;
        }
        div.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
            background-color: #238636;
            color: white;
            border-color: #238636;
        }

        /* C. BOX COLORS */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #161b22 !important;
            border: 1px solid #30363d !important;
            border-radius: 10px !important;
        }
        
        /* D. Center Images in Grid */
        div[data-testid="stImage"] {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. HELPER: LOGO URL
# ==============================================================================
def get_logo_url(ticker):
    """Retrieves logo URL based on Ticker."""
    domains = {
        # France
        "MC.PA": "lvmh.com", "TTE.PA": "totalenergies.com", "AIR.PA": "airbus.com",
        "BNP.PA": "group.bnpparibas", "SAN.PA": "sanofi.com",
        # US Tech
        "NVDA": "nvidia.com", "TSLA": "tesla.com", "AAPL": "apple.com", "MSFT": "microsoft.com",
        # Indices & ETFs
        "SPY": "ssga.com", "QQQ": "invesco.com", "CW8.PA": "amundi.fr", "GLD": "spdrgoldshares.com"
    }
    
    if "-USD" in ticker:
        symbol = ticker.split("-")[0].lower()
        return f"https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/{symbol}.png"
    
    if ticker == "GLD":
        return "https://cdn-icons-png.flaticon.com/512/1055/1055646.png"

    if ticker in domains:
        return f"https://www.google.com/s2/favicons?domain={domains[ticker]}&sz=128"
    
    return "https://cdn-icons-png.flaticon.com/512/4256/4256900.png"

# ==============================================================================
# 4. IMPORTS & ERROR HANDLING
# ==============================================================================


# ==============================================================================
# 5. HEADER & NAVIGATION
# ==============================================================================

# Main Title
st.markdown("<h1 style='text-align: center;'>Python Git Linux Project - JACKS HOAREAU</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #8b949e;'>ESILV - IF4 : HOAREAU Virgil / JACKS Arthur</h4>", unsafe_allow_html=True)

st.write("") 

# Navigation Menu
selected_tab = st.radio(
    "Navigation",
    ["🏠 Home", "🔎 Single Asset Analysis", "🌐 Multi Asset Analysis"],
    horizontal=True,
    label_visibility="collapsed"
)

# ==============================================================================
# 6. PAGE CONTENT
# ==============================================================================

if selected_tab == "🏠 Home":
    # --- HOME PAGE ---
    
    st.markdown("### 👋 Welcome to the Dashboard")
    st.info("This project was developed as part of the Python for Finance course (M1). It provides quantitative analysis tools, automated reporting, and portfolio management features.")
    
    st.divider()
    
    # --- BLOCK 1: DATA & ASSETS (FULL LIST) ---
    st.markdown("#### 📡 Data Source & Universe")
    
    with st.container(border=True):
        st.markdown("""
        We use **Yahoo Finance (`yfinance`)** to fetch real-time market data.
        The dashboard currently monitors the following **15 assets** across different sectors and regions:
        """)
        
        st.write("") 
        
        full_assets_list = [
            # 🇫🇷 FRANCE
            ("LVMH", "MC.PA"), ("TotalEnergies", "TTE.PA"), ("Airbus", "AIR.PA"), ("BNP Paribas", "BNP.PA"), ("Sanofi", "SAN.PA"),
            # 🇺🇸 US TECH
            ("NVIDIA", "NVDA"), ("Tesla", "TSLA"), ("Apple", "AAPL"), ("Microsoft", "MSFT"),
            # 🌍 INDICES / COMMODITIES
            ("S&P 500", "SPY"), ("MSCI World", "CW8.PA"), ("Nasdaq 100", "QQQ"), ("Gold", "GLD"),
            # ₿ CRYPTO
            ("Bitcoin", "BTC-USD"), ("Ethereum", "ETH-USD")
        ]
        
        cols = st.columns(5)
        for i, (name, ticker) in enumerate(full_assets_list):
            with cols[i % 5]: 
                st.image(get_logo_url(ticker), width=40)
                st.markdown(f"**{name}**")
                st.caption(f"`{ticker}`")

    st.write("") 

    # --- BLOCK 2: PROJECT OVERVIEW ---
    st.markdown("#### 📘 Project Modules")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### 🔎 Single Asset Analysis")
            st.markdown("""
            Deep dive analysis of a specific financial asset.
            * **Market Data:** OHLCV visualization.
            * **Backtesting:** MA Crossover, Momentum, Mean Reversion.
            * **Risk Metrics:** VaR (95%), Max Drawdown, Sharpe.
            * **ML Forecast:** Random Forest prediction.
            """)
            st.write("➡️ *Go to 'Single Asset Analysis' tab.*")

    with col2:
        with st.container(border=True):
            st.markdown("#### 🌐 Multi Asset Analysis")
            st.markdown("""
                Deep dive analysis of a portfolio of financial assets.
                * **Configuration and detail:** by Asset.
                * **Scoreboard::** benchmark.
                * **Interactive Simulation**.
                * **Risk Management & Diversification analysis:** Drawndown, Risk Contribution, Portfolio VaR Analysis.
            """)
            st.write("➡️ *Go to 'Multi Asset Analysis' tab.*")

    st.write("")

    # --- BLOCK 3: AUTOMATED DAILY REPORT (CRON) ---
    st.markdown("#### 📅 Automated Daily Report (Cron Job)")
    
    with st.container(border=True):
        st.markdown("""
        This report is generated automatically every day at **8:00 PM** via a Linux Cron Job running on the VM.
        It scans the entire asset universe to compute daily metrics.
        """)
        
        report_path = "reports/daily_report_latest.csv"
        
        if os.path.exists(report_path):
            df_report = pd.read_csv(report_path)
            
            st.dataframe(
                df_report.style.format({
                    "Close Price": "{:.2f} €/$",
                    "Daily Perf": "{:+.2%}",
                    "Volatility (30d)": "{:.2%}",
                    "Max Drawdown (1y)": "{:.2%}"
                }).background_gradient(subset=["Daily Perf"], cmap="RdYlGn", vmin=-0.05, vmax=0.05),
                use_container_width=True,
                hide_index=True
            )
            
            if not df_report.empty and 'Last Update' in df_report.columns:
                last_update = df_report['Last Update'].iloc[0]
                st.caption(f"✅ Last successful run: {last_update}")
        else:
            st.warning("⚠️ No daily report found. The Cron Job might not have run yet (Wait for 8pm).")

elif selected_tab == "🔎 Single Asset Analysis":
    Arthur_page.quant_a_ui()

elif selected_tab == "🌐 Multi Asset Analysis":
    Virgil_page.quant_b_ui()