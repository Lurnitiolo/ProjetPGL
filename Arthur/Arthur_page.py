import streamlit as st
from .data_loader import load_stock_data
from .strategies import AVAILABLE_ASSETS, apply_strategies

def quant_a_ui():
    st.header("Quant Module A: Univariate Analysis & Backtesting")
    
    # --- Configuration ---
    with st.expander("Asset Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            # Ensure "Other (Manual Input)" exists in your AVAILABLE_ASSETS keys if you use it
            selected_name = st.selectbox("Select Asset", options=list(AVAILABLE_ASSETS.keys()))
            
            if selected_name == "Other (Manual Input)":
                ticker = st.text_input("Yahoo Symbol", value="AIR.PA")
            else:
                ticker = AVAILABLE_ASSETS[selected_name]
        with col2:
            st.info("Interval fixed to '1d' for backtest")
            interval = "1d" 
        with col3:
            period = st.selectbox("Timeframe", ["1y", "2y", "5y", "max"], index=1)

    if st.button(f"Run Analysis on {selected_name}"):
        with st.spinner(f'Calculating for {ticker}...'):
            
            # 1. Data Loading
            df = load_stock_data(ticker, period=period, interval=interval)
            
            if df is not None and not df.empty:
                # 2. Strategy Application
                df_strat, metrics = apply_strategies(df)
                
                # 3. KPI Display
                st.write("### 📊 Performance & Risk")
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                
                kpi1.metric("Sharpe (Buy&Hold)", f"{metrics['BuyHold_Sharpe']:.2f}")
                kpi2.metric("Max Drawdown (B&H)", f"{metrics['BuyHold_MDD']:.2%}")
                kpi3.metric("Sharpe (Momentum)", f"{metrics['Momentum_Sharpe']:.2f}")
                kpi4.metric("Max Drawdown (Mom.)", f"{metrics['Momentum_MDD']:.2%}")
                
                # 4. Strategy Comparison
                st.write("### 📈 Strategy Comparison (Base 100)")
                chart_data = df_strat[['Strat_BuyHold', 'Strat_Momentum']]
                st.line_chart(chart_data)
                
                # 5. Raw Data
                with st.expander("View Raw Data"):
                    st.dataframe(df_strat.tail())
                
            else:
                st.error("Error retrieving data.")