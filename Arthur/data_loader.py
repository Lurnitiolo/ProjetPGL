import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)
def load_stock_data(ticker, period="1y", interval="1d"):
    """
    Télécharge les données depuis Yahoo Finance.
    Gère le MultiIndex et les Timezones pour éviter les bugs.
    """
    try:
        # Téléchargement
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            prepost=False,
            threads=True,
            progress=False
        )
        
        if df.empty:
            return None

        # Nettoyage MultiIndex (yfinance récent)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Index en datetime sans timezone
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Colonnes requises
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Gestion Adj Close si Close absent (rare mais possible)
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df.rename(columns={'Adj Close': 'Close'}, inplace=True)

        if not all(col in df.columns for col in required_cols):
            return None
            
        return df[required_cols]

    except Exception as e:
        st.error(f"Erreur téléchargement {ticker}: {e}")
        return None