import yfinance as yf
import pandas as pd

def load_stock_data(ticker, period="1y", interval="1d"):
    """
    Charge les données boursières et nettoie le format MultiIndex de yfinance.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, multi_level_index=False)
        
        if df.empty:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            # Parfois 'Adj Close' remplace 'Close', on gère ce cas
            if 'Adj Close' in df.columns:
                df.rename(columns={'Adj Close': 'Close'}, inplace=True)
            else:
                return None

        df = df[required_cols]
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df.dropna(inplace=True)
        
        return df

    except Exception as e:
        print(f"Erreur de chargement pour {ticker} : {e}")
        return None