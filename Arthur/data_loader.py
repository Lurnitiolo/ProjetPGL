import yfinance as yf
import pandas as pd

def load_stock_data(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df

    except Exception as e:
        print(f"Error during the chargement : {e}")
        return None