import yfinance as yf
import pandas as pd
import streamlit as st

def get_logo_url(ticker):
    # Mapping manuel pour les domaines boursiers (mis à jour)
    domains = {
        "AAPL": "apple.com", 
        "MSFT": "microsoft.com", 
        "GOOGL": "google.com",
        "AMZN": "amazon.com", 
        "TSLA": "tesla.com", 
        "NVDA": "nvidia.com",
        "MC.PA": "lvmh.com",
        "TTE.PA": "totalenergies.com",
        "AIR.PA": "airbus.com",
        "BNP.PA": "group.bnpparibas",
        "SAN.PA": "sanofi.com",
        "SPY": "ssga.com",        # State Street (S&P 500)
        "CW8.PA": "amundi.com",   # Amundi (MSCI World)
        "QQQ": "invesco.com",     # Invesco (Nasdaq 100)
        "EURUSD=X": "ecb.europa.eu"
    }
    
    # 1. Gestion des Cryptos (BTC-USD, ETH-USD)
    if "-USD" in ticker:
        symbol = ticker.split("-")[0].lower()
        return f"https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/{symbol}.png"
    
    # 2. Gestion de l'Or (Gold) - Supporte GC=F et GLD
    if ticker in ["GC=F", "GLD"]:
        return "https://cdn-icons-png.flaticon.com/512/272/272530.png"
    
    # 3. Utilisation du service Google Favicon pour les actions et ETFs
    domain = domains.get(ticker)
    if domain:
        # sz=128 permet d'obtenir la meilleure résolution possible via Google
        return f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
    
    return None

    



def load_stock_data(ticker, period="1y", interval="1d"):
    """
    Charge les données historiques pour un actif donné.
    
    Args:
        ticker (str): Le symbole de l'actif (ex: 'AAPL', 'BTC-USD', 'EURUSD=X').
        period (str): La période d'historique (ex: '1mo', '1y', 'max').
        interval (str): L'intervalle des données (ex: '1d', '1h', '5m').
    
    Returns:
        pd.DataFrame: DataFrame nettoyé avec Date et Close.
    """
    try:
        # Téléchargement des données
        print(f"Chargement des données pour {ticker}...")
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # Vérification si des données ont été trouvées
        if df.empty:
            print(f"Erreur : Aucune donnée trouvée pour le ticker {ticker}.")
            return None
            
        # Nettoyage simple : on garde l'essentiel pour le backtest
        # Parfois yfinance renvoie un MultiIndex, on simplifie si besoin
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = df.columns.get_level_values(0)

        # On s'assure que les colonnes sont propres
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"Succès : {len(df)} lignes récupérées.")
        return df

    except Exception as e:
        print(f"Erreur critique lors du chargement : {e}")
        return None

# --- Zone de Test (ne s'exécute que si on lance ce fichier directement) ---
if __name__ == "__main__":
    # Test avec Bitcoin
    test_df = load_stock_data("BTC-USD", period="1mo", interval="1d")
    if test_df is not None:
        print(test_df.head())