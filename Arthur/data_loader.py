import yfinance as yf
import pandas as pd

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