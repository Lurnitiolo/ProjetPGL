import pandas as pd
import numpy as np


AVAILABLE_ASSETS = {
    "Bitcoin (USD)": "BTC-USD",
    "Ethereum (USD)": "ETH-USD",
    "S&P 500 (Indice US)": "^GSPC",
    "CAC 40 (France)": "^FCHI",
    "Euro / Dollar": "EURUSD=X",
    "Or (Gold)": "GC=F",
    "Apple": "AAPL",
    "NVIDIA": "NVDA",
    "Tesla": "TSLA",
    "TotalEnergies": "TTE.PA",
    "Autre (Saisir manuellement)": "CUSTOM"
}