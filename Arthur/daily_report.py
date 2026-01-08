import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

AVAILABLE_ASSETS = {
    # 🇫🇷 FRANCE
    "LVMH (Luxury)": "MC.PA",
    "TotalEnergies (Energy)": "TTE.PA",
    "Airbus (Industrial)": "AIR.PA",
    "BNP Paribas (Banking)": "BNP.PA",
    "Sanofi (Health)": "SAN.PA",
    # 🇺🇸 US TECH
    "NVIDIA (AI Boom)": "NVDA",
    "Tesla (High Vol)": "TSLA",
    "Apple (Quality)": "AAPL",
    "Microsoft (Big Tech)": "MSFT",
    # 🌍 INDICES & COMMODITIES
    "S&P 500 (US Market)": "SPY",
    "MSCI World (Global)": "CW8.PA",
    "Nasdaq 100 (Tech)": "QQQ",
    "Gold (Safe Haven)": "GLD",
    # ₿ CRYPTO
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD"
}

def generate_report():
    print(f"[{datetime.now()}] Démarrage du rapport quotidien...")
    results = []

    for name, ticker in AVAILABLE_ASSETS.items():
        try:
            df = yf.download(ticker, period="1y", progress=False)
            
            if not df.empty:
                last_row = df.iloc[-1]
                open_p = float(last_row['Open'].iloc[0]) if isinstance(last_row['Open'], pd.Series) else float(last_row['Open'])
                close_p = float(last_row['Close'].iloc[0]) if isinstance(last_row['Close'], pd.Series) else float(last_row['Close'])
                
                daily_ret = (close_p - open_p) / open_p

                returns = df['Close'].pct_change().dropna()
                vol_30d = returns.tail(30).std() * np.sqrt(252)
                vol_val = float(vol_30d.iloc[0]) if isinstance(vol_30d, pd.Series) else float(vol_30d)

                rolling_max = df['Close'].cummax()
                drawdown = (df['Close'] - rolling_max) / rolling_max
                mdd = drawdown.min()
                mdd_val = float(mdd.iloc[0]) if isinstance(mdd, pd.Series) else float(mdd)

                results.append({
                    "Asset": name,
                    "Ticker": ticker,
                    "Close Price": close_p,
                    "Daily Perf": daily_ret,
                    "Volatility (30d)": vol_val,
                    "Max Drawdown (1y)": mdd_val,
                    "Last Update": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                print(f"✅ {name} processed.")
            else:
                print(f"⚠️ No data for {name}")

        except Exception as e:
            print(f"❌ Error {name}: {e}")

    df_report = pd.DataFrame(results)

    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, "daily_report_latest.csv")
    df_report.to_csv(file_path, index=False)
    print(f"[{datetime.now()}] Rapport sauvegardé : {file_path}")
    
if __name__ == "__main__":
    generate_report()