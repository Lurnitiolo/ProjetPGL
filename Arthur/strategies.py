import pandas as pd
import numpy as np

# --- NOTE IMPORTANTE ---
# Ne JAMAIS faire "from .strategies import..." ici.
# Ce fichier définit les fonctions, il ne doit pas s'importer lui-même.

def apply_strategies(df, ma_window=20):
    """
    Applique les stratégies (Buy & Hold et Momentum) sur le DataFrame.
    
    Args:
        df: DataFrame contenant au moins la colonne 'Close'.
        ma_window: Fenêtre pour la moyenne mobile (paramètre modifiable).
        
    Returns:
        df: DataFrame enrichi avec les signaux et les courbes de valeur.
    """
    if df is None or df.empty:
        return None
    
    # On travaille sur une copie pour ne pas modifier l'original par erreur
    data = df.copy()
    
    # 1. Calcul des rendements journaliers de l'actif (Variation en %)
    data['Returns'] = data['Close'].pct_change()
    
    # --- STRATÉGIE 1 : BUY AND HOLD (Référence) ---
    # On investit au début et on ne touche plus à rien.
    # On initialise à 100 (base 100) pour que ce soit lisible sur le graph.
    data['Strat_BuyHold'] = (1 + data['Returns']).cumprod() * 100

    # --- STRATÉGIE 2 : MOMENTUM (Moyenne Mobile Simple) ---
    # Calcul de la moyenne mobile sur X jours (ma_window)
    data['MA'] = data['Close'].rolling(window=ma_window).mean()
    
    # Signal : 1 (Achat/Long) si le Prix > Moyenne Mobile, sinon 0 (Cash/Neutre)
    data['Signal'] = np.where(data['Close'] > data['MA'], 1, 0)
    
    # --- POINT CRITIQUE : LE SHIFT ---
    # On décale le signal d'un jour (.shift(1)).
    # Pourquoi ? Parce qu'on prend la décision de trading ce soir à la clôture,
    # mais le rendement de cette décision ne s'appliquera que demain.
    # Sans le shift, on "tricherait" en connaissant le futur.
    data['Strat_Momentum_Returns'] = data['Signal'].shift(1) * data['Returns']
    
    # Valeur cumulée de la stratégie Momentum (Base 100)
    data['Strat_Momentum'] = (1 + data['Strat_Momentum_Returns']).cumprod() * 100
    
    # On supprime les premières lignes qui contiennent des NaN (à cause de la moyenne mobile)
    data.dropna(inplace=True)
    
    return data

def calculate_metrics(series):
    """
    Calcule la performance globale et le Max Drawdown.
    
    Args:
        series: Une série temporelle de valeurs cumulées (ex: la colonne 'Strat_Momentum')
        
    Returns:
        total_return (float): Performance totale (ex: 0.12 pour 12%)
        max_drawdown (float): Perte maximale historique (ex: -0.25 pour -25%)
    """
    if len(series) < 2:
        return 0.0, 0.0

    # 1. Performance totale 
    # (Valeur Finale / Valeur Initiale) - 1
    total_return = (series.iloc[-1] / series.iloc[0]) - 1
    
    # 2. Max Drawdown (La pire chute du portefeuille)
    # On calcule le sommet historique atteint à chaque instant (Running Max)
    running_max = series.cummax()
    # On calcule l'écart actuel par rapport à ce sommet
    drawdown = (series - running_max) / running_max
    # Le Max Drawdown est la valeur minimale de ces écarts
    max_drawdown = drawdown.min()
    
    return total_return, max_drawdown