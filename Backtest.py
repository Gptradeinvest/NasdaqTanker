import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# ==========================================
# 0. CONFIGURATION GRAPHIQUE (ROBUSTE)
# ==========================================
# Sélectionne le meilleur style "Pro" disponible selon votre version de Matplotlib
styles_to_try = ['seaborn-v0_8-darkgrid', 'seaborn-darkgrid', 'bmh', 'ggplot']
for style in styles_to_try:
    if style in plt.style.available:
        plt.style.use(style)
        break

# ==========================================
# 1. CONFIGURATION & DATA CLASSES
# ==========================================

@dataclass
class StrategyConfig:
    """Configuration des paramètres de la stratégie (Immutable)."""
    ticker: str = "QQQ"
    start_date: str = "2005-01-01"
    end_date: str = pd.Timestamp.today().strftime('%Y-%m-%d')
    risk_free_rate: float = 0.035  # Taux sans risque moyen (3.5%)
    vol_target: float = 0.15       # Cible de volatilité (15%)
    es_lookback: int = 21          # Fenêtre pour l'Expected Shortfall
    es_guard_thresh: float = 0.15  # Seuil de déclenchement du Guard
    ma_short: int = 40             # Moyenne mobile courte
    ma_long: int = 136             # Moyenne mobile longue
    leverage_cap: float = 1.80     # Levier maximum autorisé

# ==========================================
# 2. MOTEUR DE STRATÉGIE (CLASS)
# ==========================================

class GoldilocksStrategy:
    """
    Moteur de backtesting Corporate Finance :
    Volatility Targeting + Trend Filter + ES Risk Guard.
    """

    def __init__(self, config: StrategyConfig):
        self.cfg = config
        self.data: Optional[pd.DataFrame] = None
        self.results: Optional[pd.DataFrame] = None
        self.metrics: Dict = {}

    def load_data(self) -> None:
        """Récupération et nettoyage des données de marché via Yahoo Finance."""
        print(f"INFO: Chargement des données pour {self.cfg.ticker} ({self.cfg.start_date} - {self.cfg.end_date})...")
        
        try:
            raw_data = yf.download(
                self.cfg.ticker, 
                start=self.cfg.start_date, 
                end=self.cfg.end_date, 
                progress=False, 
                auto_adjust=True,
                multi_level_index=False
            )
        except Exception as e:
            raise ConnectionError(f"Erreur de connexion Yahoo Finance: {e}")

        # Gestion robuste des MultiIndex (changement récent API yfinance)
        if isinstance(raw_data.columns, pd.MultiIndex):
            try:
                raw_data.columns = raw_data.columns.get_level_values(0)
            except IndexError:
                pass # Structure déjà plate

        # Sélection et nettoyage strict
        self.data = raw_data[['Open', 'High', 'Low', 'Close']].dropna().copy()
        
        if self.data.empty:
            raise ValueError(f"ERREUR: Aucune donnée récupérée pour {self.cfg.ticker}.")
        
        print(f"SUCCÈS: {len(self.data)} jours de trading chargés.")

    @staticmethod
    def _calculate_rolling_es(returns: np.ndarray, window: int, confidence: float = 0.95) -> np.ndarray:
        """
        Calcul vectorisé de l'Expected Shortfall (CVaR) glissant.
        Optimisé pour la performance (similaire numpy stride tricks).
        """
        if len(returns) < window:
            return np.zeros_like(returns)
        
        # Création d'une vue fenêtrée du tableau (mémoire efficace)
        shape = (returns.size - window + 1, window)
        strides = (returns.strides[0], returns.strides[0])
        windows = np.lib.stride_tricks.as_strided(returns, shape=shape, strides=strides)
        
        # Seuil de coupure pour la queue de distribution
        cutoff_idx = int((1 - confidence) * window)
        
        es_values = []
        for w in windows:
            sorted_w = np.sort(w)
            # On prend les pires rendements (tail)
            tail = sorted_w[:max(1, cutoff_idx)]
            val = -np.mean(tail) if len(tail) > 0 else 0
            es_values.append(val)
            
        # Padding initial pour aligner avec l'index original
        return np.concatenate((np.zeros(window - 1), np.array(es_values)))

    def run_backtest(self) -> None:
        """Exécution du pipeline de calcul des signaux et de la performance."""
        if self.data is None:
            self.load_data()

        df = self.data.copy()
        
        # --- A. Volatilité (ATR Based) ---
        prev_close = df['Close'].shift(1)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - prev_close).abs()
        low_close = (df['Low'] - prev_close).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(20).mean()
        
        # Estimation Volatilité Annualisée
        vol_est = (atr / df['Close'].replace(0, np.nan)) * np.sqrt(252)
        vol_est = vol_est.fillna(0.15) # Fallback conservateur

        # --- B. Expected Shortfall (Risk Guard) ---
        rets = df['Close'].pct_change().fillna(0)
        es_vals = self._calculate_rolling_es(rets.values, self.cfg.es_lookback) * np.sqrt(21) # Mensualisé approx
        
        # Signal Guard (1 = Invest, 0 = Cash)
        # Si ES > Threshold, risque de krach détecté -> On coupe.
        sig_guard = np.where(es_vals > self.cfg.es_guard_thresh, 0, 1)

        # --- C. Trend Following (MA Crossover) ---
        ma_s = df['Close'].rolling(self.cfg.ma_short).mean()
        ma_l = df['Close'].rolling(self.cfg.ma_long).mean()
        sig_trend = np.where(ma_s > ma_l, 1, 0)

        # --- D. Allocation & Levier ---
        # Target Weight = Vol Target / Realized Vol
        raw_weight = (self.cfg.vol_target / vol_est)
        # Cap du levier (Risk Management)
        capped_weight = raw_weight.clip(upper=self.cfg.leverage_cap)
        
        # Application des filtres (Guard & Trend) et Lag J+1 (Trading en Open/Close next day)
        final_weight = pd.Series(
            (capped_weight * sig_guard * sig_trend).shift(1), 
            index=df.index
        ).fillna(0)

        # --- E. Attribution de Performance ---
        # Rendement Stratégie = (Poids * Market) + ((1 - Poids) * Taux Sans Risque)
        daily_rfr = self.cfg.risk_free_rate / 252
        strat_ret = (final_weight * rets) + ((1 - final_weight) * daily_rfr)

        # Stockage des résultats
        self.results = pd.DataFrame({
            'Market_Rets': rets,
            'Strat_Rets': strat_ret,
            'Eq_Curve': (1 + strat_ret).cumprod() * 100,
            'Benchmark': (1 + rets).cumprod() * 100,
            'Weight': final_weight,
            'Drawdown': np.nan # Placeholder
        })
        
        # Calcul Drawdown
        cum_max = self.results['Eq_Curve'].cummax()
        self.results['Drawdown'] = (self.results['Eq_Curve'] / cum_max) - 1

    def compute_metrics(self) -> None:
        """Calcul des KPIs financiers pour le reporting."""
        if self.results is None:
            return

        r = self.results['Strat_Rets']
        total_days = len(r)
        years = total_days / 252
        
        # CAGR
        total_ret = self.results['Eq_Curve'].iloc[-1] / 100
        cagr = (total_ret ** (1 / years)) - 1
        
        # Volatilité
        vol = r.std() * np.sqrt(252)
        
        # Sharpe (Risk Free Rate ajusté)
        sharpe = (cagr - self.cfg.risk_free_rate) / vol
        
        # Max Drawdown
        max_dd = self.results['Drawdown'].min()
        
        # Sortino (Downside Volatility)
        downside_returns = r[r < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = (cagr - self.cfg.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Calmar Ratio
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        self.metrics = {
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Max Drawdown": max_dd,
            "Total Return Multiple": total_ret
        }

    def print_tearsheet(self):
        """Affiche un rapport textuel formaté style 'Terminal Bloomberg'."""
        if not self.metrics:
            self.compute_metrics()
            
        print("\n" + "="*60)
        print(f" RAPPORT DE PERFORMANCE: {self.cfg.ticker} STRATEGY")
        print("="*60)
        print(f" Période          : {self.cfg.start_date} -> {self.cfg.end_date}")
        print(f" Paramètres       : VolTarget={self.cfg.vol_target:.0%}, Levier Max={self.cfg.leverage_cap}x")
        print("-" * 60)
        print(f" CAGR             : {self.metrics['CAGR']:>10.2%}")
        print(f" Volatilité (Ann) : {self.metrics['Volatility']:>10.2%}")
        print(f" Max Drawdown     : {self.metrics['Max Drawdown']:>10.2%}")
        print("-" * 60)
        print(f" Sharpe Ratio     : {self.metrics['Sharpe Ratio']:>10.2f}")
        print(f" Sortino Ratio    : {self.metrics['Sortino Ratio']:>10.2f}")
        print(f" Calmar Ratio     : {self.metrics['Calmar Ratio']:>10.2f}")
        print("-" * 60)
        print(f" Multiple Final   : {self.metrics['Total Return Multiple']:>10.2f}x Capital Initial")
        print("="*60 + "\n")

    def plot_performance(self):
        """Génération des graphiques professionnels."""
        if self.results is None:
            return

        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

        # 1. Equity Curve (Log Scale)
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.results.index, self.results['Eq_Curve'], label='Stratégie (Goldilocks)', color='#2E86C1', linewidth=1.5)
        ax1.plot(self.results.index, self.results['Benchmark'], label=f'Benchmark ({self.cfg.ticker})', color='#95A5A6', alpha=0.6, linewidth=1)
        ax1.set_yscale('log')
        ax1.set_ylabel('Valeur Portefeuille (Base 100)')
        ax1.set_title(f'Performance Historique ({self.cfg.ticker}) - Échelle Logarithmique', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        
        # Formattage Axe Y
        ax1.yaxis.set_major_formatter(mtick.ScalarFormatter())

        # 2. Underwater Plot (Drawdowns)
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(self.results.index, self.results['Drawdown'], color='#C0392B', linewidth=1)
        ax2.fill_between(self.results.index, self.results['Drawdown'], 0, color='#C0392B', alpha=0.15)
        ax2.axhline(-0.15, color='black', linestyle='--', linewidth=0.8, label='Seuil Alerte -15%')
        ax2.set_ylabel('Drawdown')
        ax2.set_title('Profil de Risque (Drawdown)', fontweight='bold')
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax2.legend(loc='lower right')

        # 3. Leverage / Exposition
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(self.results.index, self.results['Weight'], color='#27AE60', linewidth=1, label='Levier Utilisé')
        ax3.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
        ax3.set_ylabel('Levier (x)')
        ax3.set_title("Exposition Dynamique & Gestion du Levier", fontweight='bold')
        ax3.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

# ==========================================
# 3. EXÉCUTION (MAIN)
# ==========================================

if __name__ == "__main__":
    # Initialisation de la configuration
    config = StrategyConfig(
        ticker="QQQ",
        vol_target=0.15,
        leverage_cap=1.80,
        es_guard_thresh=0.15,
        risk_free_rate=0.035
    )

    # Instanciation et exécution
    strategy = GoldilocksStrategy(config)
    
    try:
        strategy.run_backtest()
        strategy.print_tearsheet()
        strategy.plot_performance()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")