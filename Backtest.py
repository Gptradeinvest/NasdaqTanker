import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

styles_to_try = ['seaborn-v0_8-darkgrid', 'seaborn-darkgrid', 'bmh', 'ggplot']
for style in styles_to_try:
    if style in plt.style.available:
        plt.style.use(style)
        break

@dataclass
class StrategyConfig:
    ticker: str = "QQQ"
    start_date: str = "2005-01-01"
    end_date: str = pd.Timestamp.today().strftime('%Y-%m-%d')
    risk_free_rate: float = 0.035
    vol_target: float = 0.15
    es_lookback: int = 21
    es_guard_thresh: float = 0.15
    ma_short: int = 40
    ma_long: int = 136
    leverage_cap: float = 1.80

class GoldilocksStrategy:
    def __init__(self, config: StrategyConfig):
        self.cfg = config
        self.data: Optional[pd.DataFrame] = None
        self.results: Optional[pd.DataFrame] = None
        self.metrics: Dict = {}

    def load_data(self) -> None:
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

        if isinstance(raw_data.columns, pd.MultiIndex):
            try:
                raw_data.columns = raw_data.columns.get_level_values(0)
            except IndexError:
                pass

        self.data = raw_data[['Open', 'High', 'Low', 'Close']].dropna().copy()
        
        if self.data.empty:
            raise ValueError(f"ERREUR: Aucune donnée récupérée pour {self.cfg.ticker}.")
        
        print(f"SUCCÈS: {len(self.data)} jours de trading chargés.")

    @staticmethod
    def _calculate_rolling_es(returns: pd.Series, window: int, confidence: float = 0.95) -> pd.Series:
        """Expected Shortfall rolling strict (point-in-time uniquement)."""
        cutoff = int((1 - confidence) * window)
        
        def compute_es(window_data):
            if len(window_data) < window:
                return np.nan
            sorted_rets = np.sort(window_data)
            tail = sorted_rets[:max(1, cutoff)]
            return -np.mean(tail) if len(tail) > 0 else 0
        
        return returns.rolling(window).apply(compute_es, raw=True)

    def run_backtest(self) -> None:
        if self.data is None:
            self.load_data()

        df = self.data.copy()
        
        # Rendements J
        rets = df['Close'].pct_change()
        
        # Volatilité (ATR) - Calcul sur données J, utilisation J+1
        prev_close = df['Close'].shift(1)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - prev_close).abs()
        low_close = (df['Low'] - prev_close).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(20).mean()
        vol_est = (atr / df['Close'].replace(0, np.nan)) * np.sqrt(252)
        vol_est = vol_est.fillna(0.15)
        vol_est_lagged = vol_est.shift(1)  # Connaissance J-1 pour décision J
        
        # Expected Shortfall - Calcul sur données J, utilisation J+1
        es_vals = self._calculate_rolling_es(rets, self.cfg.es_lookback) * np.sqrt(21)
        es_vals_lagged = es_vals.shift(1)  # Connaissance J-1 pour décision J
        sig_guard = (es_vals_lagged <= self.cfg.es_guard_thresh).astype(int)
        
        # Trend Following - Calcul sur Close J, utilisation J+1
        ma_s = df['Close'].rolling(self.cfg.ma_short).mean()
        ma_l = df['Close'].rolling(self.cfg.ma_long).mean()
        ma_s_lagged = ma_s.shift(1)  # Position des MAs connue à J-1
        ma_l_lagged = ma_l.shift(1)
        sig_trend = (ma_s_lagged > ma_l_lagged).astype(int)
        
        # Allocation - Basée sur vol estimée J-1
        raw_weight = (self.cfg.vol_target / vol_est_lagged)
        capped_weight = raw_weight.clip(upper=self.cfg.leverage_cap)
        
        # Poids final pour trading J (basé sur infos J-1)
        final_weight = (capped_weight * sig_guard * sig_trend).fillna(0)
        
        # Performance - Rendement J appliqué avec poids décidé sur infos J-1
        daily_rfr = self.cfg.risk_free_rate / 252
        strat_ret = (final_weight * rets) + ((1 - final_weight) * daily_rfr)

        self.results = pd.DataFrame({
            'Market_Rets': rets,
            'Strat_Rets': strat_ret,
            'Eq_Curve': (1 + strat_ret).cumprod() * 100,
            'Benchmark': (1 + rets).cumprod() * 100,
            'Weight': final_weight,
            'Drawdown': np.nan
        })
        
        cum_max = self.results['Eq_Curve'].cummax()
        self.results['Drawdown'] = (self.results['Eq_Curve'] / cum_max) - 1

    def compute_metrics(self) -> None:
        if self.results is None:
            return

        r = self.results['Strat_Rets'].dropna()
        total_days = len(r)
        years = total_days / 252
        
        total_ret = self.results['Eq_Curve'].iloc[-1] / 100
        cagr = (total_ret ** (1 / years)) - 1
        vol = r.std() * np.sqrt(252)
        sharpe = (cagr - self.cfg.risk_free_rate) / vol
        max_dd = self.results['Drawdown'].min()
        
        downside_returns = r[r < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = (cagr - self.cfg.risk_free_rate) / downside_vol if downside_vol > 0 else 0
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
        if self.results is None:
            return

        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.results.index, self.results['Eq_Curve'], label='Stratégie (Goldilocks)', color='#2E86C1', linewidth=1.5)
        ax1.plot(self.results.index, self.results['Benchmark'], label=f'Benchmark ({self.cfg.ticker})', color='#95A5A6', alpha=0.6, linewidth=1)
        ax1.set_yscale('log')
        ax1.set_ylabel('Valeur Portefeuille (Base 100)')
        ax1.set_title(f'Performance Historique ({self.cfg.ticker}) - Échelle Logarithmique', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        ax1.yaxis.set_major_formatter(mtick.ScalarFormatter())

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(self.results.index, self.results['Drawdown'], color='#C0392B', linewidth=1)
        ax2.fill_between(self.results.index, self.results['Drawdown'], 0, color='#C0392B', alpha=0.15)
        ax2.axhline(-0.15, color='black', linestyle='--', linewidth=0.8, label='Seuil Alerte -15%')
        ax2.set_ylabel('Drawdown')
        ax2.set_title('Profil de Risque (Drawdown)', fontweight='bold')
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax2.legend(loc='lower right')

        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(self.results.index, self.results['Weight'], color='#27AE60', linewidth=1, label='Levier Utilisé')
        ax3.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
        ax3.set_ylabel('Levier (x)')
        ax3.set_title("Exposition Dynamique & Gestion du Levier", fontweight='bold')
        ax3.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    config = StrategyConfig(
        ticker="QQQ",
        vol_target=0.15,
        leverage_cap=1.80,
        es_guard_thresh=0.15,
        risk_free_rate=0.035
    )

    strategy = GoldilocksStrategy(config)
    
    try:
        strategy.run_backtest()
        strategy.print_tearsheet()
        strategy.plot_performance()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
