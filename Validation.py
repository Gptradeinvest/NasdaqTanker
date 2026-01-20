import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import itertools
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

styles_to_try = ['seaborn-v0_8-darkgrid', 'seaborn-darkgrid', 'bmh', 'ggplot']
for style in styles_to_try:
    if style in plt.style.available:
        plt.style.use(style)
        break

@dataclass
class GlobalConfig:
    ticker: str = "QQQ"
    start_date: str = "2000-01-01"
    end_date: str = pd.Timestamp.today().strftime('%Y-%m-%d')
    risk_free_rate: float = 0.04

@dataclass
class StrategyParams:
    vol_target: float = 0.15      
    es_lookback: int = 21         
    es_guard_thresh: float = 0.15 
    ma_short: int = 40
    ma_long: int = 136
    leverage: float = 1.80        

@dataclass
class ValidationParams:
    n_trials_dsr: int = 2000      
    wfa_years: int = 15           
    cpcv_groups: int = 6          
    cpcv_test_groups: int = 2     
    purge_days: int = 20          
    embargo_days: int = 20
    mc_sims: int = 100000         
    proj_years: int = 5           
    retro_years: int = 5
    use_sample_weights: bool = True

class StrategyEngine:
    
    @staticmethod
    def rolling_es_fast(arr: np.ndarray, window: int, confidence: float = 0.95) -> np.ndarray:
        if len(arr) < window: return np.zeros_like(arr)
        shape = (arr.size - window + 1, window)
        strides = (arr.strides[0], arr.strides[0])
        windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
        cutoff = int((1 - confidence) * window)
        res = []
        for w in windows:
            sorted_w = np.sort(w)
            tail = sorted_w[:max(1, cutoff)]
            val = -np.mean(tail) if len(tail) > 0 else 0
            res.append(val)
        return np.concatenate((np.zeros(window-1), np.array(res)))

    @classmethod
    def compute_strategy_returns(cls, df: pd.DataFrame, params: StrategyParams, rfr: float) -> pd.Series:
        d = df.copy()
        rets = d['Close'].pct_change().fillna(0)
        
        prev_close = d['Close'].shift(1)
        tr = pd.concat([d['High']-d['Low'], (d['High']-prev_close).abs(), (d['Low']-prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(20).mean()
        vol_annual = (atr / d['Close'].replace(0, np.nan)) * np.sqrt(252)
        vol_annual = vol_annual.fillna(params.vol_target)
        vol_annual_lag = vol_annual.shift(1)

        es_vals = cls.rolling_es_fast(rets.values, params.es_lookback) * np.sqrt(21)
        es_vals = pd.Series(es_vals, index=d.index)
        es_vals_lag = es_vals.shift(1)
        sig_guard = (es_vals_lag <= params.es_guard_thresh).astype(int)
        
        ma_s = d['Close'].rolling(params.ma_short).mean()
        ma_l = d['Close'].rolling(params.ma_long).mean()
        ma_s_lag = ma_s.shift(1)
        ma_l_lag = ma_l.shift(1)
        sig_trend = (ma_s_lag > ma_l_lag).astype(int)
        
        target_exp = params.vol_target / vol_annual_lag
        w_vol = target_exp.clip(upper=params.leverage)
        w_final = (w_vol * sig_guard * sig_trend).fillna(0)
        
        daily_rfr = rfr / 252
        strategy_rets = (w_final * rets) + ((1 - w_final) * daily_rfr)
        
        return strategy_rets

class AFMLTools:
    
    @staticmethod
    def compute_sample_weights(returns: pd.Series, window: int = 63) -> pd.Series:
        """Sample weights basés sur l'unicité temporelle (Prado Ch4)."""
        idx = returns.index
        weights = pd.Series(1.0, index=idx)
        
        for i in range(len(idx)):
            t = idx[i]
            concurrent = idx[(idx >= t - pd.Timedelta(days=window)) & (idx <= t + pd.Timedelta(days=window))]
            weights.loc[t] = 1.0 / len(concurrent)
        
        return weights / weights.sum() * len(weights)
    
    @staticmethod
    def get_train_test_indices(n_groups: int, test_groups: List[int], total_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Indices train/test pour un fold CPCV."""
        group_size = total_size // n_groups
        test_idx = []
        
        for tg in test_groups:
            start = tg * group_size
            end = (tg + 1) * group_size if tg < n_groups - 1 else total_size
            test_idx.extend(range(start, end))
        
        train_idx = [i for i in range(total_size) if i not in test_idx]
        return np.array(train_idx), np.array(test_idx)
    
    @staticmethod
    def apply_purge_embargo(train_idx: np.ndarray, test_idx: np.ndarray, 
                           purge: int, embargo: int) -> Tuple[np.ndarray, np.ndarray]:
        """Applique purge avant test et embargo après test."""
        if len(test_idx) == 0:
            return train_idx, test_idx
        
        test_min, test_max = test_idx.min(), test_idx.max()
        
        # Purge: retirer train indices proches avant test
        purge_zone = range(max(0, test_min - purge), test_min)
        train_idx = np.array([i for i in train_idx if i not in purge_zone])
        
        # Embargo: retirer train indices proches après test
        embargo_zone = range(test_max + 1, min(test_max + 1 + embargo, train_idx.max() + 1) if len(train_idx) > 0 else test_max + 1)
        train_idx = np.array([i for i in train_idx if i not in embargo_zone])
        
        # Purge début du test aussi
        if purge > 0 and len(test_idx) > purge:
            test_idx = test_idx[purge:]
        
        # Embargo fin du test
        if embargo > 0 and len(test_idx) > embargo:
            test_idx = test_idx[:-embargo]
        
        return train_idx, test_idx

class QuantValidator:
    
    def __init__(self, data: pd.DataFrame, g_cfg: GlobalConfig, s_cfg: StrategyParams, v_cfg: ValidationParams):
        self.data = data
        self.g_cfg = g_cfg
        self.s_cfg = s_cfg
        self.v_cfg = v_cfg
        self.oos_series: Optional[pd.Series] = None
        self.cpcv_sharpes: List[float] = []
        self.bootstrap_results: Dict = {}

    def run_wfa(self):
        print(f"[INFO] Démarrage WFA sur {self.v_cfg.wfa_years} ans...")
        end_idx = self.data.index[-1]
        start_wfa = end_idx - pd.DateOffset(years=self.v_cfg.wfa_years)
        
        wfa_rets = []
        curr = start_wfa
        
        while curr < end_idx:
            next_yr = curr + pd.DateOffset(years=1)
            mask = (self.data.index >= curr - pd.DateOffset(days=365)) & (self.data.index < next_yr)
            chunk = self.data.loc[mask]
            
            if len(chunk) > 200:
                res = StrategyEngine.compute_strategy_returns(chunk, self.s_cfg, self.g_cfg.risk_free_rate)
                wfa_rets.append(res.loc[res.index >= curr])
            
            curr = next_yr
            
        self.oos_series = pd.concat(wfa_rets).sort_index()
        self.oos_series = self.oos_series[~self.oos_series.index.duplicated()]
        print(f"[OK] WFA Terminé. {len(self.oos_series)} jours générés.")

    def run_cpcv(self):
        print("[INFO] Démarrage CPCV (AFML complet)...")
        full_rets = StrategyEngine.compute_strategy_returns(self.data, self.s_cfg, self.g_cfg.risk_free_rate)
        n = len(full_rets)
        
        if self.v_cfg.use_sample_weights:
            weights = AFMLTools.compute_sample_weights(full_rets)
        else:
            weights = pd.Series(1.0, index=full_rets.index)
        
        combos = list(itertools.combinations(range(self.v_cfg.cpcv_groups), self.v_cfg.cpcv_test_groups))
        sharpes = []
        
        for combo in combos:
            train_idx, test_idx = AFMLTools.get_train_test_indices(
                self.v_cfg.cpcv_groups, list(combo), n
            )
            
            train_idx, test_idx = AFMLTools.apply_purge_embargo(
                train_idx, test_idx, 
                self.v_cfg.purge_days, 
                self.v_cfg.embargo_days
            )
            
            if len(test_idx) < 50:
                continue
            
            test_rets = full_rets.iloc[test_idx]
            test_weights = weights.iloc[test_idx]
            
            # Sharpe pondéré
            weighted_mean = (test_rets * test_weights).sum() / test_weights.sum()
            weighted_std = np.sqrt(((test_rets - weighted_mean)**2 * test_weights).sum() / test_weights.sum())
            
            annual_ret = weighted_mean * 252
            annual_vol = weighted_std * np.sqrt(252)
            
            sr = (annual_ret - self.g_cfg.risk_free_rate) / annual_vol if annual_vol > 0 else 0
            sharpes.append(sr)
        
        self.cpcv_sharpes = sharpes
        print(f"[OK] CPCV: {len(sharpes)} folds, Sharpe moyen: {np.mean(sharpes):.2f}")

    def calc_psr_dsr(self) -> Tuple[float, float, float]:
        if self.oos_series is None: return 0, 0, 0
        
        ret = self.oos_series
        mu = ret.mean() * 252
        sigma = ret.std() * np.sqrt(252)
        skew = stats.skew(ret)
        kurt = stats.kurtosis(ret, fisher=True)
        
        sr = (mu - self.g_cfg.risk_free_rate) / sigma if sigma > 0 else 0
        n = len(ret)
        
        sr_std = np.sqrt((1/(n/252-1)) * (1 + 0.5*sr**2 - skew*sr + (kurt/4)*sr**2))
        psr = stats.norm.cdf((sr - 0) / sr_std)
        exp_max_sr = np.sqrt(2 * np.log(self.v_cfg.n_trials_dsr)) * (1/np.sqrt(252))
        dsr = stats.norm.cdf((sr - exp_max_sr) / sr_std)
        
        return sr, psr, dsr

    def run_bootstrap(self):
        print(f"[INFO] Bootstrap {self.v_cfg.mc_sims} simulations...")
        if self.oos_series is None: return

        returns_pool = self.oos_series.values
        days_retro = self.v_cfg.retro_years * 252
        days_proj = self.v_cfg.proj_years * 252
        total_days = days_retro + days_proj

        random_indices = np.random.randint(0, len(returns_pool), size=(self.v_cfg.mc_sims, total_days))
        sim_returns = returns_pool[random_indices]
        sim_paths = (1 + sim_returns).cumprod(axis=1)

        retro_paths = sim_paths[:, :days_retro] * 100 
        proj_paths = sim_paths[:, days_retro:]
        proj_paths = proj_paths / proj_paths[:, 0][:, None] * 100

        self.bootstrap_results = {
            'retro': np.percentile(retro_paths, [1, 5, 50, 95, 99], axis=0),
            'proj': np.percentile(proj_paths, [1, 5, 50, 95, 99], axis=0),
            'real_retro': ((1 + self.oos_series.tail(days_retro)).cumprod() / (1 + self.oos_series.tail(days_retro)).iloc[0] * 100)
        }

    def generate_report(self):
        sr, psr, dsr = self.calc_psr_dsr()
        cagr = (1 + self.oos_series).prod()**(252/len(self.oos_series)) - 1
        eq = (1 + self.oos_series).cumprod()
        dd_series = eq / eq.cummax() - 1
        max_dd = dd_series.min()

        print("\n" + "█"*70)
        print(f"  RÉSULTATS VALIDATION AFML : {self.g_cfg.ticker} (Levier {self.s_cfg.leverage}x)")
        print("█"*70)
        print(f"1. PERFORMANCE OOS (WFA {self.v_cfg.wfa_years} ans)")
        print(f"   ► CAGR             : {cagr:.2%}")
        print(f"   ► Max Drawdown     : {max_dd:.2%}")
        print(f"   ► Sharpe Ratio     : {sr:.2f}")
        print("-" * 30)
        print(f"2. FIABILITÉ STATISTIQUE (Prado)")
        print(f"   ► PSR (Prob > 0)   : {psr:.2%} {'✅' if psr > 0.95 else '⚠️'}")
        print(f"   ► DSR (Robustesse) : {dsr:.2%} {'✅' if dsr > 0.50 else '⚠️'}")
        print("-" * 30)
        print(f"3. CPCV (Purge={self.v_cfg.purge_days}d, Embargo={self.v_cfg.embargo_days}d)")
        print(f"   ► Sharpe Moyen     : {np.mean(self.cpcv_sharpes):.2f}")
        print(f"   ► Sharpe Std       : {np.std(self.cpcv_sharpes):.2f}")
        print(f"   ► Sample Weights   : {'Activés' if self.v_cfg.use_sample_weights else 'Désactivés'}")
        print("-" * 30)
        
        median_gain = (self.bootstrap_results['proj'][2, -1]/100) - 1
        worst_case = (self.bootstrap_results['proj'][1, -1]/100) - 1
        print(f"4. PROJECTION 5 ANS (Monte Carlo)")
        print(f"   ► Gain Médian      : +{median_gain:.2%}")
        print(f"   ► VaR 95% (Worst)  : {worst_case:.2%}")
        print("="*70)

        fig = plt.figure(figsize=(18, 12))
        
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot((1+self.oos_series).cumprod()*100, label='Equity (WFA)', color='#2980B9')
        ax1.set_title("1. Courbe OOS Walk-Forward", fontweight='bold')
        ax1.set_yscale('log')
        ax1.yaxis.set_major_formatter(mtick.ScalarFormatter())
        ax1.grid(True, which='both', alpha=0.2)

        ax2 = plt.subplot(2, 2, 2)
        ax2.hist(self.cpcv_sharpes, bins=20, color='#F39C12', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(self.cpcv_sharpes), color='red', ls='--', label=f'Moy: {np.mean(self.cpcv_sharpes):.2f}')
        ax2.set_title("2. Stabilité Sharpe (CPCV AFML)", fontweight='bold')
        ax2.legend()

        ax3 = plt.subplot(2, 2, 3)
        days_r = len(self.bootstrap_results['retro'][0])
        x_r = np.arange(days_r)
        ax3.fill_between(x_r, self.bootstrap_results['retro'][1], self.bootstrap_results['retro'][3], color='gray', alpha=0.2, label='Intervalle 90%')
        ax3.plot(x_r, self.bootstrap_results['retro'][2], color='black', ls='--', label='Médiane Sim')
        ax3.plot(x_r, self.bootstrap_results['real_retro'].values, color='#2980B9', lw=2, label='Réalité')
        ax3.set_title("3. Réalité vs Simulations (Passé)", fontweight='bold')
        ax3.legend()

        ax4 = plt.subplot(2, 2, 4)
        days_p = len(self.bootstrap_results['proj'][0])
        x_p = np.arange(days_p)
        ax4.fill_between(x_p, self.bootstrap_results['proj'][0], self.bootstrap_results['proj'][4], color='#27AE60', alpha=0.1, label='Intervalle 99%')
        ax4.fill_between(x_p, self.bootstrap_results['proj'][1], self.bootstrap_results['proj'][3], color='#27AE60', alpha=0.3, label='Intervalle 90%')
        ax4.plot(x_p, self.bootstrap_results['proj'][2], color='#145A32', lw=2, label='Médiane Future')
        ax4.axhline(100, color='black', ls=':')
        ax4.set_title("4. Projection Probabiliste (5 Ans)", fontweight='bold')
        ax4.legend(loc='upper left')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    g_conf = GlobalConfig()
    s_conf = StrategyParams() 
    v_conf = ValidationParams()

    print(f"Chargement {g_conf.ticker}...")
    try:
        data = yf.download(g_conf.ticker, start=g_conf.start_date, end=g_conf.end_date, progress=False, auto_adjust=True, multi_level_index=False)
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data = data[['Open', 'High', 'Low', 'Close']].dropna()
    except Exception as e:
        print(f"Erreur Download: {e}")
        exit()

    validator = QuantValidator(data, g_conf, s_conf, v_conf)
    
    validator.run_wfa()
    validator.run_cpcv()
    validator.run_bootstrap()
    validator.generate_report()
