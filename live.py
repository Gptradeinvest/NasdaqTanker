import time
import os
import sys
import json
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

# ==========================================
# 1. CONFIGURATION GLOBALE
# ==========================================
REFRESH_RATE = 60  # Durée du cycle en secondes (1 minute)
LOG_FILE = "goldilocks_activity.log"
DB_FILE = "portfolio_ledger.json"

@dataclass
class StrategyConfig:
    """Paramètres de la stratégie Goldilocks."""
    ticker_signal: str = "QQQ"
    universe: List[str] = field(default_factory=lambda: ["QQQ", "QLD", "BIL"])
    ma_short: int = 40
    ma_long: int = 136
    vol_target: float = 0.15
    lev_max: float = 1.80
    guard_thresh: float = 0.15
    es_lookback: int = 21
    initial_capital: float = 10_000.0
    min_trade_amount: float = 100.0

@dataclass
class PortfolioState:
    """État comptable du portefeuille."""
    cash: float
    holdings: Dict[str, int]
    equity: float
    last_update: str

# ==========================================
# 2. OUTILS & LOGGING (SYSTEME)
# ==========================================
def setup_logging():
    """Configure les logs pour supporter l'UTF-8 sur Windows et écrire dans un fichier."""
    # Force UTF-8 pour la console Windows
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    # Logger racine
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Évite les doublons de handlers si on relance le script
    if logger.handlers:
        logger.handlers = []

    # 1. File Handler (UTF-8 forcé)
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(file_handler)

    # Note: On ne met PAS de StreamHandler (Console) ici car on veut contrôler
    # l'affichage manuellement dans la boucle Dashboard.

class QuantUtils:
    """Mathématiques financières vectorisées."""
    @staticmethod
    def rolling_es(returns: np.ndarray, window: int, confidence: float = 0.95) -> np.ndarray:
        if len(returns) < window: return np.zeros_like(returns)
        shape = (returns.size - window + 1, window)
        strides = (returns.strides[0], returns.strides[0])
        windows = np.lib.stride_tricks.as_strided(returns, shape=shape, strides=strides)
        cutoff_idx = int((1 - confidence) * window)
        es_values = []
        for w in windows:
            sorted_w = np.sort(w)
            tail = sorted_w[:max(1, cutoff_idx)]
            val = -np.mean(tail) if len(tail) > 0 else 0
            es_values.append(val)
        return np.concatenate((np.zeros(window - 1), np.array(es_values)))

# ==========================================
# 3. COMPOSANTS MÉTIER
# ==========================================
class DataProvider:
    def __init__(self, config: StrategyConfig):
        self.cfg = config

    def fetch_market_data(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Récupère historique QQQ et prix actuels."""
        tickers = list(set([self.cfg.ticker_signal] + self.cfg.universe))
        try:
            # Téléchargement silencieux
            raw = yf.download(tickers, period="2y", progress=False, auto_adjust=True, group_by='ticker')
            
            last_prices = {}
            if len(tickers) == 1:
                df = raw
                last_prices[tickers[0]] = df['Close'].iloc[-1]
                df_signal = df
            else:
                for t in tickers:
                    if t in raw:
                        df_t = raw[t].dropna()
                        last_prices[t] = df_t['Close'].iloc[-1]
                
                # Extraction Signal
                df_signal = raw[self.cfg.ticker_signal] if self.cfg.ticker_signal in raw else raw

            # Nettoyage
            if isinstance(df_signal.columns, pd.MultiIndex):
                df_signal = df_signal.xs(self.cfg.ticker_signal, level=0, axis=1)
            
            df_signal = df_signal[['Open', 'High', 'Low', 'Close']].dropna()
            return df_signal, last_prices
        except Exception as e:
            logging.error(f"Erreur Data: {e}")
            raise

class GoldilocksEngine:
    def __init__(self, config: StrategyConfig):
        self.cfg = config

    def generate_signal(self, df: pd.DataFrame) -> float:
        d = df.copy()
        # Moyennes Mobiles
        d['MA_S'] = d['Close'].rolling(self.cfg.ma_short).mean()
        d['MA_L'] = d['Close'].rolling(self.cfg.ma_long).mean()
        
        # Volatilité
        prev_close = d['Close'].shift(1)
        tr = pd.concat([d['High']-d['Low'], (d['High']-prev_close).abs(), (d['Low']-prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(20).mean()
        vol_ann = (atr / d['Close']) * np.sqrt(252)
        
        # ES Guard
        rets = d['Close'].pct_change().fillna(0)
        es_vals = QuantUtils.rolling_es(rets.values, self.cfg.es_lookback) * np.sqrt(21)

        last = d.iloc[-1]
        last_es = es_vals[-1]
        curr_vol = vol_ann.iloc[-1] if not pd.isna(vol_ann.iloc[-1]) else 0.15

        # Logique Décisionnelle
        trend_ok = last['MA_S'] > last['MA_L']
        guard_ok = last_es < self.cfg.guard_thresh
        
        if trend_ok and guard_ok:
            raw_lev = self.cfg.vol_target / curr_vol
            final_lev = min(raw_lev, self.cfg.lev_max)
            status = "RISK_ON"
        else:
            final_lev = 0.0
            status = "RISK_OFF"

        logging.info(f"🧠 ANALYSE: {status} | Trend={'UP' if trend_ok else 'DOWN'} | ES={last_es:.1%} | Vol={curr_vol:.1%} | Levier={final_lev:.2f}")
        return final_lev

    def compute_allocation(self, leverage: float) -> Dict[str, float]:
        alloc = {t: 0.0 for t in self.cfg.universe}
        if leverage <= 0.01:
            alloc["BIL"] = 1.0
        elif leverage > 1.0:
            w_qld = min(leverage - 1, 1.0)
            alloc["QLD"] = w_qld
            alloc["QQQ"] = 1.0 - w_qld
        else:
            alloc["QQQ"] = leverage
            alloc["BIL"] = 1.0 - leverage
        return alloc

class PortfolioManager:
    def __init__(self, config: StrategyConfig):
        self.cfg = config
        self.file_path = Path(DB_FILE)
        self.state = self._load_or_create()

    def _load_or_create(self) -> PortfolioState:
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                return PortfolioState(**data)
            except: pass
        return PortfolioState(self.cfg.initial_capital, {t: 0 for t in self.cfg.universe}, self.cfg.initial_capital, datetime.now().isoformat())

    def save(self):
        with open(self.file_path, 'w') as f:
            json.dump(asdict(self.state), f, indent=4)

    def execute_rebalance(self, target_alloc: Dict[str, float], prices: Dict[str, float]):
        # Mark to Market
        val_holdings = sum(self.state.holdings.get(t, 0) * p for t, p in prices.items())
        self.state.equity = self.state.cash + val_holdings
        
        logging.info(f"💰 Equity: ${self.state.equity:.2f} (Cash: ${self.state.cash:.2f})")

        for ticker, weight in target_alloc.items():
            if ticker not in prices: continue
            
            price = prices[ticker]
            qty_curr = self.state.holdings.get(ticker, 0)
            target_val = self.state.equity * weight
            diff = target_val - (qty_curr * price)

            if abs(diff) < self.cfg.min_trade_amount: continue
            
            qty_delta = int(diff / price)
            if qty_delta == 0: continue
            
            cost = qty_delta * price
            
            # Vérif Cash Achat
            if qty_delta > 0 and cost > self.state.cash:
                qty_delta = int(self.state.cash / price)
                cost = qty_delta * price
            
            self.state.cash -= cost
            self.state.holdings[ticker] = qty_curr + qty_delta
            
            action = "ACHAT" if qty_delta > 0 else "VENTE"
            logging.info(f"⚡ ORDRE: {action} {abs(qty_delta)} {ticker} @ ${price:.2f}")
        
        self.state.last_update = datetime.now().isoformat()
        self.save()

# ==========================================
# 4. ORCHESTRATEUR (CYCLE UNIQUE)
# ==========================================
def run_trading_cycle():
    """Exécute une itération complète de la stratégie."""
    cfg = StrategyConfig()
    data_prov = DataProvider(cfg)
    engine = GoldilocksEngine(cfg)
    pm = PortfolioManager(cfg)

    try:
        # 1. Data
        df_sig, prices = data_prov.fetch_market_data()
        # 2. Logic
        lev = engine.generate_signal(df_sig)
        alloc = engine.compute_allocation(lev)
        # 3. Execution
        pm.execute_rebalance(alloc, prices)
    except Exception as e:
        logging.error(f"Erreur Cycle: {e}")

# ==========================================
# 5. BOUCLE LIVE & DASHBOARD
# ==========================================
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_dashboard():
    """Affiche l'état actuel en lisant le JSON et les logs."""
    clear_screen()
    print("="*60)
    print(f"🤖 GOLDILOCKS BOT | LIVE PAPER TRADING | {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)

    # 1. Affichage Portefeuille
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r') as f:
                data = json.load(f)
            print(f"💰 CAPITAL TOTAL : ${data['equity']:,.2f}")
            print(f"💵 CASH DISPO   : ${data['cash']:,.2f}")
            print("-" * 60)
            print("POSITIONS ACTUELLES :")
            holdings = {k: v for k, v in data['holdings'].items() if v > 0}
            if holdings:
                for k, v in holdings.items():
                    print(f"   ► {k:<5} : {v} parts")
            else:
                print("   (100% Cash - Aucune position)")
        except:
            print("   (Erreur lecture portefeuille)")
    else:
        print("   (Initialisation...)")

    print("-" * 60)
    print("📜 JOURNAL D'ACTIVITÉ (Derniers événements) :")
    
    # 2. Affichage Logs (Tail)
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Affiche les 7 dernières lignes pertinentes
                for line in lines[-7:]:
                    print(f"   {line.strip()}")
        except:
            print("   (Logs vides)")
    
    print("="*60)

def main_loop():
    setup_logging()
    
    while True:
        # 1. Mise à jour Trading
        run_trading_cycle()
        
        # 2. Mise à jour Affichage
        print_dashboard()
        
        # 3. Attente Visuelle
        print(f"\n⏳ Prochain cycle dans {REFRESH_RATE} secondes...")
        
        # Barre de progression simple
        for _ in range(REFRESH_RATE):
            time.sleep(1)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n🛑 Arrêt du Bot.")