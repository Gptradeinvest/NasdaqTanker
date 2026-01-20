import time
import os
import sys
import json
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime, time as dt_time
from pathlib import Path
from enum import Enum
import traceback

# File locking cross-platform
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    try:
        import msvcrt
        HAS_MSVCRT = True
    except ImportError:
        HAS_MSVCRT = False

# ==========================================
# CONFIGURATION
# ==========================================
REFRESH_RATE = 300
LOG_FILE = "goldilocks_activity.log"
DB_FILE = "portfolio_ledger.json"
LOCK_FILE = "portfolio.lock"
ERROR_THRESHOLD = 5
SLIPPAGE_BPS = 10
COMMISSION = 1.0

class MarketHours:
    OPEN = dt_time(9, 30)
    CLOSE = dt_time(16, 0)
    TRADING_DAYS = [0, 1, 2, 3, 4]

class TradingMode(Enum):
    LIVE = "live"
    BACKTEST = "backtest"
    DRY_RUN = "dry_run"

@dataclass
class StrategyConfig:
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
    min_vol: float = 0.01
    mode: TradingMode = TradingMode.DRY_RUN

@dataclass
class PortfolioState:
    cash: float
    holdings: Dict[str, int]
    equity: float
    last_update: str
    trade_count: int = 0
    error_count: int = 0

@dataclass
class Trade:
    timestamp: str
    ticker: str
    action: str
    quantity: int
    price: float
    cost: float
    commission: float

@dataclass
class PerformanceMetrics:
    equity: float
    total_return: float
    sharpe: float
    max_drawdown: float
    trade_count: int

# ==========================================
# LOGGING
# ==========================================
def setup_logging():
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers = []

    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    ))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    console_handler.setLevel(logging.WARNING)
    logger.addHandler(console_handler)

    return logger

logger = logging.getLogger(__name__)

# ==========================================
# FILE LOCK CROSS-PLATFORM
# ==========================================
class FileLock:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filepath, 'a')
        
        if HAS_FCNTL:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
        elif HAS_MSVCRT:
            msvcrt.locking(self.file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            logger.warning("File locking non disponible sur cette plateforme")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            if HAS_FCNTL:
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            elif HAS_MSVCRT:
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            
            self.file.close()

# ==========================================
# UTILS
# ==========================================
class QuantUtils:
    @staticmethod
    def rolling_es(returns: np.ndarray, window: int, confidence: float = 0.95) -> np.ndarray:
        if len(returns) < window:
            return np.zeros_like(returns)
        
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

class MarketValidator:
    @staticmethod
    def is_market_open() -> bool:
        now = datetime.now()
        if now.weekday() not in MarketHours.TRADING_DAYS:
            return False
        current_time = now.time()
        return MarketHours.OPEN <= current_time <= MarketHours.CLOSE

    @staticmethod
    def is_trading_session() -> bool:
        now = datetime.now()
        if now.weekday() not in MarketHours.TRADING_DAYS:
            return False
        current_time = now.time()
        return MarketHours.OPEN <= current_time < MarketHours.CLOSE

# ==========================================
# DATA PROVIDER
# ==========================================
class DataProvider:
    def __init__(self, config: StrategyConfig):
        self.cfg = config
        self._cache: Optional[Tuple[pd.DataFrame, Dict[str, float], datetime]] = None
        self._cache_ttl = 60

    def fetch_market_data(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        if self._is_cache_valid():
            logger.debug("Cache hit")
            return self._cache[0], self._cache[1]

        tickers = list(set([self.cfg.ticker_signal] + self.cfg.universe))
        
        try:
            raw = yf.download(
                tickers, 
                period="2y", 
                progress=False, 
                auto_adjust=True, 
                group_by='ticker',
                threads=True
            )
            
            if raw.empty:
                raise ValueError("Données vides reçues de yfinance")
            
            last_prices = self._extract_prices(raw, tickers)
            df_signal = self._extract_signal_data(raw, tickers)
            
            self._validate_data(df_signal, last_prices)
            
            self._cache = (df_signal, last_prices, datetime.now())
            logger.info(f"Données récupérées: {len(df_signal)} jours, {len(last_prices)} tickers")
            
            return df_signal, last_prices
            
        except Exception as e:
            logger.error(f"Erreur récupération données: {e}")
            if self._cache:
                logger.warning("Utilisation cache expiré en fallback")
                return self._cache[0], self._cache[1]
            raise

    def _is_cache_valid(self) -> bool:
        if not self._cache:
            return False
        age = (datetime.now() - self._cache[2]).total_seconds()
        return age < self._cache_ttl

    def _extract_prices(self, raw: pd.DataFrame, tickers: List[str]) -> Dict[str, float]:
        last_prices = {}
        
        if len(tickers) == 1:
            last_prices[tickers[0]] = float(raw['Close'].iloc[-1])
        else:
            for t in tickers:
                try:
                    if t in raw.columns.get_level_values(0):
                        price = raw[t]['Close'].iloc[-1]
                        if not pd.isna(price):
                            last_prices[t] = float(price)
                except:
                    pass
        
        return last_prices

    def _extract_signal_data(self, raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        if len(tickers) == 1:
            df_signal = raw
        else:
            df_signal = raw[self.cfg.ticker_signal]
        
        if isinstance(df_signal.columns, pd.MultiIndex):
            df_signal = df_signal.xs(self.cfg.ticker_signal, level=0, axis=1)
        
        df_signal = df_signal[['Open', 'High', 'Low', 'Close']].dropna()
        return df_signal

    def _validate_data(self, df: pd.DataFrame, prices: Dict[str, float]):
        if len(df) < 200:
            raise ValueError(f"Données insuffisantes: {len(df)} jours")
        
        required_tickers = set(self.cfg.universe)
        missing = required_tickers - set(prices.keys())
        if missing:
            raise ValueError(f"Prix manquants: {missing}")
        
        for ticker, price in prices.items():
            if price <= 0 or np.isnan(price) or np.isinf(price):
                raise ValueError(f"Prix invalide pour {ticker}: {price}")

# ==========================================
# SIGNAL ENGINE
# ==========================================
class GoldilocksEngine:
    def __init__(self, config: StrategyConfig):
        self.cfg = config

    def generate_signal(self, df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        d = df.copy()
        
        rets = d['Close'].pct_change().fillna(0)
        
        d['MA_S'] = d['Close'].rolling(self.cfg.ma_short).mean().shift(1)
        d['MA_L'] = d['Close'].rolling(self.cfg.ma_long).mean().shift(1)
        
        prev_close = d['Close'].shift(1)
        tr = pd.concat([
            d['High'] - d['Low'], 
            (d['High'] - prev_close).abs(), 
            (d['Low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(20).mean()
        vol_ann = (atr / d['Close']) * np.sqrt(252)
        vol_ann = vol_ann.shift(1)
        
        es_vals = QuantUtils.rolling_es(rets.values, self.cfg.es_lookback) * np.sqrt(21)
        es_vals = pd.Series(es_vals, index=d.index).shift(1)

        if len(d) < max(self.cfg.ma_long, self.cfg.es_lookback):
            raise ValueError("Données historiques insuffisantes")

        last = d.iloc[-1]
        last_es = es_vals.iloc[-1]
        curr_vol = vol_ann.iloc[-1]
        
        if pd.isna(curr_vol) or curr_vol <= 0:
            curr_vol = self.cfg.vol_target
        
        curr_vol = max(curr_vol, self.cfg.min_vol)

        trend_ok = last['MA_S'] > last['MA_L']
        guard_ok = last_es < self.cfg.guard_thresh
        
        metrics = {
            'ma_short': float(last['MA_S']),
            'ma_long': float(last['MA_L']),
            'volatility': float(curr_vol),
            'es': float(last_es),
            'trend': trend_ok,
            'guard': guard_ok
        }

        if trend_ok and guard_ok:
            raw_lev = self.cfg.vol_target / curr_vol
            final_lev = min(raw_lev, self.cfg.lev_max)
            status = "RISK_ON"
            
            if raw_lev > self.cfg.lev_max:
                logger.warning(f"Levier tronqué: {raw_lev:.2f} -> {final_lev:.2f}")
        else:
            final_lev = 0.0
            status = "RISK_OFF"
            reason = []
            if not trend_ok:
                reason.append("Trend négatif")
            if not guard_ok:
                reason.append(f"ES trop élevé ({last_es:.1%})")
            logger.info(f"Risk-Off: {', '.join(reason)}")

        logger.info(f"Signal: {status} | Trend={'UP' if trend_ok else 'DOWN'} | ES={last_es:.1%} | Vol={curr_vol:.1%} | Levier={final_lev:.2f}x")
        
        return final_lev, metrics

    def compute_allocation(self, leverage: float) -> Dict[str, float]:
        alloc = {t: 0.0 for t in self.cfg.universe}
        
        if leverage <= 0.01:
            alloc["BIL"] = 1.0
        elif leverage > 1.0:
            excess = leverage - 1.0
            w_qld = min(excess, 1.0)
            alloc["QLD"] = w_qld
            alloc["QQQ"] = 1.0 - w_qld
        else:
            alloc["QQQ"] = leverage
            alloc["BIL"] = 1.0 - leverage
        
        total = sum(alloc.values())
        if abs(total - 1.0) > 0.01:
            logger.error(f"Allocation invalide: {alloc}, total={total}")
            raise ValueError("Somme allocation != 1.0")
        
        return alloc

# ==========================================
# PORTFOLIO MANAGER
# ==========================================
class PortfolioManager:
    def __init__(self, config: StrategyConfig):
        self.cfg = config
        self.file_path = Path(DB_FILE)
        self.lock_path = Path(LOCK_FILE)
        self.state = self._load_or_create()
        self.trade_log: List[Trade] = []

    def _load_or_create(self) -> PortfolioState:
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"État chargé: Equity=${data['equity']:.2f}")
                return PortfolioState(**data)
            except Exception as e:
                logger.error(f"Erreur lecture état: {e}")
        
        initial = PortfolioState(
            cash=self.cfg.initial_capital,
            holdings={t: 0 for t in self.cfg.universe},
            equity=self.cfg.initial_capital,
            last_update=datetime.now().isoformat()
        )
        logger.info(f"Nouvel état créé: ${self.cfg.initial_capital:.2f}")
        return initial

    def save(self):
        try:
            with FileLock(str(self.lock_path)):
                with open(self.file_path, 'w') as f:
                    json.dump(asdict(self.state), f, indent=4)
        except Exception as e:
            logger.error(f"Erreur sauvegarde: {e}")
            raise

    def execute_rebalance(self, target_alloc: Dict[str, float], prices: Dict[str, float]):
        self._mark_to_market(prices)
        
        logger.info(f"Rebalancing: Equity=${self.state.equity:,.2f}, Cash=${self.state.cash:,.2f}")
        
        trades = self._compute_trades(target_alloc, prices)
        
        if not trades:
            logger.info("Aucun trade requis")
            return
        
        sells = [t for t in trades if t.quantity < 0]
        buys = [t for t in trades if t.quantity > 0]
        
        for trade in sells:
            self._execute_trade(trade)
        
        for trade in buys:
            if self.state.cash < trade.cost:
                logger.warning(f"Cash insuffisant pour {trade.ticker}: requis=${trade.cost:.2f}, dispo=${self.state.cash:.2f}")
                continue
            self._execute_trade(trade)
        
        self.state.last_update = datetime.now().isoformat()
        self.save()
        
        logger.info(f"Rebalancing terminé: {len(self.trade_log)} trades exécutés")

    def _mark_to_market(self, prices: Dict[str, float]):
        val_holdings = 0.0
        
        for ticker, qty in self.state.holdings.items():
            if qty == 0:
                continue
            
            if ticker not in prices:
                logger.error(f"Prix manquant pour {ticker} (position={qty})")
                raise ValueError(f"Prix manquant: {ticker}")
            
            val_holdings += qty * prices[ticker]
        
        self.state.equity = self.state.cash + val_holdings
        
        if self.state.equity <= 0:
            logger.critical(f"Equity négative ou nulle: ${self.state.equity:.2f}")
            raise ValueError("Portfolio bankrupt")

    def _compute_trades(self, target_alloc: Dict[str, float], prices: Dict[str, float]) -> List[Trade]:
        trades = []
        
        for ticker, weight in target_alloc.items():
            price = prices[ticker]
            qty_curr = self.state.holdings.get(ticker, 0)
            target_val = self.state.equity * weight
            target_qty = int(target_val / price)
            qty_delta = target_qty - qty_curr
            
            if qty_delta == 0:
                continue
            
            gross_cost = abs(qty_delta) * price
            
            if gross_cost < self.cfg.min_trade_amount:
                continue
            
            slippage = gross_cost * (SLIPPAGE_BPS / 10000)
            commission = COMMISSION
            total_cost = gross_cost + slippage + commission
            
            if qty_delta < 0:
                total_cost = -gross_cost + slippage + commission
            
            trade = Trade(
                timestamp=datetime.now().isoformat(),
                ticker=ticker,
                action="BUY" if qty_delta > 0 else "SELL",
                quantity=abs(qty_delta),
                price=price,
                cost=total_cost,
                commission=commission
            )
            
            trades.append(trade)
        
        return trades

    def _execute_trade(self, trade: Trade):
        qty_signed = trade.quantity if trade.action == "BUY" else -trade.quantity
        qty_curr = self.state.holdings.get(trade.ticker, 0)
        qty_new = qty_curr + qty_signed
        
        if qty_new < 0:
            logger.error(f"Tentative vente à découvert: {trade.ticker} ({qty_curr} -> {qty_new})")
            raise ValueError(f"Vente à découvert interdite: {trade.ticker}")
        
        self.state.cash -= trade.cost
        
        if self.state.cash < 0:
            logger.error(f"Cash négatif après trade: ${self.state.cash:.2f}")
            raise ValueError("Cash négatif")
        
        self.state.holdings[trade.ticker] = qty_new
        self.state.trade_count += 1
        self.trade_log.append(trade)
        
        logger.info(f"TRADE {trade.action} {trade.quantity} {trade.ticker} @ ${trade.price:.2f} (cout=${trade.cost:.2f})")

    def get_metrics(self) -> PerformanceMetrics:
        total_return = (self.state.equity / self.cfg.initial_capital) - 1
        
        return PerformanceMetrics(
            equity=self.state.equity,
            total_return=total_return,
            sharpe=0.0,
            max_drawdown=0.0,
            trade_count=self.state.trade_count
        )

# ==========================================
# ORCHESTRATEUR
# ==========================================
class TradingOrchestrator:
    def __init__(self, config: StrategyConfig):
        self.cfg = config
        self.data_prov = DataProvider(config)
        self.engine = GoldilocksEngine(config)
        self.pm = PortfolioManager(config)
        self.consecutive_errors = 0

    def run_cycle(self) -> bool:
        try:
            if not MarketValidator.is_trading_session():
                logger.debug("Hors session trading")
                return False
            
            df_sig, prices = self.data_prov.fetch_market_data()
            lev, metrics = self.engine.generate_signal(df_sig)
            alloc = self.engine.compute_allocation(lev)
            
            if self.cfg.mode == TradingMode.LIVE:
                self.pm.execute_rebalance(alloc, prices)
            else:
                logger.info(f"Mode {self.cfg.mode.value}: Simulation rebalancing")
                logger.info(f"Allocation cible: {alloc}")
            
            self.consecutive_errors = 0
            return True
            
        except Exception as e:
            self.consecutive_errors += 1
            logger.error(f"Erreur cycle ({self.consecutive_errors}/{ERROR_THRESHOLD}): {e}")
            logger.debug(traceback.format_exc())
            
            if self.consecutive_errors >= ERROR_THRESHOLD:
                logger.critical(f"Seuil d'erreurs atteint ({ERROR_THRESHOLD}), arret du bot")
                raise
            
            return False

# ==========================================
# DASHBOARD
# ==========================================
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_dashboard(orchestrator: TradingOrchestrator):
    clear_screen()
    now = datetime.now()
    
    print("=" * 80)
    print(f"{'GOLDILOCKS BOT - PRODUCTION':^80}")
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S'):^80}")
    print("=" * 80)
    
    state = orchestrator.pm.state
    metrics = orchestrator.pm.get_metrics()
    
    print(f"\n{'CAPITAL':^80}")
    print("-" * 80)
    print(f"Equity Total      : ${metrics.equity:>15,.2f}")
    print(f"Cash Disponible   : ${state.cash:>15,.2f}")
    print(f"Return Total      : {metrics.total_return:>15.2%}")
    print(f"Trades Executes   : {metrics.trade_count:>15,}")
    
    print(f"\n{'POSITIONS':^80}")
    print("-" * 80)
    active = {k: v for k, v in state.holdings.items() if v > 0}
    if active:
        for ticker, qty in active.items():
            print(f"{ticker:>10} : {qty:>10,} parts")
    else:
        print(f"{'100% Cash':^80}")
    
    print(f"\n{'ETAT SYSTEME':^80}")
    print("-" * 80)
    print(f"Mode              : {orchestrator.cfg.mode.value.upper()}")
    print(f"Marche            : {'OUVERT' if MarketValidator.is_trading_session() else 'FERME'}")
    print(f"Erreurs Consec.   : {orchestrator.consecutive_errors}/{ERROR_THRESHOLD}")
    print(f"Dernier Update    : {state.last_update}")
    
    print("\n" + "=" * 80)

# ==========================================
# MAIN LOOP
# ==========================================
def main_loop():
    setup_logging()
    
    config = StrategyConfig(mode=TradingMode.DRY_RUN)
    orchestrator = TradingOrchestrator(config)
    
    logger.info("=" * 80)
    logger.info("BOT DEMARRE")
    logger.info("=" * 80)
    
    while True:
        try:
            success = orchestrator.run_cycle()
            print_dashboard(orchestrator)
            
            if not MarketValidator.is_trading_session():
                print(f"\nMarche ferme - Prochain cycle dans {REFRESH_RATE}s")
            else:
                status = "OK" if success else "WARN"
                print(f"\n[{status}] Cycle termine - Prochain dans {REFRESH_RATE}s")
            
            time.sleep(REFRESH_RATE)
            
        except KeyboardInterrupt:
            logger.info("Arret manuel du bot")
            print("\nBot arrete proprement")
            break
        except Exception as e:
            logger.critical(f"Erreur fatale: {e}")
            logger.debug(traceback.format_exc())
            print(f"\nERREUR CRITIQUE: {e}")
            break

if __name__ == "__main__":
    main_loop()
