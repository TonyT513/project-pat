"""
Project Pat — Multi-Strategy Optimizer
=======================================
Tests 3 strategy types × 7 TP values × 7 SL values = 147 combos per day
across all 5 trading days (Mon–Fri).

Strategy types tested per day:
  "crossover" — prev session High/Low crossover
  "fib50"     — Fib 0.500 crossover
  "ob"        — Order Block open-inside / close-outside

Results saved to: project_pat_results.csv
Best params printed + final backtest auto-run.

Run:
    /opt/miniconda3/bin/python project_pat_optimize.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import itertools
import vectorbt as vbt
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
TICKER      = "NQ=F"
PERIOD      = "730d"
TIMEFRAME   = "1h"
POINT_VALUE = 1.0
INIT_CASH   = 25_000
FIXED_FEES  = 2.50

FORCE_CLOSE_HOUR   = 15
SESSION_RESET_HOUR = 18

DAY_ENABLED = {
    "Mon": True,
    "Tue": True,
    "Wed": True,
    "Thu": True,
    "Fri": True,
}
DAY_MAX_TRADES = {
    "Mon": 2, "Tue": 4, "Wed": 2, "Thu": 2, "Fri": 1,
}

# Strategy types to test on each day
STRATEGY_TYPES = ["crossover", "fib50", "ob"]

TP_VALUES  = [20, 30, 40, 50, 60, 75, 100]
SL_VALUES  = [20, 30, 40, 50, 60, 75, 100]
MIN_TRADES = 5

SAVE_PATH = "/Users/tonythompson/backtest-vector/project_pat_results.csv"

# ═══════════════════════════════════════════════════════════════
# 1. FETCH DATA
# ═══════════════════════════════════════════════════════════════
print(f"Downloading {TICKER} {TIMEFRAME} ...")
raw = vbt.YFData.download(TICKER, period=PERIOD, interval=TIMEFRAME)
df  = raw.get()
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.index = pd.to_datetime(df.index)
df.index = df.index.tz_localize("America/New_York") if df.index.tz is None \
           else df.index.tz_convert("America/New_York")
df = df[df["Volume"] > 0].copy()

open_  = df["Open"]
high   = df["High"]
low    = df["Low"]
close  = df["Close"]
print(f"Bars: {len(df):,}  ({df.index[0].date()} → {df.index[-1].date()})")

# ═══════════════════════════════════════════════════════════════
# 2. LEVELS
# ═══════════════════════════════════════════════════════════════
sessions      = df.resample("24h", offset="18h").agg({"High":"max","Low":"min"}).shift(1)
prev_day_high = sessions["High"].reindex(df.index, method="ffill")
prev_day_low  = sessions["Low"].reindex(df.index, method="ffill")
fib50         = prev_day_low + 0.5 * (prev_day_high - prev_day_low)

# ═══════════════════════════════════════════════════════════════
# 3. ORDER BLOCK DETECTION
# ═══════════════════════════════════════════════════════════════
def detect_ob_series(df_h1, window=12):
    up = (df_h1["Close"] >= df_h1["Open"]).values
    his = df_h1["High"].values; los = df_h1["Low"].values
    hours = df_h1.index.hour
    ob_hi = np.full(len(df_h1), np.nan)
    ob_lo = np.full(len(df_h1), np.nan)
    ob_ex = np.zeros(len(df_h1), dtype=bool)
    cur_hi, cur_lo, cur_ex = np.nan, np.nan, False
    for i in range(len(df_h1)):
        if hours[i] == SESSION_RESET_HOUR:
            cur_hi, cur_lo, cur_ex = np.nan, np.nan, False
        if i >= window:
            dirs = up[i-window:i+1]; h = his[i-window:i+1]; l = los[i-window:i+1]
            n = len(dirs); found = False
            trail_red = 0
            for j in range(n-1,-1,-1):
                if not dirs[j]: trail_red += 1
                else: break
            ob_green = 0
            if trail_red >= 2:
                for j in range(n-1-trail_red,-1,-1):
                    if dirs[j]: ob_green += 1
                    else: break
            lead_red = 0
            if trail_red >= 2 and 1 <= ob_green <= 2:
                for j in range(n-1-trail_red-ob_green,-1,-1):
                    if not dirs[j]: lead_red += 1
                    else: break
            if trail_red >= 2 and 1 <= ob_green <= 2 and lead_red >= 2:
                fi = n-1-trail_red-ob_green+1; li = n-1-trail_red
                cur_hi = float(np.max(h[fi:li+1])); cur_lo = float(np.min(l[fi:li+1]))
                cur_ex = True; found = True
            if not found:
                trail_grn = 0
                for j in range(n-1,-1,-1):
                    if dirs[j]: trail_grn += 1
                    else: break
                ob_red = 0
                if trail_grn >= 2:
                    for j in range(n-1-trail_grn,-1,-1):
                        if not dirs[j]: ob_red += 1
                        else: break
                lead_grn = 0
                if trail_grn >= 2 and 1 <= ob_red <= 2:
                    for j in range(n-1-trail_grn-ob_red,-1,-1):
                        if dirs[j]: lead_grn += 1
                        else: break
                if trail_grn >= 2 and 1 <= ob_red <= 2 and lead_grn >= 2:
                    fi2 = n-1-trail_grn-ob_red+1; li2 = n-1-trail_grn
                    cur_hi = float(np.max(h[fi2:li2+1])); cur_lo = float(np.min(l[fi2:li2+1]))
                    cur_ex = True
        ob_hi[i] = cur_hi; ob_lo[i] = cur_lo; ob_ex[i] = cur_ex
    return (pd.Series(ob_hi, index=df_h1.index),
            pd.Series(ob_lo, index=df_h1.index),
            pd.Series(ob_ex, index=df_h1.index))

print("Detecting order blocks ...")
ob_hi_s, ob_lo_s, ob_ex_s = detect_ob_series(df)

# ═══════════════════════════════════════════════════════════════
# 4. PRE-COMPUTE CROSSOVERS
# ═══════════════════════════════════════════════════════════════
idx   = df.index
dow   = idx.dayofweek
hour_ = idx.hour

def crossover(a, b):
    r = (a > b) & ~(np.roll(a,1) > np.roll(b,1)); r[0] = False; return r
def crossunder(a, b):
    r = (a < b) & ~(np.roll(a,1) < np.roll(b,1)); r[0] = False; return r

cross_above_high = crossover (close.values, prev_day_high.values)
cross_below_low  = crossunder(close.values, prev_day_low.values)
cross_above_fib  = crossover (close.values, fib50.values)
cross_below_fib  = crossunder(close.values, fib50.values)
force_close_mask = hour_ == FORCE_CLOSE_HOUR

ob_hi_arr = ob_hi_s.values; ob_lo_arr = ob_lo_s.values; ob_ex_arr = ob_ex_s.values
open_arr  = open_.values; high_arr = high.values
low_arr   = low.values;   close_arr = close.values
n_bars    = len(df)

# ═══════════════════════════════════════════════════════════════
# 5. CORE BACKTEST FUNCTION
#    day_cfg format: {day: (enabled, max_trades, strategy, tp, sl)}
# ═══════════════════════════════════════════════════════════════
def run_backtest(day_cfg):
    entries = np.zeros(n_bars, dtype=bool)
    exits   = np.zeros(n_bars, dtype=bool)
    sizes   = np.zeros(n_bars)

    in_pos = False; pos_side = 0
    cur_tp = np.nan; cur_sl = np.nan
    dtc = 0; last_date = None

    for i in range(1, n_bars):
        bar_date = idx[i].date()
        if bar_date != last_date:
            dtc = 0; last_date = bar_date

        if force_close_mask[i] and in_pos:
            exits[i] = True; in_pos = False; pos_side = 0; continue

        if in_pos:
            if pos_side == 1:
                if low_arr[i] <= cur_sl or high_arr[i] >= cur_tp:
                    exits[i] = True; in_pos = False; pos_side = 0; continue
            else:
                if high_arr[i] >= cur_sl or low_arr[i] <= cur_tp:
                    exits[i] = True; in_pos = False; pos_side = 0; continue

        if in_pos or force_close_mask[i]:
            continue

        d = dow[i]
        if d > 4:
            continue
        day_key = ["Mon","Tue","Wed","Thu","Fri"][d]
        en, mx, strat, tp_pts, sl_pts = day_cfg[day_key]
        if not en or dtc >= mx:
            continue

        c = close_arr[i]; o = open_arr[i]
        sig_long = sig_short = False

        if strat == "crossover":
            if cross_above_high[i]:   sig_long  = True
            elif cross_below_low[i]:  sig_short = True
        elif strat == "fib50":
            if cross_above_fib[i]:    sig_long  = True
            elif cross_below_fib[i]:  sig_short = True
        elif strat == "ob":
            if ob_ex_arr[i]:
                ob_h = ob_hi_arr[i]; ob_l = ob_lo_arr[i]
                if ob_l <= o <= ob_h:
                    if c < ob_l:    sig_short = True
                    elif c > ob_h:  sig_long  = True

        if sig_long or sig_short:
            entries[i] = True; in_pos = True; dtc += 1
            if sig_long:
                sizes[i] =  1; pos_side =  1
                cur_tp = c + tp_pts * POINT_VALUE
                cur_sl = c - sl_pts * POINT_VALUE
            else:
                sizes[i] = -1; pos_side = -1
                cur_tp = c - tp_pts * POINT_VALUE
                cur_sl = c + sl_pts * POINT_VALUE

    return vbt.Portfolio.from_signals(
        close         = close, open=open_, high=high, low=low,
        entries       = pd.Series(entries & (sizes ==  1), index=df.index),
        short_entries = pd.Series(entries & (sizes == -1), index=df.index),
        exits         = pd.Series(exits,                   index=df.index),
        init_cash     = INIT_CASH, fixed_fees = FIXED_FEES, freq = TIMEFRAME,
    )

# ═══════════════════════════════════════════════════════════════
# 6. OPTIMIZATION LOOP
#    For each day: test every combo of strategy × TP × SL
# ═══════════════════════════════════════════════════════════════
# Base config: other days hold their current defaults while one is optimized
BASE_CFG = {
    "Mon": (True, 2, "crossover", 30, 30),
    "Tue": (True, 4, "ob",        50, 50),
    "Wed": (True, 2, "ob",        30, 30),
    "Thu": (True, 2, "crossover", 50, 50),
    "Fri": (True, 1, "fib50",     30, 30),
}

tp_sl_pairs  = list(itertools.product(TP_VALUES, SL_VALUES))
all_combos   = list(itertools.product(STRATEGY_TYPES, TP_VALUES, SL_VALUES))
total_combos = len(all_combos)
results      = []
ACTIVE_DAYS  = [d for d, en in DAY_ENABLED.items() if en]

print(f"\nTotal combos per day: {total_combos}  ({len(STRATEGY_TYPES)} strategies × {len(TP_VALUES)} TPs × {len(SL_VALUES)} SLs)")

for day in ACTIVE_DAYS:
    print(f"\n{'='*55}")
    print(f"Optimizing {day} ...")
    print(f"{'='*55}")
    day_results = []

    for strat, tp, sl in tqdm(all_combos, desc=day):
        cfg = dict(BASE_CFG)
        en, mx, _, _, _ = cfg[day]
        cfg[day] = (en, mx, strat, tp, sl)

        try:
            pf     = run_backtest(cfg)
            trades = pf.trades.records_readable
            if not len(trades):
                continue

            day_idx    = ["Mon","Tue","Wed","Thu","Fri"].index(day)
            day_trades = trades[
                pd.to_datetime(trades["Entry Timestamp"]).dt.dayofweek == day_idx
            ]
            if len(day_trades) < MIN_TRADES:
                continue

            win_rate  = (day_trades["Return"] > 0).mean()
            total_pnl = day_trades["PnL"].sum()
            avg_pnl   = day_trades["PnL"].mean()
            sharpe    = pf.sharpe_ratio()
            sharpe    = sharpe if np.isfinite(sharpe) else -np.inf

            day_results.append({
                "Day":       day,
                "Strategy":  strat,
                "TP":        tp,
                "SL":        sl,
                "Trades":    len(day_trades),
                "Win_Rate":  round(win_rate, 3),
                "Avg_PnL":   round(avg_pnl, 2),
                "Total_PnL": round(total_pnl, 2),
                "Sharpe":    round(sharpe, 3),
            })
        except Exception:
            continue

    if day_results:
        df_day = pd.DataFrame(day_results).sort_values("Sharpe", ascending=False)
        print(f"\n  Top 5 for {day}:")
        print(df_day.head(5).to_string(index=False))
        results.extend(day_results)
    else:
        print(f"  No valid results for {day} (fewer than {MIN_TRADES} trades per combo).")

# ═══════════════════════════════════════════════════════════════
# 7. FINAL SUMMARY & BEST PARAMS
# ═══════════════════════════════════════════════════════════════
if results:
    df_results = pd.DataFrame(results).sort_values(
        ["Day","Sharpe"], ascending=[True, False]
    )

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE — Best strategy + params per day (by Sharpe)")
    print("="*70)
    best = df_results.groupby("Day").first().reset_index()
    print(best[["Day","Strategy","TP","SL","Trades",
                "Win_Rate","Avg_PnL","Total_PnL","Sharpe"]].to_string(index=False))

    df_results.to_csv(SAVE_PATH, index=False)
    print(f"\nFull results saved to: {SAVE_PATH}")

    # ── Print DAY_CONFIG to paste into project_pat_vectorbt.py ──
    print("\n" + "="*70)
    print("PASTE THIS INTO project_pat_vectorbt.py as your DAY_CONFIG:")
    print("="*70)
    print("DAY_CONFIG = {")
    day_order = ["Mon","Tue","Wed","Thu","Fri"]
    best_dict = {}
    for _, row in best.iterrows():
        best_dict[row["Day"]] = row
    for d in day_order:
        en  = DAY_ENABLED.get(d, False)
        mx  = DAY_MAX_TRADES.get(d, 2)
        if d in best_dict:
            row = best_dict[d]
            print(f'    "{d}": ({str(en)}, {mx}, "{row["Strategy"]}", {int(row["TP"])}, {int(row["SL"])}),')
        else:
            print(f'    "{d}": ({str(en)}, {mx}, "crossover", 30, 30),  # no valid result found')
    print("}")

    # ── Run final backtest with best params ──────────────────
    print("\nRunning final backtest with best params ...")
    best_cfg = dict(BASE_CFG)
    for d in day_order:
        if d in best_dict:
            row = best_dict[d]
            en  = DAY_ENABLED.get(d, False)
            mx  = DAY_MAX_TRADES.get(d, 2)
            best_cfg[d] = (en, mx, row["Strategy"], int(row["TP"]), int(row["SL"]))

    pf_best = run_backtest(best_cfg)
    print("\n--- Optimized Portfolio Stats ---")
    print(pf_best.stats())

    trades = pf_best.trades.records_readable
    if len(trades):
        trades["DayOfWeek"] = pd.to_datetime(trades["Entry Timestamp"]).dt.day_name()
        print("\n--- Results by day ---")
        summary = trades.groupby("DayOfWeek").agg(
            Trades    = ("PnL","count"),
            Win_Rate  = ("Return", lambda x: f"{(x>0).mean():.1%}"),
            Avg_PnL   = ("PnL","mean"),
            Total_PnL = ("PnL","sum"),
        )
        day_order_full = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        summary = summary.reindex([d for d in day_order_full if d in summary.index])
        print(summary.to_string())

    pf_best.plot().show()

else:
    print("\nNo valid results found.")
