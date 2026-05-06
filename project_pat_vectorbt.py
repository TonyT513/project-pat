"""
Project Pat Strategy — NQ=F / MNQ=F — 1H Bars
WITH MANUAL INDICATOR FILTERS
===============================================
Filters applied before every entry (must pass 3 of 5):
  1. VWAP  — long above, short below
  2. MACD  — long when line > signal, short when line < signal
  3. EMA21 — long above, short below
  4. Williams %R — avoid overbought longs / oversold shorts
  5. Bollinger %B — avoid extreme band entries against signal

You can tune MIN_FILTERS (default 3) to be stricter or looser.

Run:
    /opt/miniconda3/bin/python project_pat_vectorbt.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import vectorbt as vbt

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
TICKER      = "NQ=F"
PERIOD      = "730d"
TIMEFRAME   = "1h"
POINT_VALUE = 1.0
INIT_CASH   = 25_000
FIXED_FEES  = 2.50

DAY_CONFIG = {
    "Mon": (True,  2, "crossover", 100,  30),
    "Tue": (True,  4, "ob",         50, 100),
    "Wed": (True,  2, "ob",        100,  50),
    "Thu": (True,  2, "crossover",  50, 100),
    "Fri": (True,  1, "fib50",     100,  60),
}

FORCE_CLOSE_HOUR   = 15
SESSION_RESET_HOUR = 18

# ── Filter settings ──────────────────────────────────────────
# How many of the 5 filters must agree before taking a trade
MIN_FILTERS = 3   # set to 0 to disable all filters (baseline)

# Williams %R thresholds
WILLR_OVERBOUGHT = -20   # avoid longs above this
WILLR_OVERSOLD   = -80   # avoid shorts below this

# Bollinger Band thresholds
BB_HIGH = 0.8   # avoid longs when %B above this
BB_LOW  = 0.2   # avoid shorts when %B below this

# ═══════════════════════════════════════════════════════════════
# 1. FETCH DATA
# ═══════════════════════════════════════════════════════════════
print(f"Downloading {TICKER} {TIMEFRAME} data ...")
raw = vbt.YFData.download(TICKER, period=PERIOD, interval=TIMEFRAME)
df  = raw.get()
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.index = pd.to_datetime(df.index)
if df.index.tz is None:
    df.index = df.index.tz_localize("America/New_York")
else:
    df.index = df.index.tz_convert("America/New_York")
df = df[df["Volume"] > 0].copy()

open_  = df["Open"]; high = df["High"]
low    = df["Low"];  close = df["Close"]
volume = df["Volume"]
print(f"Bars loaded: {len(df):,}  ({df.index[0].date()} → {df.index[-1].date()})")

# ═══════════════════════════════════════════════════════════════
# 2. PREV SESSION LEVELS
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
        ob_hi[i]=cur_hi; ob_lo[i]=cur_lo; ob_ex[i]=cur_ex
    return (pd.Series(ob_hi,index=df_h1.index),
            pd.Series(ob_lo,index=df_h1.index),
            pd.Series(ob_ex,index=df_h1.index))

print("Detecting order blocks ...")
ob_hi_s, ob_lo_s, ob_ex_s = detect_ob_series(df)

# ═══════════════════════════════════════════════════════════════
# 4. INDICATORS
# ═══════════════════════════════════════════════════════════════
print("Computing indicators ...")

# ── EMA 21 ───────────────────────────────────────────────────
ema_21 = close.ewm(span=21).mean()

# ── MACD (12, 26, 9) ─────────────────────────────────────────
macd_line   = close.ewm(span=12).mean() - close.ewm(span=26).mean()
macd_signal = macd_line.ewm(span=9).mean()

# ── Williams %R (14) ─────────────────────────────────────────
w_high = high.rolling(14).max()
w_low  = low.rolling(14).min()
willr  = -100 * (w_high - close) / (w_high - w_low + 1e-9)

# ── Bollinger Bands %B (20, 2) ───────────────────────────────
bb_mid   = close.rolling(20).mean()
bb_std   = close.rolling(20).std()
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std
bb_pct_b = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)

# ── VWAP (resets at 6 PM each session) ───────────────────────
grp_arr      = pd.Series(
    (df.index.hour == SESSION_RESET_HOUR).astype(int).cumsum(),
    index=df.index
)
typical_p    = (high + low + close) / 3
tp_vol       = typical_p * volume
cum_tpvol    = tp_vol.groupby(grp_arr.values).cumsum()
cum_vol      = volume.groupby(grp_arr.values).cumsum()
vwap         = cum_tpvol / cum_vol.replace(0, np.nan)

# ═══════════════════════════════════════════════════════════════
# 5. CROSSOVERS
# ═══════════════════════════════════════════════════════════════
idx   = df.index
dow   = idx.dayofweek
hour_ = idx.hour

def crossover(a, b):
    r = (a > b) & ~(np.roll(a,1) > np.roll(b,1)); r[0]=False; return r
def crossunder(a, b):
    r = (a < b) & ~(np.roll(a,1) < np.roll(b,1)); r[0]=False; return r

cross_above_high = crossover (close.values, prev_day_high.values)
cross_below_low  = crossunder(close.values, prev_day_low.values)
cross_above_fib  = crossover (close.values, fib50.values)
cross_below_fib  = crossunder(close.values, fib50.values)
force_close      = hour_ == FORCE_CLOSE_HOUR

ob_hi_arr = ob_hi_s.values; ob_lo_arr = ob_lo_s.values; ob_ex_arr = ob_ex_s.values
open_arr  = open_.values;   high_arr  = high.values
low_arr   = low.values;     close_arr = close.values
n_bars    = len(df)

# Pre-extract indicator arrays for speed
ema21_arr   = ema_21.values
macdl_arr   = macd_line.values
macds_arr   = macd_signal.values
willr_arr   = willr.values
bbpctb_arr  = bb_pct_b.values
vwap_arr    = vwap.values

# ═══════════════════════════════════════════════════════════════
# 6. FILTER FUNCTION
#    Returns number of filters the signal passes (0–5)
# ═══════════════════════════════════════════════════════════════
def count_filters(i, is_long):
    score = 0
    c = close_arr[i]

    # 1. VWAP filter
    vw = vwap_arr[i]
    if not np.isnan(vw):
        if is_long and c > vw:  score += 1
        if not is_long and c < vw: score += 1

    # 2. MACD filter
    ml = macdl_arr[i]; ms = macds_arr[i]
    if not (np.isnan(ml) or np.isnan(ms)):
        if is_long and ml > ms:     score += 1
        if not is_long and ml < ms: score += 1

    # 3. EMA 21 filter
    e21 = ema21_arr[i]
    if not np.isnan(e21):
        if is_long and c > e21:     score += 1
        if not is_long and c < e21: score += 1

    # 4. Williams %R filter
    wr = willr_arr[i]
    if not np.isnan(wr):
        # Avoid longs when overbought, avoid shorts when oversold
        if is_long and wr < WILLR_OVERBOUGHT:     score += 1
        if not is_long and wr > WILLR_OVERSOLD:   score += 1

    # 5. Bollinger %B filter
    bb = bbpctb_arr[i]
    if not np.isnan(bb):
        # Avoid longs at upper extreme, avoid shorts at lower extreme
        if is_long and bb < BB_HIGH:     score += 1
        if not is_long and bb > BB_LOW:  score += 1

    return score

# ═══════════════════════════════════════════════════════════════
# 7. BUILD SIGNALS
# ═══════════════════════════════════════════════════════════════
print(f"Building signals (MIN_FILTERS={MIN_FILTERS}) ...")

entries = np.zeros(n_bars, dtype=bool)
exits   = np.zeros(n_bars, dtype=bool)
sizes   = np.zeros(n_bars)

in_pos     = False; pos_side = 0
cur_tp     = np.nan; cur_sl  = np.nan
dtc        = 0; last_date   = None
filtered_out = 0

for i in range(1, n_bars):
    bar_date = idx[i].date()
    if bar_date != last_date: dtc = 0; last_date = bar_date

    if force_close[i] and in_pos:
        exits[i]=True; in_pos=False; pos_side=0; continue

    if in_pos:
        if pos_side == 1:
            if low_arr[i] <= cur_sl or high_arr[i] >= cur_tp:
                exits[i]=True; in_pos=False; pos_side=0; continue
        else:
            if high_arr[i] >= cur_sl or low_arr[i] <= cur_tp:
                exits[i]=True; in_pos=False; pos_side=0; continue

    if in_pos or force_close[i]: continue

    d = dow[i]
    if d > 4: continue
    day_key = ["Mon","Tue","Wed","Thu","Fri"][d]
    en, mx, strat, tp_pts, sl_pts = DAY_CONFIG[day_key]
    if not en or dtc >= mx: continue

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
            oh = ob_hi_arr[i]; ol = ob_lo_arr[i]
            if ol <= o <= oh:
                if c < ol:   sig_short = True
                elif c > oh: sig_long  = True

    if not (sig_long or sig_short): continue

    # ── Apply indicator filters ───────────────────────────────
    if MIN_FILTERS > 0:
        score = count_filters(i, sig_long)
        if score < MIN_FILTERS:
            filtered_out += 1
            continue

    # ── Record entry ─────────────────────────────────────────
    entries[i] = True; in_pos = True; dtc += 1
    side = 1 if sig_long else -1
    sizes[i]  = side; pos_side = side
    cur_tp = c + tp_pts * POINT_VALUE * side
    cur_sl = c - sl_pts * POINT_VALUE * side

print(f"Entries: {entries.sum()}  |  Exits: {exits.sum()}  |  Filtered out: {filtered_out}")

# ═══════════════════════════════════════════════════════════════
# 8. BACKTEST
# ═══════════════════════════════════════════════════════════════
print("Running backtest ...")
pf = vbt.Portfolio.from_signals(
    close=close, open=open_, high=high, low=low,
    entries      = pd.Series(entries & (sizes ==  1), index=df.index),
    short_entries= pd.Series(entries & (sizes == -1), index=df.index),
    exits        = pd.Series(exits,                   index=df.index),
    init_cash    = INIT_CASH,
    fixed_fees   = FIXED_FEES,
    freq         = TIMEFRAME,
)

# ═══════════════════════════════════════════════════════════════
# 9. RESULTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"PROJECT PAT  |  {TICKER}  |  1H  |  MIN_FILTERS={MIN_FILTERS}")
print("="*60)
print(pf.stats())

trades = pf.trades.records_readable
if len(trades):
    trades["DayOfWeek"] = pd.to_datetime(
        trades["Entry Timestamp"]
    ).dt.day_name()
    print("\n--- Results by day ---")
    summary = trades.groupby("DayOfWeek").agg(
        Trades    = ("PnL","count"),
        Win_Rate  = ("Return", lambda x: f"{(x>0).mean():.1%}"),
        Avg_PnL   = ("PnL","mean"),
        Total_PnL = ("PnL","sum"),
    )
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    summary   = summary.reindex([d for d in day_order if d in summary.index])
    print(summary.to_string())

    print("\n--- Filter settings used ---")
    print(f"  MIN_FILTERS    : {MIN_FILTERS} of 5 must pass")
    print(f"  VWAP           : long above, short below")
    print(f"  MACD           : line vs signal direction")
    print(f"  EMA 21         : price vs EMA direction")
    print(f"  Williams %R    : overbought < {WILLR_OVERBOUGHT} / oversold > {WILLR_OVERSOLD}")
    print(f"  Bollinger %B   : avoid > {BB_HIGH} for longs / < {BB_LOW} for shorts")
    print(f"\nTip: change MIN_FILTERS at the top of the file to test")
    print(f"     0 = no filter (baseline)  |  5 = strictest")

pf.plot().show()
