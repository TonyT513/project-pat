"""
Project Pat — ML Pipeline v3 — Market Structure Edition
=========================================================
Added features vs v2:

MARKET STRUCTURE:
  - Swing highs / swing lows (5-bar lookback)
  - Distance from last swing high / swing low (ATR-normalised)
  - Break of Structure (BOS) — price broke last swing high/low
  - Change of Character (CHOCH) — opposing BOS after trend
  - Higher High / Lower Low / Higher Low / Lower High flags
  - Fair Value Gap (FVG) — bullish and bearish imbalance exists
  - FVG size relative to ATR
  - Session open gap (overnight gap up/down)
  - Overnight range (session high - session low before RTH)
  - Distance from session high / session low

ADDITIONAL INDICATORS:
  - VWAP (intraday, resets at 6 PM) + distance from VWAP
  - Bollinger Bands: %B (price position) + bandwidth
  - MACD line, signal line, histogram
  - Stochastic %K and %D
  - CCI (Commodity Channel Index, 14-period)
  - Williams %R (14-period)
  - Donchian channel position (price within 20-bar channel)

Total features: ~55

Run:
    /opt/miniconda3/bin/python project_pat_ml.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import vectorbt as vbt
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
MIN_CONFIDENCE     = 0.55
N_SPLITS           = 5
RANDOM_STATE       = 42
SAVE_DIR           = "/Users/tonythompson/backtest-vector"

# ═══════════════════════════════════════════════════════════════
# 1. FETCH DATA
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("PROJECT PAT — ML PIPELINE v3 (MARKET STRUCTURE EDITION)")
print("=" * 65)
print(f"\nDownloading {TICKER} {TIMEFRAME} ...")
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
open_  = df["Open"]; high = df["High"]; low = df["Low"]
close  = df["Close"]; volume = df["Volume"]
print(f"Bars: {len(df):,}  ({df.index[0].date()} to {df.index[-1].date()})")

# ═══════════════════════════════════════════════════════════════
# 2. LEVELS
# ═══════════════════════════════════════════════════════════════
sessions      = df.resample("24h", offset="18h").agg({"High":"max","Low":"min"}).shift(1)
prev_day_high = sessions["High"].reindex(df.index, method="ffill")
prev_day_low  = sessions["Low"].reindex(df.index, method="ffill")
fib50         = prev_day_low + 0.5 * (prev_day_high - prev_day_low)
prev_day_range= prev_day_high - prev_day_low

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
    ob_wi = np.zeros(len(df_h1))
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
        ob_wi[i]=(cur_hi-cur_lo) if cur_ex and not np.isnan(cur_hi) else 0.0
    return (pd.Series(ob_hi,index=df_h1.index), pd.Series(ob_lo,index=df_h1.index),
            pd.Series(ob_ex,index=df_h1.index), pd.Series(ob_wi,index=df_h1.index))

print("Detecting order blocks ...")
ob_hi_s, ob_lo_s, ob_ex_s, ob_wi_s = detect_ob_series(df)

# ═══════════════════════════════════════════════════════════════
# 4. STANDARD TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════
print("Computing indicators ...")

def rsi(s, p=14):
    d=s.diff(); g=d.clip(lower=0).rolling(p).mean()
    l=(-d.clip(upper=0)).rolling(p).mean()
    return 100-(100/(1+g/l.replace(0,np.nan)))

def atr(h,l,c,p=14):
    tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()

# Core
rsi_14=rsi(close,14); rsi_5=rsi(close,5)
atr_14=atr(high,low,close,14); atr_5=atr(high,low,close,5)
ema_8=close.ewm(span=8).mean()
ema_21=close.ewm(span=21).mean()
ema_50=close.ewm(span=50).mean()
vol_ratio=volume/volume.rolling(20).mean().replace(0,np.nan)
volatility_10=close.pct_change().rolling(10).std()
volatility_20=close.pct_change().rolling(20).std()
momentum_3=close/close.shift(3)-1
momentum_5=close/close.shift(5)-1
momentum_10=close/close.shift(10)-1
body_size=(close-open_).abs()
upper_wick=high-pd.concat([close,open_],axis=1).max(axis=1)
lower_wick=pd.concat([close,open_],axis=1).min(axis=1)-low
bar_is_bull=(close>=open_).astype(int)
returns_sign=(close.pct_change()>0).astype(int)
recent_bull_5=returns_sign.rolling(5).mean()
recent_bull_10=returns_sign.rolling(10).mean()

def bar_pos(idx):
    h=idx.hour+idx.minute/60
    return pd.Series((h-18)%24/23, index=idx)
bar_position=bar_pos(df.index)

# ── Bollinger Bands ──────────────────────────────────────────
bb_period = 20; bb_std = 2.0
bb_mid    = close.rolling(bb_period).mean()
bb_std_s  = close.rolling(bb_period).std()
bb_upper  = bb_mid + bb_std * bb_std_s
bb_lower  = bb_mid - bb_std * bb_std_s
bb_pct_b  = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)  # 0=lower, 1=upper
bb_width  = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)  # bandwidth %

# ── MACD ─────────────────────────────────────────────────────
macd_line   = close.ewm(span=12).mean() - close.ewm(span=26).mean()
macd_signal = macd_line.ewm(span=9).mean()
macd_hist   = macd_line - macd_signal

# ── Stochastic ───────────────────────────────────────────────
stoch_period = 14
stoch_low14  = low.rolling(stoch_period).min()
stoch_high14 = high.rolling(stoch_period).max()
stoch_k      = 100 * (close - stoch_low14) / (stoch_high14 - stoch_low14 + 1e-9)
stoch_d      = stoch_k.rolling(3).mean()

# ── CCI ──────────────────────────────────────────────────────
cci_period  = 14
typical_p   = (high + low + close) / 3
cci_sma     = typical_p.rolling(cci_period).mean()
cci_mad     = typical_p.rolling(cci_period).apply(lambda x: np.mean(np.abs(x - x.mean())))
cci         = (typical_p - cci_sma) / (0.015 * cci_mad.replace(0, np.nan))

# ── Williams %R ──────────────────────────────────────────────
willr_period = 14
willr_high   = high.rolling(willr_period).max()
willr_low    = low.rolling(willr_period).min()
willr        = -100 * (willr_high - close) / (willr_high - willr_low + 1e-9)

# ── Donchian Channel ─────────────────────────────────────────
don_period  = 20
don_upper   = high.rolling(don_period).max()
don_lower   = low.rolling(don_period).min()
don_mid     = (don_upper + don_lower) / 2
don_pos     = (close - don_lower) / (don_upper - don_lower + 1e-9)  # 0=bottom, 1=top

# ═══════════════════════════════════════════════════════════════
# 5. MARKET STRUCTURE FEATURES
# ═══════════════════════════════════════════════════════════════
print("Computing market structure ...")

# ── Swing Highs / Swing Lows (5-bar pivot) ───────────────────
# A swing high: highest of 5 bars (2 left, 2 right)
# Using rolling for simplicity (look-back only, no future leakage)
swing_window = 5
swing_high = high.rolling(swing_window).max()   # recent swing high
swing_low  = low.rolling(swing_window).min()    # recent swing low

# ── Distance from swing levels ───────────────────────────────
dist_swing_high = (close - swing_high)   # negative = below swing high
dist_swing_low  = (close - swing_low)    # positive = above swing low

# ── Break of Structure (BOS) ─────────────────────────────────
# Bullish BOS: close breaks above previous swing high
# Bearish BOS: close breaks below previous swing low
prev_swing_high = swing_high.shift(1)
prev_swing_low  = swing_low.shift(1)
bos_bull = (close > prev_swing_high).astype(int)
bos_bear = (close < prev_swing_low).astype(int)
bos_any  = (bos_bull | bos_bear).astype(int)

# ── Higher High / Lower Low / Higher Low / Lower High ────────
# Compare current swing vs 5 bars ago
prev5_swing_high = swing_high.shift(5)
prev5_swing_low  = swing_low.shift(5)
higher_high = (swing_high > prev5_swing_high).astype(int)
lower_low   = (swing_low  < prev5_swing_low).astype(int)
higher_low  = (swing_low  > prev5_swing_low).astype(int)
lower_high  = (swing_high < prev5_swing_high).astype(int)

# Bullish structure: higher highs AND higher lows
# Bearish structure: lower highs AND lower lows
bull_structure = (higher_high & higher_low).astype(int)
bear_structure = (lower_high  & lower_low).astype(int)

# ── Change of Character (CHOCH) ──────────────────────────────
# After a bear structure, a bullish BOS = CHOCH up (potential reversal)
# After a bull structure, a bearish BOS = CHOCH down
choch_up   = ((bear_structure.shift(1).fillna(0).astype(int) == 1) & (bos_bull == 1)).astype(int)
choch_down = ((bull_structure.shift(1).fillna(0).astype(int) == 1) & (bos_bear == 1)).astype(int)

# ── Fair Value Gap (FVG) ─────────────────────────────────────
# Bullish FVG: low[i] > high[i-2]  (gap up between 3 bars)
# Bearish FVG: high[i] < low[i-2]  (gap down between 3 bars)
fvg_bull = (low > high.shift(2)).astype(int)
fvg_bear = (high < low.shift(2)).astype(int)
fvg_any  = (fvg_bull | fvg_bear).astype(int)
# FVG size relative to ATR
fvg_bull_size = (low - high.shift(2)).clip(lower=0) / atr_14.replace(0, np.nan)
fvg_bear_size = (low.shift(2) - high).clip(lower=0) / atr_14.replace(0, np.nan)

# ── Session group: cumulative count of 6 PM resets ──────────
grp = pd.Series(
    (df.index.hour == SESSION_RESET_HOUR).astype(int).cumsum(),
    index=df.index
)
grp_arr = grp.values   # integer array, same length as df

# ── VWAP (resets each session) ───────────────────────────────
typical_price = (high + low + close) / 3
tp_vol        = typical_price * volume
cum_tpvol     = tp_vol.groupby(grp_arr).cumsum()
cum_vol_grp   = volume.groupby(grp_arr).cumsum()
vwap          = cum_tpvol / cum_vol_grp.replace(0, np.nan)
dist_vwap     = (close - vwap) / atr_14.replace(0, np.nan)

# ── Session High / Low ───────────────────────────────────────
session_high      = high.groupby(grp_arr).cummax()
session_low       = low.groupby(grp_arr).cummin()
dist_session_high = (close - session_high) / atr_14.replace(0, np.nan)
dist_session_low  = (close - session_low)  / atr_14.replace(0, np.nan)

# ── Session open price ───────────────────────────────────────
session_open_price = open_.groupby(grp_arr).transform("first")

# ── Overnight gap ────────────────────────────────────────────
# For each session, find the last close of the PREVIOUS session
last_close_by_session = close.groupby(grp_arr).last()          # indexed by grp int
prev_last_close       = last_close_by_session.shift(1)         # shift by 1 session
# Map each bar back to its session's prev close
prev_close_mapped = grp.map(prev_last_close)
overnight_gap = (session_open_price - prev_close_mapped) / atr_14.replace(0, np.nan)

# ═══════════════════════════════════════════════════════════════
# 6. CROSSOVERS
# ═══════════════════════════════════════════════════════════════
idx=df.index; dow=idx.dayofweek; hour_=idx.hour
def crossover(a,b):
    r=(a>b)&~(np.roll(a,1)>np.roll(b,1)); r[0]=False; return r
def crossunder(a,b):
    r=(a<b)&~(np.roll(a,1)<np.roll(b,1)); r[0]=False; return r
cross_above_high=crossover(close.values,prev_day_high.values)
cross_below_low =crossunder(close.values,prev_day_low.values)
cross_above_fib =crossover(close.values,fib50.values)
cross_below_fib =crossunder(close.values,fib50.values)
force_close_mask=hour_==FORCE_CLOSE_HOUR

ob_hi_arr=ob_hi_s.values; ob_lo_arr=ob_lo_s.values
ob_ex_arr=ob_ex_s.values; ob_wi_arr=ob_wi_s.values
open_arr=open_.values; high_arr=high.values
low_arr=low.values; close_arr=close.values
n_bars=len(df)

# ═══════════════════════════════════════════════════════════════
# 7. GENERATE SIGNALS + FEATURES + LABELS
# ═══════════════════════════════════════════════════════════════
print("Generating signals, features, and labels ...")
signal_records=[]; in_pos=False; pos_side=0
cur_tp=np.nan; cur_sl=np.nan; dtc=0; last_date=None; entry_side=0

for i in range(1, n_bars):
    bar_date=idx[i].date()
    if bar_date!=last_date: dtc=0; last_date=bar_date

    if force_close_mask[i] and in_pos:
        pnl=(close_arr[i]-signal_records[-1]["_entry_price"])*entry_side
        signal_records[-1].update({"exit_price":close_arr[i],"pnl":pnl,
                                    "label_win":int(pnl>0),"exit_reason":"force_close"})
        in_pos=False; pos_side=0; continue

    if in_pos:
        hit_tp=hit_sl=False
        if pos_side==1:
            if low_arr[i]<=cur_sl: hit_sl=True
            if high_arr[i]>=cur_tp: hit_tp=True
        else:
            if high_arr[i]>=cur_sl: hit_sl=True
            if low_arr[i]<=cur_tp: hit_tp=True
        if hit_tp or hit_sl:
            ep=cur_tp if hit_tp else cur_sl
            pnl=(ep-signal_records[-1]["_entry_price"])*entry_side
            signal_records[-1].update({"exit_price":ep,"pnl":pnl,
                                        "label_win":int(pnl>0),
                                        "exit_reason":"tp" if hit_tp else "sl"})
            in_pos=False; pos_side=0; continue

    if in_pos or force_close_mask[i]: continue
    d=dow[i]
    if d>4: continue
    day_key=["Mon","Tue","Wed","Thu","Fri"][d]
    en,mx,strat,tp_pts,sl_pts=DAY_CONFIG[day_key]
    if not en or dtc>=mx: continue

    c=close_arr[i]; o=open_arr[i]
    sig_long=sig_short=False
    if strat=="crossover":
        if cross_above_high[i]: sig_long=True
        elif cross_below_low[i]: sig_short=True
    elif strat=="fib50":
        if cross_above_fib[i]: sig_long=True
        elif cross_below_fib[i]: sig_short=True
    elif strat=="ob":
        if ob_ex_arr[i]:
            oh=ob_hi_arr[i]; ol=ob_lo_arr[i]
            if ol<=o<=oh:
                if c<ol: sig_short=True
                elif c>oh: sig_long=True
    if not (sig_long or sig_short): continue

    side=1 if sig_long else -1
    tp_price=c+tp_pts*POINT_VALUE*side
    sl_price=c-sl_pts*POINT_VALUE*side
    atv=max(atr_14.iloc[i], 0.01)
    pdh=prev_day_high.iloc[i]; pdl=prev_day_low.iloc[i]
    pdr=max(prev_day_range.iloc[i], atv)

    def _s(series): return float(series.iloc[i]) if not np.isnan(series.iloc[i]) else 0.0

    rec={
        # ── TIME ─────────────────────────────────────────────
        "hour":              hour_[i],
        "day_of_week":       d,
        "bar_position":      _s(bar_position),

        # ── DISTANCE FROM LEVELS ─────────────────────────────
        "dist_pdh_atr":      (c-pdh)/atv,
        "dist_pdl_atr":      (c-pdl)/atv,
        "dist_fib50_atr":    (c-fib50.iloc[i])/atv,
        "dist_ema8_atr":     (c-ema_8.iloc[i])/atv,
        "dist_ema21_atr":    (c-ema_21.iloc[i])/atv,
        "dist_ema50_atr":    (c-ema_50.iloc[i])/atv,

        # ── TREND ────────────────────────────────────────────
        "ema8_above_ema21":  int(ema_8.iloc[i]>ema_21.iloc[i]),
        "ema21_above_ema50": int(ema_21.iloc[i]>ema_50.iloc[i]),

        # ── MOMENTUM ─────────────────────────────────────────
        "momentum_3":        _s(momentum_3),
        "momentum_5":        _s(momentum_5),
        "momentum_10":       _s(momentum_10),
        "recent_bull_5":     _s(recent_bull_5),
        "recent_bull_10":    _s(recent_bull_10),

        # ── VOLATILITY ───────────────────────────────────────
        "atr_14":            atv,
        "atr_ratio":         atr_5.iloc[i]/atv,
        "volatility_10":     _s(volatility_10),
        "volatility_20":     _s(volatility_20),

        # ── VOLUME ───────────────────────────────────────────
        "volume_ratio":      _s(vol_ratio),

        # ── RSI ──────────────────────────────────────────────
        "rsi_14":            _s(rsi_14),
        "rsi_5":             _s(rsi_5),
        "rsi_diff":          _s(rsi_5)-_s(rsi_14),

        # ── CANDLE STRUCTURE ─────────────────────────────────
        "body_ratio":        body_size.iloc[i]/atv,
        "upper_wick_atr":    upper_wick.iloc[i]/atv,
        "lower_wick_atr":    lower_wick.iloc[i]/atv,
        "bar_is_bull":       bar_is_bull.iloc[i],

        # ── ORDER BLOCK ──────────────────────────────────────
        "ob_exists":         int(ob_ex_arr[i]),
        "ob_width_atr":      ob_wi_arr[i]/atv,

        # ── PREV DAY RANGE ───────────────────────────────────
        "prev_day_range_atr": pdr/atv,

        # ══ NEW: BOLLINGER BANDS ═════════════════════════════
        "bb_pct_b":          _s(bb_pct_b),     # 0=at lower, 1=at upper
        "bb_width":          _s(bb_width),     # wider = more volatile

        # ══ NEW: MACD ════════════════════════════════════════
        "macd_line":         _s(macd_line)/atv,
        "macd_signal":       _s(macd_signal)/atv,
        "macd_hist":         _s(macd_hist)/atv,
        "macd_above_signal": int(_s(macd_line)>_s(macd_signal)),

        # ══ NEW: STOCHASTIC ══════════════════════════════════
        "stoch_k":           _s(stoch_k),
        "stoch_d":           _s(stoch_d),
        "stoch_k_above_d":   int(_s(stoch_k)>_s(stoch_d)),
        "stoch_overbought":  int(_s(stoch_k)>80),
        "stoch_oversold":    int(_s(stoch_k)<20),

        # ══ NEW: CCI ════════════════════════════════════════
        "cci":               _s(cci),
        "cci_overbought":    int(_s(cci)>100),
        "cci_oversold":      int(_s(cci)<-100),

        # ══ NEW: WILLIAMS %R ════════════════════════════════
        "willr":             _s(willr),

        # ══ NEW: DONCHIAN ════════════════════════════════════
        "don_position":      _s(don_pos),      # 0=bottom, 1=top of 20-bar range

        # ══ NEW: VWAP ════════════════════════════════════════
        "dist_vwap_atr":     _s(dist_vwap),    # above/below VWAP
        "price_above_vwap":  int(c > _s(vwap)) if not np.isnan(_s(vwap)) else 0,

        # ══ NEW: MARKET STRUCTURE ════════════════════════════
        "dist_swing_high_atr": _s(dist_swing_high)/atv,
        "dist_swing_low_atr":  _s(dist_swing_low)/atv,
        "bos_bull":            _s(bos_bull),   # bullish break of structure
        "bos_bear":            _s(bos_bear),   # bearish break of structure
        "higher_high":         _s(higher_high),
        "lower_low":           _s(lower_low),
        "higher_low":          _s(higher_low),
        "lower_high":          _s(lower_high),
        "bull_structure":      _s(bull_structure),  # HH + HL = bullish
        "bear_structure":      _s(bear_structure),  # LH + LL = bearish
        "choch_up":            _s(choch_up),   # change of character up
        "choch_down":          _s(choch_down), # change of character down
        "fvg_bull":            _s(fvg_bull),   # bullish fair value gap present
        "fvg_bear":            _s(fvg_bear),   # bearish fair value gap present
        "fvg_bull_size_atr":   _s(fvg_bull_size),
        "fvg_bear_size_atr":   _s(fvg_bear_size),
        "dist_session_high_atr": _s(dist_session_high),
        "dist_session_low_atr":  _s(dist_session_low),
        "overnight_gap_atr":     _s(overnight_gap),

        # ── META (not used as features) ──────────────────────
        "_bar_idx":   i, "_timestamp": idx[i], "_entry_price": c,
        "_tp_price":  tp_price, "_sl_price": sl_price,
        "_side":      side, "_day": day_key, "_strat": strat,
        "exit_price": np.nan, "pnl": np.nan, "label_win": np.nan, "exit_reason":"",
    }
    signal_records.append(rec)
    dtc+=1; in_pos=True; pos_side=side
    cur_tp=tp_price; cur_sl=sl_price; entry_side=side

sig_df=pd.DataFrame(signal_records).dropna(subset=["label_win"]).copy()
sig_df["label_win"]=sig_df["label_win"].astype(int)
print(f"Total labelled signals: {len(sig_df)}")
print(f"Overall win rate: {sig_df['label_win'].mean():.1%}")
print(f"\nWin rate by day:")
print(sig_df.groupby("_day")["label_win"].agg(["count","mean"]).rename(
    columns={"count":"Trades","mean":"Win_Rate"}).to_string())

# ═══════════════════════════════════════════════════════════════
# 8. FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════
META_COLS=["_bar_idx","_timestamp","_entry_price","_tp_price","_sl_price",
           "_side","_day","_strat","exit_price","pnl","label_win","exit_reason"]
feature_cols=[c for c in sig_df.columns if c not in META_COLS]
X=sig_df[feature_cols].fillna(0).astype(float)
y_win=sig_df["label_win"].astype(int)
y_dir=(sig_df["_side"]==1).astype(int)

print(f"\nTotal features: {len(feature_cols)}")
print("Feature groups:")
print(f"  Time:             hour, day_of_week, bar_position")
print(f"  Price levels:     dist_pdh/pdl/fib50/ema8/ema21/ema50")
print(f"  Trend:            ema8_above_ema21, ema21_above_ema50")
print(f"  Momentum:         momentum_3/5/10, recent_bull_5/10")
print(f"  Volatility:       atr_14, atr_ratio, volatility_10/20")
print(f"  Volume:           volume_ratio")
print(f"  RSI:              rsi_14, rsi_5, rsi_diff")
print(f"  Candle:           body_ratio, upper/lower_wick, bar_is_bull")
print(f"  Order Block:      ob_exists, ob_width_atr")
print(f"  Bollinger Bands:  bb_pct_b, bb_width")
print(f"  MACD:             macd_line/signal/hist, macd_above_signal")
print(f"  Stochastic:       stoch_k/d, k_above_d, overbought/oversold")
print(f"  CCI:              cci, overbought/oversold")
print(f"  Williams %R:      willr")
print(f"  Donchian:         don_position")
print(f"  VWAP:             dist_vwap_atr, price_above_vwap")
print(f"  Market Structure: swing H/L, BOS, CHOCH, HH/LL/HL/LH,")
print(f"                    bull/bear structure, FVG, session H/L,")
print(f"                    overnight gap")

# ═══════════════════════════════════════════════════════════════
# 9. WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"WALK-FORWARD VALIDATION ({N_SPLITS} folds, TimeSeriesSplit)")
print(f"{'='*65}")

tscv=TimeSeriesSplit(n_splits=N_SPLITS)
xgb_params=dict(n_estimators=400,max_depth=4,learning_rate=0.04,
                subsample=0.8,colsample_bytree=0.7,min_child_weight=3,
                eval_metric="logloss",random_state=RANDOM_STATE,n_jobs=-1)

win_scores=[]; dir_scores=[]
for fold,(train_idx,test_idx) in enumerate(tscv.split(X)):
    Xtr,Xte=X.iloc[train_idx],X.iloc[test_idx]
    yw_tr,yw_te=y_win.iloc[train_idx],y_win.iloc[test_idx]
    yd_tr,yd_te=y_dir.iloc[train_idx],y_dir.iloc[test_idx]
    mw=XGBClassifier(**xgb_params); mw.fit(Xtr,yw_tr)
    md=XGBClassifier(**xgb_params); md.fit(Xtr,yd_tr)
    wa=accuracy_score(yw_te,mw.predict(Xte))
    da=accuracy_score(yd_te,md.predict(Xte))
    win_scores.append(wa); dir_scores.append(da)
    print(f"  Fold {fold+1}: Win filter={wa:.3f}  Direction={da:.3f}  "
          f"(train={len(train_idx)}, test={len(test_idx)})")

avg_win=np.mean(win_scores); avg_dir=np.mean(dir_scores)
print(f"\n  Avg Win filter: {avg_win:.3f} +/- {np.std(win_scores):.3f}")
print(f"  Avg Direction:  {avg_dir:.3f} +/- {np.std(dir_scores):.3f}")
print(f"\n  Interpretation:")
if avg_win > 0.56:
    verdict_win = "GOOD — model finding real patterns above baseline"
elif avg_win > 0.52:
    verdict_win = "MODEST — slight edge, use carefully"
else:
    verdict_win = "WEAK — no reliable edge found"
if avg_dir > 0.56:
    verdict_dir = "GOOD — genuine directional edge"
elif avg_dir > 0.52:
    verdict_dir = "MODEST — marginal directional signal"
else:
    verdict_dir = "WEAK — essentially random"
print(f"  Win filter ({avg_win:.1%}): {verdict_win}")
print(f"  Direction  ({avg_dir:.1%}): {verdict_dir}")

# ═══════════════════════════════════════════════════════════════
# 10. TRAIN FINAL MODELS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TRAINING FINAL MODELS")
print(f"{'='*65}")
final_win=XGBClassifier(**xgb_params); final_win.fit(X,y_win)
final_dir=XGBClassifier(**xgb_params); final_dir.fit(X,y_dir)

sig_df["ml_win_prob"]=final_win.predict_proba(X)[:,1]
sig_df["ml_dir_prob"]=final_dir.predict_proba(X)[:,1]
sig_df["ml_approved"]=sig_df["ml_win_prob"]>=MIN_CONFIDENCE

kept=sig_df[sig_df["ml_approved"]]
dropped=sig_df[~sig_df["ml_approved"]]
print(f"\nML filter at >= {MIN_CONFIDENCE:.0%} confidence:")
print(f"  Signals kept:     {len(kept)} / {len(sig_df)}")
print(f"  Win rate kept:    {kept['label_win'].mean():.1%}  (baseline: {sig_df['label_win'].mean():.1%})")
print(f"  Win rate dropped: {dropped['label_win'].mean():.1%}")
lift = kept['label_win'].mean() - sig_df['label_win'].mean()
print(f"  Lift vs baseline: {lift:+.1%}")

fi_win=pd.Series(final_win.feature_importances_,index=feature_cols).sort_values(ascending=False)
print(f"\nTop 15 features (win filter):")
print(fi_win.head(15).to_string())

fi_dir=pd.Series(final_dir.feature_importances_,index=feature_cols).sort_values(ascending=False)
print(f"\nTop 15 features (direction model):")
print(fi_dir.head(15).to_string())

# ═══════════════════════════════════════════════════════════════
# 11. ML BACKTEST
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("FINAL BACKTEST — ML-APPROVED SIGNALS ONLY")
print(f"{'='*65}")

approved=sig_df[sig_df["ml_approved"]].sort_values("_bar_idx").reset_index(drop=True)
ent_long=np.zeros(n_bars,dtype=bool); ent_short=np.zeros(n_bars,dtype=bool)
exits_ml=np.zeros(n_bars,dtype=bool)
in_pos=False; cur_tp=np.nan; cur_sl=np.nan; pos_side=0; ai=0; na_=len(approved)

for i in range(1,n_bars):
    if force_close_mask[i] and in_pos:
        exits_ml[i]=True; in_pos=False; pos_side=0; continue
    if in_pos:
        if pos_side==1:
            if low_arr[i]<=cur_sl or high_arr[i]>=cur_tp:
                exits_ml[i]=True; in_pos=False; pos_side=0; continue
        else:
            if high_arr[i]>=cur_sl or low_arr[i]<=cur_tp:
                exits_ml[i]=True; in_pos=False; pos_side=0; continue
    if in_pos or force_close_mask[i]: continue
    if ai<na_:
        s=approved.iloc[ai]
        if s["_bar_idx"]==i:
            ai+=1; side=int(s["_side"])
            if side==1: ent_long[i]=True
            else: ent_short[i]=True
            in_pos=True; pos_side=side
            cur_tp=s["_tp_price"]; cur_sl=s["_sl_price"]

pf_ml=vbt.Portfolio.from_signals(
    close=close,open=open_,high=high,low=low,
    entries=pd.Series(ent_long,index=df.index),
    short_entries=pd.Series(ent_short,index=df.index),
    exits=pd.Series(exits_ml,index=df.index),
    init_cash=INIT_CASH,fixed_fees=FIXED_FEES,freq=TIMEFRAME)

print("\n--- ML Portfolio Stats ---")
print(pf_ml.stats())

trades_ml=pf_ml.trades.records_readable
if len(trades_ml):
    trades_ml["DayOfWeek"]=pd.to_datetime(trades_ml["Entry Timestamp"]).dt.day_name()
    print("\n--- ML Results by day ---")
    sm=trades_ml.groupby("DayOfWeek").agg(
        Trades=("PnL","count"),
        Win_Rate=("Return",lambda x:f"{(x>0).mean():.1%}"),
        Avg_PnL=("PnL","mean"),Total_PnL=("PnL","sum"))
    sm=sm.reindex([d for d in ["Monday","Tuesday","Wednesday","Thursday","Friday"] if d in sm.index])
    print(sm.to_string())

# ═══════════════════════════════════════════════════════════════
# 12. BASELINE BACKTEST
# ═══════════════════════════════════════════════════════════════
ent_b=np.zeros(n_bars,dtype=bool); ext_b=np.zeros(n_bars,dtype=bool); sz_b=np.zeros(n_bars)
in_pos=False; pos_side=0; cur_tp=np.nan; cur_sl=np.nan; dtc=0; last_date=None
for i in range(1,n_bars):
    bar_date=idx[i].date()
    if bar_date!=last_date: dtc=0; last_date=bar_date
    if force_close_mask[i] and in_pos:
        ext_b[i]=True; in_pos=False; pos_side=0; continue
    if in_pos:
        if pos_side==1:
            if low_arr[i]<=cur_sl or high_arr[i]>=cur_tp:
                ext_b[i]=True; in_pos=False; pos_side=0; continue
        else:
            if high_arr[i]>=cur_sl or low_arr[i]<=cur_tp:
                ext_b[i]=True; in_pos=False; pos_side=0; continue
    if in_pos or force_close_mask[i]: continue
    d=dow[i]
    if d>4: continue
    dk=["Mon","Tue","Wed","Thu","Fri"][d]
    en,mx,strat,tp_pts,sl_pts=DAY_CONFIG[dk]
    if not en or dtc>=mx: continue
    c=close_arr[i]; o=open_arr[i]
    sl=sh=False
    if strat=="crossover":
        if cross_above_high[i]: sl=True
        elif cross_below_low[i]: sh=True
    elif strat=="fib50":
        if cross_above_fib[i]: sl=True
        elif cross_below_fib[i]: sh=True
    elif strat=="ob":
        if ob_ex_arr[i]:
            oh=ob_hi_arr[i]; ol=ob_lo_arr[i]
            if ol<=o<=oh:
                if c<ol: sh=True
                elif c>oh: sl=True
    if not (sl or sh): continue
    if sl: sz_b[i]=1; pos_side=1; cur_tp=c+tp_pts*POINT_VALUE; cur_sl=c-sl_pts*POINT_VALUE
    else:  sz_b[i]=-1; pos_side=-1; cur_tp=c-tp_pts*POINT_VALUE; cur_sl=c+sl_pts*POINT_VALUE
    ent_b[i]=True; in_pos=True; dtc+=1

pf_base=vbt.Portfolio.from_signals(
    close=close,open=open_,high=high,low=low,
    entries=pd.Series(ent_b&(sz_b==1),index=df.index),
    short_entries=pd.Series(ent_b&(sz_b==-1),index=df.index),
    exits=pd.Series(ext_b,index=df.index),
    init_cash=INIT_CASH,fixed_fees=FIXED_FEES,freq=TIMEFRAME)

# ═══════════════════════════════════════════════════════════════
# 13. COMPARISON
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("COMPARISON — No ML vs With ML")
print(f"{'='*65}")
b_ret=pf_base.total_return()*100; m_ret=pf_ml.total_return()*100
b_sr=pf_base.sharpe_ratio();      m_sr=pf_ml.sharpe_ratio()
b_dd=pf_base.max_drawdown()*100;  m_dd=pf_ml.max_drawdown()*100
b_wr=pf_base.trades.win_rate()*100 if len(pf_base.trades.records) else 0
m_wr=pf_ml.trades.win_rate()*100   if len(pf_ml.trades.records) else 0
print(f"\n{'Metric':<28}{'No ML':>12}{'With ML':>12}{'Change':>12}")
print("-"*64)
print(f"{'Total Return %':<28}{b_ret:>11.2f}%{m_ret:>11.2f}%{m_ret-b_ret:>+11.2f}%")
print(f"{'Sharpe Ratio':<28}{b_sr:>12.3f}{m_sr:>12.3f}{m_sr-b_sr:>+12.3f}")
print(f"{'Max Drawdown %':<28}{b_dd:>11.2f}%{m_dd:>11.2f}%{m_dd-b_dd:>+11.2f}%")
print(f"{'Win Rate %':<28}{b_wr:>11.1f}%{m_wr:>11.1f}%{m_wr-b_wr:>+11.1f}%")
print(f"{'Total Trades':<28}{len(pf_base.trades.records):>12}{len(pf_ml.trades.records):>12}")
print(f"{'End Value $':<28}{pf_base.final_value():>12,.2f}{pf_ml.final_value():>12,.2f}")
print(f"\n  Walk-forward win filter: {avg_win:.1%} — {verdict_win}")
print(f"  Walk-forward direction:  {avg_dir:.1%} — {verdict_dir}")

# ═══════════════════════════════════════════════════════════════
# 14. SAVE CHARTS + DATA
# ═══════════════════════════════════════════════════════════════
fig,axes=plt.subplots(1,2,figsize=(18,8))
fi_win_s=pd.Series(final_win.feature_importances_,index=feature_cols).sort_values()
fi_win_s.tail(20).plot(kind="barh",ax=axes[0],color="#2196F3")
axes[0].set_title("Top 20 Features — Win Filter",fontsize=12)
fi_dir_s=pd.Series(final_dir.feature_importances_,index=feature_cols).sort_values()
fi_dir_s.tail(20).plot(kind="barh",ax=axes[1],color="#4CAF50")
axes[1].set_title("Top 20 Features — Direction Model",fontsize=12)
plt.tight_layout()
chart_path=f"{SAVE_DIR}/feature_importance.png"
plt.savefig(chart_path,dpi=150,bbox_inches="tight")
print(f"\nFeature importance chart: {chart_path}")
sig_df.to_csv(f"{SAVE_DIR}/project_pat_ml_signals.csv",index=False)
print(f"Signals + ML scores: {SAVE_DIR}/project_pat_ml_signals.csv")

pf_ml.plot().show()
print("\nDone.")
