# Project Pat — NQ Futures Algorithmic Trading System

> **Disclaimer:** This project is for educational and portfolio purposes only. Nothing here constitutes financial advice. Past backtest results do not guarantee future performance.

---

## Overview

Project Pat is a full end-to-end algorithmic trading system built for **NQ=F (Nasdaq-100 E-mini futures)** and **MNQ=F (Micro Nasdaq-100 futures)** on the **1-hour timeframe**. The project covers the complete quant workflow:

1. **Strategy design** — rule-based day-of-week strategies in TradingView Pine Script v6
2. **Python backtesting** — vectorbt-powered historical simulation
3. **Parameter optimization** — grid search across TP/SL and strategy types per day
4. **Machine learning** — XGBoost models with 68 leakage-free features including market structure
5. **Indicator filtering** — VWAP, MACD, EMA 21, Williams %R, Bollinger Bands applied as pre-trade filters
6. **TradingView deployment** — production-ready Pine Script v6 strategy with alerts

---

## Strategy Logic

Each day of the week runs a different signal type, optimized independently:

| Day | Strategy | Signal | TP | SL |
|---|---|---|---|---|
| Monday | Prev High/Low Crossover | Price breaks prev session high or low | 100 pts | 30 pts |
| Tuesday | Order Block Reject | Open inside OB zone, close breaks outside | 50 pts | 100 pts |
| Wednesday | Order Block Reject | Same as Tuesday | 100 pts | 50 pts |
| Thursday | Prev High/Low Crossover | Price breaks prev session high or low | 50 pts | 100 pts |
| Friday | Fib 50% Crossover | Price crosses the 50% fib of prev session | 100 pts | 60 pts |

**Order Block Detection:** Pattern-based identification of 1H candle sequences following the structure: `leadSameDir(≥2) → obCandles(1-2 opposite) → trailSameDir(≥2)`. The OB zone is drawn from the opposing candles and reset at each 6 PM session open.

**Session handling:** NQ futures sessions open at 6 PM ET. All daily trade counters and OB zones reset at 6 PM. Force-close fires at 3:55 PM ET.

---

## Indicator Filters

Before any entry is taken, the following 5 indicators are checked. A minimum of 3 must agree with the signal direction (configurable):

| Filter | Long condition | Short condition |
|---|---|---|
| **VWAP** | Price above VWAP | Price below VWAP |
| **MACD** | MACD line above signal | MACD line below signal |
| **EMA 21** | Price above EMA 21 | Price below EMA 21 |
| **Williams %R** | Not overbought (< -20) | Not oversold (> -80) |
| **Bollinger %B** | %B below 0.8 | %B above 0.2 |

---

## Repository Structure

```
project-pat/
├── project_pat_vectorbt.py     # Main backtest with indicator filters
├── project_pat_optimize.py     # Grid search optimizer (3 strategies × 7 TP × 7 SL)
├── project_pat_ml.py           # Full ML pipeline (XGBoost, walk-forward validation)
├── tradingview/
│   └── project_pat_v2.pine     # TradingView Pine Script v6 (paste into editor)
├── results/
│   └── ML_RESULTS.txt          # Full ML pipeline output
├── docs/
│   └── strategy_overview.md    # Detailed strategy documentation
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### Requirements
- Python 3.10+
- macOS, Linux, or Windows
- Conda (recommended) or pip

### Step 1 — Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/project-pat.git
cd project-pat
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

Or with conda:
```bash
conda create -n projectpat python=3.13
conda activate projectpat
pip install -r requirements.txt
```

### Step 3 — Run the backtest
```bash
python project_pat_vectorbt.py
```

### Step 4 — Run the optimizer (optional, takes ~5 minutes)
```bash
python project_pat_optimize.py
```

### Step 5 — Run the ML pipeline (optional, takes ~10 minutes)
```bash
python project_pat_ml.py
```

---

## Dependencies

```
vectorbt>=1.0.0
yfinance>=0.2.60
xgboost>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.23.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

All installed via:
```bash
pip install -r requirements.txt
```

---

## Results

### Backtest (No ML filters) — 730 days, NQ=F 1H
| Metric | Value |
|---|---|
| Total Return | -5.39% |
| Sharpe Ratio | -0.114 |
| Max Drawdown | -20.21% |
| Win Rate | 47.0% |
| Total Trades | 496 |

### After ML Filtering (XGBoost, 55% confidence threshold)
| Metric | No ML | With ML | Change |
|---|---|---|---|
| Total Return | -5.39% | +102.73% | +108.12% |
| Sharpe Ratio | -0.114 | 2.731 | +2.845 |
| Max Drawdown | -20.21% | -9.05% | +11.16% |
| Win Rate | 47.0% | 77.8% | +30.8% |
| Total Trades | 496 | 257 | — |
| End Value ($25k start) | $23,651 | $50,681 | — |

> **Note on ML results:** The final backtest return of 102% uses a model trained on all data — this is an in-sample result. The honest walk-forward accuracy is **54.2%** for the win filter (modest but real edge). The 77.8% backtest win rate reflects the model having seen this data before. Forward performance should be validated with live paper trading.

### ML Walk-Forward Validation (5-fold TimeSeriesSplit)
| Fold | Win Filter | Direction |
|---|---|---|
| 1 | 56.7% | 100.0% |
| 2 | 51.0% | 100.0% |
| 3 | 48.1% | 99.0% |
| 4 | 53.8% | 99.0% |
| 5 | 61.5% | 99.0% |
| **Average** | **54.2%** | **99.8%** |

### Results by Day (Optimized params, no ML)
| Day | Strategy | Trades | Win Rate | Total PnL |
|---|---|---|---|---|
| Monday | Crossover | 79 | 38.0% | -$1,034 |
| Tuesday | OB Reject | 155 | 56.1% | -$1,263 |
| Wednesday | OB Reject | 136 | 39.0% | +$669 |
| **Thursday** | **Crossover** | **85** | **55.3%** | **+$3,046** |
| Friday | Fib 50% | 41 | 39.0% | -$2,765 |

---

## Machine Learning Pipeline

### Features (68 total, leakage-free)

| Category | Features |
|---|---|
| Time | hour, day_of_week, bar_position |
| Price levels | dist from prev day H/L, fib50, EMA 8/21/50 (ATR-normalised) |
| Trend | EMA 8 > EMA 21, EMA 21 > EMA 50 |
| Momentum | 3/5/10-bar returns, recent bullish bar ratio |
| Volatility | ATR 14, ATR ratio (short/long), rolling std 10/20 |
| Volume | ratio vs 20-bar average |
| RSI | 14-period, 5-period, difference |
| Candle structure | body ratio, upper/lower wick, bar direction |
| Bollinger Bands | %B position, bandwidth |
| MACD | line, signal, histogram, line vs signal |
| Stochastic | %K, %D, crossover, overbought/oversold |
| CCI | value, overbought/oversold flags |
| Williams %R | raw value |
| Donchian | price position within 20-bar channel |
| VWAP | distance from session VWAP |
| **Market Structure** | swing H/L distance, BOS bull/bear, CHOCH up/down, HH/LL/HL/LH, bull/bear structure, FVG bull/bear + size, session H/L, overnight gap |

### Models
- **Win Filter:** XGBoost Classifier — predicts whether signal will be a winner
- **Direction Model:** XGBoost Classifier — predicts long vs short direction
- **Validation:** TimeSeriesSplit (5 folds) — no future data leakage
- **Hyperparameters:** n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.8

### Top Features (Win Filter)
1. `day_of_week` — strongest signal, confirms day-specific edge
2. `willr` — Williams %R momentum state
3. `dist_fib50_atr` — distance from Fib 50% level
4. `dist_vwap_atr` — VWAP distance
5. `macd_signal` — MACD signal line state

---

## TradingView Setup

1. Open TradingView on **NQ=F** or **MNQ=F**, set chart to **1H**
2. Open **Pine Script Editor** (bottom panel)
3. Paste contents of `tradingview/project_pat_v2.pine`
4. Click **Save** → **Add to chart**

### What you'll see
- **Green triangles (▲)** — long entry signals with filter score e.g. `Thu Long [4/5]`
- **Red triangles (▼)** — short entry signals
- **Purple line** — VWAP
- **Blue line** — EMA 21
- **Red step lines** — Prev Day High and Low
- **Orange step line** — Fib 50%
- **Teal background** — all 5 filters bullish
- **Red background** — all 5 filters bearish

---

## Known Issues & Troubleshooting

### Python

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: vectorbt` | Wrong Python environment | Run with `/opt/miniconda3/bin/python` |
| `AttributeError: .values on ndarray` | Extra `.values` call on numpy array | Already fixed in current version |
| `TypeError: Cannot compare dtypes int64 and datetime64` | Index mismatch in session groupby | Use `groupby(array).cumsum()` not resample ngroup |
| `SyntaxError: invalid syntax` (optimize.py line 102) | Bad generator expression with `else` clause | Fixed — remove that line entirely |
| `OSError: Read-only file system` | CSV saving to `/` root | Change `SAVE_PATH` to your project folder |
| `No trades found` | Data gap or wrong ticker | Check `df["Volume"] > 0` filter, try `PERIOD = "60d"` |

### TradingView Pine Script

| Error | Cause | Fix |
|---|---|---|
| `CE10013 — shorttitle too long` | Short title over 10 chars | Use `"ProjPat v2"` |
| `CE10088 — cannot modify global variable in function` | `var` mutation inside function | Move `dayTradeCount += 1` outside function to top-level |
| `CE10013 — mismatched input ":"` | Inline `if cond : statement` syntax | Split to two lines: `if cond` then indented statement |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.13 | Core language |
| vectorbt | Vectorized backtesting engine |
| yfinance | Historical market data (Yahoo Finance) |
| XGBoost | Gradient boosted tree ML models |
| scikit-learn | Walk-forward validation (TimeSeriesSplit) |
| pandas + numpy | Data processing |
| matplotlib | Feature importance charts |
| TradingView Pine Script v6 | Live strategy deployment |

---

## Future Work

- [ ] Walk-forward optimization (rolling window re-optimization)
- [ ] Live paper trading integration via TradersPost webhooks
- [ ] Expand to ES=F (S&P 500 futures)
- [ ] Add market regime filter (VIX-based)
- [ ] Streamlit dashboard for real-time signal monitoring
- [ ] More historical data (5+ years) via Polygon.io or Databento

---

## Author

Built by **Tony Thompson**

- Strategy design, backtesting, optimization, and ML pipeline
- Languages: Pine Script v6, Python
- Libraries: vectorbt, XGBoost, scikit-learn, pandas

---

## License

MIT License — free to use, modify, and distribute with attribution.
