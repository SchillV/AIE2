# Implementation Plan: IDR/RON Exchange Rate – Model Training & Prediction

## 1. Project Structure

```
AIE2/
├── app.py                                         # Streamlit web interface (primary entry point)
├── requirements.txt
├── plan_implementare_antrenare_modele.md
├── core/                                          # auxiliary package
│   ├── __init__.py
│   ├── scraper.py                                 # fetches BNR rates → resources/data/
│   ├── models.py                                  # tuning, CV, model comparison
│   ├── pipeline.py                                # CLI retraining pipeline + logging
│   └── visualize.py                               # Plotly interactive + matplotlib diagnostics
├── resources/                                     # gitignored — generated artifacts
│   ├── data/
│   │   └── idr_exchange_rates.csv                 # produced by core/scraper.py
│   └── models/                                    # produced by pipeline / Streamlit app
│       ├── all_results.json                       # CV results for all four models
│       ├── best_params.json                       # best hyperparams (human-readable)
│       ├── best_model_<timestamp>.pkl             # overall best fitted model
│       ├── arima_<timestamp>.pkl
│       ├── exponentialsmoothing_<timestamp>.pkl
│       ├── naive_<timestamp>.pkl
│       ├── naivedrift_<timestamp>.pkl
│       └── diagnostics.png                        # combined 2×3 diagnostic figure
└── logs/                                          # gitignored — per-run training diagnostics
    └── <timestamp>/
        ├── results_summary.csv                    # MAE, AIC, params per model
        ├── cv_folds.csv                           # fold-by-fold MAE per model
        ├── arima_diagnostics.png                  # 2×2 diagnostic figure
        ├── exponentialsmoothing_diagnostics.png
        ├── diagnostics.png                        # combined 2×3 all-models figure
        ├── forecast_arima.png                     # rolling one-step retroactive forecast chart
        ├── forecast_exponentialsmoothing.png
        ├── forecast_naive.png
        ├── forecast_naivedrift.png
        └── forecast.csv                           # wide-format actual vs predicted for all models
```

---

## 2. Data Loading (`core/models.py → load_series`)

- Read `resources/data/idr_exchange_rates.csv` produced by `core/scraper.py`
- Auto-detect date column (hints: "data", "date", "zi", "day") and rate column (hints: "curs", "rate", "idr", "valoare")
- Parse dates with `format="%d.%m.%Y"`, coerce numerics (handle comma decimals)
- Deduplicate by date with `groupby().last()` (guards against duplicate BNR entries)
- Scale to **100 IDR → RON** (`× 100`) for readability
- Reindex to business-day frequency (`"B"`) and forward-fill gaps (public holidays)
  - Trade-off: ffill keeps the regular `B`-frequency index statsmodels requires, but injects
    artificial zero-return days that slightly deflate volatility and inflate autocorrelation
- Return `pd.Series` with `DatetimeIndex`, `name = "100IDR_RON"`

---

## 3. Models

### 3.1 ARIMA(p, d, q) — `statsmodels.tsa.arima.model.ARIMA`

**Integration order (d) — determined automatically via ADF test:**

```
adfuller(levels)  → p < 0.05  ⟹  d = 0  (stationary)
adfuller(diff(1)) → p < 0.05  ⟹  d = 1  (random walk, typical for FX)
otherwise                      ⟹  d = 2
```

AIC ranking is only valid across models fit to the same differencing order, so `d` is fixed once
before the grid search and the same `d` is used for all ARIMA candidates.

**Search space (p, q at fixed d):**
| Param | Range |
|-------|-------|
| p     | 0 – 3 |
| q     | 0 – 3 |

**Strategy:**
1. Compute AIC for all 16 combinations at fixed `d`.
2. Keep top 5 by AIC → run walk-forward CV using `append(refit=False)` (one full fit per fold,
   Kalman state extended cheaply for each subsequent step).
3. Select by minimum `mean_MAE`; use `std_MAE` as stability tiebreaker.

### 3.2 Exponential Smoothing — `statsmodels.tsa.holtwinters.ExponentialSmoothing`

Seasonal options removed: daily FX rates exhibit no meaningful weekly or monthly seasonality.

**Search space:**
| Param        | Options                    |
|--------------|----------------------------|
| trend        | None, "add", "mul"         |
| damped_trend | True, False                |

**Strategy:** Prune invalid combos (damped requires trend) → 5 valid combinations →
walk-forward CV on all; select by minimum `mean_MAE`.

Confidence intervals use bootstrap resampling of in-sample residuals (1 000 draws),
since Holt-Winters has no closed-form prediction interval.

### 3.3 Naive (last-value baseline)

Predicts next value = most recent observed value.  Equivalent to assuming a pure random walk
with zero drift.  Serves as the primary benchmark — any model worth deploying should beat this.

### 3.4 Naive + Drift

Predicts next value = last value + mean daily change over all history so far.  Captures a
long-term directional trend in the rate.  The drift is recomputed fresh at each walk-forward
step, matching the CV setup exactly.

---

## 4. TimeSeriesCrossValidation (`walk_forward_cv`)

- Uses `sklearn.model_selection.TimeSeriesSplit` with **5 folds**
- `test_size = max(10, len(series) // 6)` per fold (expanding window)
- For each fold: at each test step, predict one step ahead, then append the actual observation
- **ARIMA**: fit once per fold; extend via `ARIMAResults.append(refit=False)` for subsequent
  steps — only the Kalman state is updated, avoiding repeated full refits
- **ES / Naive / NaiveDrift**: full refit per step (fast)
- Returns `(mean_MAE, std_MAE, fold_maes_list)`

### Model Selection Criterion
```
best = argmin(mean_MAE)
ties broken by argmin(std_MAE)
```

---

## 5. Retraining Pipeline (`core/pipeline.py`)

Triggered via CLI (run from the project root):
```bash
python -m core.pipeline [--csv resources/data/idr_exchange_rates.csv]
```
Or from the **Streamlit app** (Overview → Retrain All Models / per-model page → Retrain).

Steps:
1. Load latest CSV with `load_series()`
2. Run `tune_arima()`, `tune_exp_smoothing()`, `tune_naive()`, `tune_naive_drift()` sequentially
3. Call `compare_models()` → select winner (lowest mean MAE)
4. `fit_final_model()` on **entire** series for each model
5. Persist to `resources/models/`:
   - `arima_<ts>.pkl`, `exponentialsmoothing_<ts>.pkl`, `naive_<ts>.pkl`, `naivedrift_<ts>.pkl`
   - `best_model_<ts>.pkl`, `all_results.json`, `best_params.json`
6. Regenerate `resources/models/diagnostics.png` (combined 2×3 figure)
7. Write debug logs to `logs/<timestamp>/` (see §8)

---

## 6. Streamlit Web Interface (`app.py`)

```bash
streamlit run app.py
```

**Pages** (sidebar navigation):

| Page | Content |
|------|---------|
| **Overview** | Model comparison table ranked by CV MAE, retroactive forecast charts for all models, Update Data and Retrain All buttons |
| **ARIMA** | Per-model metrics, hyperparameters, forecast chart, 2×2 diagnostic plots, CV fold bar chart, model description, individual retrain |
| **Exponential Smoothing** | Same layout |
| **Naive (last value)** | Same layout |
| **Naive + Drift** | Same layout |
| **About** | This document + tech stack + quick-start |

**Key implementation details:**
- Mtime-based cache keys: `@st.cache_data` is keyed on file modification time, so the cache
  auto-invalidates whenever `core/pipeline.py` writes new files
- Live progress: `st.status` + `st.progress` during retraining (full page width)
- Deferred retrain action: training flag is set by the Confirm button and checked at the **top** of
  `page_model()` / `page_overview()` so the status panel renders before any charts

---

## 7. Visualisations (`core/visualize.py`)

### 7.1 Interactive Plotly Forecast Chart — served by Streamlit
- **X-axis:** last ~2 months of history + rolling one-step-ahead retroactive window (last 2 weeks)
- Rolling one-step-ahead: for each day `d` in the test window, the model is fit on all data
  prior to `d` and makes a single one-step prediction — mirrors walk-forward CV exactly so
  the plotted MAE is consistent with the reported CV MAE and the prediction line is not flat
- **Traces:**
  - Blue solid line — historical data (full window, no gap into retroactive slice)
  - Orange dashed line + markers — rolling one-step-ahead prediction
  - Orange shaded band — 95% confidence interval
  - Shaded background rect marking the retroactive window
- Confidence intervals:
  - ARIMA: `get_forecast(steps=1).conf_int(alpha=0.05)` from the Kalman filter
  - ES / Naive / NaiveDrift: bootstrap (500 resampled residual draws per step)
- `rangebreaks=[dict(bounds=["sat","mon"])]` hides weekend gaps for business-day data
- Hover mode: `"x unified"` for easy cross-trace reading

### 7.2 Per-Model Diagnostic Panel — 2 × 2 (`make_per_model_diagnostic_figure`)
| Position | Plot |
|----------|------|
| (0,0) | Residuals distribution histogram + normal-fit overlay (burn-in trimmed for ARIMA) |
| (0,1) | Residuals over time (burn-in trimmed for ARIMA) |
| (1,0) | MAE per CV fold bar chart |
| (1,1) | Actual vs Predicted scatter (in-sample, last 100 observations) |

**Burn-in trimming:** ARIMA's Kalman diffuse initialisation inflates the first `d` innovations
by orders of magnitude.  Those observations are dropped (`k = max(d, 1)`) before plotting
residuals and fitting the normal overlay, so the histogram is not dominated by a single spike.
ES, Naive, and NaiveDrift have no Kalman initialisation and are left untrimmed.

### 7.3 Combined Diagnostic Panel — 2 × 3 (`diagnostics.png`)
| Position | Plot |
|----------|------|
| (0,0) | Residuals distribution (best model, burn-in trimmed) |
| (0,1) | Residuals over time (best model, burn-in trimmed) |
| (0,2) | MAE per CV fold — one line per model |
| (1,0) | Model comparison bar chart — mean MAE ± std (best highlighted in gold) |
| (1,1) | Actual vs Predicted scatter |
| (1,2) | Cumulative mean MAE across folds (best model) |

---

## 8. Training Logs (`logs/<timestamp>/`)

Written automatically by `core/pipeline.py` after every full pipeline run.

| File | Contents |
|------|----------|
| `results_summary.csv` | One row per model: mean MAE, std MAE, AIC, best-params, is_best flag |
| `cv_folds.csv` | One row per fold per model: fold index, MAE — use to spot unstable folds |
| `arima_diagnostics.png` | 2×2 figure: residual distribution, residuals over time, MAE per fold bar, actual vs predicted scatter |
| `exponentialsmoothing_diagnostics.png` | Same layout for Exponential Smoothing |
| `diagnostics.png` | Combined 2×3 all-models figure (identical format to `resources/models/diagnostics.png`) |
| `forecast_arima.png` | Rolling one-step retroactive forecast chart for ARIMA |
| `forecast_exponentialsmoothing.png` | Same for Exponential Smoothing |
| `forecast_naive.png` | Same for Naive |
| `forecast_naivedrift.png` | Same for Naive + Drift |
| `forecast.csv` | Wide-format table: date, actual, then `{model}_predicted / _lower_95 / _upper_95` for each model |

Logs are gitignored. Each run creates a new timestamped subfolder so old runs are preserved for comparison.

---

## 9. Dependencies (`requirements.txt`)

```
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=5.0.0
statsmodels>=0.14.0
scikit-learn>=1.4.0
plotly>=5.20.0
pandas>=2.1.0
numpy>=1.26.0
matplotlib>=3.8.0
scipy>=1.12.0
streamlit>=1.35.0
```

---

## 10. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| ADF-based `d` selection | AIC is only comparable across models fit to identically differenced data; fixing `d` once via a unit-root test ensures the AIC ranking in `tune_arima` is statistically valid |
| AIC pre-filter before CV | CV with walk-forward refitting is O(n · folds · test_size) — pre-filtering cuts wall-time by ~4× for ARIMA |
| `append(refit=False)` in CV | One full fit per fold; Kalman state extended cheaply for each additional test step — substantially faster with negligible accuracy change |
| `std_MAE` as tiebreaker | Unstable models (high variance across folds) are less reliable in production |
| Bootstrap CI for ES / Naive | statsmodels ES has no closed-form prediction interval; Naive has no concept of a model CI at all — bootstrap over in-sample residuals is calibration-free |
| SARIMAX removed | On daily FX data SARIMAX collapses to ARIMA (seasonal orders all select to zero); keeping it added search time and produced duplicate results in the comparison table |
| ES seasonal options removed | Daily FX rates have no meaningful weekly or monthly seasonality; removing those paths cuts the ES grid from 36 to 5 combos without sacrificing accuracy |
| Naive baselines included | Necessary sanity check — a model that can't beat the last-value forecast is not useful; also serves as an honest reference point in the comparison table |
| Rolling one-step retroactive plot | A single multi-step `get_forecast(n)` produces a flat or rapidly-diverging line whose displayed MAE is not the same statistic as the CV MAE; rolling one-step predictions track the data and are directly comparable |
| Burn-in trimming (ARIMA residuals) | The Kalman diffuse prior inflates the first innovation by ~100× making the residual histogram uninterpretable; dropping the first `d` residuals restores a meaningful distribution |
| Business-day frequency | BNR publishes rates on weekdays only; `"B"` freq + ffill avoids spurious gaps in statsmodels |
| Retroactive 2-week evaluation | Uses data the model never saw, giving an honest picture of real-world performance |
| `resources/` folder | Keeps all generated artifacts separate from source code; the whole folder is gitignored |
| `logs/` folder | Timestamped per-run diagnostics survive across retraining cycles for post-hoc comparison |
| Mtime cache invalidation | Allows the Streamlit app to detect externally-run pipeline results without manual cache clearing |
| Deferred retrain action | `st.status` is created before any chart rendering so the progress panel appears at full page width at the top of the page |
| `core/` package | Groups auxiliary modules (scraper, models, pipeline, visualize) away from the top-level entry points |
