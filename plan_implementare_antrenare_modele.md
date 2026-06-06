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
│       ├── all_results.json                       # CV results for all three models
│       ├── best_params.json                       # best hyperparams (human-readable)
│       ├── best_model_<timestamp>.pkl             # overall best fitted model
│       ├── arima_<timestamp>.pkl
│       ├── sarimax_<timestamp>.pkl
│       ├── exponentialsmoothing_<timestamp>.pkl
│       └── diagnostics.png                        # combined 2×3 diagnostic figure
└── logs/                                          # gitignored — per-run training diagnostics
    └── <timestamp>/
        ├── results_summary.csv                    # MAE, AIC, params per model
        ├── cv_folds.csv                           # fold-by-fold MAE per model
        ├── arima_diagnostics.png                  # 2×2 diagnostic figure
        ├── sarimax_diagnostics.png
        ├── exponentialsmoothing_diagnostics.png
        └── diagnostics.png                        # combined 2×3 all-models figure
```

---

## 2. Data Loading (`core/models.py → load_series`)

- Read `resources/data/idr_exchange_rates.csv` produced by `core/scraper.py`
- Auto-detect date column (hints: "data", "date", "zi", "day") and rate column (hints: "curs", "rate", "idr", "valoare")
- Parse dates with `format="%d.%m.%Y"`, coerce numerics (handle comma decimals)
- Deduplicate by date with `groupby().last()` (guards against duplicate BNR entries)
- Scale to **100 IDR → RON** (`× 100`) for readability
- Reindex to business-day frequency (`"B"`) and forward-fill gaps (public holidays)
- Return `pd.Series` with `DatetimeIndex`, `name = "100IDR_RON"`

---

## 3. Models

### 3.1 ARIMA(p, d, q) — `statsmodels.tsa.arima.model.ARIMA`

**Search space:**
| Param | Range |
|-------|-------|
| p     | 0 – 3 |
| d     | 0 – 2 |
| q     | 0 – 3 |

**Strategy:**
1. Compute AIC for all 48 combinations (fast fit on full series).
2. Keep top 10 by AIC → run walk-forward CV on top 5.
3. Select by minimum `mean_MAE`; use `std_MAE` as stability tiebreaker.

### 3.2 SARIMAX(p, d, q)(P, D, Q, s) — `statsmodels.tsa.statespace.sarimax.SARIMAX`

**Search space:**
| Param | Range      |
|-------|------------|
| p,q   | 0 – 2      |
| d     | 0 – 1      |
| P,Q   | 0 – 1      |
| D     | 0 – 1      |
| s     | 5 (weekly), 21 (monthly) |

**Strategy:** Same AIC pre-filter → top 5 CV, same selection criterion.

### 3.3 Exponential Smoothing — `statsmodels.tsa.holtwinters.ExponentialSmoothing`

**Search space:**
| Param            | Options                    |
|------------------|----------------------------|
| trend            | None, "add", "mul"         |
| seasonal         | None, "add", "mul"         |
| damped_trend     | True, False                |
| seasonal_periods | 5, 21                      |

**Strategy:** Prune invalid combos (damped requires trend; seasonal requires period ≥ 2) → direct walk-forward CV on all valid combinations.

---

## 4. TimeSeriesCrossValidation (`walk_forward_cv`)

- Uses `sklearn.model_selection.TimeSeriesSplit` with **5 folds**
- `test_size = max(10, len(series) // 6)` per fold (expanding window)
- For each fold: fit model on training slice, then **one-step-ahead walk-forward**:
  at each test step, the model is retrained on all history including prior test observations
- Returns `(mean_MAE, std_MAE, fold_maes_list)`
- `std_MAE` captures model **stability** across folds

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
2. Run `tune_arima()`, `tune_sarimax()`, `tune_exp_smoothing()` sequentially
3. Call `compare_models()` → select winner
4. `fit_final_model()` on **entire** series for each model
5. Persist to `resources/models/`:
   - `arima_<ts>.pkl`, `sarimax_<ts>.pkl`, `exponentialsmoothing_<ts>.pkl`
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
| **ARIMA / SARIMAX / Exponential Smoothing** | Per-model metrics, hyperparameters, forecast chart, 2×2 diagnostic plots, CV fold bar chart, individual retrain |
| **About** | This document + tech stack + quick-start |

**Key implementation details:**
- Mtime-based cache keys: `@st.cache_data` is keyed on file modification time, so the cache auto-invalidates whenever `core/pipeline.py` writes new files
- Live progress: `st.status` + `st.progress` during retraining (full page width — action is deferred until outside any column context)
- Navigation: `st.session_state` + `st.rerun()` for immediate button colour update

---

## 7. Visualisations (`core/visualize.py`)

### 7.1 Interactive Plotly Chart — served by Streamlit
- **X-axis:** last ~2 months of history + retroactive 2-week window
- **Traces:**
  - Blue solid line — historical data (full window, no gap into retroactive slice)
  - Orange dashed line + markers — model retroactive prediction
  - Orange shaded band — 95% confidence interval
  - Shaded background rect marking the retroactive window
- Confidence intervals:
  - ARIMA/SARIMAX: `get_forecast().conf_int(alpha=0.05)`
  - ES: bootstrap (1 000 resampled residual draws)
- `rangebreaks=[dict(bounds=["sat","mon"])]` hides weekend gaps for financial data
- Hover mode: `"x unified"` for easy cross-trace reading

### 7.2 Per-Model Diagnostic Panel — 2 × 2 (`make_per_model_diagnostic_figure`)
| Position | Plot |
|----------|------|
| (0,0) | Residuals distribution histogram + normal-fit overlay |
| (0,1) | Residuals over time |
| (1,0) | MAE per CV fold bar chart |
| (1,1) | Actual vs Predicted scatter (in-sample, last 100 observations) |

### 7.3 Combined Diagnostic Panel — 2 × 3 (`diagnostics.png`)
| Position | Plot |
|----------|------|
| (0,0) | Residuals distribution (best model) |
| (0,1) | Residuals over time (best model) |
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
| `sarimax_diagnostics.png` | Same layout for SARIMAX |
| `exponentialsmoothing_diagnostics.png` | Same layout for ES |
| `diagnostics.png` | Combined 2×3 all-models figure (identical format to `resources/models/diagnostics.png`) |

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
| AIC pre-filter before CV | CV with walk-forward refitting is O(n · folds · test_size) — pre-filtering cuts wall-time by ~10× |
| `std_MAE` as tiebreaker | Unstable models (high variance across folds) are less reliable in production |
| Bootstrap CI for ES | statsmodels ES has no closed-form PI; bootstrap over residuals is straightforward and calibration-free |
| Business-day frequency | BNR publishes rates on weekdays only; `"B"` freq + ffill avoids spurious gaps |
| Retroactive 2-week evaluation | Uses data the model never saw, giving an honest picture of real-world performance |
| `resources/` folder | Keeps all generated artifacts separate from source code; the whole folder is gitignored |
| `logs/` folder | Timestamped per-run diagnostics survive across retraining cycles for post-hoc comparison |
| Mtime cache invalidation | Allows the Streamlit app to detect externally-run pipeline results without manual cache clearing |
| Deferred retrain action | `st.status` is created outside any column context so the progress panel renders at full page width |
| `core/` package | Groups auxiliary modules (scraper, models, pipeline, visualize) away from the top-level entry points |
