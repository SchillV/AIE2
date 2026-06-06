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
│   ├── visualize.py                               # Plotly interactive + matplotlib diagnostics
│   └── chatbot.py                                 # LLM chatbot backend (Claude API + Ollama)
├── resources/                                     # gitignored — generated artifacts
│   ├── data/
│   │   └── idr_exchange_rates.csv                 # produced by core/scraper.py
│   ├── models/                                    # produced by pipeline / Streamlit app
│   │   ├── all_results.json                       # CV results for all four models
│   │   ├── best_params.json                       # best hyperparams (human-readable)
│   │   ├── best_model_<timestamp>.pkl             # overall best fitted model
│   │   ├── arima_<timestamp>.pkl
│   │   ├── exponentialsmoothing_<timestamp>.pkl
│   │   ├── naive_<timestamp>.pkl
│   │   ├── naivedrift_<timestamp>.pkl
│   │   └── diagnostics.png                        # combined 2×3 diagnostic figure
│   └── settings.json                              # persisted user settings (gitignored)
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

Confidence intervals use bootstrap resampling of in-sample residuals (configurable draws,
default 500), since Holt-Winters has no closed-form prediction interval.

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
Or from the **Streamlit app** (Overview → Retrain All Models / per-model page → Retrain /
Settings → Training section).

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

**Pages** (sidebar navigation — fixed, non-collapsible):

| Page | Content |
|------|---------|
| **Overview** | Data status metrics, retroactive forecast charts (best model first), model performance comparison table, Update Data and Retrain All buttons |
| **ARIMA** | Per-model metrics, hyperparameters, forecast chart, 2×2 interactive diagnostic plots, CV fold bar chart, model description, individual retrain |
| **Exponential Smoothing** | Same layout |
| **Naive (last value)** | Same layout |
| **Naive + Drift** | Same layout |
| **Chatbot** | LLM-powered assistant scoped to app data; quick model selector; retry/clear controls |
| **Settings** | Data status + Update shortcut; Training shortcuts; Forecast Display tuning; Chat/LLM configuration |
| **About** | Implementation plan + tech stack + quick-start + file reference |

Overview, each model page, and About all display a subtle watermark footer:
*"Made by Vlad Schiller · Agentically coded by Claude"*

**Key implementation details:**
- Mtime-based cache keys: `@st.cache_data` is keyed on file modification time, so the cache
  auto-invalidates whenever `core/pipeline.py` writes new files
- Live progress: `st.status` + `st.progress` during retraining (full page width)
- Deferred retrain action: training flag is set by the Confirm button and checked at the **top** of
  `page_model()` / `page_overview()` / `page_settings()` so the status panel renders before any charts
- Non-collapsible sidebar: CSS hides `[data-testid="stSidebarCollapseButton"]` and
  `[data-testid="collapsedControl"]`; `initial_sidebar_state="expanded"` ensures it starts open

---

## 7. Visualisations (`core/visualize.py`)

### 7.1 Interactive Plotly Forecast Chart — served by Streamlit
- **X-axis:** last N months of history (configurable, default 2) + rolling one-step-ahead retroactive window
- Rolling one-step-ahead: for each day `d` in the test window, the model is fit on all data
  prior to `d` and makes a single one-step prediction — mirrors walk-forward CV exactly so
  the plotted MAE is consistent with the reported CV MAE and the prediction line is not flat
- **Configurable parameters** (via Settings → Forecast Display):
  - `n_retro_days` (default 10): retroactive window length
  - `n_history_days` (derived from `history_months × 22`, default 44): history shown
  - `alpha` (derived from `ci_pct`, default 95%): confidence interval width
  - `n_boot` (default 500): bootstrap draws for ES/Naive CI
- **Traces:**
  - Blue solid line — historical data (full window, no gap into retroactive slice)
  - Orange dashed line + markers — rolling one-step-ahead prediction
  - Orange shaded band — configurable-width confidence interval
  - Shaded background rect marking the retroactive window
- Confidence intervals:
  - ARIMA: `get_forecast(steps=1).conf_int(alpha=alpha)` from the Kalman filter
  - ES / Naive / NaiveDrift: bootstrap (configurable draws per step)
- `rangebreaks=[dict(bounds=["sat","mon"])]` hides weekend gaps for business-day data
- Hover mode: `"x unified"` for easy cross-trace reading
- Colors: grid lines use `rgba(128,128,128,0.2)` for dark-mode compatibility

### 7.2 Per-Model Diagnostic Panel — 2 × 2 (`make_per_model_diagnostic_figure_plotly`)
Interactive Plotly figure (replaces static matplotlib version in the app).

| Position | Plot |
|----------|------|
| (1,1) | Residuals distribution histogram + normal-fit overlay (burn-in trimmed for ARIMA) |
| (1,2) | Residuals over time (burn-in trimmed for ARIMA) |
| (2,1) | MAE per CV fold bar chart |
| (2,2) | Actual vs Predicted scatter (in-sample, last 100 observations) |

**Burn-in trimming:** ARIMA's Kalman diffuse initialisation inflates the first `d` innovations
by orders of magnitude.  Those observations are dropped (`k = max(d, 1)`) before plotting
residuals and fitting the normal overlay, so the histogram is not dominated by a single spike.
ES, Naive, and NaiveDrift have no Kalman initialisation and are left untrimmed.

### 7.3 Combined Diagnostic Panel — 2 × 3 (`diagnostics.png`)
Static matplotlib figure saved to disk by the CLI pipeline.

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

## 9. Chatbot (`core/chatbot.py`)

An LLM-powered assistant embedded in the Streamlit app, scoped strictly to data present in the
app.  It cannot access the internet and will not fabricate information.

### 9.1 LLM Backends

| Backend | Package | Notes |
|---------|---------|-------|
| **Claude (Anthropic API)** | `anthropic` | Requires API key (`sk-ant-…`); supports full tool-use loop |
| **Local model (Ollama)** | `ollama` | Requires running `ollama serve`; uses tag-based action parsing |

The active backend and model are selectable from the Chatbot page or from Settings.

### 9.2 System Prompt (`build_app_context`)

Built fresh for every API call from live session data:
- Current exchange-rate series stats (last 15 business days, all-time min/max/mean, daily/weekly change)
- All model CV results (MAE, std, per-fold, best params)
- Available tool definitions

### 9.3 Tools

**Visualisation tools** (render inside the chat bubble):
| Tool | Action |
|------|--------|
| `show_forecast_chart(model_name)` | Interactive rolling one-step retroactive forecast chart |
| `show_model_comparison()` | Model performance comparison table |
| `show_diagnostic_plots(model_name)` | 2×2 diagnostic figure for one model |
| `get_data_table(n_rows)` | Last N exchange rate observations |

**App action tools** (trigger real side effects):
| Tool | Action |
|------|--------|
| `fetch_bnr_data()` | Scrapes BNR and saves the latest rates to CSV |
| `retrain_all_models()` | Runs the full retraining pipeline (15–30 min) |
| `retrain_model(model_name)` | Retrains a single model |

The Claude backend runs a full agentic tool-use loop (tool results are fed back to the model
until `stop_reason == "end_turn"`).  The Ollama backend uses structured tags (`[SHOW:…]`,
`[RUN:…]`) parsed from the response text.

### 9.4 Message History & Display Actions

Messages are stored in `st.session_state["chat_messages"]` as dicts with `role`, `content`,
and `actions` (list of display-action dicts rendered inside chat bubbles).  Action types:
`show_forecast`, `show_comparison`, `show_diagnostics`.

### 9.5 Sidebar Quick Chat

A collapsed expander in the sidebar shows the last two message previews and a text-input form
for sending messages without navigating to the Chatbot page.  Uses the same `_send_chat()`
pipeline as the full chatbot page.

---

## 10. Settings (`app.py → page_settings`)

Settings are persisted to `resources/settings.json` on every widget change and loaded into
`st.session_state` on first render.

### 10.1 Data Section
- Status indicators for exchange-rate CSV and trained models (observation count, last-trained timestamp)
- **Update Data** shortcut button — triggers the BNR scraper without navigating to Overview

### 10.2 Training Section
- **Retrain All Models** button (with two-step confirmation; shows estimated time)
- Per-model retrain buttons in a 4-column row (ARIMA / ES / Naive / Naive+Drift), each with
  its own confirmation dialog and time estimate

### 10.3 Forecast Display Section
| Setting | Key | Default | Effect |
|---------|-----|---------|--------|
| Retroactive window | `retro_days` | 10 days | Length of rolling one-step-ahead test window in forecast charts |
| History window | `history_months` | 2 months | Months of history shown before the retroactive slice (`× 22 business days`) |
| Confidence interval | `ci_pct` | 95% | CI width (converted to `alpha = 1 − ci_pct/100`) for all models |
| Bootstrap CI samples | `n_boot` | 500 | Residual bootstrap draws for ES/Naive CI; ARIMA uses Kalman covariance |

All four settings are read by `_forecast_kwargs()` in `app.py` and forwarded to every
`make_forecast_figure()` call so charts update immediately without retraining.

### 10.4 Chat / LLM Section
| Setting | Key | Default |
|---------|-----|---------|
| Backend | `chat_backend` | `"claude"` |
| Claude model | `chat_claude_model` | `claude-sonnet-4-6` |
| Claude API key | `chat_claude_key` | `""` |
| Ollama model | `chat_ollama_model` | `"llama3.2"` |

Ollama settings include a live status check and a model-pull panel (streams `ollama pull`
output line-by-line into the UI).

---

## 11. Dependencies (`requirements.txt`)

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
anthropic>=0.25.0
ollama>=0.2.0
```

---

## 12. Key Design Decisions

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
| Configurable forecast display settings | Retroactive window, history window, CI level, and bootstrap samples are user-tunable from the Settings page and take effect immediately on all charts without retraining |
| Non-collapsible sidebar | CSS hides the collapse button so navigation is always visible; the fixed sidebar width (230–260 px) accommodates all page labels without truncation |
| Dual-backend chatbot | Claude API gives the best reasoning and full tool-use; Ollama provides an offline fallback; both are scoped to app data only |
| Settings persisted to JSON | `resources/settings.json` survives app restarts; it is gitignored so API keys are never committed |
| `core/` package | Groups auxiliary modules (scraper, models, pipeline, visualize, chatbot) away from the top-level entry points |
