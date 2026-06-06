"""
Streamlit front-end for IDR/RON exchange rate forecasting.

Run with:
    streamlit run app.py
"""

import contextlib
import io
import json
import pickle
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ─── Constants ───────────────────────────────────────────────────────────────
CSV_PATH = Path("resources") / "data" / "idr_exchange_rates.csv"
MODELS_DIR = Path("resources") / "models"
ALL_RESULTS_PATH = MODELS_DIR / "all_results.json"
PLAN_PATH = Path("plan_implementare_antrenare_modele.md")
SETTINGS_PATH = Path("resources") / "settings.json"

# Keys that are persisted to disk; values are the defaults.
_SETTINGS_KEYS: dict = {
    "chat_backend": "claude",
    "chat_claude_model": "claude-sonnet-4-6",
    "chat_claude_key": "",
    "chat_ollama_model": "llama3.2",
}

MODEL_NAMES = ["ARIMA", "ExponentialSmoothing", "Naive", "NaiveDrift"]
_PKL_PREFIX = {
    "ARIMA": "arima",
    "ExponentialSmoothing": "exponentialsmoothing",
    "Naive": "naive",
    "NaiveDrift": "naivedrift",
}
_DISPLAY = {
    "ARIMA": "ARIMA",
    "ExponentialSmoothing": "Exponential Smoothing",
    "Naive": "Naive (last value)",
    "NaiveDrift": "Naive + Drift",
}
_PAGES = ["Overview", "ARIMA", "Exponential Smoothing", "Naive", "Naive + Drift", "Chatbot", "Settings", "About"]

_MODEL_DESCRIPTION = {
    "ARIMA": """\
**ARIMA (AutoRegressive Integrated Moving Average)**

ARIMA(p, d, q) combines three components:

- **AR(p)** — the forecast is a linear combination of the `p` most recent values
- **I(d)** — the series is differenced `d` times to remove the unit root and make it stationary
- **MA(q)** — the forecast also depends on the `q` most recent one-step forecast errors

For IDR/RON, the integration order `d` is determined automatically each run via an **ADF
(Augmented Dickey-Fuller) unit-root test**.  Daily FX rates are typically I(1) (random walk),
so the model is usually fit on first differences.  Fixing `d` before the grid search ensures
AIC values are comparable across all (p, q) candidates.

The best (p, q) pair is selected by AIC pre-filter followed by walk-forward cross-validation.
95% confidence intervals are produced by the Kalman filter's covariance matrix.
""",

    "ExponentialSmoothing": """\
**Exponential Smoothing (Holt method, non-seasonal)**

Exponential Smoothing assigns exponentially decreasing weights to past observations — recent
values matter more than distant ones.  This implementation uses the **Holt** (two-component)
variant with an optional trend:

- **Simple ES** (`trend=None`) — smoothed level only; equivalent to an EWMA forecast
- **Additive trend** (`trend="add"`) — level + linear trend component
- **Multiplicative trend** (`trend="mul"`) — level + proportionally scaled trend
- **Damped trend** — the trend is gradually pulled toward zero at long horizons, avoiding
  unrealistic straight-line extrapolation

Seasonal options are excluded: daily FX rates show no meaningful weekly or monthly
periodicity, and including them would add noise without improving accuracy.

All smoothing parameters (α, β) are estimated by maximum likelihood.  95% confidence
intervals are computed by bootstrap resampling of in-sample residuals (500 draws).
""",

    "Naive": """\
**Naive Forecast (last-value baseline)**

The simplest possible forecast: **tomorrow's rate equals today's rate**.

```
ŷ_{t+1} = y_t
```

This is equivalent to assuming the exchange rate follows a pure random walk with no drift.
Under the weak form of the Efficient Market Hypothesis, this is the best forecast an
outsider can make — future price movements are unpredictable from past prices alone.

The Naive model serves as the **primary benchmark**: any statistically motivated model that
cannot consistently outperform it on walk-forward CV is not adding value.  The CV MAE here
is the floor that ARIMA and Exponential Smoothing are competing against.

Confidence intervals are constructed by bootstrap resampling of in-sample first differences
(the empirical distribution of one-day rate changes).
""",

    "NaiveDrift": """\
**Naive with Drift**

An extension of the Naive forecast that adds the **mean daily change** over all historical
data seen so far:

```
ŷ_{t+1} = y_t + mean(Δy_1, Δy_2, …, Δy_t)
```

If the exchange rate has been trending — appreciating or depreciating on average over the
training window — the drift term captures that momentum.  The drift is recalculated at every
step of walk-forward CV, so it reflects only information available before each prediction.

For a currency pair that has been gradually moving in one direction this will modestly
outperform the pure Naive model; for a mean-reverting pair the drift can hurt.  Comparing
this model against the plain Naive shows whether the historical trend carries any predictive
signal.

Confidence intervals are constructed by bootstrap resampling of in-sample residuals.
""",
}

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDR/RON Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Content padding */
    .main .block-container { padding: 2.5rem 3rem 2rem; max-width: 1200px; }
    /* Sidebar width */
    section[data-testid="stSidebar"] { min-width: 230px; max-width: 260px; }
    /* Nav buttons: collapse spacing, left-align text */
    section[data-testid="stSidebar"] [data-testid="stButton"] { margin-bottom: 3px; }
    section[data-testid="stSidebar"] [data-testid="stButton"] > button {
        text-align: left;
        border-radius: 6px;
        font-size: 0.92rem;
    }
    /* Metric cards — rgba background adapts to both light and dark mode */
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
    }
    h1 { margin-bottom: 0.25rem; }
    h2 { margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── Cache helpers ────────────────────────────────────────────────────────────
# Use file-mtime as a cache-key parameter so the cache auto-invalidates whenever
# the file on disk changes (even if pipeline.py was run from outside the app).

@st.cache_data(show_spinner=False)
def _cached_load_series(mtime: float) -> "pd.Series | None":
    if not CSV_PATH.exists():
        return None
    from core.models import load_series
    return load_series(str(CSV_PATH))


@st.cache_data(show_spinner=False)
def _cached_load_results(mtime: float) -> "dict | None":
    if not ALL_RESULTS_PATH.exists():
        return None
    with open(ALL_RESULTS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_resource(show_spinner=False)
def _cached_load_pkl(model_name: str, mtime: float) -> "dict | None":
    prefix = _PKL_PREFIX.get(model_name, model_name.lower())
    matches = sorted(MODELS_DIR.glob(f"{prefix}_*.pkl"))
    if not matches:
        return None
    with open(matches[-1], "rb") as fh:
        return pickle.load(fh)


def _get_series() -> "pd.Series | None":
    mtime = CSV_PATH.stat().st_mtime if CSV_PATH.exists() else 0.0
    return _cached_load_series(mtime)


def _get_all_results() -> "dict | None":
    mtime = ALL_RESULTS_PATH.stat().st_mtime if ALL_RESULTS_PATH.exists() else 0.0
    return _cached_load_results(mtime)


def _get_model_pkl(model_name: str) -> "dict | None":
    prefix = _PKL_PREFIX.get(model_name, model_name.lower())
    matches = sorted(MODELS_DIR.glob(f"{prefix}_*.pkl"))
    mtime = matches[-1].stat().st_mtime if matches else 0.0
    return _cached_load_pkl(model_name, mtime)


def _clear_caches() -> None:
    _cached_load_series.clear()
    _cached_load_results.clear()
    _cached_load_pkl.clear()


def _load_settings() -> None:
    """Read persisted settings into session_state on first load."""
    saved: dict = {}
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH) as fh:
                saved = json.load(fh)
        except Exception:
            pass
    for key, default in _SETTINGS_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = saved.get(key, default)


def _save_settings() -> None:
    """Persist current settings to disk (used as on_change callback for settings widgets)."""
    data = {k: st.session_state.get(k, v) for k, v in _SETTINGS_KEYS.items()}
    try:
        SETTINGS_PATH.parent.mkdir(exist_ok=True)
        with open(SETTINGS_PATH, "w") as fh:
            json.dump(data, fh, indent=2)
    except Exception:
        pass


@st.cache_data(ttl=30, show_spinner=False)
def _ollama_status() -> tuple[bool, list[str]]:
    from core.chatbot import is_ollama_running, list_ollama_models
    running = is_ollama_running()
    return running, (list_ollama_models() if running else [])


# ─── Action helpers ──────────────────────────────────────────────────────────

def _do_update_data() -> bool:
    from core.scraper import fetch_exchange_rates, save_to_csv
    try:
        headers, rows = fetch_exchange_rates()
        if not rows:
            st.error("Scraper returned no data rows.")
            return False
        save_to_csv(headers, rows, str(CSV_PATH))
        _clear_caches()
        return True
    except Exception as exc:
        st.error(f"Scraping failed: {exc}")
        st.code(traceback.format_exc())
        return False


def _do_retrain_all() -> bool:
    """Run the full pipeline with a live st.status progress panel."""
    from core.models import (
        load_series, tune_arima, tune_exp_smoothing,
        tune_naive, tune_naive_drift, compare_models, fit_final_model,
    )
    from core.pipeline import _serialise, _PKL_PREFIX as PKL, OUTPUT_DIR, _write_logs
    from core.visualize import generate_all_plots

    MODELS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Progress bar lives outside the collapsible status box so it's always visible.
        prog = st.progress(0, text="Starting …")

        with st.status("Retraining all models …", expanded=True) as status:

            # ── Load data ─────────────────────────────────────────────────
            prog.progress(2, text="Loading exchange-rate data …")
            series = load_series(str(CSV_PATH))
            st.write(
                f"✅ **Data loaded** — {len(series):,} observations "
                f"({series.index[0].date()} → {series.index[-1].date()})"
            )

            # ── ARIMA ─────────────────────────────────────────────────────
            prog.progress(8, text="Tuning ARIMA — ADF test + AIC candidates …")
            st.write("---")
            st.write("🔍 **ARIMA** — ADF integration test + AIC pre-filter + walk-forward CV …")
            _buf = io.StringIO()
            with contextlib.redirect_stdout(_buf):
                arima_res = tune_arima(series)
            if _out := _buf.getvalue():
                st.code(_out, language=None)
            st.write(
                f"✅ ARIMA best: `order={arima_res['order']}`  "
                f"MAE = `{arima_res['mean_mae']:.6f} ± {arima_res['std_mae']:.6f}`"
            )

            # ── Exponential Smoothing ─────────────────────────────────────
            prog.progress(40, text="Tuning Exponential Smoothing …")
            st.write("---")
            st.write("🔍 **Exponential Smoothing** — walk-forward CV on all valid combos …")
            _buf = io.StringIO()
            with contextlib.redirect_stdout(_buf):
                es_res = tune_exp_smoothing(series)
            if _out := _buf.getvalue():
                st.code(_out, language=None)
            st.write(
                f"✅ ES best: `trend={es_res['trend']}  seasonal={es_res['seasonal']}  "
                f"damped={es_res['damped_trend']}`  "
                f"MAE = `{es_res['mean_mae']:.6f} ± {es_res['std_mae']:.6f}`"
            )

            # ── Naive ─────────────────────────────────────────────────────
            prog.progress(65, text="Evaluating Naive baseline …")
            st.write("---")
            st.write("🔍 **Naive** — walk-forward CV (last-value baseline) …")
            _buf = io.StringIO()
            with contextlib.redirect_stdout(_buf):
                naive_res = tune_naive(series)
            if _out := _buf.getvalue():
                st.code(_out, language=None)
            st.write(
                f"✅ Naive: MAE = `{naive_res['mean_mae']:.6f} ± {naive_res['std_mae']:.6f}`"
            )

            # ── Naive+Drift ───────────────────────────────────────────────
            prog.progress(75, text="Evaluating Naive+Drift baseline …")
            st.write("---")
            st.write("🔍 **Naive+Drift** — walk-forward CV (last value + mean drift) …")
            _buf = io.StringIO()
            with contextlib.redirect_stdout(_buf):
                naive_drift_res = tune_naive_drift(series)
            if _out := _buf.getvalue():
                st.code(_out, language=None)
            st.write(
                f"✅ Naive+Drift: MAE = `{naive_drift_res['mean_mae']:.6f} ± {naive_drift_res['std_mae']:.6f}`"
            )

            # ── Select best ───────────────────────────────────────────────
            prog.progress(83, text="Selecting best model …")
            st.write("---")
            best = compare_models(arima_res, es_res, naive_res, naive_drift_res)
            st.write(
                f"🏆 **Best model: {best['model']}** — "
                f"MAE = `{best['mean_mae']:.6f} ± {best['std_mae']:.6f}`"
            )

            # ── Fit & save ────────────────────────────────────────────────
            prog.progress(87, text="Fitting all models on full series …")
            st.write("---")
            st.write("💾 Fitting and saving …")
            fitted_map: dict = {}
            for name, res in [
                ("ARIMA", arima_res),
                ("ExponentialSmoothing", es_res),
                ("Naive", naive_res),
                ("NaiveDrift", naive_drift_res),
            ]:
                fitted = fit_final_model(series, res)
                fitted_map[name] = fitted
                pkl_path = OUTPUT_DIR / f"{PKL[name]}_{timestamp}.pkl"
                with open(pkl_path, "wb") as fh:
                    pickle.dump({"params": res, "fitted": fitted}, fh)
                st.write(f"  · `{pkl_path.name}` saved")

            best_path = OUTPUT_DIR / f"best_model_{timestamp}.pkl"
            with open(best_path, "wb") as fh:
                pickle.dump({"params": best, "fitted": fitted_map[best["model"]]}, fh)
            st.write(f"  · `{best_path.name}` (best) saved")

            # Save all_results.json
            all_results_payload = {
                "ARIMA": _serialise(arima_res),
                "ExponentialSmoothing": _serialise(es_res),
                "Naive": _serialise(naive_res),
                "NaiveDrift": _serialise(naive_drift_res),
                "best": _serialise(best),
                "timestamp": timestamp,
                "n_observations": len(series),
            }
            with open(ALL_RESULTS_PATH, "w", encoding="utf-8") as fh:
                json.dump(all_results_payload, fh, indent=2)
            st.write("  · `all_results.json` updated")

            # Diagnostics PNG
            prog.progress(94, text="Generating diagnostic plots …")
            all_results_dict = {
                "ARIMA": arima_res,
                "ExponentialSmoothing": es_res,
                "Naive": naive_res,
                "NaiveDrift": naive_drift_res,
                "best": best,
            }
            with contextlib.redirect_stdout(io.StringIO()):
                generate_all_plots(series, fitted_map[best["model"]], all_results_dict, OUTPUT_DIR)
            st.write("  · `resources/models/diagnostics.png` saved")

            # Training logs
            prog.progress(97, text="Writing training logs …")
            with contextlib.redirect_stdout(io.StringIO()):
                _write_logs(timestamp, all_results_dict, fitted_map, series)
            st.write(f"  · `logs/{timestamp}/` written")

            prog.progress(100, text="Done!")
            status.update(label="✅ All models retrained successfully!", state="complete")

        _clear_caches()
        return True

    except Exception as exc:
        st.error(f"Retraining failed: {exc}")
        st.code(traceback.format_exc())
        return False


def _do_retrain_single(model_name: str) -> bool:
    """Retune one model with a live st.status progress panel."""
    from core.models import (
        load_series, tune_arima, tune_exp_smoothing,
        tune_naive, tune_naive_drift, fit_final_model,
    )
    from core.pipeline import _serialise, _PKL_PREFIX as PKL, OUTPUT_DIR, MODEL_NAMES, _write_logs

    MODELS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    display = _DISPLAY[model_name]

    try:
        # Progress bar outside the collapsible status box.
        prog = st.progress(0, text="Starting …")

        with st.status(f"Retraining {display} …", expanded=True) as status:

            prog.progress(5, text="Loading data …")
            series = load_series(str(CSV_PATH))
            st.write(
                f"✅ **Data loaded** — {len(series):,} observations "
                f"({series.index[0].date()} → {series.index[-1].date()})"
            )

            prog.progress(12, text=f"Tuning {display} …")
            st.write(f"🔍 **{display}** — hyperparameter search + walk-forward CV …")
            _buf = io.StringIO()
            with contextlib.redirect_stdout(_buf):
                if model_name == "ARIMA":
                    result = tune_arima(series)
                elif model_name == "Naive":
                    result = tune_naive(series)
                elif model_name == "NaiveDrift":
                    result = tune_naive_drift(series)
                else:
                    result = tune_exp_smoothing(series)
            if _out := _buf.getvalue():
                st.code(_out, language=None)
            st.write(
                f"✅ Best params: `{_params_short(result)}`  "
                f"MAE = `{result['mean_mae']:.6f} ± {result['std_mae']:.6f}`"
            )

            prog.progress(80, text="Fitting on full series and saving …")
            fitted = fit_final_model(series, result)
            pkl_path = OUTPUT_DIR / f"{PKL[model_name]}_{timestamp}.pkl"
            with open(pkl_path, "wb") as fh:
                pickle.dump({"params": result, "fitted": fitted}, fh)
            st.write(f"  · `{pkl_path.name}` saved")

            # Update all_results.json
            if ALL_RESULTS_PATH.exists():
                with open(ALL_RESULTS_PATH, encoding="utf-8") as fh:
                    stored = json.load(fh)
            else:
                stored = {}

            stored[model_name] = _serialise(result)
            stored["timestamp"] = timestamp

            available = [stored[n] for n in MODEL_NAMES if n in stored]
            if available:
                best = min(available, key=lambda x: (x["mean_mae"], x["std_mae"]))
                stored["best"] = best
                if best["model"] == model_name:
                    best_path = OUTPUT_DIR / f"best_model_{timestamp}.pkl"
                    with open(best_path, "wb") as fh:
                        pickle.dump({"params": result, "fitted": fitted}, fh)
                    st.write(f"  · `{best_path.name}` (new best model) saved")

            with open(ALL_RESULTS_PATH, "w", encoding="utf-8") as fh:
                json.dump(stored, fh, indent=2)
            st.write("  · `all_results.json` updated")

            # Training logs
            prog.progress(95, text="Writing training logs …")
            partial_results = {model_name: result, "best": stored.get("best", {})}
            with contextlib.redirect_stdout(io.StringIO()):
                _write_logs(timestamp, partial_results, {model_name: fitted}, series)
            st.write(f"  · `logs/{timestamp}/` written")

            prog.progress(100, text="Done!")
            status.update(label=f"✅ {display} retrained successfully!", state="complete")

        _clear_caches()
        return True

    except Exception as exc:
        st.error(f"Retraining {model_name} failed: {exc}")
        st.code(traceback.format_exc())
        return False


# ─── UI helpers ───────────────────────────────────────────────────────────────

def _normalise_params(params: dict) -> dict:
    out = dict(params)
    for key in ("order", "seasonal_order"):
        if isinstance(out.get(key), list):
            out[key] = tuple(out[key])
    return out


def _params_short(r: dict) -> str:
    name = r.get("model", "")
    if name == "ARIMA":
        return f"order={tuple(r['order'])}"
    if name == "SARIMAX":
        return f"order={tuple(r['order'])}  seasonal={tuple(r['seasonal_order'])}"
    if name in ("Naive", "NaiveDrift"):
        return "—"
    return (
        f"trend={r.get('trend')}  seasonal={r.get('seasonal')}  "
        f"damped={r.get('damped_trend')}  sp={r.get('seasonal_periods')}"
    )


def _params_display(r: dict) -> dict:
    skip = {"fold_maes", "mean_mae", "std_mae", "aic", "model"}
    return {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in r.items() if k not in skip
    }


def _sorted_models(all_results: dict) -> list[dict]:
    candidates = [all_results[n] for n in MODEL_NAMES if n in all_results]
    return sorted(candidates, key=lambda x: (x["mean_mae"], x["std_mae"]))


def _confirm_retrain_widget(key: str, label: str) -> None:
    """Render trigger → warning → confirm/cancel buttons.

    Sets st.session_state[f"run_{key}"] = True and reruns on confirm.
    The CALLER must check that flag at the top of the page function and run
    the actual training there, so the status panel appears before page content.
    """
    if st.button(label, key=f"btn_{key}"):
        st.session_state[f"pending_{key}"] = True

    if st.session_state.get(f"pending_{key}"):
        st.warning("This operation may take **15–30 minutes**. The app will be unresponsive during training.")
        col_go, col_cancel, _ = st.columns([1, 1, 5])
        with col_go:
            if st.button("✅ Confirm", key=f"confirm_{key}", width="stretch"):
                st.session_state[f"pending_{key}"] = False
                st.session_state[f"run_{key}"] = True
                st.rerun()
        with col_cancel:
            if st.button("✖ Cancel", key=f"cancel_{key}", width="stretch"):
                st.session_state[f"pending_{key}"] = False
                st.rerun()


# ─── Pages ───────────────────────────────────────────────────────────────────

def page_overview() -> None:
    st.title("IDR / RON Exchange Rate Forecaster")
    st.caption("Source: [cursbnr.ro](https://www.cursbnr.ro) — BNR official rates · scale: **100 IDR → RON**")

    # Active operations render at the top before any buttons or data.
    if st.session_state.get("do_retrain_all_now"):
        st.session_state["do_retrain_all_now"] = False
        _do_retrain_all()
        st.rerun()
        return

    if st.session_state.get("do_update_data_now"):
        st.session_state["do_update_data_now"] = False
        with st.status("Fetching latest rates from BNR …", expanded=True) as s:
            ok = _do_update_data()
        if ok:
            s.update(label="✅ Data updated!", state="complete")
        st.rerun()
        return

    series = _get_series()
    all_results = _get_all_results()

    # ── Action buttons ────────────────────────────────────────────────────
    c_upd, c_retrain, _ = st.columns([1.4, 1.8, 4.8])
    with c_upd:
        if st.button("🔄 Update Data", width="stretch"):
            st.session_state["do_update_data_now"] = True
            st.rerun()
    with c_retrain:
        if st.button("🤖 Retrain All Models", key="btn_retrain_all", width="stretch"):
            st.session_state["pending_retrain_all"] = True

    if st.session_state.get("pending_retrain_all"):
        st.warning(
            "Retraining all three models with hyperparameter tuning may take "
            "**15–30 minutes**. The app will be unresponsive during this time."
        )
        col_go, col_cancel, _ = st.columns([1, 1, 5])
        with col_go:
            if st.button("✅ Confirm retrain", key="confirm_retrain_all", width="stretch"):
                st.session_state["pending_retrain_all"] = False
                st.session_state["do_retrain_all_now"] = True
                st.rerun()
        with col_cancel:
            if st.button("✖ Cancel", key="cancel_retrain_all", width="stretch"):
                st.session_state["pending_retrain_all"] = False
                st.rerun()

    st.divider()

    # ── Data status ───────────────────────────────────────────────────────
    if series is None:
        st.info(
            "No exchange-rate data found.  \n"
            "Click **Update Data** to fetch from BNR, then **Retrain All Models** to train the forecasting models."
        )
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Observations", f"{len(series):,}")
    m2.metric("First date", str(series.index[0].date()))
    m3.metric("Latest date", str(series.index[-1].date()))
    m4.metric("Latest rate (100 IDR → RON)", f"{series.iloc[-1]:.4f}")

    st.divider()

    if all_results is None:
        st.info(
            "Models have not been trained yet.  \n"
            "Click **Retrain All Models** above to run the hyperparameter-tuning pipeline."
        )
        return

    best_name = all_results.get("best", {}).get("model", "")
    sorted_models = _sorted_models(all_results)

    # ── Forecast charts ───────────────────────────────────────────────────
    st.subheader("Retroactive Forecast Charts (strongest model first)")
    st.caption(
        "Each model is trained on data up to 2 weeks ago and predicts that 2-week window. "
        "Shaded band = 95% confidence interval."
    )

    from core.visualize import make_forecast_figure
    for r in sorted_models:
        model_name = r["model"]
        is_best = model_name == best_name
        label = (
            f"{'★ Best · ' if is_best else ''}"
            f"{_DISPLAY.get(model_name, model_name)}  —  "
            f"CV MAE: {r['mean_mae']:.6f} ± {r['std_mae']:.6f}"
        )
        with st.expander(label, expanded=is_best):
            try:
                fig = make_forecast_figure(series, _normalise_params(r))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Chart error: {exc}")

    st.divider()

    # ── Model comparison table ────────────────────────────────────────────
    st.subheader("Model Performance Comparison")
    table_rows = []
    for rank, r in enumerate(sorted_models, 1):
        name = r["model"]
        table_rows.append({
            "Rank": f"{'★ ' if name == best_name else ''}#{rank}",
            "Model": _DISPLAY.get(name, name),
            "Mean CV MAE": f"{r['mean_mae']:.6f}",
            "Std CV MAE": f"{r['std_mae']:.6f}",
            "Best Params": _params_short(r),
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Trained on timestamp
    ts = all_results.get("timestamp", "")
    if ts:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        st.caption(f"Last trained: {dt.strftime('%d %b %Y %H:%M')}")


def page_model(model_name: str) -> None:
    display_name = _DISPLAY.get(model_name, model_name)
    st.title(display_name)

    # If retraining was confirmed, show only the training panel — skip all charts.
    run_key = f"retrain_{model_name}"
    if st.session_state.get(f"run_{run_key}"):
        st.session_state[f"run_{run_key}"] = False
        _do_retrain_single(model_name)
        st.rerun()
        return

    series = _get_series()
    all_results = _get_all_results()

    if series is None or all_results is None:
        st.info("No data or trained models found. Use the **Overview** page to get started.")
        return

    result = all_results.get(model_name)
    if result is None:
        st.info(f"No results for {display_name} yet.")
        return

    is_best = all_results.get("best", {}).get("model") == model_name

    # ── Metrics ───────────────────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Mean CV MAE", f"{result['mean_mae']:.6f}")
    mc2.metric("Std CV MAE", f"{result['std_mae']:.6f}")
    mc3.metric("Best model overall?", "Yes ★" if is_best else "No")
    if "aic" in result:
        mc4.metric("AIC", f"{result['aic']:.2f}")

    st.divider()

    # ── Hyperparameters ───────────────────────────────────────────────────
    st.subheader("Best Hyperparameters")
    st.json(_params_display(result))

    st.divider()

    # ── Forecast chart ────────────────────────────────────────────────────
    st.subheader("Retroactive Forecast — last 2 months · 2-week prediction window")
    from core.visualize import make_forecast_figure
    try:
        fig = make_forecast_figure(series, _normalise_params(result))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Forecast chart error: {exc}")

    st.divider()

    # ── Diagnostic plots ──────────────────────────────────────────────────
    st.subheader("Diagnostic Plots")
    pkl_data = _get_model_pkl(model_name)
    if pkl_data is not None:
        fitted = pkl_data.get("fitted")
        if fitted is not None:
            from core.visualize import make_per_model_diagnostic_figure_plotly
            try:
                diag_fig = make_per_model_diagnostic_figure_plotly(series, fitted, result)
                st.plotly_chart(diag_fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Diagnostic plot error: {exc}")
        else:
            st.info("The saved pickle does not contain a fitted model object.")
    else:
        st.info(
            f"No saved {display_name} pickle found in `{MODELS_DIR}/`.  \n"
            "Run **Retrain All Models** from the Overview page or **Retrain This Model** below."
        )

    st.divider()

    # ── CV fold detail ────────────────────────────────────────────────────
    fold_maes = result.get("fold_maes", [])
    if fold_maes:
        st.subheader("Cross-Validation — MAE per Fold")
        fold_df = pd.DataFrame(
            {"Fold": [f"Fold {i}" for i in range(1, len(fold_maes) + 1)], "MAE": fold_maes}
        ).set_index("Fold")
        st.bar_chart(fold_df, use_container_width=True)
        st.divider()

    # ── Model description ─────────────────────────────────────────────────
    desc = _MODEL_DESCRIPTION.get(model_name)
    if desc:
        st.subheader("About This Model")
        st.markdown(desc)
        st.divider()

    # ── Retrain ───────────────────────────────────────────────────────────
    st.subheader(f"Retrain {display_name}")
    st.caption(
        f"Re-runs hyperparameter tuning for {display_name} only, "
        "then re-fits on all available data. Other models are not affected."
    )

    _confirm_retrain_widget(run_key, f"🔁 Retrain {display_name}")


def page_about() -> None:
    st.title("About")

    if PLAN_PATH.exists():
        st.markdown(PLAN_PATH.read_text(encoding="utf-8"))
    else:
        st.warning(f"Implementation plan not found at `{PLAN_PATH}`.")

    st.divider()
    st.subheader("Tech Stack")
    st.markdown("""
| Component | Library |
|-----------|---------|
| Data fetching | `requests`, `beautifulsoup4` |
| Time-series models | `statsmodels` — ARIMA, SARIMAX, ExponentialSmoothing |
| Hyperparameter tuning | AIC pre-filter + walk-forward one-step-ahead CV |
| Cross-validation | `scikit-learn` `TimeSeriesSplit` (5 folds) |
| Performance metric | MAE (mean); Std MAE for stability |
| Interactive charts | `plotly` |
| Diagnostic plots | `matplotlib`, `scipy` |
| Web interface | `streamlit` |
""")

    st.divider()
    st.subheader("Quick Start")
    st.markdown("""
```bash
pip install -r requirements.txt     # install dependencies
python -m core.scraper              # fetch BNR exchange-rate data
python -m core.pipeline             # train all models  (~15–30 min)
streamlit run app.py                # launch this app
```
Data updates and retraining can also be triggered directly from the **Overview** page.
""")

    st.divider()
    st.subheader("File Reference")
    st.markdown("""
| File | Purpose |
|------|---------|
| `app.py` | This Streamlit application (primary entry point) |
| `core/scraper.py` | Scraper — POSTs to cursbnr.ro, saves `resources/data/idr_exchange_rates.csv` |
| `core/models.py` | Data loading (100 IDR scale), walk-forward CV, ARIMA/SARIMAX/ES tuning |
| `core/pipeline.py` | CLI retraining pipeline + `retrain_single_model()` |
| `core/visualize.py` | `make_forecast_figure()`, `make_per_model_diagnostic_figure()` |
| `resources/data/idr_exchange_rates.csv` | Exchange-rate data fetched by `core/scraper.py` |
| `resources/models/all_results.json` | CV results for all three models |
| `resources/models/*_<timestamp>.pkl` | Per-model fitted objects |
| `resources/models/diagnostics.png` | Combined diagnostic figure (CLI pipeline only) |
""")


# ─── Chatbot helpers ──────────────────────────────────────────────────────────

def _chat_messages() -> list:
    """Return (and initialise) the shared chat message history."""
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    return st.session_state["chat_messages"]


def _render_chat_action(action: dict, series, all_results) -> None:
    """Render a display action produced by the chatbot inside a chat bubble."""
    action_type = action.get("type")
    model_name = action.get("model_name", "")

    if action_type == "show_forecast":
        if series is None or all_results is None:
            st.warning("No data/models available to render chart.")
            return
        result = all_results.get(model_name)
        if result is None:
            st.warning(f"No results for {model_name}.")
            return
        from core.visualize import make_forecast_figure
        try:
            fig = make_forecast_figure(series, _normalise_params(result))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Chart error: {exc}")

    elif action_type == "show_comparison":
        if all_results is None:
            st.warning("No model results available.")
            return
        sorted_models = _sorted_models(all_results)
        best_name = all_results.get("best", {}).get("model", "")
        rows = []
        for rank, r in enumerate(sorted_models, 1):
            name = r["model"]
            rows.append({
                "Rank": f"{'★ ' if name == best_name else ''}#{rank}",
                "Model": _DISPLAY.get(name, name),
                "Mean CV MAE": f"{r['mean_mae']:.6f}",
                "Std CV MAE": f"{r['std_mae']:.6f}",
                "Best Params": _params_short(r),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    elif action_type == "show_diagnostics":
        pkl_data = _get_model_pkl(model_name)
        if pkl_data is None or all_results is None:
            st.warning(f"No saved model pickle for {model_name}.")
            return
        fitted = pkl_data.get("fitted")
        result = all_results.get(model_name)
        if fitted is None or result is None:
            st.warning("Pickle exists but model object missing.")
            return
        from core.visualize import make_per_model_diagnostic_figure_plotly
        try:
            fig = make_per_model_diagnostic_figure_plotly(series, fitted, result)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Diagnostic plot error: {exc}")


def _on_active_model_change() -> None:
    """on_change callback for the combined model selectbox in the chatbot page."""
    from core.chatbot import CLAUDE_MODELS
    model = st.session_state.get("chat_active_model", "")
    if model in CLAUDE_MODELS:
        st.session_state["chat_backend"] = "claude"
        st.session_state["chat_claude_model"] = model
    else:
        st.session_state["chat_backend"] = "ollama"
        st.session_state["chat_ollama_model"] = model
    _save_settings()


def _build_api_messages(chat_msgs: list) -> list[dict]:
    """Convert ChatMessage list → plain dicts for API calls."""
    return [{"role": m["role"], "content": m["content"]} for m in chat_msgs]


def _send_chat(
    user_text: str,
    series,
    all_results,
) -> None:
    """
    Append user message, call selected LLM backend, append assistant response.
    All state lives in st.session_state.
    """
    from core.chatbot import (
        build_app_context, get_response_claude, get_response_ollama,
        DEFAULT_CLAUDE_MODEL, DEFAULT_OLLAMA_MODEL, execute_run_action,
    )

    messages = _chat_messages()
    messages.append({"role": "user", "content": user_text, "actions": []})

    system = build_app_context(series, all_results)
    api_msgs = _build_api_messages(messages)

    backend = st.session_state.get("chat_backend", "claude")

    with st.spinner("Working …"):
        if backend == "claude":
            api_key = st.session_state.get("chat_claude_key", "")
            model = st.session_state.get("chat_claude_model", DEFAULT_CLAUDE_MODEL)
            if not api_key:
                response_text = (
                    "No Claude API key provided. "
                    "Go to **Settings** to enter your API key, or switch to a local model."
                )
                response_actions: list = []
            else:
                resp = get_response_claude(
                    api_msgs, system, api_key, model, series, all_results
                )
                response_text = resp.text
                response_actions = resp.actions
        else:
            model = st.session_state.get("chat_ollama_model", DEFAULT_OLLAMA_MODEL)
            resp = get_response_ollama(
                api_msgs, system, model, series, all_results
            )
            response_text = resp.text
            response_actions = resp.actions

    # Split actions: display ones persist in history; execute ones run once and are discarded.
    display_actions = [
        a for a in response_actions
        if not a.get("type", "").startswith(("run_", "did_"))
    ]
    run_actions = [
        a for a in response_actions
        if a.get("type", "").startswith(("run_", "did_"))
    ]

    # Hallucination guard (Claude only): if no tool action was emitted but the user asked
    # for data to be fetched and data is still absent, the model responded in text only.
    if backend == "claude" and not run_actions and series is None:
        _fetch_keywords = ("fetch", "download", "scrape", "get data", "update data", "retrieve")
        if any(kw in user_text.lower() for kw in _fetch_keywords):
            response_text += (
                "\n\n> ⚠️ **Note:** No fetch tool was called during this response. "
                "If you intended to download exchange-rate data, please ask again — "
                "e.g. *\"Please fetch the BNR data now.\"*"
            )

    # Execute pending actions (Ollama path); Claude already ran them inside _execute_tool.
    execution_notes: list[str] = []
    for action in run_actions:
        if action.get("type", "").startswith("run_"):
            note = execute_run_action(action)
            if note:
                execution_notes.append(note)

    if run_actions:
        _clear_caches()
        # Re-read after action so we can confirm what actually changed in the app
        fresh_series = _get_series()
        if fresh_series is not None and series is None:
            # Data was absent before; confirm it's now loaded
            first = fresh_series.index[0].date()
            last = fresh_series.index[-1].date()
            execution_notes.append(
                f"**Data is now available in the app** — "
                f"{len(fresh_series):,} observations ({first} → {last})."
            )

    if execution_notes:
        suffix = "\n\n" + "\n\n".join(execution_notes)
        response_text = (response_text + suffix).strip() if response_text else suffix.strip()

    messages.append({
        "role": "assistant",
        "content": response_text,
        "actions": display_actions,
    })


def page_chatbot() -> None:
    from core.chatbot import CLAUDE_MODELS, DEFAULT_OLLAMA_MODEL

    series = _get_series()
    all_results = _get_all_results()
    messages = _chat_messages()

    st.title("Chatbot")
    st.caption(
        "Answers are based only on the data loaded in this app. "
        "Configure the LLM backend in **Settings**."
    )

    # ── Message history ───────────────────────────────────────────────────────
    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["content"]:
                st.markdown(msg["content"])
            for action in msg.get("actions", []):
                _render_chat_action(action, series, all_results)

    # ── Action row (regenerate / clear) ──────────────────────────────────────
    if messages:
        regen_col, clear_col, _ = st.columns([1, 1, 6])
        with regen_col:
            regen_clicked = st.button("↺ Retry", key="btn_regen", use_container_width=True,
                                      help="Remove the last response and generate a new one")
        with clear_col:
            if st.button("🗑 Clear", key="btn_clear_chat", use_container_width=True):
                st.session_state["chat_messages"] = []
                st.rerun()

        if regen_clicked and len(messages) >= 2 and messages[-1]["role"] == "assistant":
            messages.pop()                     # remove assistant message
            last_user = messages.pop()         # remove user message (will be re-added)
            _send_chat(last_user["content"], series, all_results)
            st.rerun()

    # ── Quick model selector ──────────────────────────────────────────────────
    _, installed = _ollama_status()
    ollama_models = installed if installed else [DEFAULT_OLLAMA_MODEL]
    all_models = CLAUDE_MODELS + ollama_models

    # Derive the currently active model from backend + specific model keys.
    backend = st.session_state.get("chat_backend", "claude")
    current_model = (
        st.session_state.get("chat_claude_model", CLAUDE_MODELS[0])
        if backend == "claude"
        else st.session_state.get("chat_ollama_model", ollama_models[0])
    )
    if st.session_state.get("chat_active_model") not in all_models:
        st.session_state["chat_active_model"] = (
            current_model if current_model in all_models else all_models[0]
        )

    sel_col, _ = st.columns([2, 6])
    with sel_col:
        st.selectbox(
            "Model",
            options=all_models,
            format_func=lambda m: f"Claude › {m}" if m in CLAUDE_MODELS else m,
            key="chat_active_model",
            label_visibility="collapsed",
            on_change=_on_active_model_change,
        )

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input(
        "Ask about exchange rates, models, forecasts …",
        key="chatbot_page_input",
    ):
        with st.chat_message("user"):
            st.markdown(prompt)
        _send_chat(prompt, series, all_results)
        last = messages[-1] if messages else None
        if last and last["role"] == "assistant":
            with st.chat_message("assistant"):
                if last["content"]:
                    st.markdown(last["content"])
                for action in last.get("actions", []):
                    _render_chat_action(action, series, all_results)


def page_settings() -> None:
    from core.chatbot import CLAUDE_MODELS, DEFAULT_OLLAMA_MODEL, download_ollama_model

    st.title("Settings")

    # ── Chat / LLM ────────────────────────────────────────────────────────────
    st.subheader("Chat")
    st.radio(
        "LLM Backend",
        options=["claude", "ollama"],
        format_func=lambda x: "Claude (Anthropic API)" if x == "claude" else "Local model (Ollama)",
        key="chat_backend",
        horizontal=True,
        on_change=_save_settings,
    )
    st.divider()

    if st.session_state.get("chat_backend", "claude") == "claude":
        st.markdown(
            "**Getting an API key**  \n"
            "Sign in at [console.anthropic.com](https://console.anthropic.com), open "
            "**API Keys** in the left sidebar, and click **Create Key**. "
            "Keys start with `sk-ant-…`.  \n"
            "Your key is saved to `resources/settings.json` on this machine "
            "and is not included in git."
        )
        k1, k2 = st.columns([2, 1])
        with k1:
            st.text_input(
                "API Key",
                type="password",
                key="chat_claude_key",
                placeholder="sk-ant-…",
                on_change=_save_settings,
            )
        with k2:
            st.selectbox(
                "Model",
                options=CLAUDE_MODELS,
                key="chat_claude_model",
                on_change=_save_settings,
            )
    else:
        ollama_running, installed = _ollama_status()
        if ollama_running:
            st.success("Ollama is running.")
        else:
            st.warning(
                "Ollama is not reachable. Start it with `ollama serve`, "
                "or install it from [ollama.com](https://ollama.com)."
            )
        model_options = installed if installed else [DEFAULT_OLLAMA_MODEL]
        st.selectbox(
            "Model",
            options=model_options,
            key="chat_ollama_model",
            on_change=_save_settings,
        )

        st.divider()
        st.subheader("Download a model")
        st.caption(
            "Enter any model name from [ollama.com/search](https://ollama.com/search). "
            "Examples: `llama3.2`, `mistral`, `phi4`."
        )
        p1, p2 = st.columns([3, 1])
        with p1:
            st.text_input(
                "Model name",
                key="_ollama_pull_input",
                placeholder=DEFAULT_OLLAMA_MODEL,
                label_visibility="collapsed",
            )
        with p2:
            pull_clicked = st.button("⬇ Pull", key="btn_pull_model", use_container_width=True)

        if pull_clicked:
            pull_name = (
                st.session_state.get("_ollama_pull_input", "").strip() or DEFAULT_OLLAMA_MODEL
            )
            status_msg = st.empty()
            status_msg.info(f"Pulling `{pull_name}` …")
            pull_lines: list[str] = []
            with st.container(height=220):
                output_area = st.empty()
            for line in download_ollama_model(pull_name):
                pull_lines.append(line)
                output_area.code("\n".join(pull_lines[-60:]), language=None)
            if any(ln.startswith("[DONE]") for ln in pull_lines):
                status_msg.success(f"✅ `{pull_name}` downloaded. Reload the page to use it.")
                _ollama_status.clear()
            else:
                status_msg.error("Download failed — see output above.")


# ─── Sidebar navigation ───────────────────────────────────────────────────────

def main() -> None:
    if "page" not in st.session_state:
        st.session_state["page"] = "Overview"
    if "chat_backend" not in st.session_state:
        _load_settings()

    with st.sidebar:
        st.markdown("## 📈 IDR/RON Forecaster")
        st.markdown("---")

        for page in _PAGES:
            is_active = st.session_state["page"] == page
            if st.button(
                page,
                key=f"nav_{page}",
                type="primary" if is_active else "secondary",
                width="stretch",
            ):
                st.session_state["page"] = page
                st.rerun()  # re-render immediately so button colours update

        st.markdown("---")
        # Status indicators (auto-refresh every render thanks to mtime cache)
        csv_ok = CSV_PATH.exists()
        models_ok = ALL_RESULTS_PATH.exists()
        st.markdown(
            f"{'🟢' if csv_ok else '🔴'} Exchange-rate data  \n"
            f"{'🟢' if models_ok else '🔴'} Trained models"
        )
        if models_ok:
            ts = (_get_all_results() or {}).get("timestamp", "")
            if ts:
                dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                st.caption(f"Trained: {dt.strftime('%d %b %Y %H:%M')}")

        # ── Sidebar quick-chat ────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("💬 Quick Chat", expanded=False):
            msgs = _chat_messages()
            for m in msgs[-2:]:
                icon = "🧑" if m["role"] == "user" else "🤖"
                preview = m["content"][:110] + ("…" if len(m["content"]) > 110 else "")
                st.caption(f"**{icon}** {preview}")
                if msgs[-2:] and m != msgs[-2:][-1]:
                    st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)

            with st.form("sidebar_quick_chat", clear_on_submit=True):
                quick_input = st.text_input(
                    "Message",
                    label_visibility="collapsed",
                    placeholder="Ask about the data …",
                )
                submitted = st.form_submit_button("Send", use_container_width=True)

            if st.button("Open chatbot", key="sidebar_open_chat", use_container_width=True):
                st.session_state["page"] = "Chatbot"
                st.rerun()

            if submitted and quick_input.strip():
                _send_chat(quick_input.strip(), _get_series(), _get_all_results())
                st.rerun()

    page = st.session_state["page"]
    if page == "Overview":
        page_overview()
    elif page == "ARIMA":
        page_model("ARIMA")
    elif page == "Exponential Smoothing":
        page_model("ExponentialSmoothing")
    elif page == "Naive":
        page_model("Naive")
    elif page == "Naive + Drift":
        page_model("NaiveDrift")
    elif page == "Chatbot":
        page_chatbot()
    elif page == "Settings":
        page_settings()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
