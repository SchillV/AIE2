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
_PAGES = ["Overview", "ARIMA", "Exponential Smoothing", "Naive", "Naive + Drift", "About"]

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


# ─── Sidebar navigation ───────────────────────────────────────────────────────

def main() -> None:
    if "page" not in st.session_state:
        st.session_state["page"] = "Overview"

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
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
