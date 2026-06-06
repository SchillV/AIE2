"""
Streamlit front-end for IDR/RON exchange rate forecasting.

Run with:
    streamlit run app.py
"""

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

MODEL_NAMES = ["ARIMA", "SARIMAX", "ExponentialSmoothing"]
_PKL_PREFIX = {
    "ARIMA": "arima",
    "SARIMAX": "sarimax",
    "ExponentialSmoothing": "exponentialsmoothing",
}
_DISPLAY = {
    "ARIMA": "ARIMA",
    "SARIMAX": "SARIMAX",
    "ExponentialSmoothing": "Exponential Smoothing",
}
_PAGES = ["Overview", "ARIMA", "SARIMAX", "Exponential Smoothing", "About"]

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
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #f8f9fb;
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
        load_series, tune_arima, tune_sarimax, tune_exp_smoothing,
        compare_models, fit_final_model,
    )
    from core.pipeline import _serialise, _PKL_PREFIX as PKL, OUTPUT_DIR
    from core.visualize import generate_all_plots

    MODELS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        with st.status("Retraining all models …", expanded=True) as status:
            prog = st.progress(0, text="Starting …")

            # ── Load data ─────────────────────────────────────────────────
            prog.progress(2, text="Loading exchange-rate data …")
            series = load_series(str(CSV_PATH))
            st.write(
                f"✅ **Data loaded** — {len(series):,} observations "
                f"({series.index[0].date()} → {series.index[-1].date()})"
            )

            # ── ARIMA ─────────────────────────────────────────────────────
            prog.progress(8, text="Tuning ARIMA — computing AIC candidates …")
            st.write("---")
            st.write("🔍 **ARIMA** — AIC pre-filter + walk-forward CV …")
            arima_res = tune_arima(series)
            st.write(
                f"✅ ARIMA best: `order={arima_res['order']}`  "
                f"MAE = `{arima_res['mean_mae']:.6f} ± {arima_res['std_mae']:.6f}`"
            )

            # ── SARIMAX ───────────────────────────────────────────────────
            prog.progress(35, text="Tuning SARIMAX — AIC grid search …")
            st.write("---")
            st.write("🔍 **SARIMAX** — AIC pre-filter + walk-forward CV …")
            sarimax_res = tune_sarimax(series)
            st.write(
                f"✅ SARIMAX best: `order={sarimax_res['order']}  "
                f"seasonal={sarimax_res['seasonal_order']}`  "
                f"MAE = `{sarimax_res['mean_mae']:.6f} ± {sarimax_res['std_mae']:.6f}`"
            )

            # ── Exponential Smoothing ─────────────────────────────────────
            prog.progress(65, text="Tuning Exponential Smoothing …")
            st.write("---")
            st.write("🔍 **Exponential Smoothing** — walk-forward CV on all valid combos …")
            es_res = tune_exp_smoothing(series)
            st.write(
                f"✅ ES best: `trend={es_res['trend']}  seasonal={es_res['seasonal']}  "
                f"damped={es_res['damped_trend']}`  "
                f"MAE = `{es_res['mean_mae']:.6f} ± {es_res['std_mae']:.6f}`"
            )

            # ── Select best ───────────────────────────────────────────────
            prog.progress(80, text="Selecting best model …")
            st.write("---")
            best = compare_models(arima_res, sarimax_res, es_res)
            st.write(
                f"🏆 **Best model: {best['model']}** — "
                f"MAE = `{best['mean_mae']:.6f} ± {best['std_mae']:.6f}`"
            )

            # ── Fit & save ────────────────────────────────────────────────
            prog.progress(85, text="Fitting all models on full series …")
            st.write("---")
            st.write("💾 Fitting and saving …")
            fitted_map: dict = {}
            for name, res in [
                ("ARIMA", arima_res),
                ("SARIMAX", sarimax_res),
                ("ExponentialSmoothing", es_res),
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
                "SARIMAX": _serialise(sarimax_res),
                "ExponentialSmoothing": _serialise(es_res),
                "best": _serialise(best),
                "timestamp": timestamp,
                "n_observations": len(series),
            }
            with open(ALL_RESULTS_PATH, "w", encoding="utf-8") as fh:
                json.dump(all_results_payload, fh, indent=2)
            st.write(f"  · `all_results.json` updated")

            # Diagnostics PNG
            prog.progress(95, text="Generating diagnostic plots …")
            all_results_dict = {
                "ARIMA": arima_res, "SARIMAX": sarimax_res,
                "ExponentialSmoothing": es_res, "best": best,
            }
            generate_all_plots(series, fitted_map[best["model"]], all_results_dict)
            st.write("  · `diagnostics.png` saved")

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
    from core.models import load_series, tune_arima, tune_sarimax, tune_exp_smoothing, fit_final_model
    from core.pipeline import _serialise, _PKL_PREFIX as PKL, OUTPUT_DIR, MODEL_NAMES

    MODELS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    display = _DISPLAY[model_name]

    try:
        with st.status(f"Retraining {display} …", expanded=True) as status:
            prog = st.progress(0, text="Starting …")

            prog.progress(5, text="Loading data …")
            series = load_series(str(CSV_PATH))
            st.write(f"✅ {len(series):,} observations loaded")

            prog.progress(12, text=f"Tuning {display} …")
            st.write(f"🔍 **{display}** — hyperparameter search + walk-forward CV …")
            if model_name == "ARIMA":
                result = tune_arima(series)
            elif model_name == "SARIMAX":
                result = tune_sarimax(series)
            else:
                result = tune_exp_smoothing(series)
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


def _confirm_retrain_widget(key: str, label: str, action) -> None:
    """Reusable confirm-before-retrain widget.

    The trigger button, warning, and confirm/cancel buttons are rendered here.
    The action itself is deferred until *after* the column block closes so that
    any st.status / st.progress panels it creates render at full page width.
    """
    """Reusable confirm-before-retrain widget.

    The trigger button, warning, and confirm/cancel buttons are rendered here.
    The action itself is deferred until *after* the column block closes so that
    any st.status / st.progress panels it creates render at full page width.
    """
    if st.button(label, key=f"btn_{key}"):
        st.session_state[f"pending_{key}"] = True

    if st.session_state.get(f"pending_{key}"):
        st.warning("This operation may take **15–30 minutes**. The app will be unresponsive during training.")
        col_go, col_cancel, _ = st.columns([1, 1, 5])
        with col_go:
            if st.button("✅ Confirm", key=f"confirm_{key}", use_container_width=True):
                st.session_state[f"pending_{key}"] = False
                st.session_state[f"run_{key}"] = True   # defer to outside the column
        with col_cancel:
            if st.button("✖ Cancel", key=f"cancel_{key}", use_container_width=True):
                st.session_state[f"pending_{key}"] = False
                st.rerun()

    # Run action at full page width — outside the columns above.
    if st.session_state.get(f"run_{key}"):
        st.session_state[f"run_{key}"] = False
        action()
        st.rerun()

    # Run action at full page width — outside the columns above.
    if st.session_state.get(f"run_{key}"):
        st.session_state[f"run_{key}"] = False
        action()
        st.rerun()


# ─── Pages ───────────────────────────────────────────────────────────────────

def page_overview() -> None:
    st.title("IDR / RON Exchange Rate Forecaster")
    st.caption("Source: [cursbnr.ro](https://www.cursbnr.ro) — BNR official rates · scale: **100 IDR → RON**")

    series = _get_series()
    all_results = _get_all_results()

    # ── Action buttons ────────────────────────────────────────────────────
    c_upd, c_retrain, _ = st.columns([1.4, 1.8, 4.8])
    with c_upd:
        if st.button("🔄 Update Data", use_container_width=True):
            st.session_state["do_update_data_now"] = True   # defer to full-width context

    with c_retrain:
        # Only the trigger button lives inside the narrow column.
        if st.button("🤖 Retrain All Models", key="btn_retrain_all", use_container_width=True):
            st.session_state["pending_retrain_all"] = True

    # Run at full page width — outside all columns above.
    if st.session_state.get("do_update_data_now"):
        st.session_state["do_update_data_now"] = False
        with st.status("Fetching latest rates from BNR …", expanded=True) as s:
            ok = _do_update_data()
        if ok:
            s.update(label="✅ Data updated!", state="complete")
            st.rerun()

    # Confirmation dialog rendered at full page width, outside the columns above.
    if st.session_state.get("pending_retrain_all"):
        st.warning(
            "Retraining all three models with hyperparameter tuning may take "
            "**15–30 minutes**. The app will be unresponsive during this time."
        )
        col_go, col_cancel, _ = st.columns([1, 1, 5])
        with col_go:
            if st.button("✅ Confirm retrain", key="confirm_retrain_all", use_container_width=True):
                st.session_state["pending_retrain_all"] = False
                st.session_state["do_retrain_all_now"] = True   # defer to full-width context
                st.session_state["do_retrain_all_now"] = True   # defer to full-width context
        with col_cancel:
            if st.button("✖ Cancel", key="cancel_retrain_all", use_container_width=True):
                st.session_state["pending_retrain_all"] = False
                st.rerun()

    # Run at full page width — outside all columns above.
    if st.session_state.get("do_retrain_all_now"):
        st.session_state["do_retrain_all_now"] = False
        _do_retrain_all()
        st.rerun()

    # Run at full page width — outside all columns above.
    if st.session_state.get("do_retrain_all_now"):
        st.session_state["do_retrain_all_now"] = False
        _do_retrain_all()
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

    # ── Model comparison table ────────────────────────────────────────────
    st.subheader("Model Performance Comparison")
    best_name = all_results.get("best", {}).get("model", "")
    sorted_models = _sorted_models(all_results)
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

    st.divider()

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


def page_model(model_name: str) -> None:
    display_name = _DISPLAY.get(model_name, model_name)
    st.title(display_name)

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
            from core.visualize import make_per_model_diagnostic_figure
            try:
                diag_fig = make_per_model_diagnostic_figure(series, fitted, result)
                st.pyplot(diag_fig, use_container_width=True)
                plt.close(diag_fig)
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

    # ── Retrain ───────────────────────────────────────────────────────────
    st.subheader(f"Retrain {display_name}")
    st.caption(
        f"Re-runs hyperparameter tuning for {display_name} only, "
        "then re-fits on all available data. Other models are not affected."
    )

    def _retrain_action():
        _do_retrain_single(model_name)

    _confirm_retrain_widget(f"retrain_{model_name}", f"🔁 Retrain {display_name}", _retrain_action)


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
                use_container_width=True,
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
    elif page == "SARIMAX":
        page_model("SARIMAX")
    elif page == "Exponential Smoothing":
        page_model("ExponentialSmoothing")
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
