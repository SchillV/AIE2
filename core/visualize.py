"""
Visualisation utilities for IDR/RON exchange rate forecasting.

Public API
----------
make_forecast_figure()              → plotly Figure  (used by Streamlit app)
make_per_model_diagnostic_figure()  → matplotlib Figure (used by Streamlit app)
generate_all_plots()                → saves diagnostics.png  (used by CLI pipeline)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from scipy import stats

from .models import fit_final_model


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _normalise_params(params: dict) -> dict:
    """Ensure tuple fields are tuples (they become lists when round-tripped via JSON)."""
    out = dict(params)
    for key in ("order", "seasonal_order"):
        if isinstance(out.get(key), list):
            out[key] = tuple(out[key])
    return out


def _retroactive_forecast(
    series: pd.Series,
    model_params: dict,
    n_retro_days: int = 10,
    alpha: float = 0.05,
    n_boot: int = 1000,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Fit model on series[:-n_retro_days], forecast n_retro_days steps ahead.
    Returns (actual_test, pred_mean, lower_95, upper_95) all on test.index.
    """
    model_params = _normalise_params(model_params)
    train = series.iloc[:-n_retro_days]
    test = series.iloc[-n_retro_days:]
    fitted = fit_final_model(train, model_params)
    model_name = model_params["model"]

    if model_name in ("ARIMA", "SARIMAX"):
        fc = fitted.get_forecast(steps=n_retro_days)
        pred_mean = fc.predicted_mean
        ci = fc.conf_int(alpha=alpha)
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values
    else:
        pred_mean_values = fitted.forecast(steps=n_retro_days).values
        residuals = fitted.resid.dropna().values
        boots = np.stack(
            [
                pred_mean_values
                + np.random.choice(residuals, size=n_retro_days, replace=True)
                for _ in range(n_boot)
            ]
        )
        lower = np.percentile(boots, 100 * alpha / 2, axis=0)
        upper = np.percentile(boots, 100 * (1 - alpha / 2), axis=0)
        pred_mean = pd.Series(pred_mean_values, index=test.index)

    pred_s = pd.Series(np.asarray(pred_mean), index=test.index, name="Predicted")
    lower_s = pd.Series(lower, index=test.index, name="Lower 95%")
    upper_s = pd.Series(upper, index=test.index, name="Upper 95%")
    return test, pred_s, lower_s, upper_s


# --------------------------------------------------------------------------- #
# Public: interactive Plotly forecast chart                                   #
# --------------------------------------------------------------------------- #

def make_forecast_figure(
    series: pd.Series,
    model_params: dict,
    n_history_days: int = 44,
    n_retro_days: int = 10,
) -> go.Figure:
    """
    Build and return an interactive Plotly figure (does NOT save to disk).

    Traces
    ------
    - Blue solid   : historical data (last ~2 months)
    - Orange dashed: retroactive model prediction (last 2 weeks)
    - Orange band  : 95% confidence interval
    - Green solid  : actual values for the retroactive window
    """
    if len(series) <= n_retro_days:
        raise ValueError(
            f"Series too short ({len(series)} obs) for a {n_retro_days}-day retroactive window."
        )

    model_params = _normalise_params(model_params)
    _test, pred, lower, upper = _retroactive_forecast(series, model_params, n_retro_days)

    # History spans the full window (pre-retro + retro) so there is no gap.
    # The prediction is overlaid on the retroactive slice.
    history_start = max(0, len(series) - n_history_days - n_retro_days)
    history = series.iloc[history_start:]

    retro_start_str = str(series.index[-n_retro_days])
    retro_end_str   = str(series.index[-1])

    fig = go.Figure()

    # Shaded background for the retroactive window (drawn first, sits behind everything)
    fig.add_shape(
        type="rect",
        x0=retro_start_str, x1=retro_end_str,
        y0=0, y1=1, yref="paper",
        fillcolor="rgba(255, 127, 14, 0.07)",
        line_width=0, layer="below",
    )
    fig.add_annotation(
        x=retro_start_str, y=1, yref="paper",
        text="Retroactive window", showarrow=False,
        xanchor="left", yanchor="bottom",
        font=dict(size=11, color="#888888"),
    )

    # Continuous historical + actual line (no gap)
    fig.add_trace(go.Scatter(
        x=history.index, y=history.values,
        mode="lines", name="Actual rate",
        line=dict(color="#2c7bb6", width=2),
    ))

    # CI band (behind prediction line)
    fig.add_trace(go.Scatter(
        x=list(pred.index) + list(pred.index[::-1]),
        y=list(upper.values) + list(lower.values[::-1]),
        fill="toself", fillcolor="rgba(255,127,14,0.20)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Interval", hoverinfo="skip",
    ))

    # Retroactive prediction overlaid on the retro window
    fig.add_trace(go.Scatter(
        x=pred.index, y=pred.values,
        mode="lines+markers",
        name=f"Predicted – {model_params['model']}",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
        marker=dict(size=6, symbol="circle-open"),
    ))

    mae = model_params["mean_mae"]
    std_mae = model_params["std_mae"]
    model_label = model_params["model"]

    fig.update_layout(
        title=dict(
            text=(
                f"IDR/RON Exchange Rate – {model_label} Retroactive Forecast<br>"
                f"<sup>CV MAE = {mae:.6f}  ±  {std_mae:.6f}</sup>"
            ),
            x=0.5, font=dict(size=16),
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="#e8e8e8",
            # hide weekend gaps so business-day data appears continuous
            rangebreaks=[dict(bounds=["sat", "mon"])],
        ),
        yaxis=dict(title="100 IDR / RON", showgrid=True, gridcolor="#e8e8e8"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=520, margin=dict(t=100),
    )
    return fig


# --------------------------------------------------------------------------- #
# Public: per-model diagnostic matplotlib figure                              #
# --------------------------------------------------------------------------- #

def make_per_model_diagnostic_figure(
    series: pd.Series,
    fitted_model,
    model_result: dict,
) -> plt.Figure:
    """
    Return a 2×2 matplotlib Figure with diagnostics for a single model.
    Does NOT save to disk — caller is responsible for display or saving.
    """
    model_name = model_result.get("model", "Model")
    residuals = fitted_model.resid.dropna()
    fold_maes = model_result.get("fold_maes", [])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    # ── (0,0) Residuals histogram ──────────────────────────────────────────
    ax = axes[0, 0]
    ax.hist(residuals, bins=40, density=True, alpha=0.72, color="#2c7bb6", edgecolor="white")
    x_range = np.linspace(residuals.min(), residuals.max(), 300)
    ax.plot(x_range, stats.norm.pdf(x_range, residuals.mean(), residuals.std()),
            "r-", lw=2, label="Normal fit")
    ax.set_title("Residuals Distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.legend()

    # ── (0,1) Residuals over time ─────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(residuals.index, residuals.values, color="#2c7bb6", lw=1, alpha=0.8)
    ax.axhline(0, color="red", linestyle="--", lw=1.2)
    ax.set_title("Residuals Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")

    # ── (1,0) MAE per CV fold ─────────────────────────────────────────────
    ax = axes[1, 0]
    if fold_maes:
        folds = range(1, len(fold_maes) + 1)
        ax.bar(folds, fold_maes, color="#2c7bb6", alpha=0.8, edgecolor="black", linewidth=0.6)
        ax.axhline(np.mean(fold_maes), color="red", linestyle="--", lw=1.5,
                   label=f"Mean = {np.mean(fold_maes):.6f}")
        ax.set_title("MAE per CV Fold")
        ax.set_xlabel("Fold")
        ax.set_ylabel("MAE")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No CV data", ha="center", va="center", transform=ax.transAxes)

    # ── (1,1) Actual vs Predicted (in-sample, last 100) ───────────────────
    ax = axes[1, 1]
    try:
        fitted_vals = fitted_model.fittedvalues.dropna().iloc[-100:]
        actual_vals = series.loc[fitted_vals.index]
        ax.scatter(actual_vals, fitted_vals, alpha=0.45, s=14, color="#2c7bb6")
        lo = min(actual_vals.min(), fitted_vals.min())
        hi = max(actual_vals.max(), fitted_vals.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
        ax.set_title("Actual vs Predicted (in-sample, last 100)")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend()
    except Exception as exc:
        ax.text(0.5, 0.5, f"Could not plot:\n{exc}", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)

    fig.suptitle(f"{model_name} – Diagnostic Plots", fontsize=13, fontweight="bold")
    return fig


# --------------------------------------------------------------------------- #
# Private: combined all-models diagnostic panel (CLI pipeline only)           #
# --------------------------------------------------------------------------- #

def _plot_diagnostics(
    series: pd.Series,
    fitted_model,
    all_results: dict,
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    best_name = all_results["best"]["model"]
    residuals = fitted_model.resid.dropna()

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(residuals, bins=40, density=True, alpha=0.72, color="#2c7bb6", edgecolor="white")
    x_range = np.linspace(residuals.min(), residuals.max(), 300)
    ax.plot(x_range, stats.norm.pdf(x_range, residuals.mean(), residuals.std()),
            "r-", lw=2, label="Normal fit")
    ax.set_title(f"Residuals Distribution ({best_name})")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(residuals.index, residuals.values, color="#2c7bb6", lw=1, alpha=0.8)
    ax.axhline(0, color="red", linestyle="--", lw=1.2)
    ax.set_title(f"Residuals Over Time ({best_name})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")

    ax = fig.add_subplot(gs[0, 2])
    model_keys = ["ARIMA", "SARIMAX", "ExponentialSmoothing"]
    palette = ["#2c7bb6", "#d7191c", "#1a9641"]
    for key, color in zip(model_keys, palette):
        fold_maes = all_results.get(key, {}).get("fold_maes", [])
        if fold_maes:
            ax.plot(range(1, len(fold_maes) + 1), fold_maes, marker="o", label=key, color=color)
    ax.set_title("MAE per CV Fold (all models)")
    ax.set_xlabel("CV Fold")
    ax.set_ylabel("MAE")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    short_names = ["ARIMA", "SARIMAX", "ES"]
    mean_maes = [all_results.get(k, {}).get("mean_mae", np.nan) for k in model_keys]
    std_maes = [all_results.get(k, {}).get("std_mae", np.nan) for k in model_keys]
    bars = ax.bar(short_names, mean_maes, yerr=std_maes, capsize=6,
                  color=palette, alpha=0.82, edgecolor="black", linewidth=0.8)
    winner_idx = model_keys.index(best_name)
    bars[winner_idx].set_edgecolor("gold")
    bars[winner_idx].set_linewidth(3)
    ax.set_title("Mean MAE ± Std (best = gold border)")
    ax.set_ylabel("MAE")

    ax = fig.add_subplot(gs[1, 1])
    try:
        fitted_vals = fitted_model.fittedvalues.dropna().iloc[-100:]
        actual_vals = series.loc[fitted_vals.index]
        ax.scatter(actual_vals, fitted_vals, alpha=0.45, s=14, color="#2c7bb6")
        lo = min(actual_vals.min(), fitted_vals.min())
        hi = max(actual_vals.max(), fitted_vals.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
        ax.set_title("Actual vs Predicted (in-sample, last 100)")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend()
    except Exception as exc:
        ax.text(0.5, 0.5, f"Could not plot:\n{exc}", ha="center", va="center",
                transform=ax.transAxes)

    ax = fig.add_subplot(gs[1, 2])
    fold_maes = all_results["best"].get("fold_maes", [])
    if fold_maes:
        n = len(fold_maes)
        cum_mean = np.cumsum(fold_maes) / np.arange(1, n + 1)
        ax.plot(range(1, n + 1), cum_mean, marker="o", color="#d7191c", lw=2)
        ax.set_title(f"Cumulative Mean MAE – {best_name}")
        ax.set_xlabel("CV Fold")
        ax.set_ylabel("Cumulative Mean MAE")
    else:
        ax.text(0.5, 0.5, "No fold data", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("IDR/RON Exchange Rate – Model Diagnostic Plots",
                 fontsize=14, fontweight="bold", y=1.01)

    out = output_dir / "diagnostics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic plots saved → {out}")


# --------------------------------------------------------------------------- #
# Public entry point (CLI pipeline)                                           #
# --------------------------------------------------------------------------- #

def generate_all_plots(
    series: pd.Series,
    fitted_model,
    all_results: dict,
    output_dir: str | Path = ".",
) -> None:
    """Save the combined all-models diagnostic PNG. (forecast.html is now served by the app.)"""
    output_dir = Path(output_dir)
    _plot_diagnostics(series, fitted_model, all_results, output_dir)


# --------------------------------------------------------------------------- #
# Public: forecast data extraction and static chart (used by pipeline logs)   #
# --------------------------------------------------------------------------- #

def retroactive_forecast_df(
    series: pd.Series,
    model_params: dict,
    n_retro_days: int = 10,
) -> pd.DataFrame:
    """
    Run the retroactive forecast and return results as a DataFrame.

    Columns: actual, predicted, lower_95, upper_95.
    Index: DatetimeIndex of the retroactive evaluation window.
    """
    model_params = _normalise_params(model_params)
    actual, pred, lower, upper = _retroactive_forecast(series, model_params, n_retro_days)
    return pd.DataFrame(
        {
            "actual": actual.values,
            "predicted": pred.values,
            "lower_95": lower.values,
            "upper_95": upper.values,
        },
        index=actual.index,
    )


def make_forecast_figure_static(
    series: pd.Series,
    model_params: dict,
    n_history_days: int = 44,
    n_retro_days: int = 10,
) -> plt.Figure:
    """Matplotlib forecast chart for saving to PNG in training logs.

    Shows the same retroactive window as the interactive Plotly chart in the app,
    but as a static matplotlib figure that can be saved without kaleido.
    """
    model_params = _normalise_params(model_params)
    actual, pred, lower, upper = _retroactive_forecast(series, model_params, n_retro_days)

    history_start = max(0, len(series) - n_history_days - n_retro_days)
    history = series.iloc[history_start:]

    model_name = model_params["model"]
    mae = model_params.get("mean_mae", float("nan"))
    std_mae = model_params.get("std_mae", float("nan"))

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(history.index, history.values, color="#2c7bb6", lw=2, label="Actual rate")
    ax.fill_between(
        pred.index, lower.values, upper.values,
        color="#ff7f0e", alpha=0.20, label="95% CI",
    )
    ax.plot(
        pred.index, pred.values,
        color="#ff7f0e", lw=2, linestyle="--", marker="o", markersize=5,
        label=f"Predicted – {model_name}",
    )
    ax.axvspan(actual.index[0], actual.index[-1], alpha=0.05, color="#ff7f0e")
    ax.axvline(actual.index[0], color="#888888", lw=1, linestyle=":", alpha=0.8)

    ax.set_title(
        f"IDR/RON – {model_name} Retroactive Forecast\n"
        f"CV MAE = {mae:.6f}  ±  {std_mae:.6f}",
        fontsize=12,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("100 IDR / RON")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
