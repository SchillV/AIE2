"""
Interactive Plotly forecast chart and matplotlib diagnostic panel
for IDR/RON exchange rate predictions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from scipy import stats

from models import fit_final_model


# --------------------------------------------------------------------------- #
# Retroactive forecast helper                                                  #
# --------------------------------------------------------------------------- #

def _retroactive_forecast(
    series: pd.Series,
    best_params: dict,
    n_retro_days: int = 10,
    alpha: float = 0.05,
    n_boot: int = 1000,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Fit model on series[:-n_retro_days], forecast forward n_retro_days steps.
    Returns (actual_test, pred_mean, lower_95, upper_95) all on test.index.
    """
    train = series.iloc[:-n_retro_days]
    test = series.iloc[-n_retro_days:]
    fitted = fit_final_model(train, best_params)
    model_name = best_params["model"]

    if model_name in ("ARIMA", "SARIMAX"):
        fc = fitted.get_forecast(steps=n_retro_days)
        pred_mean = fc.predicted_mean
        ci = fc.conf_int(alpha=alpha)
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values

    else:  # ExponentialSmoothing — bootstrap CI
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
# Interactive Plotly chart                                                     #
# --------------------------------------------------------------------------- #

def plot_interactive_forecast(
    series: pd.Series,
    all_results: dict,
    n_history_days: int = 44,
    n_retro_days: int = 10,
    output_html: str = "forecast.html",
) -> go.Figure:
    """
    Interactive chart:
      - Blue line  : historical data (last ~2 months)
      - Green line : actual values for the retroactive 2-week window
      - Orange band: 95% confidence interval
      - Orange dash: retroactive model prediction
    """
    best_params = all_results["best"]
    test, pred, lower, upper = _retroactive_forecast(series, best_params, n_retro_days)

    # Historical slice excludes the retroactive window
    history_end = series.index[-n_retro_days - 1]
    history_start_idx = max(0, len(series) - n_history_days - n_retro_days)
    history = series.iloc[history_start_idx : len(series) - n_retro_days]

    fig = go.Figure()

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="Historical (last 2 months)",
            line=dict(color="#2c7bb6", width=2),
        )
    )

    # 95% CI band (drawn before the prediction line so it sits behind)
    fig.add_trace(
        go.Scatter(
            x=list(pred.index) + list(pred.index[::-1]),
            y=list(upper.values) + list(lower.values[::-1]),
            fill="toself",
            fillcolor="rgba(255, 127, 14, 0.20)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence Interval",
            hoverinfo="skip",
        )
    )

    # Retroactive prediction
    fig.add_trace(
        go.Scatter(
            x=pred.index,
            y=pred.values,
            mode="lines+markers",
            name=f"Predicted – {best_params['model']}",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            marker=dict(size=6, symbol="circle-open"),
        )
    )

    # Actual values for the retroactive window
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=test.values,
            mode="lines+markers",
            name="Actual (retroactive period)",
            line=dict(color="#1a9641", width=2),
            marker=dict(size=6),
        )
    )

    # Separator line at the start of the retroactive window
    # (add_vline with annotations fails on datetime axes in recent plotly/pandas;
    #  use add_shape + add_annotation instead)
    separator_x = str(history.index[-1])
    fig.add_shape(
        type="line",
        x0=separator_x,
        x1=separator_x,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#888888", width=1, dash="dot"),
    )
    fig.add_annotation(
        x=separator_x,
        y=1,
        yref="paper",
        text="Retroactive window",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=11, color="#888888"),
    )

    mae = best_params["mean_mae"]
    std_mae = best_params["std_mae"]
    model_label = best_params["model"]

    fig.update_layout(
        title=dict(
            text=(
                f"IDR/RON Exchange Rate – {model_label} Retroactive Forecast<br>"
                f"<sup>CV MAE = {mae:.6f}  ±  {std_mae:.6f}</sup>"
            ),
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(title="Date", showgrid=True, gridcolor="#e8e8e8"),
        yaxis=dict(title="IDR / RON Rate", showgrid=True, gridcolor="#e8e8e8"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=560,
        margin=dict(t=100),
    )

    fig.write_html(output_html)
    print(f"Interactive chart saved → {output_html}")
    return fig


# --------------------------------------------------------------------------- #
# Matplotlib diagnostic panel                                                  #
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

    # ── (0,0) Residuals histogram ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(residuals, bins=40, density=True, alpha=0.72, color="#2c7bb6", edgecolor="white")
    x_range = np.linspace(residuals.min(), residuals.max(), 300)
    ax.plot(
        x_range,
        stats.norm.pdf(x_range, residuals.mean(), residuals.std()),
        "r-",
        lw=2,
        label="Normal fit",
    )
    ax.set_title(f"Residuals Distribution ({best_name})")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.legend()

    # ── (0,1) Residuals over time ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(residuals.index, residuals.values, color="#2c7bb6", lw=1, alpha=0.8)
    ax.axhline(0, color="red", linestyle="--", lw=1.2)
    ax.set_title(f"Residuals Over Time ({best_name})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")

    # ── (0,2) MAE per CV fold — all models ────────────────────────────────
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

    # ── (1,0) Model comparison bar chart ──────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    short_names = ["ARIMA", "SARIMAX", "ES"]
    mean_maes = [all_results.get(k, {}).get("mean_mae", np.nan) for k in model_keys]
    std_maes = [all_results.get(k, {}).get("std_mae", np.nan) for k in model_keys]
    bars = ax.bar(
        short_names,
        mean_maes,
        yerr=std_maes,
        capsize=6,
        color=palette,
        alpha=0.82,
        edgecolor="black",
        linewidth=0.8,
    )
    # Gold border for winner
    winner_idx = model_keys.index(best_name)
    bars[winner_idx].set_edgecolor("gold")
    bars[winner_idx].set_linewidth(3)
    ax.set_title("Mean MAE ± Std (best = gold border)")
    ax.set_ylabel("MAE")

    # ── (1,1) Actual vs Predicted scatter (last 100 in-sample) ────────────
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
        ax.text(0.5, 0.5, f"Could not plot:\n{exc}", ha="center", va="center", transform=ax.transAxes)

    # ── (1,2) Cumulative mean MAE across folds (best model) ───────────────
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

    fig.suptitle(
        "IDR/RON Exchange Rate – Model Diagnostic Plots",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    out = output_dir / "diagnostics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic plots saved → {out}")


# --------------------------------------------------------------------------- #
# Public entry point                                                           #
# --------------------------------------------------------------------------- #

def generate_all_plots(
    series: pd.Series,
    fitted_model,
    all_results: dict,
    output_dir: str | Path = ".",
) -> None:
    """Generate the interactive Plotly chart and all matplotlib diagnostic plots."""
    output_dir = Path(output_dir)
    plot_interactive_forecast(series, all_results, output_html=str(output_dir / "forecast.html"))
    _plot_diagnostics(series, fitted_model, all_results, output_dir)
