"""
Time-series model training, hyperparameter tuning, and one-step-ahead
walk-forward cross-validation for IDR/RON exchange rate data.
"""

import warnings
import itertools
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Search spaces                                                                #
# --------------------------------------------------------------------------- #

ARIMA_P = range(0, 4)
ARIMA_Q = range(0, 4)
# d is determined per-run via ADF test in tune_arima, not searched over.

# SARIMAX removed: on daily FX data it collapses to ARIMA; use ARIMA + ES instead.

ES_TREND = [None, "add", "mul"]
ES_DAMPED = [True, False]
# ES seasonal options removed: daily FX has no meaningful weekly/monthly seasonality.

N_CV_SPLITS = 5
TOP_AIC = 10
TOP_CV = 5

# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #

_DATE_HINTS = ["data", "date", "zi", "day"]
_RATE_HINTS = ["curs", "rate", "valoare", "idr", "value"]


def load_series(csv_path: str | Path) -> pd.Series:
    """
    Load exchange-rate CSV produced by main.py.
    Returns a business-day-frequency pd.Series indexed by date.
    """
    df = pd.read_csv(csv_path)

    # When the CSV has N header columns but N+1 data columns, pandas silently
    # absorbs the first data column (usually the date) into the index.
    # Detect this and bring it back as a regular column named "date".
    if not isinstance(df.index, pd.RangeIndex):
        df.index.name = "date"
        df = df.reset_index()

    df.columns = df.columns.str.strip().str.lower()

    date_col = next(
        (c for c in df.columns if any(h in c for h in _DATE_HINTS)),
        df.columns[0],
    )
    rate_col = next(
        (c for c in df.columns if any(h in c for h in _RATE_HINTS)),
        df.columns[1],
    )

    # Handle multiple date formats: DD.MM.YYYY, DD/MM/YYYY, YYYY-MM-DD, etc.
    raw_dates = df[date_col].astype(str).str.strip()
    parsed = pd.to_datetime(raw_dates, format="%d.%m.%Y", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(raw_dates, dayfirst=True, errors="coerce")
    df[date_col] = parsed

    df[rate_col] = (
        df[rate_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df = (
        df.dropna(subset=[date_col, rate_col])
        .sort_values(date_col)
        .set_index(date_col)
    )

    series = df[rate_col].rename("IDR_RON")
    # Keep the last entry per date in case the source has duplicate rows
    series = series.groupby(series.index).last()
    # Express as "100 IDR → RON" (more readable than the raw per-unit value)
    series = series * 100
    series.name = "100IDR_RON"
    # Trade-off: ffill for missing business days (public holidays) keeps a
    # regular B-frequency index that statsmodels requires, but it injects
    # artificial zero-return days which slightly deflate volatility estimates
    # and inflate autocorrelation.  The alternative (drop missing rather than
    # ffill) requires an irregular-index model or explicit gap handling.
    series = series.asfreq("B").ffill()
    return series


# --------------------------------------------------------------------------- #
# Walk-forward one-step-ahead cross-validation                                #
# --------------------------------------------------------------------------- #

def walk_forward_cv(
    series: pd.Series,
    fit_fn: Callable[[list[float]], float],
    n_splits: int = N_CV_SPLITS,
) -> tuple[float, float, list[float]]:
    """
    Walk-forward one-step-ahead CV using expanding training windows.

    fit_fn(history) must return a single float prediction for the next step.
    Returns (mean_mae, std_mae, fold_maes).
    """
    test_size = max(10, len(series) // (n_splits + 1))
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    fold_maes: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(series), 1):
        train = list(series.iloc[train_idx])
        test = list(series.iloc[test_idx])
        preds: list[float] = []
        history = train.copy()

        for actual in test:
            try:
                preds.append(fit_fn(history))
            except Exception:
                preds.append(float("nan"))
            history.append(actual)

        pairs = [(p, a) for p, a in zip(preds, test) if not np.isnan(p)]
        if pairs:
            p_arr, a_arr = zip(*pairs)
            mae = mean_absolute_error(a_arr, p_arr)
            fold_maes.append(mae)

    if not fold_maes:
        return float("inf"), float("inf"), []
    return float(np.mean(fold_maes)), float(np.std(fold_maes)), fold_maes


# --------------------------------------------------------------------------- #
# ARIMA                                                                        #
# --------------------------------------------------------------------------- #

def _select_d(series: pd.Series, max_d: int = 2) -> tuple[int, str]:
    """Determine ARIMA differencing order via sequential ADF test.

    Tests levels; if non-stationary (p ≥ 0.05) tests first differences; etc.
    Returns (d, log_string) so tune_arima can print the rationale.
    """
    candidate = series.dropna().copy()
    for d in range(max_d + 1):
        stat, p, *_ = adfuller(candidate, autolag="AIC")
        label = "levels" if d == 0 else f"diff({d})"
        if p < 0.05:
            return d, f"ADF stat={stat:.3f} p={p:.4f} on {label} → stationary"
        if d == max_d:
            return d, f"ADF stat={stat:.3f} p={p:.4f} on {label} → still non-stationary; using d={d}"
        candidate = candidate.diff().dropna()
    return 0, "fallback d=0"


def _arima_walk_forward_cv(
    series: pd.Series,
    order: tuple,
    n_splits: int = N_CV_SPLITS,
) -> tuple[float, float, list[float]]:
    """Walk-forward CV for ARIMA: one full fit per fold, then append(refit=False).

    Extending via Kalman state update is substantially faster than re-fitting
    from scratch at each test step, with negligible accuracy change.
    """
    test_size = max(10, len(series) // (n_splits + 1))
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    fold_maes: list[float] = []

    for train_idx, test_idx in tscv.split(series):
        train = series.iloc[train_idx]
        test_vals = series.iloc[test_idx]

        try:
            fitted = ARIMA(train, order=order).fit()
        except Exception:
            continue

        preds: list[float] = []
        actuals: list[float] = []

        for i in range(len(test_vals)):
            if i > 0:
                try:
                    fitted = fitted.append(test_vals.iloc[[i - 1]], refit=False)
                except Exception:
                    # Fallback: full refit if state extension fails.
                    history = pd.concat([train, test_vals.iloc[:i]])
                    try:
                        fitted = ARIMA(history, order=order).fit()
                    except Exception:
                        break
            try:
                fc = fitted.get_forecast(steps=1)
                preds.append(float(fc.predicted_mean.iloc[0]))
                actuals.append(float(test_vals.iloc[i]))
            except Exception:
                pass

        if preds and actuals:
            fold_maes.append(mean_absolute_error(actuals, preds))

    if not fold_maes:
        return float("inf"), float("inf"), []
    return float(np.mean(fold_maes)), float(np.std(fold_maes)), fold_maes


def tune_arima(series: pd.Series) -> dict:
    """Fix d via ADF test, AIC pre-filter over (p, q), then walk-forward CV."""
    d, d_reason = _select_d(series)
    print(f"[ARIMA] Integration order: d={d}  ({d_reason})", flush=True)

    grid = list(itertools.product(ARIMA_P, [d], ARIMA_Q))
    print(f"[ARIMA] AIC search — {len(grid)} candidates at fixed d={d} …", flush=True)
    aic_rows: list[dict] = []

    for p, _, q in grid:
        try:
            aic = ARIMA(series, order=(p, d, q)).fit().aic
            aic_rows.append({"order": (p, d, q), "aic": aic})
        except Exception:
            pass

    aic_rows.sort(key=lambda x: x["aic"])
    candidates = aic_rows[:TOP_CV]
    print(
        f"[ARIMA] {len(aic_rows)}/{len(grid)} converged — "
        f"CV on top {len(candidates)} by AIC …",
        flush=True,
    )

    best: dict = {"mean_mae": float("inf")}
    n = len(candidates)
    for i, cand in enumerate(candidates, 1):
        mean_mae, std_mae, fold_maes = _arima_walk_forward_cv(series, cand["order"])
        print(
            f"  [{i}/{n}] order={cand['order']}  AIC={cand['aic']:.2f}  "
            f"MAE={mean_mae:.6f} ± {std_mae:.6f}",
            flush=True,
        )
        if mean_mae < best["mean_mae"] or (
            mean_mae == best["mean_mae"] and std_mae < best.get("std_mae", float("inf"))
        ):
            best = {
                "model": "ARIMA",
                "order": cand["order"],
                "aic": cand["aic"],
                "mean_mae": mean_mae,
                "std_mae": std_mae,
                "fold_maes": fold_maes,
            }

    print(
        f"[ARIMA] Best → order={best['order']}  "
        f"MAE={best['mean_mae']:.6f} ± {best['std_mae']:.6f}",
        flush=True,
    )
    return best


# --------------------------------------------------------------------------- #
# Exponential Smoothing (non-seasonal)                                         #
# --------------------------------------------------------------------------- #

def _es_fit_fn(
    trend: str | None,
    damped_trend: bool,
) -> Callable[[list[float]], float]:
    def fit(history: list[float]) -> float:
        model = ExponentialSmoothing(
            history,
            trend=trend,
            seasonal=None,
            damped_trend=damped_trend if trend else False,
            initialization_method="estimated",
        )
        return float(model.fit(optimized=True).forecast(steps=1)[0])
    return fit


def tune_exp_smoothing(series: pd.Series) -> dict:
    """Walk-forward CV over non-seasonal Exponential Smoothing combinations."""
    valid = [
        (trend, damped)
        for trend, damped in itertools.product(ES_TREND, ES_DAMPED)
        if not (damped and not trend)
    ]
    print(f"[ES] Walk-forward CV on {len(valid)} non-seasonal combos …", flush=True)

    best: dict = {"mean_mae": float("inf")}
    for i, (trend, damped) in enumerate(valid, 1):
        mean_mae, std_mae, fold_maes = walk_forward_cv(series, _es_fit_fn(trend, damped))
        print(
            f"  [{i}/{len(valid)}] trend={trend} damped={damped}  "
            f"MAE={mean_mae:.6f} ± {std_mae:.6f}",
            flush=True,
        )
        if mean_mae < best["mean_mae"] or (
            mean_mae == best["mean_mae"] and std_mae < best.get("std_mae", float("inf"))
        ):
            best = {
                "model": "ExponentialSmoothing",
                "trend": trend,
                "seasonal": None,
                "damped_trend": damped,
                "seasonal_periods": None,
                "mean_mae": mean_mae,
                "std_mae": std_mae,
                "fold_maes": fold_maes,
            }

    print(
        f"[ES] Best → trend={best['trend']} damped={best['damped_trend']}  "
        f"MAE={best['mean_mae']:.6f} ± {best['std_mae']:.6f}",
        flush=True,
    )
    return best


# --------------------------------------------------------------------------- #
# Naive baselines                                                              #
# --------------------------------------------------------------------------- #

def tune_naive(series: pd.Series) -> dict:
    """Walk-forward CV using the last-value (naive) forecast."""
    print("[Naive] Walk-forward CV …", flush=True)
    mean_mae, std_mae, fold_maes = walk_forward_cv(series, lambda h: h[-1])
    print(f"[Naive] MAE = {mean_mae:.6f} ± {std_mae:.6f}", flush=True)
    return {
        "model": "Naive",
        "mean_mae": mean_mae,
        "std_mae": std_mae,
        "fold_maes": fold_maes,
    }


def tune_naive_drift(series: pd.Series) -> dict:
    """Walk-forward CV using naive-with-drift (last value + mean historical change)."""
    def _drift_fn(h: list[float]) -> float:
        return h[-1] + float(np.mean(np.diff(h))) if len(h) >= 2 else h[-1]

    print("[Naive+Drift] Walk-forward CV …", flush=True)
    mean_mae, std_mae, fold_maes = walk_forward_cv(series, _drift_fn)
    print(f"[Naive+Drift] MAE = {mean_mae:.6f} ± {std_mae:.6f}", flush=True)
    return {
        "model": "NaiveDrift",
        "mean_mae": mean_mae,
        "std_mae": std_mae,
        "fold_maes": fold_maes,
    }


# --------------------------------------------------------------------------- #
# Model selection and final fit                                                #
# --------------------------------------------------------------------------- #

def compare_models(*models: dict) -> dict:
    """Return the model with the lowest mean_mae (std_mae used as tiebreaker)."""
    return min(models, key=lambda x: (x["mean_mae"], x["std_mae"]))


class _NaiveFitted:
    """Minimal result wrapper for naive / naive-with-drift baselines.

    Exposes .fittedvalues, .resid, and .forecast() so that visualize.py
    can treat it identically to a statsmodels result object.
    """

    def __init__(self, series: pd.Series, drift: float = 0.0) -> None:
        self._last = float(series.iloc[-1])
        self._drift = drift
        fitted_arr = series.values[:-1] + drift
        self.fittedvalues = pd.Series(fitted_arr, index=series.index[1:])
        self.resid = pd.Series(series.values[1:] - fitted_arr, index=series.index[1:])

    def forecast(self, steps: int) -> "pd.Series":
        return pd.Series(
            [self._last + self._drift * (k + 1) for k in range(steps)]
        )


def fit_final_model(series: pd.Series, best_params: dict):
    """Fit the winning model on the provided series and return the result object."""
    name = best_params["model"]
    if name == "ARIMA":
        return ARIMA(series, order=best_params["order"]).fit()
    if name == "SARIMAX":
        return SARIMAX(
            series,
            order=best_params["order"],
            seasonal_order=best_params["seasonal_order"],
        ).fit(disp=False)
    if name == "Naive":
        return _NaiveFitted(series, drift=0.0)
    if name == "NaiveDrift":
        drift = float(np.mean(np.diff(series.values)))
        return _NaiveFitted(series, drift=drift)
    # ExponentialSmoothing (non-seasonal)
    return ExponentialSmoothing(
        series,
        trend=best_params["trend"],
        seasonal=None,
        damped_trend=best_params["damped_trend"] if best_params["trend"] else False,
        initialization_method="estimated",
    ).fit(optimized=True)
