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

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Search spaces                                                                #
# --------------------------------------------------------------------------- #

ARIMA_P = range(0, 4)
ARIMA_D = range(0, 3)
ARIMA_Q = range(0, 4)

SARIMAX_P = range(0, 3)
SARIMAX_D = range(0, 2)
SARIMAX_Q = range(0, 3)
SARIMAX_SP = range(0, 2)
SARIMAX_SD = range(0, 2)
SARIMAX_SQ = range(0, 2)
SARIMAX_S = [5, 21]

ES_TREND = [None, "add", "mul"]
ES_SEASONAL = [None, "add", "mul"]
ES_DAMPED = [True, False]
ES_PERIODS = [5, 21]

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

def _arima_fit_fn(order: tuple) -> Callable[[list[float]], float]:
    def fit(history: list[float]) -> float:
        return float(ARIMA(history, order=order).fit().forecast(steps=1)[0])
    return fit


def tune_arima(series: pd.Series) -> dict:
    """AIC pre-filter then walk-forward CV for ARIMA(p,d,q)."""
    grid = list(itertools.product(ARIMA_P, ARIMA_D, ARIMA_Q))
    print(f"[ARIMA] AIC search — {len(grid)} candidates …", flush=True)
    aic_rows: list[dict] = []

    for p, d, q in grid:
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
        mean_mae, std_mae, fold_maes = walk_forward_cv(series, _arima_fit_fn(cand["order"]))
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
# SARIMAX                                                                      #
# --------------------------------------------------------------------------- #

def _sarimax_fit_fn(order: tuple, seasonal_order: tuple) -> Callable[[list[float]], float]:
    def fit(history: list[float]) -> float:
        res = SARIMAX(history, order=order, seasonal_order=seasonal_order).fit(disp=False)
        return float(res.forecast(steps=1)[0])
    return fit


def tune_sarimax(series: pd.Series) -> dict:
    """AIC pre-filter then walk-forward CV for SARIMAX."""
    grid = list(itertools.product(
        SARIMAX_P, SARIMAX_D, SARIMAX_Q,
        SARIMAX_SP, SARIMAX_SD, SARIMAX_SQ,
        SARIMAX_S,
    ))
    print(f"[SARIMAX] AIC search — {len(grid)} candidates …", flush=True)
    aic_rows: list[dict] = []

    for p, d, q, P, D, Q, s in grid:
        try:
            aic = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=False).aic
            aic_rows.append({"order": (p, d, q), "seasonal_order": (P, D, Q, s), "aic": aic})
        except Exception:
            pass

    aic_rows.sort(key=lambda x: x["aic"])
    candidates = aic_rows[:TOP_CV]
    print(
        f"[SARIMAX] {len(aic_rows)}/{len(grid)} converged — "
        f"CV on top {len(candidates)} by AIC …",
        flush=True,
    )

    best: dict = {"mean_mae": float("inf")}
    n = len(candidates)
    for i, cand in enumerate(candidates, 1):
        mean_mae, std_mae, fold_maes = walk_forward_cv(
            series, _sarimax_fit_fn(cand["order"], cand["seasonal_order"])
        )
        print(
            f"  [{i}/{n}] order={cand['order']} seasonal={cand['seasonal_order']}  "
            f"AIC={cand['aic']:.2f}  MAE={mean_mae:.6f} ± {std_mae:.6f}",
            flush=True,
        )
        if mean_mae < best["mean_mae"] or (
            mean_mae == best["mean_mae"] and std_mae < best.get("std_mae", float("inf"))
        ):
            best = {
                "model": "SARIMAX",
                "order": cand["order"],
                "seasonal_order": cand["seasonal_order"],
                "aic": cand["aic"],
                "mean_mae": mean_mae,
                "std_mae": std_mae,
                "fold_maes": fold_maes,
            }

    print(
        f"[SARIMAX] Best → order={best['order']} seasonal={best['seasonal_order']}  "
        f"MAE={best['mean_mae']:.6f} ± {best['std_mae']:.6f}",
        flush=True,
    )
    return best


# --------------------------------------------------------------------------- #
# Exponential Smoothing                                                        #
# --------------------------------------------------------------------------- #

def _es_fit_fn(
    trend: str | None,
    seasonal: str | None,
    damped_trend: bool,
    seasonal_periods: int,
) -> Callable[[list[float]], float]:
    def fit(history: list[float]) -> float:
        model = ExponentialSmoothing(
            history,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods if seasonal else None,
            damped_trend=damped_trend if trend else False,
            initialization_method="estimated",
        )
        return float(model.fit(optimized=True).forecast(steps=1)[0])
    return fit


def tune_exp_smoothing(series: pd.Series) -> dict:
    """Walk-forward CV over all valid Exponential Smoothing combinations."""
    valid = [
        (t, s, d, sp)
        for t, s, d, sp in itertools.product(ES_TREND, ES_SEASONAL, ES_DAMPED, ES_PERIODS)
        if not (d and not t) and not (s and sp < 2)
    ]
    print(f"[ES] Walk-forward CV on {len(valid)} valid combos …", flush=True)

    best: dict = {"mean_mae": float("inf")}
    for i, (trend, seasonal, damped, sp) in enumerate(valid, 1):
        mean_mae, std_mae, fold_maes = walk_forward_cv(
            series, _es_fit_fn(trend, seasonal, damped, sp)
        )
        print(
            f"  [{i}/{len(valid)}] trend={trend} seasonal={seasonal} "
            f"damped={damped} sp={sp}  MAE={mean_mae:.6f} ± {std_mae:.6f}",
            flush=True,
        )

        if mean_mae < best["mean_mae"] or (
            mean_mae == best["mean_mae"] and std_mae < best.get("std_mae", float("inf"))
        ):
            best = {
                "model": "ExponentialSmoothing",
                "trend": trend,
                "seasonal": seasonal,
                "damped_trend": damped,
                "seasonal_periods": sp,
                "mean_mae": mean_mae,
                "std_mae": std_mae,
                "fold_maes": fold_maes,
            }

    print(
        f"[ES] Best → trend={best['trend']} seasonal={best['seasonal']}  "
        f"MAE={best['mean_mae']:.6f} ± {best['std_mae']:.6f}",
        flush=True,
    )
    return best


# --------------------------------------------------------------------------- #
# Model selection and final fit                                                #
# --------------------------------------------------------------------------- #

def compare_models(arima: dict, sarimax: dict, es: dict) -> dict:
    """Return the model with the lowest mean_mae (std_mae used as tiebreaker)."""
    return min([arima, sarimax, es], key=lambda x: (x["mean_mae"], x["std_mae"]))


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
    # ExponentialSmoothing
    return ExponentialSmoothing(
        series,
        trend=best_params["trend"],
        seasonal=best_params["seasonal"],
        seasonal_periods=best_params["seasonal_periods"] if best_params["seasonal"] else None,
        damped_trend=best_params["damped_trend"] if best_params["trend"] else False,
        initialization_method="estimated",
    ).fit(optimized=True)
