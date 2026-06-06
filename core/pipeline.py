"""
Automatic retraining pipeline.

Usage (run from the project root):
    python -m core.pipeline                          # uses default CSV
    python -m core.pipeline --csv path/to/file.csv   # custom CSV path
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .models import (
    load_series,
    tune_arima,
    tune_sarimax,
    tune_exp_smoothing,
    compare_models,
    fit_final_model,
)
from .visualize import generate_all_plots, make_per_model_diagnostic_figure

OUTPUT_DIR = Path("resources") / "models"
LOG_DIR = Path("logs")
DEFAULT_CSV = str(Path("resources") / "data" / "idr_exchange_rates.csv")

MODEL_NAMES = ["ARIMA", "SARIMAX", "ExponentialSmoothing"]
_PKL_PREFIX = {
    "ARIMA": "arima",
    "SARIMAX": "sarimax",
    "ExponentialSmoothing": "exponentialsmoothing",
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _serialise(result: dict) -> dict:
    """Convert tuples → lists so the dict can be saved as JSON."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in result.items()}


def _banner(text: str, width: int = 62) -> None:
    print(f"\n{'=' * width}")
    print(f" {text}")
    print(f"{'=' * width}\n")


def _save_all_results(
    arima: dict, sarimax: dict, es: dict, best: dict,
    timestamp: str, n_obs: int,
) -> None:
    path = OUTPUT_DIR / "all_results.json"
    payload = {
        "ARIMA": _serialise(arima),
        "SARIMAX": _serialise(sarimax),
        "ExponentialSmoothing": _serialise(es),
        "best": _serialise(best),
        "timestamp": timestamp,
        "n_observations": n_obs,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"All results   → {path}")


def _save_model_pkl(model_name: str, result: dict, fitted, timestamp: str) -> Path:
    prefix = _PKL_PREFIX[model_name]
    path = OUTPUT_DIR / f"{prefix}_{timestamp}.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"params": result, "fitted": fitted}, fh)
    print(f"{model_name:25s} → {path}")
    return path


def _write_logs(
    timestamp: str,
    all_results: dict,
    fitted_map: dict,
    series: "pd.Series",
) -> None:
    """Write per-run diagnostics to logs/<timestamp>/.

    Outputs
    -------
    results_summary.csv              MAE, AIC, best-params for each model
    cv_folds.csv                     fold-by-fold MAE for each model
    <model>_diagnostics.png          2×2 diagnostic figure per model
    diagnostics.png                  combined 2×3 all-models figure
    """
    log_dir = LOG_DIR / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    best_name = all_results.get("best", {}).get("model", "")

    # ── results_summary.csv ───────────────────────────────────────────────────
    rows: list[dict] = []
    for name in MODEL_NAMES:
        res = all_results.get(name)
        if res is None:
            continue
        row: dict = {
            "model": name,
            "is_best": name == best_name,
            "mean_mae": res.get("mean_mae"),
            "std_mae": res.get("std_mae"),
            "aic": res.get("aic", ""),
        }
        if "order" in res:
            row["order"] = str(res["order"])
        if "seasonal_order" in res:
            row["seasonal_order"] = str(res["seasonal_order"])
        if "trend" in res:
            row["trend"] = res.get("trend")
            row["seasonal"] = res.get("seasonal")
            row["damped_trend"] = res.get("damped_trend")
            row["seasonal_periods"] = res.get("seasonal_periods")
        rows.append(row)

    pd.DataFrame(rows).to_csv(log_dir / "results_summary.csv", index=False)
    print(f"  · results_summary.csv")

    # ── cv_folds.csv ──────────────────────────────────────────────────────────
    fold_rows: list[dict] = []
    for name in MODEL_NAMES:
        res = all_results.get(name) or {}
        for i, mae in enumerate(res.get("fold_maes", []), 1):
            fold_rows.append({"model": name, "fold": i, "mae": mae})

    pd.DataFrame(fold_rows).to_csv(log_dir / "cv_folds.csv", index=False)
    print(f"  · cv_folds.csv")

    # ── per-model 2×2 diagnostic PNGs ─────────────────────────────────────────
    for name in MODEL_NAMES:
        fitted = fitted_map.get(name)
        res = all_results.get(name)
        if fitted is None or res is None:
            continue
        prefix = _PKL_PREFIX.get(name, name.lower())
        fig = make_per_model_diagnostic_figure(series, fitted, res)
        fig.savefig(log_dir / f"{prefix}_diagnostics.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  · {prefix}_diagnostics.png")

    # ── combined 2×3 all-models PNG ───────────────────────────────────────────
    best_fitted = fitted_map.get(best_name)
    if best_fitted is not None:
        generate_all_plots(series, best_fitted, all_results, log_dir)

    print(f"Training logs → {log_dir}/")


# --------------------------------------------------------------------------- #
# Full pipeline                                                                #
# --------------------------------------------------------------------------- #

def retrain_pipeline(csv_path: str = DEFAULT_CSV) -> dict:
    """
    Full retraining pipeline:
      1. Load latest CSV
      2. Hyperparameter-tune ARIMA, SARIMAX, ExponentialSmoothing
      3. Select best model (lowest mean CV MAE; std as tiebreaker)
      4. Fit all three models on full series and persist each
      5. Save all_results.json + best_params.json
      6. Regenerate diagnostics PNG
      7. Write per-run debug logs to logs/<timestamp>/
    Returns the all_results dict.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    _banner(f"IDR Exchange Rate – Retraining Pipeline  [{timestamp}]")

    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print("        Run  python -m core.scraper  first to fetch exchange-rate data.")
        sys.exit(1)

    series = load_series(csv_path)
    print(
        f"Loaded {len(series)} observations "
        f"[{series.index[0].date()} → {series.index[-1].date()}]\n"
    )

    if len(series) < 50:
        print("[ERROR] Too few observations for reliable model training (need ≥ 50).")
        sys.exit(1)

    # Tune
    arima_res = tune_arima(series)
    print()
    sarimax_res = tune_sarimax(series)
    print()
    es_res = tune_exp_smoothing(series)
    print()

    best = compare_models(arima_res, sarimax_res, es_res)
    _banner(
        f"Best model: {best['model']}  "
        f"MAE = {best['mean_mae']:.6f}  ±  {best['std_mae']:.6f}"
    )

    # Fit all three on full series and save individual pickles
    print("Fitting all models on full series and saving …")
    fitted_map: dict[str, object] = {}
    for name, res in [("ARIMA", arima_res), ("SARIMAX", sarimax_res), ("ExponentialSmoothing", es_res)]:
        fitted = fit_final_model(series, res)
        fitted_map[name] = fitted
        _save_model_pkl(name, res, fitted, timestamp)

    # Also save the best model separately (for backward-compat lookups)
    best_path = OUTPUT_DIR / f"best_model_{timestamp}.pkl"
    with open(best_path, "wb") as fh:
        pickle.dump({"params": best, "fitted": fitted_map[best["model"]]}, fh)
    print(f"Best model        → {best_path}")

    # Persist metadata
    _save_all_results(arima_res, sarimax_res, es_res, best, timestamp, len(series))

    serialisable_best = _serialise(best)
    serialisable_best["timestamp"] = timestamp
    serialisable_best["n_observations"] = len(series)
    best_params_path = OUTPUT_DIR / "best_params.json"
    with open(best_params_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable_best, fh, indent=2)
    print(f"Best params       → {best_params_path}")

    # Diagnostics PNG (combined, to resources/models/)
    all_results = {
        "ARIMA": arima_res,
        "SARIMAX": sarimax_res,
        "ExponentialSmoothing": es_res,
        "best": best,
    }
    print()
    generate_all_plots(series, fitted_map[best["model"]], all_results, OUTPUT_DIR)

    # Per-run debug logs
    print("\nWriting training logs …")
    _write_logs(timestamp, all_results, fitted_map, series)

    print("\nPipeline complete.")
    return all_results


# --------------------------------------------------------------------------- #
# Single-model retrain (called by Streamlit app)                              #
# --------------------------------------------------------------------------- #

def retrain_single_model(model_name: str, csv_path: str = DEFAULT_CSV) -> dict:
    """
    Retune one model, fit on the full series, update all_results.json.
    Returns the updated all_results dict.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {MODEL_NAMES}.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    series = load_series(csv_path)
    print(f"\nRetrain {model_name}: {len(series)} observations\n")

    if model_name == "ARIMA":
        new_result = tune_arima(series)
    elif model_name == "SARIMAX":
        new_result = tune_sarimax(series)
    else:
        new_result = tune_exp_smoothing(series)

    fitted = fit_final_model(series, new_result)
    _save_model_pkl(model_name, new_result, fitted, timestamp)

    # Load existing all_results (or start fresh)
    all_results_path = OUTPUT_DIR / "all_results.json"
    if all_results_path.exists():
        with open(all_results_path, encoding="utf-8") as fh:
            all_results_json = json.load(fh)
    else:
        all_results_json = {}

    all_results_json[model_name] = _serialise(new_result)
    all_results_json["timestamp"] = timestamp

    # Re-determine best among available models
    available = [all_results_json[n] for n in MODEL_NAMES if n in all_results_json]
    if available:
        best_json = min(available, key=lambda x: (x["mean_mae"], x["std_mae"]))
        all_results_json["best"] = best_json

        # If this model is now the best, update the best_model pickle too
        if best_json["model"] == model_name:
            best_path = OUTPUT_DIR / f"best_model_{timestamp}.pkl"
            with open(best_path, "wb") as fh:
                pickle.dump({"params": new_result, "fitted": fitted}, fh)
            print(f"New best model    → {best_path}")

    with open(all_results_path, "w", encoding="utf-8") as fh:
        json.dump(all_results_json, fh, indent=2)
    print(f"all_results.json updated → {all_results_path}")

    return all_results_json


# --------------------------------------------------------------------------- #
# CLI entry point                                                              #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IDR/RON exchange rate – automatic retraining pipeline"
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=f"Path to the exchange-rate CSV (default: {DEFAULT_CSV})",
    )
    args = parser.parse_args()
    retrain_pipeline(args.csv)


if __name__ == "__main__":
    main()
