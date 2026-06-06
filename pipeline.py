"""
Automatic retraining pipeline.

Usage:
    python pipeline.py                          # uses idr_exchange_rates.csv
    python pipeline.py --csv path/to/file.csv   # custom CSV path
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

from models import (
    load_series,
    tune_arima,
    tune_sarimax,
    tune_exp_smoothing,
    compare_models,
    fit_final_model,
)
from visualize import generate_all_plots

OUTPUT_DIR = Path("trained_models")
DEFAULT_CSV = "idr_exchange_rates.csv"


def retrain_pipeline(csv_path: str = DEFAULT_CSV) -> dict:
    """
    Full retraining pipeline:
      1. Load latest CSV data
      2. Hyperparameter tune ARIMA, SARIMAX, ExponentialSmoothing
      3. Select best model (lowest mean CV MAE, std as tiebreaker)
      4. Fit best model on entire series
      5. Persist model + params
      6. Regenerate all visualisations
    Returns a dict with all model results.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    _banner(f"IDR Exchange Rate – Retraining Pipeline  [{timestamp}]")

    # 1. Load data
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print("        Run  python main.py  first to fetch exchange-rate data.")
        sys.exit(1)

    series = load_series(csv_path)
    print(
        f"Loaded {len(series)} observations "
        f"[{series.index[0].date()} → {series.index[-1].date()}]\n"
    )

    if len(series) < 50:
        print("[ERROR] Too few observations for reliable model training (need ≥ 50).")
        sys.exit(1)

    # 2. Tune all models
    arima_res = tune_arima(series)
    print()
    sarimax_res = tune_sarimax(series)
    print()
    es_res = tune_exp_smoothing(series)
    print()

    # 3. Select best
    best = compare_models(arima_res, sarimax_res, es_res)
    _banner(
        f"Best model: {best['model']}  "
        f"MAE = {best['mean_mae']:.6f}  ±  {best['std_mae']:.6f}"
    )

    # 4. Fit on full series
    print("Fitting best model on full series …")
    fitted = fit_final_model(series, best)
    print("Done.\n")

    # 5a. Persist fitted model (pickle)
    model_path = OUTPUT_DIR / f"best_model_{timestamp}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump({"params": best, "fitted": fitted}, fh)
    print(f"Model pickle  → {model_path}")

    # 5b. Persist params as JSON (human-readable, no pickle dependency)
    serialisable = {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in best.items()
        if k != "fold_maes"
    }
    serialisable["fold_maes"] = best.get("fold_maes", [])
    serialisable["timestamp"] = timestamp
    serialisable["n_observations"] = len(series)
    params_path = OUTPUT_DIR / "best_params.json"
    with open(params_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)
    print(f"Best params   → {params_path}")

    # 6. Visualisations
    all_results = {
        "ARIMA": arima_res,
        "SARIMAX": sarimax_res,
        "ExponentialSmoothing": es_res,
        "best": best,
    }
    print()
    generate_all_plots(series, fitted, all_results)

    print("\nPipeline complete.")
    return all_results


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _banner(text: str, width: int = 62) -> None:
    print(f"\n{'=' * width}")
    print(f" {text}")
    print(f"{'=' * width}\n")


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
