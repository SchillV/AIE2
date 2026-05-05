"""
optimize_models.py
==================
Optimizare hiperparametri – Tema AIE2 – COVID-19 Mortality Prediction

DIAGNOSTIC REZULTATE INIȚIALE:
-------------------------------
La prima rulare metricile arătau astfel:
  • ROC-AUC  ≈ 0.92  (excelent)
  • Recall   ≈ 0.85  (bun)
  • Precision ≈ 0.33 (slab)
  • F1-score ≈ 0.45  (mediocru)

Aparent pare overfitting, DAR ROC-AUC mare (0.92) dovedește contrariul — modelul
rankuiește corect cazurile. Problema reală este pragul de decizie, deplasat
din cauza downsampling-ului:

  • Train (downsample 1:3): ~25% cazuri pozitive
  • Test (distribuție naturală): ~8% cazuri pozitive
  • La prag default 0.5, modelul etichetează ~19% ca pozitive pe test
    (reflectă distribuția de antrenare) → multe false pozitive → Precision mică

SOLUȚIE:
--------
1. Split în 3 seturi: train_ds (downsample) / val (natural) / test (natural)
2. Tunare prag pe VAL (maximizare F1) — val are distribuția naturală, deci
   pragul găsit se transferă corect pe test
3. Raportare metrici la AMBELE praguri (default 0.5 + tunat)
4. Adăugare PR-AUC (average_precision) — mai informativ pentru date dezechilibrate
5. Grafice: bar chart comparativ + PR curves cu pragurile marcate

Modelele cached din rularea anterioară rămân valabile (train_ds identic).

Metodologie optimizare (nemodificată):
  • Logistic Regression  : GridSearchCV – 6 valori C, cv=5
  • Random Forest        : RandomizedSearchCV – 20 iterații, cv=3
  • XGBoost              : Optuna 50 trials + MedianPruner
  • Ensemble             : VotingClassifier soft (ponderi din val AUC)

Runtime: ~2–5 min dacă modelele sunt cached, ~15–25 min de la zero.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pathlib as pth
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)
from scipy.stats import randint

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from xgboost import XGBClassifier

# ──────────────────────────────────────────────────────────────
# Constante
# ──────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
MODELS_DIR     = pth.Path("optimized_models")
MODELS_DIR.mkdir(exist_ok=True)
ENSEMBLE_CACHE = MODELS_DIR / "best_ensemble.pkl"


# ══════════════════════════════════════════════════════════════
# 1. PREPROCESARE  (identic cu notebook-ul)
# ══════════════════════════════════════════════════════════════
def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """
    Încarcă și preprocesează dataset-ul COVID-19.
    Logică identică cu Tema_1_AIE2_-_Schiller_Vlad-Radu.ipynb.
    """
    df = pd.read_csv(csv_path)

    df["DEATH"] = [0 if r == "9999-99-99" else 1 for r in df["DATE_DIED"]]
    df.drop("DATE_DIED", axis=1, inplace=True)

    categoric = [
        "USMER", "MEDICAL_UNIT", "SEX", "PATIENT_TYPE", "INTUBED", "PNEUMONIA",
        "PREGNANT", "DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION",
        "OTHER_DISEASE", "CARDIOVASCULAR", "OBESITY", "RENAL_CHRONIC",
        "TOBACCO", "CLASIFFICATION_FINAL", "ICU", "DEATH",
    ]

    for col in categoric:
        if col in df.columns:
            df[col] = df[col].replace(2, 0)

    df["PREGNANT"] = df["PREGNANT"].replace([97, 98], 2)

    for col in categoric:
        if col in df.columns:
            df[col] = df[col].replace([97, 98, 99], np.nan)

    for col in ["MEDICAL_UNIT", "CLASIFFICATION_FINAL", "PATIENT_TYPE"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    df = df.loc[:, df.isna().mean() <= 0.5]
    df.loc[(df["AGE"] <= 0) | (df["AGE"] > 122), "AGE"] = np.nan
    df = df.dropna(subset=["AGE"])

    existing = set(df.columns)
    cats = [c for c in categoric if c in existing]
    for col in cats:
        df[col] = df[col].replace(2, 0)
        df = df.dropna(subset=[col])

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# 2. SPLIT TRAIN / VAL / TEST  (modificat pentru tunare prag)
# ══════════════════════════════════════════════════════════════
def prepare_splits(df: pd.DataFrame):
    """
    Pregătește 3 seturi stratificate:
      • train_ds: 80% din date, downsampled 1:3 (~25% pozitive)
                  → folosit pentru ANTRENARE modele
      • val:      10% din date, distribuție naturală (~8% pozitive)
                  → folosit pentru TUNARE PRAG (F1 optim)
      • test:     10% din date, distribuție naturală (~8% pozitive)
                  → folosit pentru EVALUARE FINALĂ (set complet izolat)

    IMPORTANT: train_idx este identic cu versiunea anterioară a script-ului
    (același RANDOM_SEED), deci modelele cached rămân valabile.
    Doar setul original de 20% test este împărțit acum 50/50 în val + test.
    """
    X = df.drop(columns=["DEATH"])
    y = df["DEATH"]
    indices = np.arange(len(df))

    # Split principal 80/20 — identic cu rularea anterioară
    train_idx, testval_idx = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    # Sub-split 50/50 în val + test (ambele cu distribuție naturală)
    testval_y = df.iloc[testval_idx]["DEATH"]
    val_idx, test_idx = train_test_split(
        testval_idx, test_size=0.5, stratify=testval_y, random_state=RANDOM_SEED
    )

    # Downsample doar pe train (identic cu notebook-ul)
    train_idx = np.array(train_idx)
    dead_idx  = train_idx[df.iloc[train_idx]["DEATH"] == 1]
    alive_idx = train_idx[df.iloc[train_idx]["DEATH"] == 0]
    alive_ds  = resample(
        alive_idx, replace=False,
        n_samples=3 * len(dead_idx),
        random_state=RANDOM_SEED,
    )
    train_ds = np.concatenate([dead_idx, alive_ds])
    np.random.shuffle(train_ds)

    X_train = df.iloc[train_ds].drop(columns=["DEATH"]).reset_index(drop=True)
    y_train = df.iloc[train_ds]["DEATH"].reset_index(drop=True)
    X_val   = df.iloc[val_idx].drop(columns=["DEATH"]).reset_index(drop=True)
    y_val   = df.iloc[val_idx]["DEATH"].reset_index(drop=True)
    X_test  = df.iloc[test_idx].drop(columns=["DEATH"]).reset_index(drop=True)
    y_test  = df.iloc[test_idx]["DEATH"].reset_index(drop=True)

    # Standardizare pentru LR (scaler fitat pe train, aplicat pe val + test)
    scaler      = StandardScaler()
    X_train_std = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_val_std   = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns
    )
    X_test_std  = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            X_train_std, X_val_std, X_test_std, scaler)


# ══════════════════════════════════════════════════════════════
# 3. EVALUARE + TUNARE PRAG
# ══════════════════════════════════════════════════════════════
def evaluate(name: str, model, X: pd.DataFrame, y: pd.Series,
             threshold: float = 0.5) -> dict:
    """
    Calculează metricile de performanță la pragul specificat.
    Include ROC-AUC și PR-AUC (threshold-independent) + metrici la prag.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Model":     name,
        "Threshold": round(float(threshold), 4),
        "Accuracy":  round(accuracy_score(y, y_pred),                    4),
        "Precision": round(precision_score(y, y_pred, zero_division=0),   4),
        "Recall":    round(recall_score(y, y_pred,    zero_division=0),   4),
        "F1-score":  round(f1_score(y, y_pred,        zero_division=0),   4),
        "ROC-AUC":   round(roc_auc_score(y, y_prob),                      4),
        "PR-AUC":    round(average_precision_score(y, y_prob),            4),
    }


def tune_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple:
    """
    Găsește pragul care maximizează F1-score pe setul dat.
    Rezolvă shift-ul de prag cauzat de downsampling.

    :param y_true: Etichete binare adevărate
    :param y_prob: Probabilități prezise pentru clasa 1
    :return:       Tuple (prag_optim, F1_optim)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # F1 pentru fiecare prag; exclud ultimul punct (degenerat)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx  = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


# ══════════════════════════════════════════════════════════════
# 4. GRIDSEARCH – LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════
def optimize_lr(X_train_std: pd.DataFrame,
                y_train: pd.Series) -> LogisticRegression:
    """GridSearchCV pentru Logistic Regression cu grid extins."""
    model_path = MODELS_DIR / "best_lr.pkl"
    if model_path.exists():
        print("  [LR] Model existent → încarc din cache.")
        return joblib.load(model_path)

    print("  [LR] Rulare GridSearchCV (6 valori C × cv=5) …")
    param_grid = {
        "C":        [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty":  ["l2"],
        "solver":   ["lbfgs"],
        "max_iter": [1000],
    }
    gs = GridSearchCV(
        LogisticRegression(class_weight="balanced"),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train_std, y_train)
    print(f"  [LR] Best params : {gs.best_params_}")
    print(f"  [LR] CV ROC-AUC  : {gs.best_score_:.4f}")
    joblib.dump(gs.best_estimator_, model_path)
    return gs.best_estimator_


# ══════════════════════════════════════════════════════════════
# 5. RANDOMIZEDSEARCH – RANDOM FOREST
# ══════════════════════════════════════════════════════════════
def optimize_rf(X_train: pd.DataFrame,
                y_train: pd.Series) -> RandomForestClassifier:
    """RandomizedSearchCV pentru Random Forest cu spațiu extins."""
    model_path = MODELS_DIR / "best_rf.pkl"
    if model_path.exists():
        print("  [RF] Model existent → încarc din cache.")
        return joblib.load(model_path)

    print("  [RF] Rulare RandomizedSearchCV (n_iter=20, cv=3) …")
    param_dist = {
        "n_estimators":      randint(200, 500),
        "max_depth":         [None, 10, 15, 20, 30],
        "min_samples_split": randint(2, 15),
        "min_samples_leaf":  randint(1, 8),
        "max_features":      ["sqrt", "log2", 0.4],
    }
    rs = RandomizedSearchCV(
        RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0,
    )
    rs.fit(X_train, y_train)
    print(f"  [RF] Best params : {rs.best_params_}")
    print(f"  [RF] CV ROC-AUC  : {rs.best_score_:.4f}")
    joblib.dump(rs.best_estimator_, model_path)
    return rs.best_estimator_


# ══════════════════════════════════════════════════════════════
# 6. OPTUNA – XGBOOST
# ══════════════════════════════════════════════════════════════
def optimize_xgb(X_train: pd.DataFrame,
                 y_train: pd.Series) -> XGBClassifier:
    """Optuna cu 50 trials + MedianPruner + spațiu extins."""
    model_path = MODELS_DIR / "best_xgb.pkl"
    study_path = MODELS_DIR / "optuna_study_xgb.pkl"

    if model_path.exists():
        print("  [XGB] Model existent → încarc din cache.")
        return joblib.load(model_path)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators",     200, 600),
            "max_depth":        trial.suggest_int("max_depth",          3,   9),
            "learning_rate":    trial.suggest_float("learning_rate",  0.01, 0.25),
            "subsample":        trial.suggest_float("subsample",       0.5,  1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight",   1,  10),
            "gamma":            trial.suggest_float("gamma",            0.0, 0.5),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric":      "logloss",
            "random_state":     RANDOM_SEED,
            "n_jobs":           -1,
        }
        score = cross_val_score(
            XGBClassifier(**params), X_train, y_train,
            cv=3, scoring="roc_auc", n_jobs=-1,
        ).mean()
        return score

    print("  [XGB] Rulare Optuna (50 trials, MedianPruner, TPE sampler) …")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(f"  [XGB] Best params : {study.best_params}")
    print(f"  [XGB] CV ROC-AUC  : {study.best_value:.4f}")

    best_params = study.best_params.copy()
    best_params.update({
        "scale_pos_weight": scale_pos_weight,
        "eval_metric":      "logloss",
        "random_state":     RANDOM_SEED,
        "n_jobs":           -1,
    })
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    joblib.dump(final_model, model_path)
    joblib.dump(study, study_path)
    print(f"  [XGB] Study Optuna salvat → {study_path}")
    return final_model


# ══════════════════════════════════════════════════════════════
# 7. ENSEMBLE – VotingClassifier soft (ponderi din val AUC)
# ══════════════════════════════════════════════════════════════
def build_ensemble(
    best_lr: LogisticRegression,
    best_rf: RandomForestClassifier,
    best_xgb: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    auc_lr_val: float,
    auc_rf_val: float,
    auc_xgb_val: float,
) -> VotingClassifier:
    """
    VotingClassifier soft cu ponderi ∝ ROC-AUC pe VAL (nu pe test).
    Folosind val (nu test) pentru ponderi, test rămâne 100% separat.
    Ensemble-ul e mereu rebuild-uit pentru a reflecta ponderile corecte.
    """
    # Ștergem cache vechi (dacă a fost construit cu ponderi din test)
    if ENSEMBLE_CACHE.exists():
        print("  [ENS] Șterg cache vechi → rebuild cu ponderi din validare.")
        ENSEMBLE_CACHE.unlink()

    total = auc_lr_val + auc_rf_val + auc_xgb_val
    w_lr, w_rf, w_xgb = auc_lr_val / total, auc_rf_val / total, auc_xgb_val / total
    print(f"  [ENS] Ponderi (val AUC): "
          f"LR={w_lr:.3f}  RF={w_rf:.3f}  XGB={w_xgb:.3f}")

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(**best_lr.get_params())),
    ])

    ensemble = VotingClassifier(
        estimators=[
            ("lr",  lr_pipeline),
            ("rf",  best_rf),
            ("xgb", best_xgb),
        ],
        voting="soft",
        weights=[w_lr, w_rf, w_xgb],
        n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)
    joblib.dump(ensemble, ENSEMBLE_CACHE)
    return ensemble


# ══════════════════════════════════════════════════════════════
# 8. PLOT – COMPARAȚIE DEFAULT vs. TUNED
# ══════════════════════════════════════════════════════════════
def plot_comparison(results_df: pd.DataFrame,
                    out_path: str = "optimization_comparison.png") -> None:
    """
    Grafic bar cu 3 paneluri (F1, Precision, Recall): Default vs Tuned.
    """
    default_df = results_df[results_df["Model"].str.endswith("(default)")].reset_index(drop=True)
    tuned_df   = results_df[results_df["Model"].str.endswith("(tuned)")].reset_index(drop=True)

    model_names = default_df["Model"].str.replace(r" \(default\)", "", regex=True).tolist()
    x     = np.arange(len(model_names))
    width = 0.38

    metrics = ["F1-score", "Precision", "Recall"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, metric in zip(axes, metrics):
        b1 = ax.bar(
            x - width / 2, default_df[metric], width,
            label="Prag default (0.5)", color="#ff7f0e", alpha=0.85,
        )
        b2 = ax.bar(
            x + width / 2, tuned_df[metric], width,
            label="Prag tunat (val)", color="#2ca02c", alpha=0.85,
        )
        for bar in list(b1) + list(b2):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}: Default vs. Tuned")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="lower right", fontsize=9)

    plt.suptitle(
        "Impactul Tunării Pragului – COVID-19 Mortality\n"
        "Precision crește semnificativ; F1 se îmbunătățește prin balansarea P vs R",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafic salvat → {out_path}")


# ══════════════════════════════════════════════════════════════
# 9. PLOT – PRECISION-RECALL CURVES + CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════
def plot_threshold_analysis(
    models_data: list,
    y_test: pd.Series,
    out_path: str = "threshold_analysis.png",
) -> None:
    """
    Grafic cu 2 paneluri:
      • Stânga: PR curves cu pragurile marcate (○ tunat, ■ default 0.5)
      • Dreapta: Confusion matrix pentru cel mai bun model (prag tunat)

    :param models_data: Listă de tuple (nume, y_prob_test, prag_tunat)
    :param y_test:      Etichete adevărate pe test
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # ── Panel stânga: Precision-Recall Curves ──
    ax = axes[0]
    for (name, y_prob, thresh), color in zip(models_data, colors):
        precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(recalls, precisions, color=color, lw=2,
                label=f"{name} (PR-AUC = {ap:.3f})")

        # Pragul tunat (cerc gol)
        y_pred_t = (y_prob >= thresh).astype(int)
        p_t = precision_score(y_test, y_pred_t, zero_division=0)
        r_t = recall_score(y_test, y_pred_t, zero_division=0)
        ax.plot(r_t, p_t, "o", color=color, markersize=13,
                markerfacecolor="white", markeredgewidth=2.5, zorder=5)

        # Pragul default 0.5 (pătrat plin)
        y_pred_d = (y_prob >= 0.5).astype(int)
        p_d = precision_score(y_test, y_pred_d, zero_division=0)
        r_d = recall_score(y_test, y_pred_d, zero_division=0)
        ax.plot(r_d, p_d, "s", color=color, markersize=9,
                alpha=0.7, zorder=4)

    baseline = float(y_test.mean())
    ax.axhline(baseline, color="gray", linestyle=":", lw=1.2,
               label=f"Random baseline = {baseline:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves\n○ = prag tunat (F1 max pe val)   ■ = prag default 0.5")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # ── Panel dreapta: Confusion Matrix ──
    ax = axes[1]
    f1_per_model = [
        f1_score(y_test, (yp >= th).astype(int), zero_division=0)
        for _, yp, th in models_data
    ]
    best_idx = int(np.argmax(f1_per_model))
    best_name, best_prob, best_thresh = models_data[best_idx]
    best_pred = (best_prob >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, best_pred)

    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    labels = ["Supraviețuit", "Decedat"]
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=15, fontweight="bold",
            )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicție")
    ax.set_ylabel("Realitate")
    tp, fp = cm[1, 1], cm[0, 1]
    fn, tn = cm[1, 0], cm[0, 0]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    ax.set_title(
        f"Matricea Confuziei – {best_name}\n"
        f"(prag tunat = {best_thresh:.3f}  |  P={prec:.3f}  R={rec:.3f})"
    )
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafic salvat → {out_path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    csv_path = next(pth.Path.cwd().glob("DateProiect9.csv"), None)
    if csv_path is None:
        raise FileNotFoundError(
            "DateProiect9.csv nu a fost găsit. "
            "Rulați script-ul din același folder cu fișierul CSV."
        )

    print("=" * 68)
    print("  OPTIMIZARE HIPERPARAMETRI + TUNARE PRAG – Tema AIE2")
    print("  GridSearch (LR) · RandomizedSearch (RF) · Optuna (XGB) · Ensemble")
    print("=" * 68)

    # ── 1. Date ─────────────────────────────────────────────
    print("\n[1] Încarc și preprocesez datele …")
    df = load_and_preprocess(str(csv_path))
    print(f"    Shape: {df.shape}  |  Rata DEATH=1: {df['DEATH'].mean():.2%}")

    # ── 2. Split în 3 seturi ────────────────────────────────
    print("\n[2] Split train / val / test …")
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     X_train_std, X_val_std, X_test_std,
     scaler) = prepare_splits(df)
    print(f"    Train (downsample 1:3) : {len(X_train):>8,}  |  DEATH=1: {y_train.mean():.1%}")
    print(f"    Val   (natural)        : {len(X_val):>8,}  |  DEATH=1: {y_val.mean():.1%}")
    print(f"    Test  (natural)        : {len(X_test):>8,}  |  DEATH=1: {y_test.mean():.1%}")

    # ── 3-5. Optimizare modele individuale ──────────────────
    print("\n[3] Optimizare Logistic Regression …")
    best_lr = optimize_lr(X_train_std, y_train)

    print("\n[4] Optimizare Random Forest …")
    best_rf = optimize_rf(X_train, y_train)

    print("\n[5] Optimizare XGBoost …")
    best_xgb = optimize_xgb(X_train, y_train)

    # ── 6. Probabilități pe val + tunare prag ───────────────
    print("\n[6] Tunare prag de decizie pe setul de validare …")
    prob_lr_val  = best_lr.predict_proba(X_val_std)[:, 1]
    prob_rf_val  = best_rf.predict_proba(X_val)[:, 1]
    prob_xgb_val = best_xgb.predict_proba(X_val)[:, 1]

    thresh_lr,  f1_lr_val  = tune_threshold(y_val, prob_lr_val)
    thresh_rf,  f1_rf_val  = tune_threshold(y_val, prob_rf_val)
    thresh_xgb, f1_xgb_val = tune_threshold(y_val, prob_xgb_val)

    # AUC pe val — folosit pentru ponderi ensemble (evităm leak din test)
    auc_lr_val  = roc_auc_score(y_val, prob_lr_val)
    auc_rf_val  = roc_auc_score(y_val, prob_rf_val)
    auc_xgb_val = roc_auc_score(y_val, prob_xgb_val)

    print(f"    [LR]  prag={thresh_lr:.3f}  F1_val={f1_lr_val:.4f}  AUC_val={auc_lr_val:.4f}")
    print(f"    [RF]  prag={thresh_rf:.3f}  F1_val={f1_rf_val:.4f}  AUC_val={auc_rf_val:.4f}")
    print(f"    [XGB] prag={thresh_xgb:.3f}  F1_val={f1_xgb_val:.4f}  AUC_val={auc_xgb_val:.4f}")

    # ── 7. Ensemble cu ponderi din val ──────────────────────
    print("\n[7] Construiesc Ensemble (VotingClassifier soft) …")
    best_ensemble = build_ensemble(
        best_lr, best_rf, best_xgb,
        X_train, y_train,
        auc_lr_val, auc_rf_val, auc_xgb_val,
    )
    prob_ens_val = best_ensemble.predict_proba(X_val)[:, 1]
    thresh_ens, f1_ens_val = tune_threshold(y_val, prob_ens_val)
    print(f"    [ENS] prag={thresh_ens:.3f}  F1_val={f1_ens_val:.4f}")

    # ── 8. Evaluare pe test la 2 praguri ────────────────────
    print("\n[8] Evaluez pe test – prag DEFAULT (0.5) și prag TUNAT …")

    res_default = [
        evaluate("Logistic Regression (default)", best_lr,       X_test_std, y_test, 0.5),
        evaluate("Random Forest (default)",        best_rf,       X_test,     y_test, 0.5),
        evaluate("XGBoost (default)",              best_xgb,      X_test,     y_test, 0.5),
        evaluate("Ensemble (default)",             best_ensemble, X_test,     y_test, 0.5),
    ]
    res_tuned = [
        evaluate("Logistic Regression (tuned)",    best_lr,       X_test_std, y_test, thresh_lr),
        evaluate("Random Forest (tuned)",          best_rf,       X_test,     y_test, thresh_rf),
        evaluate("XGBoost (tuned)",                best_xgb,      X_test,     y_test, thresh_xgb),
        evaluate("Ensemble (tuned)",               best_ensemble, X_test,     y_test, thresh_ens),
    ]

    all_results = []
    for d, t in zip(res_default, res_tuned):
        all_results.extend([d, t])
    results_df = pd.DataFrame(all_results)

    print("\n" + results_df.to_string(index=False))
    results_df.to_csv("optimization_results.csv", index=False)
    print("\nTabel salvat → optimization_results.csv")

    # ── 9. Câștig din tunare ────────────────────────────────
    print("\n[9] Câștig din tunarea pragului (test set):")
    for d, t in zip(res_default, res_tuned):
        name = d["Model"].replace(" (default)", "")
        df1 = t["F1-score"] - d["F1-score"]
        dp  = t["Precision"] - d["Precision"]
        print(f"    {name:22s}  ΔF1={df1:+.4f}  ΔPrecision={dp:+.4f}")

    tuned_df = pd.DataFrame(res_tuned)
    best_row = tuned_df.loc[tuned_df["F1-score"].idxmax()]
    print(f"\n  Cel mai bun model (F1 cu prag tunat): {best_row['Model']}")
    print(f"  F1 = {best_row['F1-score']:.4f}  |  "
          f"P = {best_row['Precision']:.4f}  |  R = {best_row['Recall']:.4f}")
    print(f"  ROC-AUC = {best_row['ROC-AUC']:.4f}  |  PR-AUC = {best_row['PR-AUC']:.4f}")

    # ── 10. Grafice ─────────────────────────────────────────
    print("\n[10] Generez grafice …")
    plot_comparison(results_df)

    prob_lr_test  = best_lr.predict_proba(X_test_std)[:, 1]
    prob_rf_test  = best_rf.predict_proba(X_test)[:, 1]
    prob_xgb_test = best_xgb.predict_proba(X_test)[:, 1]
    prob_ens_test = best_ensemble.predict_proba(X_test)[:, 1]
    models_data = [
        ("Logistic Regression", prob_lr_test,  thresh_lr),
        ("Random Forest",        prob_rf_test,  thresh_rf),
        ("XGBoost",              prob_xgb_test, thresh_xgb),
        ("Ensemble",             prob_ens_test, thresh_ens),
    ]
    plot_threshold_analysis(models_data, y_test)

    # ── 11. Salvare praguri ─────────────────────────────────
    thresholds_df = pd.DataFrame([
        {"Model": "Logistic Regression", "Threshold": thresh_lr,  "F1_val": f1_lr_val},
        {"Model": "Random Forest",        "Threshold": thresh_rf,  "F1_val": f1_rf_val},
        {"Model": "XGBoost",              "Threshold": thresh_xgb, "F1_val": f1_xgb_val},
        {"Model": "Ensemble",             "Threshold": thresh_ens, "F1_val": f1_ens_val},
    ])
    thresholds_df.to_csv("tuned_thresholds.csv", index=False)
    print("Praguri tunate salvate → tuned_thresholds.csv")

    print("\n" + "=" * 68)
    print("Fișiere generate:")
    print("  • optimized_models/best_lr.pkl / best_rf.pkl / best_xgb.pkl")
    print("  • optimized_models/best_ensemble.pkl")
    print("  • optimized_models/optuna_study_xgb.pkl")
    print("  • optimization_results.csv      (metrici default + tuned)")
    print("  • tuned_thresholds.csv          (praguri optime per model)")
    print("  • optimization_comparison.png   (F1/P/R: default vs tuned)")
    print("  • threshold_analysis.png        (PR curves + confusion matrix)")
    print("=" * 68)
