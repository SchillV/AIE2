"""
ROI Analysis – COVID-19 Mortality Prediction
============================================
Tema AIE2 – Lift / Cumulative Gains / ROI Charts
cu 100 de înregistrări alese aleatoriu din setul de test.

Cum să rulezi:
  1. Asigură-te că `DateProiect9.csv` este în același folder.
  2. python roi_analysis.py
     → va antrena modelul XGBoost (best din notebook), îl salvează ca
       `xgb_model.pkl`, alege 100 înregistrări aleatoare și generează
       graficele în `roi_charts.png` + tabelul în `lift_table.csv`.

Dacă ai deja modelul salvat:
  → script-ul îl detectează automat și NU îl reantrenează.
"""

import numpy as np
import pandas as pd
import pathlib as pth
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Parametri business (ca în exemplul telcom din curs)
# ──────────────────────────────────────────────────────────────
PATIENT_VALUE      = 500   # Valoarea unui pacient salvat ($)
INTERVENTION_COST  = 100   # Costul intervenției/cuponului ($)
# Dacă actual DEATH=1 și intervenim → profit = 500 - 100 = +400
# Dacă actual DEATH=0 și intervenim → risipă   = -100
PROFIT_IF_POSITIVE = PATIENT_VALUE - INTERVENTION_COST   # +400
PROFIT_IF_NEGATIVE = -INTERVENTION_COST                  # -100
RANDOM_SEED        = 42
N_SAMPLES          = 100  # numărul de înregistrări aleatorii


# ══════════════════════════════════════════════════════════════
# 1. Preprocesare date (identic cu notebook-ul)
# ══════════════════════════════════════════════════════════════
def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Derivăm DEATH din DATE_DIED
    df["DEATH"] = [0 if r == "9999-99-99" else 1 for r in df["DATE_DIED"]]
    df.drop("DATE_DIED", axis=1, inplace=True)

    categoric = [
        "USMER", "MEDICAL_UNIT", "SEX", "PATIENT_TYPE", "INTUBED", "PNEUMONIA",
        "PREGNANT", "DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION",
        "OTHER_DISEASE", "CARDIOVASCULAR", "OBESITY", "RENAL_CHRONIC",
        "TOBACCO", "CLASIFFICATION_FINAL", "ICU", "DEATH",
    ]
    # Înlocuim 2 → 0
    for col in categoric:
        if col in df.columns:
            df[col] = df[col].replace(2, 0)

    # PREGNANT: 97 = bărbat (N/A), 98 = femei nedeclarată
    df["PREGNANT"] = df["PREGNANT"].replace([97, 98], 2)

    # Valori speciale → NaN
    for col in categoric:
        if col in df.columns:
            df[col] = df[col].replace([97, 98, 99], np.nan)

    # Eliminăm coloane indezirabile
    for col in ["MEDICAL_UNIT", "CLASIFFICATION_FINAL", "PATIENT_TYPE"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Păstrăm doar coloanele cu ≤ 50% NaN
    df = df.loc[:, df.isna().mean() <= 0.5]

    # Curățăm AGE
    df.loc[(df["AGE"] <= 0) | (df["AGE"] > 122), "AGE"] = np.nan
    df = df.dropna(subset=["AGE"])

    # Eliminăm rândurile cu NaN pe categorice rămase
    existing = set(df.columns)
    cats = [c for c in categoric if c in existing]
    for col in cats:
        df[col] = df[col].replace(2, 0)
        df = df.dropna(subset=[col])

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# 2. Antrenare / salvare / încărcare model
# ══════════════════════════════════════════════════════════════
MODEL_PATH = "xgb_model.pkl"

def train_and_save(df: pd.DataFrame):
    """Antrenează XGBoost cu cei mai buni hiperparametri din notebook."""
    from sklearn.model_selection import train_test_split
    from sklearn.utils import resample
    from xgboost import XGBClassifier

    X = df.drop(columns=["DEATH"])
    y = df["DEATH"]

    indices = np.arange(len(df))
    train_idx, _ = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    # Downsample-ul clasei majority (identic cu notebook-ul)
    dead_idx  = train_idx[df.iloc[train_idx]["DEATH"] == 1]
    alive_idx = train_idx[df.iloc[train_idx]["DEATH"] == 0]
    alive_ds_idx = resample(
        alive_idx, replace=False,
        n_samples=3 * len(dead_idx),
        random_state=RANDOM_SEED
    )
    tr = np.concatenate([dead_idx, alive_ds_idx])
    np.random.shuffle(tr)

    X_tr, y_tr = df.iloc[tr].drop(columns=["DEATH"]), df.iloc[tr]["DEATH"]
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()

    # Hiperparametrii optimi din Optuna (notebook)
    model = XGBClassifier(
        n_estimators=407, max_depth=5,
        learning_rate=0.017660929814067364,
        subsample=0.76897737730628,
        colsample_bytree=0.8960941818833018,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    print("  Antrenez modelul XGBoost (poate dura ~1-2 min) …")
    model.fit(X_tr, y_tr)
    joblib.dump(model, MODEL_PATH)
    print(f"  Model salvat → {MODEL_PATH}")
    return model


def get_model(df: pd.DataFrame):
    if pth.Path(MODEL_PATH).exists():
        print(f"Model găsit la {MODEL_PATH}. Îl încarc …")
        return joblib.load(MODEL_PATH)
    print("Modelul nu a fost găsit. Antrenez …")
    return train_and_save(df)


# ══════════════════════════════════════════════════════════════
# 3. Selectăm 100 înregistrări aleatorii din setul de TEST
# ══════════════════════════════════════════════════════════════
def get_test_sample(df: pd.DataFrame, model, n=N_SAMPLES):
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=["DEATH"])
    y = df["DEATH"]
    indices = np.arange(len(df))
    _, test_idx = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    # 100 aleatorii din test
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(test_idx, size=n, replace=False)

    X_sample = X.iloc[sample_idx].reset_index(drop=True)
    y_sample = y.iloc[sample_idx].reset_index(drop=True)

    # ── predict_proba[:, 1] ── probabilitatea de DEATH=1
    proba = model.predict_proba(X_sample)[:, 1]

    print(f"\n100 înregistrări aleatorii din test:")
    print(f"  DEATH=1 (decedat):  {y_sample.sum()}")
    print(f"  DEATH=0 (suprav.):  {(y_sample == 0).sum()}")
    print(f"  Rata deceselor:     {y_sample.mean():.1%}")

    return X_sample, y_sample, proba


# ══════════════════════════════════════════════════════════════
# 4. Calculul lift / ROI pentru cele 3 scenarii
# ══════════════════════════════════════════════════════════════
def build_scenario(y_true: pd.Series, proba: np.ndarray,
                   sort_mode: str = "model") -> pd.DataFrame:
    """
    sort_mode:
      'model'  – sortat DESC după scor model (scenariul AI)
      'random' – ordine aleatorie          (scenariul "la voia întâmplării")
      'oracle' – toate cazurile pozitive primele (scenariul perfect)
    """
    df = pd.DataFrame({"ActualDeath": y_true.values, "Proba": proba})

    if sort_mode == "model":
        df = df.sort_values("Proba", ascending=False).reset_index(drop=True)
    elif sort_mode == "oracle":
        df = df.sort_values("ActualDeath", ascending=False).reset_index(drop=True)
    elif sort_mode == "random":
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    else:
        raise ValueError(f"sort_mode necunoscut: {sort_mode}")

    n = len(df)
    total_positive = df["ActualDeath"].sum()
    baseline_rate  = total_positive / n   # rata deceselor în eșantion

    df["CumulativeCases"]  = np.arange(1, n + 1)
    df["CumulativeDeaths"] = df["ActualDeath"].cumsum()
    df["PropCases"]        = df["CumulativeCases"] / n
    df["PropDeathsCaught"] = df["CumulativeDeaths"] / max(total_positive, 1)

    # LIFT = (prop_deaths_caught) / (prop_cases) — față de baza aleatorie
    with np.errstate(invalid="ignore", divide="ignore"):
        df["Lift"] = np.where(
            df["PropCases"] > 0,
            df["PropDeathsCaught"] / df["PropCases"],
            np.nan
        )
    df["BaselineLift"] = 1.0  # linia aleatorie pe graficul de lift

    # Profit/Loss per înregistrare
    df["ProfitLoss"] = df["ActualDeath"].map(
        {1: PROFIT_IF_POSITIVE, 0: PROFIT_IF_NEGATIVE}
    )
    df["NetProfit"]  = df["ProfitLoss"].cumsum()

    return df


# ══════════════════════════════════════════════════════════════
# 5. Decile Table (ca în Excel)
# ══════════════════════════════════════════════════════════════
def decile_table(model_df: pd.DataFrame, total_positive: int) -> pd.DataFrame:
    """Tabel cu 10 decile – fiecare decilă = 10% din înregistrări."""
    n = len(model_df)
    rows = []
    for d in range(1, 11):
        cutoff = int(np.ceil(d * n / 10))
        cum_pos = model_df.iloc[:cutoff]["ActualDeath"].sum()
        # "fără model" = câte pozitive ai prinde la întâmplare la același cutoff
        baseline_pos = cutoff * total_positive / n
        lift = cum_pos / baseline_pos if baseline_pos > 0 else 0
        rows.append({
            "Decile": d,
            "Cutoff (#)": cutoff,
            "Cum. Deaths (Model)": int(cum_pos),
            "Cum. Deaths (Random)": round(baseline_pos, 1),
            "Lift": round(lift, 3),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# 6. Grafice
# ══════════════════════════════════════════════════════════════
def make_charts(model_df, random_df, oracle_df, total_positive, out_path="roi_charts.png"):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle(
        "Analiza ROI – Predicția Mortalității COVID-19\n"
        f"(100 înregistrări aleatorii din setul de test  |  "
        f"valoare pacient = ${PATIENT_VALUE}  |  cost intervenție = ${INTERVENTION_COST})",
        fontsize=12, y=1.01
    )

    x_pct_m = model_df["PropCases"] * 100
    x_pct_r = random_df["PropCases"] * 100
    x_pct_o = oracle_df["PropCases"] * 100

    # ── Grafic 1: Cumulative Gains ─────────────────────────────
    ax = axes[0]
    ax.plot(x_pct_m, model_df["PropDeathsCaught"] * 100,
            color="#1f77b4", lw=2.5, label="Model (XGBoost)")
    ax.plot(x_pct_r, random_df["PropDeathsCaught"] * 100,
            color="#ff7f0e", lw=1.8, ls="--", label="Aleatoriu")
    ax.plot(x_pct_o, oracle_df["PropDeathsCaught"] * 100,
            color="#2ca02c", lw=1.8, ls=":", label="Oracol (perfect)")
    ax.set_xlabel("% Pacienți contactați")
    ax.set_ylabel("% Decese identificate")
    ax.set_title("Cumulative Gains Chart")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    # ── Grafic 2: Lift Chart ───────────────────────────────────
    ax = axes[1]
    ax.plot(x_pct_m, model_df["Lift"],
            color="#1f77b4", lw=2.5, label="Model (XGBoost)")
    ax.axhline(1.0, color="#ff7f0e", lw=1.8, ls="--", label="Aleatoriu (lift = 1)")
    ax.set_xlabel("% Pacienți contactați")
    ax.set_ylabel("Lift")
    ax.set_title("Lift Chart")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    # ── Grafic 3: ROI – Profit net cumulat ────────────────────
    ax = axes[2]
    ax.plot(model_df["CumulativeCases"], model_df["NetProfit"],
            color="#1f77b4", lw=2.5, label="Model (AI)")
    ax.plot(random_df["CumulativeCases"], random_df["NetProfit"],
            color="#ff7f0e", lw=1.8, ls="--", label="Aleatoriu")
    ax.plot(oracle_df["CumulativeCases"], oracle_df["NetProfit"],
            color="#2ca02c", lw=1.8, ls=":", label="Oracol (perfect)")
    ax.axhline(0, color="red", lw=1, ls="-", alpha=0.6, label="Break-even")
    ax.set_xlabel("# Pacienți contactați")
    ax.set_ylabel("Profit net cumulat ($)")
    ax.set_title("ROI: Profit net vs. # Pacienți contactați")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGrafice salvate → {out_path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── 1. Găsim CSV-ul ──────────────────────────────────────
    csv_path = next(pth.Path.cwd().glob("DateProiect9.csv"), None)
    if csv_path is None:
        raise FileNotFoundError(
            "DateProiect9.csv nu a fost găsit. "
            "Asigurați-vă că fișierul este în același folder cu script-ul."
        )

    print("=" * 60)
    print("ANALIZA ROI – Predicția Mortalității COVID-19")
    print("=" * 60)

    # ── 2. Preprocesăm datele ─────────────────────────────────
    print("\n[1] Încarc și preprocesez datele …")
    df = load_and_preprocess(str(csv_path))
    print(f"    Shape final: {df.shape}")
    print(f"    Rata DEATH=1: {df['DEATH'].mean():.2%}")

    # ── 3. Obținem modelul (antrenăm dacă nu există) ──────────
    print("\n[2] Obțin modelul …")
    model = get_model(df)

    # ── 4. Eșantion de 100 ───────────────────────────────────
    print("\n[3] Selectez 100 înregistrări aleatorii din test …")
    X_sample, y_sample, proba = get_test_sample(df, model, n=N_SAMPLES)
    total_pos = int(y_sample.sum())

    # ── 5. Construim cele 3 scenarii ─────────────────────────
    print("\n[4] Calculez scenariile …")
    model_df  = build_scenario(y_sample, proba, "model")
    random_df = build_scenario(y_sample, proba, "random")
    oracle_df = build_scenario(y_sample, proba, "oracle")

    # ── 6. Tabel decile ──────────────────────────────────────
    dec_tbl = decile_table(model_df, total_pos)
    print("\n[5] Tabel decile (Model):")
    print(dec_tbl.to_string(index=False))
    dec_tbl.to_csv("lift_table.csv", index=False)
    print("    Tabel salvat → lift_table.csv")

    # ── 7. Rezumat ROI ───────────────────────────────────────
    final_model  = model_df["NetProfit"].iloc[-1]
    final_random = random_df["NetProfit"].iloc[-1]
    final_oracle = oracle_df["NetProfit"].iloc[-1]

    print("\n[6] Rezumat ROI (dacă intervenim pe toți cei 100):")
    print(f"    Oracol (perfect): ${final_oracle:>8,.0f}")
    print(f"    Model (XGBoost):  ${final_model:>8,.0f}")
    print(f"    Aleatoriu:        ${final_random:>8,.0f}")
    print(f"    Câștig vs aleatoriu: ${final_model - final_random:>+,.0f}")

    # ── 8. Grafice ───────────────────────────────────────────
    print("\n[7] Generez graficele …")
    make_charts(model_df, random_df, oracle_df, total_pos)

    print("\nGata! Fișiere generate:")
    print(f"  • xgb_model.pkl  – modelul XGBoost salvat")
    print(f"  • lift_table.csv – tabelul decile")
    print(f"  • roi_charts.png – cele 3 grafice")
    print("=" * 60)
