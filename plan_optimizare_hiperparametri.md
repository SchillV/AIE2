# Plan Optimizare Hiperparametri – Tema AIE2
**Dataset:** DateProiect9.csv (COVID-19 Mortality Prediction)
**Fișier implementare:** `optimize_models.py`

---

## Situația actuală (notebook)

| Model | Metodă | Spațiu de căutare | CV Score |
|---|---|---|---|
| Logistic Regression | GridSearchCV, cv=3 | C: [0.01, 0.1, 1, 10] → **4 valori** | 0.9244 |
| Random Forest | GridSearchCV, cv=3 | 2×3×2 = **12 combinații** | 0.9257 |
| XGBoost | Optuna, **30 trials**, cv=3 | 5 parametri continui | 0.9269 |

Punctele slabe: LR și RF au spații de căutare mici. XGBoost e deja mai bine optimizat.

---

## Filozofia optimizării (aceeași ca semestrul 1)

- **GridSearch** pentru modele cu puțini parametri (LR, RF) → spațiu discret, reproductibil
- **Optuna** pentru XGBoost → mulți parametri corelați, spațiu continuu mai potrivit

---

## Pasul 1 – Logistic Regression: GridSearchCV extins

**Ce se schimbă față de notebook:**
- C: de la 4 valori `[0.01, 0.1, 1, 10]` → 6 valori `[0.001, 0.01, 0.1, 1, 10, 100]`
- cv: de la 3 → 5 (estimare mai stabilă pentru model simplu)
- Total: 6 combinații × cv=5 = 30 fits (~30 secunde)

```python
param_grid = {
    "C":        [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty":  ["l2"],
    "solver":   ["lbfgs"],
    "max_iter": [1000],
}
```

---

## Pasul 2 – Random Forest: RandomizedSearchCV (grid extins)

**Ce se schimbă față de notebook:**
- Din GridSearch cu 12 combinații fixe → RandomizedSearch cu 20 combinații aleatorii
- Adăugăm `min_samples_leaf` și `max_features` în spațiu
- Exploram mai larg `n_estimators` (200–500 vs. [200, 400]) și `max_depth`
- Total: 20 iterații × cv=3 = 60 fits (~5–10 minute cu n_jobs=-1)

```python
param_dist = {
    "n_estimators":      randint(200, 500),
    "max_depth":         [None, 10, 15, 20, 30],
    "min_samples_split": randint(2, 15),
    "min_samples_leaf":  randint(1, 8),
    "max_features":      ["sqrt", "log2", 0.4],
}
```

---

## Pasul 3 – XGBoost: Optuna extins (50 trials)

**Ce se schimbă față de notebook:**
- Trials: 30 → **50**
- Adăugăm `min_child_weight` și `gamma` în spațiu
- Adăugăm **MedianPruner** (oprește trial-urile slabe devreme)
- Sampler: TPESampler cu seed fix (reproductibil)
- Studiul Optuna se salvează pentru vizualizări ulterioare (optuna-dashboard)

```python
params = {
    "n_estimators":     [200, 600],        # int
    "max_depth":        [3, 9],            # int
    "learning_rate":    [0.01, 0.25],      # float
    "subsample":        [0.5, 1.0],        # float
    "colsample_bytree": [0.5, 1.0],        # float
    "min_child_weight": [1, 10],           # int  ← NOU
    "gamma":            [0.0, 0.5],        # float ← NOU
}
```

---

## Pasul 4 – Ensemble: VotingClassifier soft

**Idee din `idei_de_imbunatatire.md`**: medie ponderată a predicțiilor.

- **Ponderi** = proporționale cu ROC-AUC pe test pentru fiecare model
- LR wrapuit în `Pipeline(StandardScaler → LR)` pentru a primi date brute
- RF și XGB primesc date brute direct
- `voting="soft"` → combină probabilitățile, nu clasele

```python
ensemble = VotingClassifier(
    estimators=[("lr", lr_pipeline), ("rf", best_rf), ("xgb", best_xgb)],
    voting="soft",
    weights=[w_lr, w_rf, w_xgb],  # normalizate din ROC-AUC
)
```

---

## Pasul 5 – Salvare + Comparație

**Modele salvate** (cu auto-cache: dacă există, nu se reantrenează):
```
optimized_models/
├── best_lr.pkl
├── best_rf.pkl
├── best_xgb.pkl
├── best_ensemble.pkl
└── optuna_study_xgb.pkl   ← pentru vizualizări Optuna
```

**Outputs:**
- `optimization_results.csv` – tabel cu toate metricile
- `optimization_comparison.png` – grafic bar comparativ

---

## Estimare runtime

| Pas | Metrie | Estimat |
|---|---|---|
| LR GridSearch | 6 × cv=5 = 30 fits | ~1 min |
| RF RandomizedSearch | 20 × cv=3 = 60 fits (paralel) | ~5–10 min |
| XGB Optuna | 50 trials × cv=3 = 150 fits | ~5–10 min |
| Ensemble fit | 1 fit (modele cu params optimi) | ~2 min |
| **Total** | | **~15–25 min** |

---

## Structura fișierelor

```
├── Tema_1_AIE2_-_Schiller_Vlad-Radu.ipynb   (existent – nemodificat)
├── roi_analysis.py                            (existent – nemodificat)
├── optimize_models.py                         ← NOU (implementare plan)
├── plan_optimizare_hiperparametri.md          ← acest fișier
├── DateProiect9.csv                           (existent)
└── optimized_models/
    ├── best_lr.pkl
    ├── best_rf.pkl
    ├── best_xgb.pkl
    ├── best_ensemble.pkl
    └── optuna_study_xgb.pkl
```
