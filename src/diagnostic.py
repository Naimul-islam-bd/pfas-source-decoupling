"""
Module 4, step 2b: diagnostics to make the decoupling result publishable.

The step-2 run showed a small decoupling delta (full AUC - covariates_only
AUC ~= 0.035-0.044) but ALSO showed State dominating SHAP importance by 4-6x
over any distance feature. Before writing up, we must rule out that State's
dominance -- and hence the whole covariates_only baseline -- is a
high-cardinality memorization artifact rather than real geographic
confounding. A reviewer will ask this first.

Three diagnostics, no new dependencies beyond step 2:

  1. STATE-ONLY vs COVARIATES-ONLY AUC. If State alone ~= covariates_only,
     State carries essentially the whole covariate baseline. That's fine IF
     the signal is real (diagnostic 3 decides that).

  2. 5-FOLD CROSS-VALIDATION for every target x variant, replacing the single
     80/20 split. Reports mean +/- std AUC, and -- the key number -- a 95%
     confidence interval on the decoupling delta (full - covariates_only),
     computed fold-wise (paired per fold). If that CI is tight and low, the
     "distance adds little beyond confounders" claim is defensible; if it's
     wide or straddles a meaningful value, the single-split delta was noise.

  3. STATE-SHUFFLE MEMORIZATION TEST. Refit the covariates_only model with
     State randomly permuted (target association destroyed), 5-fold. If
     shuffled-State AUC ~= real-State AUC, State's contribution is largely
     memorized base-rate noise. If it drops sharply, State carries real,
     generalizable geographic signal (legitimate confounder). This is the
     cleanest separator between the two interpretations.

This script does NOT rewrite model.py or change any step-2 output; it only
reads the same model_table.csv and reports diagnostics.

Reads:
    data/interim/model_table.csv
Writes:
    outputs/tables/diagnostic_cv.csv          per target x variant CV AUCs
    outputs/tables/diagnostic_delta_ci.csv    decoupling delta + 95% CI
    outputs/tables/diagnostic_state_shuffle.csv  real vs shuffled State AUC

Usage: python src/diagnostic.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

IN = Path("data/interim/model_table.csv")
CV_OUT = Path("outputs/tables/diagnostic_cv.csv")
CI_OUT = Path("outputs/tables/diagnostic_delta_ci.csv")
SHUF_OUT = Path("outputs/tables/diagnostic_state_shuffle.csv")

DIST_COLS = ["dist_industrial_km", "dist_wwtp_km", "dist_airport_km", "dist_military_km"]
COV_COLS = ["State", "Region", "Size", "FacilityWaterType", "n_facilities", "n_watertypes"]
CAT_COLS = ["State", "Region", "Size", "FacilityWaterType"]

REGULATED = ["PFOA", "PFOS"]
UNREGULATED_HOOK = ["PFPeA", "PFHxA", "PFBA"]

RANDOM_STATE = 42
N_SPLITS = 5


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["regulated_detected"] = (df[REGULATED].sum(axis=1) > 0).astype(int)
    df["unregulated_detected"] = (df[UNREGULATED_HOOK].sum(axis=1) > 0).astype(int)
    return df


def cols_for(variant: str, df: pd.DataFrame) -> list[str]:
    if variant == "distances_only":
        return list(DIST_COLS)
    if variant == "covariates_only":
        return [c for c in COV_COLS if c in df.columns]
    if variant == "state_only":
        return ["State"]
    if variant == "full":
        return list(DIST_COLS) + [c for c in COV_COLS if c in df.columns]
    raise ValueError(variant)


def as_categorical(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def new_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        enable_categorical=True, eval_metric="auc", random_state=RANDOM_STATE,
    )


def cv_auc(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Return per-fold test AUCs (length N_SPLITS)."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr, te in skf.split(X, y):
        model = new_model()
        model.fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], proba))
    return np.array(aucs)


def cv_auc_state_shuffled(df: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """covariates_only, but State permuted INSIDE each training fold only
    (test fold keeps real State so we measure loss of generalizable signal)."""
    cols = cols_for("covariates_only", df)
    X = as_categorical(df[cols])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)
    aucs = []
    for tr, te in skf.split(X, y):
        X_tr = X.iloc[tr].copy()
        perm = rng.permutation(X_tr["State"].to_numpy())
        X_tr["State"] = pd.Categorical(perm, categories=X["State"].cat.categories)
        model = new_model()
        model.fit(X_tr, y.iloc[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], proba))
    return np.array(aucs)


def main() -> None:
    print("Loading model table")
    df = pd.read_csv(IN, dtype={"location_id": str})
    df = make_targets(df)
    print(f"  {df.shape[0]:,} rows")

    targets = ["any_pfas_detected", "regulated_detected", "unregulated_detected"]
    variants = ["distances_only", "state_only", "covariates_only", "full"]

    for p in (CV_OUT, CI_OUT, SHUF_OUT):
        p.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n1+2. {N_SPLITS}-fold CV per target x variant (mean +/- std AUC)")
    cv_rows = []
    fold_store = {}  # (target, variant) -> per-fold array
    for target in targets:
        y = df[target]
        print(f"\n  -- {target}  (prevalence {y.mean()*100:.1f}%) --")
        for variant in variants:
            X = as_categorical(df[cols_for(variant, df)])
            aucs = cv_auc(X, y)
            fold_store[(target, variant)] = aucs
            cv_rows.append({
                "target": target, "variant": variant,
                "auc_mean": round(aucs.mean(), 4), "auc_std": round(aucs.std(ddof=1), 4),
            })
            print(f"     {variant:<16} AUC {aucs.mean():.4f} +/- {aucs.std(ddof=1):.4f}")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(CV_OUT, index=False)
    print(f"\n  saved -> {CV_OUT}")

    print("\n  Decoupling delta (full - covariates_only), paired per fold, 95% CI:")
    ci_rows = []
    for target in targets:
        d = fold_store[(target, "full")] - fold_store[(target, "covariates_only")]
        mean_d = d.mean()
        # t-based 95% CI on the paired fold differences (n=5 folds)
        se = d.std(ddof=1) / np.sqrt(len(d))
        t95 = 2.776  # t_{0.975, df=4}
        lo, hi = mean_d - t95 * se, mean_d + t95 * se
        ci_rows.append({
            "target": target, "delta_mean": round(mean_d, 4),
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
        })
        print(f"     {target:<22} delta {mean_d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    pd.DataFrame(ci_rows).to_csv(CI_OUT, index=False)
    print(f"  saved -> {CI_OUT}")

    print("\n3. State-shuffle memorization test (covariates_only, State permuted")
    print("   in training folds; test folds keep real State):")
    shuf_rows = []
    for target in targets:
        y = df[target]
        real = fold_store[(target, "covariates_only")]
        shuffled = cv_auc_state_shuffled(df, y)
        drop = real.mean() - shuffled.mean()
        shuf_rows.append({
            "target": target,
            "covariates_real_state_auc": round(real.mean(), 4),
            "covariates_shuffled_state_auc": round(shuffled.mean(), 4),
            "auc_drop": round(drop, 4),
        })
        print(f"     {target:<22} real {real.mean():.4f} -> shuffled "
              f"{shuffled.mean():.4f}   drop {drop:+.4f}")
    pd.DataFrame(shuf_rows).to_csv(SHUF_OUT, index=False)
    print(f"  saved -> {SHUF_OUT}")

    print("\nHOW TO READ:")
    print("  * state_only AUC ~= covariates_only AUC  -> State carries the whole")
    print("    covariate baseline (expected given the SHAP ranking).")
    print("  * delta 95% CI tight & low (e.g. entirely < ~0.05) -> the 'distance")
    print("    adds little beyond confounders' claim is statistically defensible.")
    print("  * State-shuffle: LARGE auc_drop -> State holds real generalizable")
    print("    geographic signal (legitimate confounder, finding stands).")
    print("    SMALL auc_drop -> State was mostly memorized base rates; the")
    print("    covariates_only baseline is inflated and must be reported with")
    print("    that caveat (or State regularized / grouped to region level).")


if __name__ == "__main__":
    main()
