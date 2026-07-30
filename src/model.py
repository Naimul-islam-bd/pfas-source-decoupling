"""
Module 4, step 2: the decoupling test -- does proximity to PFAS sources
predict detection?

Three binary classification targets, each modeled with XGBoost against the
four distance-to-source features and the location covariates, then explained
with SHAP:

    any_pfas_detected     -- primary test, all 29 analytes
    regulated_detected    -- PFOA and/or PFOS detected (the regulated pair
                              used as the benchmark in Salvatore et al. 2022)
    unregulated_detected  -- PFPeA, PFHxA, and/or PFBA detected -- the three
                              unregulated short-chains that are the project's
                              empirical hook (most-detected nationally, ahead
                              of PFOA/PFOS)

For each target, THREE model variants are fit:
    distances_only   -- the 4 distance features alone.
    covariates_only  -- covariates alone, no distances. This is the baseline
                         the decoupling test actually needs: it shows how much
                         is predictable from confounders before distance is
                         even in the picture.
    full             -- distances + covariates. SHAP is computed on this model
                         to rank the distance features against the covariates.

THE ACTUAL DECOUPLING COMPARISON is covariates_only vs full, not
distances_only vs full: if adding the 4 distance features to covariates_only
barely moves AUC, that is the evidence distance carries little information
beyond confounders (decoupled). distances_only vs full instead just shows how
much covariates add on top of distance, which is a different question.
(distances_only is still reported since it's informative on its own -- it's
the raw, unconditional distance signal -- just not the primary decoupling
test.)

Categorical covariates are passed to XGBoost natively (enable_categorical),
no one-hot encoding, so State/Region etc. don't blow up into dozens of columns.

KNOWN RISK (flagging per project rules, not hiding it): SHAP's TreeExplainer
occasionally trips on categorical dtypes depending on shap/xgboost version
pairing. If shap_summary_plot or the SHAP loop throws, paste the exact
traceback back -- it's a quick fix, not a redesign.

Reads:
    data/interim/model_table.csv
Writes:
    outputs/tables/model_results.csv      AUC per target x variant
    outputs/tables/shap_importance.csv    mean |SHAP| per feature x target
    outputs/figures/shap_summary_<target>.png   one per target, full model

Usage: python src/model.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

IN = Path("data/interim/model_table.csv")
RESULTS = Path("outputs/tables/model_results.csv")
SHAP_TABLE = Path("outputs/tables/shap_importance.csv")
FIG_DIR = Path("outputs/figures")

DIST_COLS = ["dist_industrial_km", "dist_wwtp_km", "dist_airport_km", "dist_military_km"]
COV_COLS = ["State", "Region", "Size", "FacilityWaterType", "n_facilities", "n_watertypes"]
CAT_COLS = ["State", "Region", "Size", "FacilityWaterType"]

# Regulated pair benchmarked to Salvatore et al. (2022); unregulated hook is
# the three top-detected short-chains established in Module 2.
REGULATED = ["PFOA", "PFOS"]
UNREGULATED_HOOK = ["PFPeA", "PFHxA", "PFBA"]

RANDOM_STATE = 42


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REGULATED + UNREGULATED_HOOK if c not in df.columns]
    if missing:
        raise ValueError(f"expected analyte columns missing from model_table.csv: {missing}")
    df = df.copy()
    df["regulated_detected"] = (df[REGULATED].sum(axis=1) > 0).astype(int)
    df["unregulated_detected"] = (df[UNREGULATED_HOOK].sum(axis=1) > 0).astype(int)
    return df


def prep_features(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "distances_only":
        cols = list(DIST_COLS)
    elif variant == "covariates_only":
        cols = [c for c in COV_COLS if c in df.columns]
    elif variant == "full":
        cols = list(DIST_COLS) + [c for c in COV_COLS if c in df.columns]
    else:
        raise ValueError(f"unknown variant: {variant}")
    X = df[cols].copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def fit_and_evaluate(X: pd.DataFrame, y: pd.Series, label: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"  [{label}] n={len(X):,}  prevalence={y.mean()*100:.1f}%  "
          f"test AUC={auc:.3f}")
    return model, X_train, auc


def shap_importance(model, X_train, target_name, variant_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    imp = pd.DataFrame({
        "feature": X_train.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    imp["target"] = target_name
    imp["variant"] = variant_name
    return imp, shap_values


def main() -> None:
    print("1. Loading model table")
    df = pd.read_csv(IN, dtype={"location_id": str})
    print(f"  {df.shape[0]:,} rows x {df.shape[1]} cols")
    df = make_targets(df)
    print(f"  regulated_detected prevalence:   {df['regulated_detected'].mean()*100:.1f}%")
    print(f"  unregulated_detected prevalence: {df['unregulated_detected'].mean()*100:.1f}%")

    targets = ["any_pfas_detected", "regulated_detected", "unregulated_detected"]

    results_rows = []
    shap_frames = []
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    print("\n2. Fitting models (distances-only vs distances+covariates) per target")
    for target_col in targets:
        y = df[target_col]
        print(f"\n  -- target: {target_col} --")

        X_dist = prep_features(df, "distances_only")
        _, _, auc_d = fit_and_evaluate(X_dist, y, f"{target_col} | distances-only")
        results_rows.append({"target": target_col, "variant": "distances_only", "n": len(df),
                              "prevalence_pct": round(y.mean() * 100, 2), "test_auc": round(auc_d, 4)})

        X_cov = prep_features(df, "covariates_only")
        _, _, auc_c = fit_and_evaluate(X_cov, y, f"{target_col} | covariates-only")
        results_rows.append({"target": target_col, "variant": "covariates_only", "n": len(df),
                              "prevalence_pct": round(y.mean() * 100, 2), "test_auc": round(auc_c, 4)})

        X_full = prep_features(df, "full")
        model_f, Xtr_f, auc_f = fit_and_evaluate(X_full, y, f"{target_col} | full")
        results_rows.append({"target": target_col, "variant": "full", "n": len(df),
                              "prevalence_pct": round(y.mean() * 100, 2), "test_auc": round(auc_f, 4)})

        delta_decoupling = auc_f - auc_c
        print(f"    >> decoupling delta (full AUC - covariates_only AUC): {delta_decoupling:+.4f}")
        print("       (small delta = distance adds little beyond confounders = decoupled)")

        imp, shap_values = shap_importance(model_f, Xtr_f, target_col, "full")
        shap_frames.append(imp)
        print("    full feature-ranking SHAP importance (all features, mean |SHAP|):")
        for _, r in imp.iterrows():
            flag = "  <- distance" if r["feature"] in DIST_COLS else ""
            print(f"      {r['feature']:<20} {r['mean_abs_shap']:.4f}{flag}")

        plt.figure()
        shap.summary_plot(shap_values, Xtr_f, show=False)
        plt.tight_layout()
        fig_path = FIG_DIR / f"shap_summary_{target_col}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"    saved -> {fig_path}")

    print("\n3. Writing results tables")
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(RESULTS, index=False)
    shap_df = pd.concat(shap_frames, ignore_index=True)
    shap_df.to_csv(SHAP_TABLE, index=False)
    print(f"  saved -> {RESULTS}")
    print(f"  saved -> {SHAP_TABLE}")

    print("\n4. Summary: AUC per target x variant")
    pivot = results_df.pivot(index="target", columns="variant", values="test_auc")
    pivot = pivot[["distances_only", "covariates_only", "full"]]
    pivot["decoupling_delta_full_minus_cov"] = pivot["full"] - pivot["covariates_only"]
    print(pivot.to_string())
    print("\n  PRIMARY decoupling read: the decoupling_delta column (full AUC minus")
    print("  covariates_only AUC). Small delta = distance adds little on top of")
    print("  confounders = supports decoupling. Large delta = distance still")
    print("  matters even after controlling for water-source type, size, etc.")
    print("  SECONDARY read: distances_only AUC on its own, and where distance")
    print("  features land in the full-model SHAP ranking above (top vs bottom).")
    print("  Compare regulated_detected vs unregulated_detected rows for the")
    print("  specific hook-related question: do unregulated short-chains show a")
    print("  smaller decoupling_delta than regulated PFOA/PFOS?")
    print("\nnext: write up results; consider sensitivity toggles from HANDOFF_02")
    print("(Salvatore-38 NAICS, major-POTW-only WWTP, FUDS munitions-response subset).")


if __name__ == "__main__":
    main()
