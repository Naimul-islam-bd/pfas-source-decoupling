"""
Module 4, step 2c: robustness of the decoupling result under CV schemes that
respect spatial structure and sampling non-independence.

WHY THIS EXISTS (read before running):
The step-2b diagnostics used random 5-fold CV. Two problems make random folds
too optimistic for a SPATIAL question:

  (A) Spatial autocorrelation. Nearby water systems have similar detection.
      With random folds, a location's near-neighbours can sit in BOTH train
      and test, so the model partly memorises local geography and every AUC
      is inflated. The honest test is SPATIAL-BLOCK CV: whole geographic
      blocks are held out, so the model must generalise to unseen regions.

  (B) Non-independence. location_id values like 010106001_TP1 / _EP001 are
      entry points / treatment plants WITHIN one public water system (PWSID).
      Multiple rows per PWS are not independent samples; random folds split a
      PWS across train and test, leaking system-level signal and shrinking
      CIs artificially. Fix: PWS-GROUPED CV (all rows of a PWS in one fold).

This script re-estimates, for each target, the decoupling delta
(full AUC - covariates_only AUC) under THREE CV schemes side by side:
      random        -- reference, reproduces step-2b
      pws_grouped   -- StratifiedGroupKFold on PWSID  (fixes B)
      spatial_block -- StratifiedGroupKFold on a lat/lon grid cell (fixes A)
If the delta stays small and stable across all three, the "distance adds
little beyond regional structure" conclusion is robust to these threats.
If it moves a lot, we learn exactly which assumption was carrying it.

WHAT THIS SCRIPT DOES **NOT** FIX (stated plainly, no pretending):
  * ZIP-CENTROID GEOCODING. Every location is placed at its ZIP centroid
    (coord_source == zip_centroid for all 24,640). ZIP-centroid position
    error is often a few km -- comparable to or LARGER than the median
    industrial (2.2 km) and WWTP (6.6 km) distances. That attenuates the true
    distance-detection association toward zero (classical regression
    dilution). So a SMALL distance signal here is partly real, partly a
    measurement artifact, and this script cannot separate the two. Removing
    this confound needs facility-level coordinates (EPA SDWIS/FRS geocoding),
    which is a new data-acquisition module, not a modelling tweak. Until then
    the distance result is a LOWER BOUND on the true association and must be
    reported that way.

PWSID source: model_table.csv has no PWSID column, so we recover it from
locations_geocoded.csv (which has location_id + PWSID) via a join. We do NOT
guess it by slicing location_id strings.

Reads:
    data/interim/model_table.csv
    data/interim/locations_geocoded.csv   (for the PWSID grouping key)
Writes:
    outputs/tables/robustness_cv_delta.csv

Usage: python src/robustness.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

MODEL_TABLE = Path("data/interim/model_table.csv")
GEO = Path("data/interim/locations_geocoded.csv")
OUT = Path("outputs/tables/robustness_cv_delta.csv")

DIST_COLS = ["dist_industrial_km", "dist_wwtp_km", "dist_airport_km", "dist_military_km"]
COV_COLS = ["State", "Region", "Size", "FacilityWaterType", "n_facilities", "n_watertypes"]
CAT_COLS = ["State", "Region", "Size", "FacilityWaterType"]

REGULATED = ["PFOA", "PFOS"]
UNREGULATED_HOOK = ["PFPeA", "PFHxA", "PFBA"]

RANDOM_STATE = 42
N_SPLITS = 5
GRID_DEG = 1.0  # spatial block size in degrees (~111 km lat); coarse on purpose
T95_DF4 = 2.776  # t_{0.975, df = N_SPLITS - 1}


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["regulated_detected"] = (df[REGULATED].sum(axis=1) > 0).astype(int)
    df["unregulated_detected"] = (df[UNREGULATED_HOOK].sum(axis=1) > 0).astype(int)
    return df


def attach_group_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add pws_group (from geocoded PWSID) and grid_group (lat/lon cell)."""
    df = df.copy()

    geo = pd.read_csv(GEO, dtype={"location_id": str})
    pwsid_col = next((c for c in ("PWSID", "pwsid", "pws_id") if c in geo.columns), None)
    if pwsid_col is None:
        raise ValueError(f"no PWSID column in {GEO}; columns = {list(geo.columns)}")
    geo = (geo[["location_id", pwsid_col]]
           .rename(columns={pwsid_col: "pws_group"})
           .drop_duplicates(subset="location_id"))  # 1 PWSID per location_id; never fan out rows
    df["location_id"] = df["location_id"].astype(str)
    n_before = len(df)
    df = df.merge(geo, on="location_id", how="left", validate="many_to_one")
    assert len(df) == n_before, "PWSID join changed row count -- aborting"

    # Any location whose PWSID didn't join falls back to its own id as its group
    # (treated as a singleton system) rather than being silently dropped.
    n_missing_pws = int(df["pws_group"].isna().sum())
    df["pws_group"] = df["pws_group"].fillna(df["location_id"])

    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("model_table.csv is missing lat/lon needed for spatial blocks")
    lat_cell = np.floor(df["lat"] / GRID_DEG).astype("Int64")
    lon_cell = np.floor(df["lon"] / GRID_DEG).astype("Int64")
    df["grid_group"] = lat_cell.astype(str) + "_" + lon_cell.astype(str)

    print(f"  PWS groups: {df['pws_group'].nunique():,} distinct "
          f"(rows/group median {df.groupby('pws_group').size().median():.0f}, "
          f"unjoined PWSID -> singleton: {n_missing_pws:,})")
    print(f"  spatial blocks ({GRID_DEG} deg): {df['grid_group'].nunique():,} distinct "
          f"cells (rows/cell median {df.groupby('grid_group').size().median():.0f})")
    return df


def build_X(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    cols = list(DIST_COLS) if variant == "full" else []
    if variant == "full":
        cols += [c for c in COV_COLS if c in df.columns]
    elif variant == "covariates_only":
        cols = [c for c in COV_COLS if c in df.columns]
    else:
        raise ValueError(variant)
    X = df[cols].copy()
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


def grouped_fold_auc(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> np.ndarray:
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr, te in sgkf.split(X, y, groups=groups):
        model = new_model()
        model.fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], proba))
    return np.array(aucs)


def main() -> None:
    print("Loading model table + group keys")
    df = pd.read_csv(MODEL_TABLE, dtype={"location_id": str})
    df = make_targets(df)
    df = attach_group_keys(df)
    print(f"  {df.shape[0]:,} rows\n")

    targets = ["any_pfas_detected", "regulated_detected", "unregulated_detected"]
    schemes = {"pws_grouped": "pws_group", "spatial_block": "grid_group"}

    rows = []
    for target in targets:
        y = df[target]
        print(f"-- {target}  (prevalence {y.mean()*100:.1f}%) --")
        X_cov = build_X(df, "covariates_only")
        X_full = build_X(df, "full")

        for scheme, gcol in schemes.items():
            groups = df[gcol]
            auc_cov = grouped_fold_auc(X_cov, y, groups)
            auc_full = grouped_fold_auc(X_full, y, groups)
            d = auc_full - auc_cov  # paired per fold
            se = d.std(ddof=1) / np.sqrt(len(d))
            lo, hi = d.mean() - T95_DF4 * se, d.mean() + T95_DF4 * se
            rows.append({
                "target": target, "cv_scheme": scheme,
                "cov_auc": round(auc_cov.mean(), 4), "full_auc": round(auc_full.mean(), 4),
                "delta_mean": round(d.mean(), 4),
                "delta_ci95_low": round(lo, 4), "delta_ci95_high": round(hi, 4),
            })
            print(f"   {scheme:<14} cov {auc_cov.mean():.4f}  full {auc_full.mean():.4f}  "
                  f"delta {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        print()

    out_df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)
    print(f"saved -> {OUT}\n")

    print("HOW TO READ:")
    print("  * Compare each scheme's AUCs to the step-2b random-CV numbers.")
    print("    Spatial-block and PWS-grouped AUCs are EXPECTED to be somewhat")
    print("    lower than random-CV -- that drop is the inflation random folds")
    print("    were hiding, not a failure. The realistic performance is the")
    print("    spatial-block row.")
    print("  * The key quantity is still delta (distance's marginal lift over")
    print("    confounders). If delta stays small with a tight CI under BOTH")
    print("    grouped schemes, the decoupling conclusion is robust to")
    print("    autocorrelation and PWS non-independence.")
    print("  * REMEMBER the un-fixed confound: ZIP-centroid geocoding still")
    print("    attenuates the distance signal, so even these deltas are an")
    print("    UPPER bound on how decoupled things look / LOWER bound on the")
    print("    true distance association. Report accordingly.")


if __name__ == "__main__":
    main()
