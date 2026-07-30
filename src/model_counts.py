"""
Module 4, step 2e: does the decoupling survive Hu-style source COUNTS?

The detection decoupling was established with nearest-DISTANCE features. The
concentration-source literature (Hu et al. 2016; recent buffer-count studies)
instead uses source COUNTS within a radius. A reviewer's sharpest remaining
objection: "your weak signal is because you used distance, not counts -- counts
are what track PFAS." This test answers it directly by adding the 5 km / 10 km
source counts (from source_counts.py) and seeing whether they restore
predictive power beyond confounders.

Three nested variants per target, spatial-block + PWS-grouped CV, delta vs the
covariates_only baseline:
    covariates_only            baseline (State, water type, size, ...)
    cov + distances            our original 'full'
    cov + distances + counts   adds n_{industrial,wwtp,military}_{5,10}km

Read the two deltas:
    delta_dist  = (cov+dist) - cov            marginal lift of distances
    delta_all   = (cov+dist+counts) - cov     marginal lift of distances+counts
If delta_all is still small (like delta_dist), the decoupling holds even with
the literature's own feature type -- the strongest possible rebuttal. If
delta_all jumps up, counts carry source signal that distance missed, and the
decoupling claim must be softened to "weak" rather than "absent".

Reads:
    data/interim/model_table.csv
    data/interim/source_counts.csv
    data/interim/locations_geocoded.csv   (PWSID + coords for grouping)
Writes:
    outputs/tables/robustness_counts_delta.csv

Usage: python src/model_counts.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

MODEL_TABLE = Path("data/interim/model_table.csv")
COUNTS = Path("data/interim/source_counts.csv")
GEO = Path("data/interim/locations_geocoded.csv")
OUT = Path("outputs/tables/robustness_counts_delta.csv")

DIST_COLS = ["dist_industrial_km", "dist_wwtp_km", "dist_airport_km", "dist_military_km"]
COUNT_COLS = ["n_industrial_5km", "n_wwtp_5km", "n_military_5km",
              "n_industrial_10km", "n_wwtp_10km", "n_military_10km"]
COV_COLS = ["State", "Region", "Size", "FacilityWaterType", "n_facilities", "n_watertypes"]
CAT_COLS = ["State", "Region", "Size", "FacilityWaterType"]

REGULATED = ["PFOA", "PFOS"]
UNREGULATED_HOOK = ["PFPeA", "PFHxA", "PFBA"]

RANDOM_STATE = 42
N_SPLITS = 5
GRID_DEG = 1.0
T95_DF4 = 2.776


def make_targets(df):
    df = df.copy()
    df["regulated_detected"] = (df[REGULATED].sum(axis=1) > 0).astype(int)
    df["unregulated_detected"] = (df[UNREGULATED_HOOK].sum(axis=1) > 0).astype(int)
    return df


def attach(df):
    df = df.copy()
    df["location_id"] = df["location_id"].astype(str)
    df = df.drop(columns=[c for c in ["pws_group"] if c in df.columns])

    cnt = pd.read_csv(COUNTS, dtype={"location_id": str})
    n0 = len(df)
    df = df.merge(cnt, on="location_id", how="left", validate="many_to_one")
    assert len(df) == n0, "counts join changed row count"
    # locations with no count row (unplaced) -> 0 sources
    for c in COUNT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    geo = pd.read_csv(GEO, dtype={"location_id": str})
    pcol = next((c for c in ("PWSID", "pwsid") if c in geo.columns), None)
    keep = ["location_id"] + ([pcol] if pcol else [])
    geo = geo[keep].drop_duplicates("location_id")
    if pcol:
        geo = geo.rename(columns={pcol: "pws_group"})
    df = df.merge(geo, on="location_id", how="left", validate="many_to_one")
    if "pws_group" not in df.columns:
        df["pws_group"] = df["location_id"]
    df["pws_group"] = df["pws_group"].fillna(df["location_id"])

    df["grid_group"] = (np.floor(df["lat"] / GRID_DEG).astype("Int64").astype(str)
                        + "_" + np.floor(df["lon"] / GRID_DEG).astype("Int64").astype(str))
    return df


def build_X(df, variant):
    if variant == "covariates_only":
        cols = [c for c in COV_COLS if c in df.columns]
    elif variant == "cov_dist":
        cols = list(DIST_COLS) + [c for c in COV_COLS if c in df.columns]
    elif variant == "cov_dist_counts":
        cols = list(DIST_COLS) + [c for c in COUNT_COLS if c in df.columns] + \
               [c for c in COV_COLS if c in df.columns]
    else:
        raise ValueError(variant)
    X = df[cols].copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def new_model():
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        enable_categorical=True, eval_metric="auc", random_state=RANDOM_STATE)


def grouped_auc(X, y, groups):
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    out = []
    for tr, te in sgkf.split(X, y, groups=groups):
        m = new_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        out.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    return np.array(out)


def ci(delta):
    se = delta.std(ddof=1) / np.sqrt(len(delta))
    return delta.mean(), delta.mean() - T95_DF4 * se, delta.mean() + T95_DF4 * se


def main():
    print("Loading model table + counts + grouping keys")
    df = make_targets(pd.read_csv(MODEL_TABLE, dtype={"location_id": str}))
    df = attach(df)
    print(f"  {len(df):,} rows")
    present_counts = [c for c in COUNT_COLS if c in df.columns]
    print(f"  count features present: {present_counts}")

    targets = ["any_pfas_detected", "regulated_detected", "unregulated_detected"]
    schemes = {"pws_grouped": "pws_group", "spatial_block": "grid_group"}
    rows = []

    for target in targets:
        y = df[target]
        print(f"\n-- {target} (prev {y.mean()*100:.1f}%) --")
        for scheme, gcol in schemes.items():
            g = df[gcol]
            a_cov = grouped_auc(build_X(df, "covariates_only"), y, g)
            a_dist = grouped_auc(build_X(df, "cov_dist"), y, g)
            a_all = grouped_auc(build_X(df, "cov_dist_counts"), y, g)
            d_dist_m, d_dist_lo, d_dist_hi = ci(a_dist - a_cov)
            d_all_m, d_all_lo, d_all_hi = ci(a_all - a_cov)
            rows.append({
                "target": target, "cv_scheme": scheme,
                "cov_auc": round(a_cov.mean(), 4),
                "cov_dist_auc": round(a_dist.mean(), 4),
                "cov_dist_counts_auc": round(a_all.mean(), 4),
                "delta_dist": round(d_dist_m, 4),
                "delta_dist_ci_lo": round(d_dist_lo, 4), "delta_dist_ci_hi": round(d_dist_hi, 4),
                "delta_all": round(d_all_m, 4),
                "delta_all_ci_lo": round(d_all_lo, 4), "delta_all_ci_hi": round(d_all_hi, 4),
            })
            print(f"   {scheme:<14} cov {a_cov.mean():.4f} | +dist {a_dist.mean():.4f} "
                  f"(delta {d_dist_m:+.4f}) | +dist+counts {a_all.mean():.4f} "
                  f"(delta {d_all_m:+.4f} [{d_all_lo:+.4f},{d_all_hi:+.4f}])")

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}")

    print("\n=== SIDE-BY-SIDE: marginal lift over confounders ===")
    for scheme in ["spatial_block", "pws_grouped"]:
        print(f"\n  [{scheme}]  distances-only lift vs distances+counts lift")
        sub = res[res["cv_scheme"] == scheme][
            ["target", "delta_dist", "delta_all", "delta_all_ci_lo", "delta_all_ci_hi"]]
        print(sub.to_string(index=False))

    print("\nHOW TO READ (the rebuttal):")
    print("  delta_all is the lift from distances+counts together over confounders.")
    print("  If delta_all stays small (close to delta_dist), source proximity AND")
    print("  source density -- the literature's own feature -- add little beyond")
    print("  regional structure: the decoupling is not a feature-choice artifact.")
    print("  If delta_all is clearly larger than delta_dist, counts recover signal")
    print("  distance missed; report the finding as 'weak association' not 'absent'.")


if __name__ == "__main__":
    main()
