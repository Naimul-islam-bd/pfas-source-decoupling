"""
Module 4, step 2f: the core decoupling test with REPEATED cross-validation, for
statistically stable confidence intervals.

WHY: earlier robustness scripts estimated the decoupling delta from a single
5-fold split -> the CI rested on just 5 fold-differences (t with df=4), which is
fragile and can swing with the random seed. Here we repeat the whole 5-fold
StratifiedGroupKFold R times with different seeds, giving 5*R paired fold
deltas. We report both a t-based CI (df = 5R-1) and a percentile CI, which no
longer hinges on a single unlucky split. This is the number that should go in
the paper.

Everything else matches the established design exactly so results are
comparable: targets any/regulated/unregulated; variants covariates_only vs
full (covariates + 4 distances); CV schemes spatial-block (lat/lon grid) and
PWS-grouped (PWSID); delta = AUC(full) - AUC(covariates_only), paired per fold.

Reads:
    data/interim/model_table.csv
    data/interim/locations_geocoded.csv   (PWSID + coords for grouping)
Writes:
    outputs/tables/robustness_repeated_delta.csv

Usage: python src/robustness_repeated.py
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
OUT = Path("outputs/tables/robustness_repeated_delta.csv")

DIST_COLS = ["dist_industrial_km", "dist_wwtp_km", "dist_airport_km", "dist_military_km"]
COV_COLS = ["State", "Region", "Size", "FacilityWaterType", "n_facilities", "n_watertypes"]
CAT_COLS = ["State", "Region", "Size", "FacilityWaterType"]

REGULATED = ["PFOA", "PFOS"]
UNREGULATED_HOOK = ["PFPeA", "PFHxA", "PFBA"]

N_SPLITS = 5
N_REPEATS = 5           # 5 x 5 = 25 paired fold deltas
GRID_DEG = 1.0


def make_targets(df):
    df = df.copy()
    df["regulated_detected"] = (df[REGULATED].sum(axis=1) > 0).astype(int)
    df["unregulated_detected"] = (df[UNREGULATED_HOOK].sum(axis=1) > 0).astype(int)
    return df


def attach(df):
    df = df.copy()
    df["location_id"] = df["location_id"].astype(str)
    geo = pd.read_csv(GEO, dtype={"location_id": str})
    pcol = next((c for c in ("PWSID", "pwsid") if c in geo.columns), None)
    keep = ["location_id"] + ([pcol] if pcol else [])
    geo = geo[keep].drop_duplicates("location_id")
    if pcol:
        geo = geo.rename(columns={pcol: "pws_group"})
    n0 = len(df)
    df = df.merge(geo, on="location_id", how="left", validate="many_to_one")
    assert len(df) == n0
    if "pws_group" not in df.columns:
        df["pws_group"] = df["location_id"]
    df["pws_group"] = df["pws_group"].fillna(df["location_id"])
    df["grid_group"] = (np.floor(df["lat"] / GRID_DEG).astype("Int64").astype(str)
                        + "_" + np.floor(df["lon"] / GRID_DEG).astype("Int64").astype(str))
    return df


def build_X(df, variant):
    cols = ([] if variant == "covariates_only" else list(DIST_COLS)) + \
           [c for c in COV_COLS if c in df.columns]
    X = df[cols].copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def new_model():
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        enable_categorical=True, eval_metric="auc", random_state=0)


def repeated_deltas(Xc, Xf, y, groups):
    """Return array of 5*N_REPEATS paired (full - cov) fold AUC deltas."""
    deltas = []
    for rep in range(N_REPEATS):
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=rep)
        for tr, te in sgkf.split(Xc, y, groups=groups):
            mc = new_model(); mc.fit(Xc.iloc[tr], y.iloc[tr])
            ac = roc_auc_score(y.iloc[te], mc.predict_proba(Xc.iloc[te])[:, 1])
            mf = new_model(); mf.fit(Xf.iloc[tr], y.iloc[tr])
            af = roc_auc_score(y.iloc[te], mf.predict_proba(Xf.iloc[te])[:, 1])
            deltas.append(af - ac)
    return np.array(deltas)


def main():
    print(f"Repeated CV: {N_SPLITS} folds x {N_REPEATS} repeats "
          f"= {N_SPLITS*N_REPEATS} paired deltas per cell")
    df = make_targets(pd.read_csv(MODEL_TABLE, dtype={"location_id": str}))
    df = attach(df)
    print(f"  {len(df):,} rows")

    targets = ["any_pfas_detected", "regulated_detected", "unregulated_detected"]
    schemes = {"pws_grouped": "pws_group", "spatial_block": "grid_group"}
    from scipy import stats
    dof = N_SPLITS * N_REPEATS - 1
    tcrit = stats.t.ppf(0.975, dof)

    rows = []
    for target in targets:
        y = df[target]
        print(f"\n-- {target} (prev {y.mean()*100:.1f}%) --")
        Xc, Xf = build_X(df, "covariates_only"), build_X(df, "full")
        for scheme, gcol in schemes.items():
            d = repeated_deltas(Xc, Xf, y, df[gcol])
            mean = d.mean()
            se = d.std(ddof=1) / np.sqrt(len(d))
            t_lo, t_hi = mean - tcrit * se, mean + tcrit * se
            p_lo, p_hi = np.percentile(d, [2.5, 97.5])
            rows.append({
                "target": target, "cv_scheme": scheme, "n_estimates": len(d),
                "delta_mean": round(mean, 4),
                "t_ci_lo": round(t_lo, 4), "t_ci_hi": round(t_hi, 4),
                "pct_ci_lo": round(p_lo, 4), "pct_ci_hi": round(p_hi, 4),
            })
            print(f"   {scheme:<14} delta {mean:+.4f}  "
                  f"t95 [{t_lo:+.4f},{t_hi:+.4f}]  "
                  f"pctile95 [{p_lo:+.4f},{p_hi:+.4f}]")

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}")
    print("\nThese CIs rest on 25 estimates (df=24), not 5 (df=4): report these as")
    print("the definitive decoupling deltas. t-CI and percentile-CI agreeing is a")
    print("good sign of stability.")


if __name__ == "__main__":
    main()
