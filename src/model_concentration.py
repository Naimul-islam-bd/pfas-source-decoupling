"""
Module 4, Stage 2: the concentration decoupling test (the second stage of the
hurdle). Stage 1 asked WHERE PFAS is detected; Stage 2 asks, AMONG locations
where it IS detected, whether source proximity predicts HOW MUCH.

This mirrors the Stage-1 decoupling test exactly, so the two are directly
comparable:
    outcome   log10(total_pfas_conc)   (detected locations only; right-skewed,
                                        so log-transformed)
    variants  covariates_only  vs  full (covariates + 4 source distances)
    CV        spatial-block + PWS-grouped StratifiedGroupKFold analogue
              (for regression we use GroupKFold on the same keys)
    metric    R^2 (not AUC -- this is regression); decoupling delta = R^2(full)
              - R^2(covariates_only), paired per fold with a 95% CI

INTERPRETATION:
  * delta small  -> source proximity adds little to concentration beyond
                    regional structure: concentration is decoupled too
                    (strengthens the overall decoupling story).
  * delta clearly positive -> among detected systems, proximity DOES raise
                    concentration, even though it didn't drive detection. That
                    is the elegant two-stage result: detection is regional,
                    magnitude is local -- and it reconciles this work with the
                    concentration-based source literature (Hu et al. etc.).
Either outcome is publishable; we report what the data says.

Per-analyte models are also run for the analytes with enough detects
(>~1500 detected locations): PFOA, PFOS, PFPeA, PFHxA, PFBA, PFBS, PFHxS.

Reads:
    data/interim/concentration_matrix.csv   (detected locations, ug/L)
    data/interim/model_table.csv            (distances + covariates + coords)
    data/interim/locations_geocoded.csv     (PWSID + coord_source)
Writes:
    outputs/tables/stage2_concentration_delta.csv

Usage: python src/model_concentration.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

CONC = Path("data/interim/concentration_matrix.csv")
MODEL_TABLE = Path("data/interim/model_table.csv")
GEO = Path("data/interim/locations_geocoded.csv")
OUT = Path("outputs/tables/stage2_concentration_delta.csv")

DIST_COLS = ["dist_industrial_km", "dist_wwtp_km", "dist_airport_km", "dist_military_km"]
COV_COLS = ["State", "Region", "Size", "FacilityWaterType", "n_facilities", "n_watertypes"]
CAT_COLS = ["State", "Region", "Size", "FacilityWaterType"]

# analytes with enough detected locations for a per-analyte model
PER_ANALYTE = ["PFPeA", "PFHxA", "PFBS", "PFBA", "PFOS", "PFOA", "PFHxS"]
MIN_DETECTS = 800

RANDOM_STATE = 42
N_SPLITS = 5
GRID_DEG = 1.0
T95_DF4 = 2.776


def build_X(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "covariates_only":
        cols = [c for c in COV_COLS if c in df.columns]
    elif variant == "full":
        cols = list(DIST_COLS) + [c for c in COV_COLS if c in df.columns]
    else:
        raise ValueError(variant)
    X = df[cols].copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def new_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        enable_categorical=True, random_state=RANDOM_STATE,
    )


def grouped_r2(X, y, groups) -> np.ndarray:
    gkf = GroupKFold(n_splits=N_SPLITS)
    out = []
    for tr, te in gkf.split(X, y, groups=groups):
        m = new_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        out.append(r2_score(y.iloc[te], m.predict(X.iloc[te])))
    return np.array(out)


def delta_ci(full_r2, cov_r2):
    d = full_r2 - cov_r2
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - T95_DF4 * se, d.mean() + T95_DF4 * se


def run_outcome(df: pd.DataFrame, yname: str, y: pd.Series) -> list[dict]:
    print(f"\n=== outcome: {yname}  (n={len(df):,}) ===")
    schemes = {"pws_grouped": "pws_group", "spatial_block": "grid_group"}
    rows = []
    for scheme, gcol in schemes.items():
        g = df[gcol]
        Xc = build_X(df, "covariates_only")
        Xf = build_X(df, "full")
        rc = grouped_r2(Xc, y, g)
        rf = grouped_r2(Xf, y, g)
        dm, lo, hi = delta_ci(rf, rc)
        rows.append({"outcome": yname, "cv_scheme": scheme, "n": len(df),
                     "cov_r2": round(rc.mean(), 4), "full_r2": round(rf.mean(), 4),
                     "delta_mean": round(dm, 4),
                     "delta_ci95_low": round(lo, 4), "delta_ci95_high": round(hi, 4)})
        print(f"   {scheme:<14} covR2 {rc.mean():.4f}  fullR2 {rf.mean():.4f}  "
              f"delta {dm:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    return rows


def main() -> None:
    print("1. Loading concentration matrix (detected locations) + features")
    conc = pd.read_csv(CONC, dtype={"location_id": str, "PWSID": str})
    mt = pd.read_csv(MODEL_TABLE, dtype={"location_id": str})
    geo = pd.read_csv(GEO, dtype={"location_id": str})

    pwsid_col = next((c for c in ("PWSID", "pwsid") if c in geo.columns), None)
    keep_geo = ["location_id"] + ([pwsid_col] if pwsid_col else [])
    geo = geo[keep_geo].drop_duplicates("location_id")
    if pwsid_col:
        geo = geo.rename(columns={pwsid_col: "pws_group"})

    # join distances+covariates onto detected locations
    feat_cols = ["location_id"] + DIST_COLS + [c for c in COV_COLS if c in mt.columns] + ["lat", "lon"]
    df = conc.merge(mt[feat_cols], on="location_id", how="inner")
    df = df.merge(geo, on="location_id", how="left")
    if "pws_group" not in df.columns:
        df["pws_group"] = df["location_id"]
    df["pws_group"] = df["pws_group"].fillna(df["location_id"])
    df["grid_group"] = (np.floor(df["lat"] / GRID_DEG).astype("Int64").astype(str)
                        + "_" + np.floor(df["lon"] / GRID_DEG).astype("Int64").astype(str))
    print(f"   detected locations with full features: {len(df):,}")

    # primary outcome: log10(total_pfas_conc)
    df = df[df["total_pfas_conc"] > 0].copy()
    df["log_total"] = np.log10(df["total_pfas_conc"])
    print(f"   log10(total_pfas_conc): min={df['log_total'].min():.2f}, "
          f"median={df['log_total'].median():.2f}, max={df['log_total'].max():.2f}")

    results = []
    results += run_outcome(df, "log_total_pfas", df["log_total"])

    # per-analyte outcomes where power allows
    for a in PER_ANALYTE:
        if a not in df.columns:
            continue
        sub = df[df[a].notna() & (df[a] > 0)].copy()
        if len(sub) < MIN_DETECTS:
            print(f"\n   (skip {a}: only {len(sub)} detected locations < {MIN_DETECTS})")
            continue
        sub["logy"] = np.log10(sub[a])
        results += run_outcome(sub, f"log_{a}", sub["logy"])

    res = pd.DataFrame(results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}")

    print("\n=== SIDE-BY-SIDE: concentration decoupling delta (R^2 lift) ===")
    for scheme in ["spatial_block", "pws_grouped"]:
        print(f"\n  [{scheme}]")
        piv = res[res["cv_scheme"] == scheme][["outcome", "delta_mean",
                                               "delta_ci95_low", "delta_ci95_high"]]
        print(piv.to_string(index=False))

    print("\nHOW TO READ:")
    print("  Compare these concentration deltas to the Stage-1 DETECTION deltas.")
    print("  * If concentration delta is also small -> magnitude is decoupled too")
    print("    (overall decoupling stronger).")
    print("  * If concentration delta is clearly positive while detection delta")
    print("    was ~0 -> two-stage story: detection regional, concentration local.")
    print("  Note: total_pfas_conc dynamic range is narrow (most detects near MRL),")
    print("  so modest R^2 is expected; the delta (lift from distances) is the")
    print("  quantity of interest, not absolute R^2.")


if __name__ == "__main__":
    main()
