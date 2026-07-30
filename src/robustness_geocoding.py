"""
Module 4, step 2d: the geocoding-quality robustness check -- the headline
artifact-killer for the decoupling result.

THE ARGUMENT THIS SETTLES:
A reviewer's first objection to "PFAS detection is weakly tied to source
proximity" is: "your coordinates are ZIP centroids, off by kilometres, so of
course distance looks weak -- that's measurement error, not real decoupling."
geolocate_v2 upgraded ~50% of locations to facility-grade FRS coordinates
(median accuracy ~30 m; the ZIP->facility shift had median ~3.25 km, confirming
the ZIP error was real and comparable to the distances being measured).

This script re-runs the decoupling test (covariates_only vs full, spatial-block
+ PWS-grouped CV) on TWO row sets, side by side:
    full            -- every modelled location (facility-grade + ZIP mix)
    facility_only   -- ONLY coord_source == 'frs_facility' rows (~30 m coords,
                       no ZIP position noise at all)

If the decoupling delta stays small in facility_only -- where geocoding is
essentially exact -- the decoupling is NOT a geocoding artifact. That is the
sentence the paper needs. If instead the delta jumps up in facility_only, the
earlier "weak distance" story was partly ZIP attenuation and must be softened.
Either way we learn the truth.

coord_source lives in locations_geocoded.csv (model_table.csv doesn't carry
it), so we join it on location_id.

Reads:
    data/interim/model_table.csv
    data/interim/locations_geocoded.csv   (for coord_source + PWSID)
Writes:
    outputs/tables/robustness_geocoding_quality.csv

Usage: python src/robustness_geocoding.py
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
OUT = Path("outputs/tables/robustness_geocoding_quality.csv")

DIST_COLS = ["dist_industrial_km", "dist_wwtp_km", "dist_airport_km", "dist_military_km"]
COV_COLS = ["State", "Region", "Size", "FacilityWaterType", "n_facilities", "n_watertypes"]
CAT_COLS = ["State", "Region", "Size", "FacilityWaterType"]

REGULATED = ["PFOA", "PFOS"]
UNREGULATED_HOOK = ["PFPeA", "PFHxA", "PFBA"]

RANDOM_STATE = 42
N_SPLITS = 5
GRID_DEG = 1.0
T95_DF4 = 2.776


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["regulated_detected"] = (df[REGULATED].sum(axis=1) > 0).astype(int)
    df["unregulated_detected"] = (df[UNREGULATED_HOOK].sum(axis=1) > 0).astype(int)
    return df


def attach_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add coord_source, pws_group, grid_group from locations_geocoded.csv."""
    df = df.copy()
    df["location_id"] = df["location_id"].astype(str)
    # model_table normally doesn't carry these; drop if present so the join is clean
    df = df.drop(columns=[c for c in ["coord_source", "pws_group"] if c in df.columns])

    geo = pd.read_csv(GEO, dtype={"location_id": str})
    pwsid_col = next((c for c in ("PWSID", "pwsid", "pws_id") if c in geo.columns), None)
    cols = ["location_id", "coord_source"]
    if pwsid_col:
        cols.append(pwsid_col)
    geo = geo[cols].drop_duplicates(subset="location_id")
    if pwsid_col:
        geo = geo.rename(columns={pwsid_col: "pws_group"})

    n0 = len(df)
    df = df.merge(geo, on="location_id", how="left", validate="many_to_one")
    assert len(df) == n0, "coord_source join changed row count"

    if "pws_group" not in df.columns:
        df["pws_group"] = df["location_id"]
    df["pws_group"] = df["pws_group"].fillna(df["location_id"])

    lat_cell = np.floor(df["lat"] / GRID_DEG).astype("Int64")
    lon_cell = np.floor(df["lon"] / GRID_DEG).astype("Int64")
    df["grid_group"] = lat_cell.astype(str) + "_" + lon_cell.astype(str)

    df["coord_source"] = df["coord_source"].fillna("unknown")
    print("  coord_source breakdown (modelled rows): "
          f"{df['coord_source'].value_counts().to_dict()}")
    return df


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


def new_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        enable_categorical=True, eval_metric="auc", random_state=RANDOM_STATE,
    )


def grouped_auc(X, y, groups) -> np.ndarray:
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    out = []
    for tr, te in sgkf.split(X, y, groups=groups):
        m = new_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        out.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    return np.array(out)


def delta_ci(full_aucs, cov_aucs):
    d = full_aucs - cov_aucs
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - T95_DF4 * se, d.mean() + T95_DF4 * se


def run_rowset(df: pd.DataFrame, label: str, rows) -> list[dict]:
    sub = df.loc[rows].copy()
    print(f"\n=== rowset: {label}  (n={len(sub):,}) ===")
    targets = ["any_pfas_detected", "regulated_detected", "unregulated_detected"]
    schemes = {"pws_grouped": "pws_group", "spatial_block": "grid_group"}
    out = []
    for target in targets:
        y = sub[target]
        if y.nunique() < 2:
            print(f"  {target}: only one class present, skipping")
            continue
        Xc = build_X(sub, "covariates_only")
        Xf = build_X(sub, "full")
        print(f"  -- {target} (prev {y.mean()*100:.1f}%) --")
        for scheme, gcol in schemes.items():
            g = sub[gcol]
            ac = grouped_auc(Xc, y, g)
            af = grouped_auc(Xf, y, g)
            dm, lo, hi = delta_ci(af, ac)
            out.append({"rowset": label, "target": target, "cv_scheme": scheme,
                        "n": len(sub), "cov_auc": round(ac.mean(), 4),
                        "full_auc": round(af.mean(), 4), "delta_mean": round(dm, 4),
                        "delta_ci95_low": round(lo, 4), "delta_ci95_high": round(hi, 4)})
            print(f"     {scheme:<14} cov {ac.mean():.4f}  full {af.mean():.4f}  "
                  f"delta {dm:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    return out


def main() -> None:
    print("Loading model table + geocoding quality keys")
    df = pd.read_csv(MODEL_TABLE, dtype={"location_id": str})
    df = make_targets(df)
    df = attach_keys(df)
    print(f"  {len(df):,} modelled rows")

    rows_full = df.index
    rows_fac = df.index[df["coord_source"] == "frs_facility"]
    print(f"\n  full rowset:          {len(rows_full):,}")
    print(f"  facility-grade only:  {len(rows_fac):,} "
          f"({100*len(rows_fac)/len(df):.1f}% of modelled)")

    results = []
    results += run_rowset(df, "full", rows_full)
    results += run_rowset(df, "facility_only", rows_fac)

    res = pd.DataFrame(results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}")

    print("\n=== SIDE-BY-SIDE: decoupling delta, full vs facility_only ===")
    for scheme in ["spatial_block", "pws_grouped"]:
        print(f"\n  [{scheme}]")
        piv = (res[res["cv_scheme"] == scheme]
               .pivot(index="target", columns="rowset", values="delta_mean"))
        piv = piv[[c for c in ["full", "facility_only"] if c in piv.columns]]
        print(piv.to_string())

    print("\nHOW TO READ (the headline):")
    print("  Compare 'facility_only' vs 'full' delta per target. If facility_only")
    print("  delta is about the same (still small), the weak distance signal is")
    print("  NOT a ZIP-geocoding artifact -- it holds where coordinates are exact")
    print("  (~30 m). That is the decisive robustness result for the paper.")
    print("  If facility_only delta is clearly LARGER, part of the earlier")
    print("  'decoupling' was ZIP attenuation and the claim must be softened.")


if __name__ == "__main__":
    main()
