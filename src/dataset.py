"""
Module 4, step 1: assemble the analysis-ready modeling table.

Joins the per-location PFAS detection matrix (Module 2) to the four
distance-to-source features (Module 3b) and the geocoded coordinates
(Module 3a), producing one tidy table for the decoupling analysis.

The detection matrix carries 9 metadata columns plus the PFAS analytes. Analyte
columns are identified by EXCLUDING the known non-analyte columns, so facility
IDs / counts / water-type can never be mistaken for a PFAS measurement. The
script prints the analyte list and the actual cell values so the encoding is
confirmed, not assumed.

Decoupling-test framing: the OUTCOME is PFAS detection; the PREDICTORS are the
four nearest-source distances. Location covariates (water-source type, system
size, region, facility counts) are carried through so the model can control for
them -- holding these confounders constant is what isolates the source signal.

Reads:
    data/interim/detection_matrix.csv      per-location metadata + per-PFAS values
    data/interim/source_distances.csv      4 distance features + is_remote_outlier
    data/interim/locations_geocoded.csv    location_id, lat, lon
Writes:
    data/interim/model_table.csv           one row per modelled location

Usage: python src/dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DET = Path("data/interim/detection_matrix.csv")
DIST = Path("data/interim/source_distances.csv")
GEO = Path("data/interim/locations_geocoded.csv")
OUT = Path("data/interim/model_table.csv")

DIST_COLS = [
    "dist_industrial_km", "dist_wwtp_km", "dist_airport_km", "dist_military_km"
]

# Every column in the detection matrix that is NOT a PFAS analyte (lower-cased).
# Analytes are defined as "everything else", so this list must be complete --
# it was extended after the real schema was revealed.
NON_ANALYTE = {
    "location_id", "pwsid", "pws_id",
    "state", "state_code", "state_name",
    "region", "size", "system_size",
    "facilityid", "facility_id",
    "facilitywatertype", "facility_water_type", "watertype", "water_type",
    "n_facilities", "n_watertypes", "n_water_types", "n_samples",
    "county", "county_name", "fips", "zip", "zipcode", "postal_code",
    "lat", "lon", "latitude", "longitude", "coord_source",
}

# Location-level covariates (real, non-distance predictors / confounders) to
# carry into the model table when present in the detection matrix.
COVARIATES = ["State", "Region", "Size", "FacilityWaterType",
              "n_facilities", "n_watertypes"]


def main() -> None:
    print("1. Loading + inspecting the detection matrix")
    det = pd.read_csv(DET, dtype={"location_id": str})
    print(f"  detection_matrix: {det.shape[0]:,} rows x {det.shape[1]} cols")
    print(f"  columns: {', '.join(map(str, det.columns))}")

    analytes = [c for c in det.columns if str(c).lower() not in NON_ANALYTE]
    covs = [c for c in COVARIATES if c in det.columns]
    print(f"  PFAS analyte columns ({len(analytes)}):")
    print("    " + ", ".join(map(str, analytes)))
    print(f"  covariates carried ({len(covs)}): {', '.join(covs)}")

    vals = pd.unique(det[analytes].head(4000).to_numpy().ravel())
    vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
    print(f"  distinct analyte cell values (clean sample): "
          f"{sorted(map(str, vals))[:15]}")

    print("\n2. Binarising detection (value > 0 -> detected)")
    keep = det[["location_id", *covs]].copy()
    keep["location_id"] = keep["location_id"].astype(str)
    n_nan = 0
    for c in analytes:
        num = pd.to_numeric(det[c], errors="coerce")
        n_nan += int(num.isna().sum())
        keep[c] = (num > 0).astype("int8")
    keep["n_pfas_detected"] = keep[analytes].sum(axis=1).astype(int)
    keep["any_pfas_detected"] = (keep["n_pfas_detected"] > 0).astype(int)
    print(f"  non-numeric/NaN analyte cells -> non-detect: {n_nan:,}")
    print(f"  any-PFAS detection prevalence: "
          f"{keep['any_pfas_detected'].mean() * 100:.1f}%")
    print(f"  mean PFAS detected per location: "
          f"{keep['n_pfas_detected'].mean():.2f}  "
          f"(range {keep['n_pfas_detected'].min()}-{keep['n_pfas_detected'].max()})")
    # per-analyte detection rate, sorted -- the empirical hook lives here
    rates = (keep[analytes].mean() * 100).sort_values(ascending=False)
    print("  top analyte detection rates (%):")
    for name, rate in rates.head(8).items():
        print(f"    {str(name):<14} {rate:5.1f}")

    print("\n3. Loading distance features + coordinates")
    dist = pd.read_csv(DIST, dtype={"location_id": str})
    geo = pd.read_csv(GEO, dtype={"location_id": str})
    geo = geo[["location_id", *[c for c in ("lat", "lon") if c in geo.columns]]]
    print(f"  source_distances: {len(dist):,} rows | "
          f"locations_geocoded: {len(geo):,} rows")

    print("\n4. Joining on location_id (inner)")
    tab = keep.merge(dist, on="location_id", how="inner").merge(
        geo, on="location_id", how="inner")
    print(f"  detection rows:                  {len(keep):,}")
    print(f"  after join (distances + coords): {len(tab):,}")
    print(f"  dropped, no distance/coords (unplaced): {len(keep) - len(tab):,}")

    print("\n5. Removing remote outliers")
    before = len(tab)
    if "is_remote_outlier" in tab.columns:
        tab = tab[~tab["is_remote_outlier"].astype(bool)].copy()
        tab = tab.drop(columns=["is_remote_outlier"])
    print(f"  removed: {before - len(tab):,} | modelled locations: {len(tab):,}")

    print("\n6. Writing model table")
    front = ["location_id"] + covs
    front += [c for c in ("lat", "lon") if c in tab.columns]
    front += DIST_COLS + ["n_pfas_detected", "any_pfas_detected"]
    ordered = front + [c for c in analytes if c in tab.columns]
    tab = tab[[c for c in ordered if c in tab.columns]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT, index=False)

    print(f"  shape: {tab.shape[0]:,} rows x {tab.shape[1]} cols")
    print("  median distance (km): " + ", ".join(
        f"{c.replace('dist_', '').replace('_km', '')}={tab[c].median():.1f}"
        for c in DIST_COLS if c in tab.columns))
    if "FacilityWaterType" in tab.columns:
        print("  by water-source type (any-PFAS %):")
        wt = tab.groupby("FacilityWaterType")["any_pfas_detected"].mean() * 100
        print("    " + wt.round(1).to_string().replace("\n", "\n    "))

    print("\nsample:")
    show = ["location_id", *covs[:3], *DIST_COLS,
            "n_pfas_detected", "any_pfas_detected"]
    print(tab[[c for c in show if c in tab.columns]].head().to_string(index=False))
    print(f"\nsaved -> {OUT}")
    print("next: Module 4 step 2 -- model detection vs distance (XGBoost + SHAP).")


if __name__ == "__main__":
    main()
