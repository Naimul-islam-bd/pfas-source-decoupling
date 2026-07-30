"""
Module 3a-v2: upgrade location coordinates from ZIP-centroid to facility-grade
where possible, using FRS SFDW (drinking-water program) points.

WHY: the original locations_geocoded.csv places every location at its ZIP
centroid (position error commonly a few km). Median nearest-source distances
are 2.2-9 km, so ZIP error is comparable to the signal and attenuates the
distance-detection association (regression dilution). FRS SFDW facility points
carry real coordinates (probe: median accuracy 30 m) for a large subset of our
PWSIDs. We swap those in and keep ZIP for the rest -> a hybrid, honestly
labelled in coord_source.

METHOD (decided after probing the real data):
  * From FRS NATIONAL_SINGLE.CSV, keep rows whose PGM_SYS_ACRNMS contains an
    SFDW token; parse the PWSID from that token (format seen in probe:
    "SFDW:AL0000001" or "SFDW:AL0000001 32664" -> PWSID = AL0000001).
  * Keep rows with valid LATITUDE83/LONGITUDE83 and ACCURACY_VALUE <= 1000 m
    (same Salvatore-style filter as Module 3b).
  * A PWS can have several facilities; pick the BEST-ACCURACY facility per PWS
    (a real, verifiable point, avoiding the phantom-location problem a centroid
    of scattered wells would create). Report within-PWS spread so the choice is
    transparent.
  * Broadcast the chosen PWS coordinate to every UCMR5 location of that PWS
    (locations are entry points of the same system, essentially co-located).
  * Build hybrid coordinates: facility-grade where available, else the existing
    ZIP centroid. Label coord_source in {frs_facility, zip_centroid, none}.

OUTPUT keeps the EXACT schema of the original locations_geocoded.csv
(location_id, PWSID, State, lat, lon, coord_source) so the UNCHANGED
spatial.py -> dataset.py -> robustness.py chain can be re-run on top of it with
no code edits. The old file is backed up first.

Reads:
    data/raw/frs/NATIONAL_SINGLE.CSV
    data/interim/detection_matrix.csv        (location_id, PWSID, State)
    data/interim/locations_geocoded.csv      (existing ZIP coords -> fallback)
Writes:
    data/interim/locations_geocoded.csv           (OVERWRITTEN, hybrid)
    data/interim/locations_geocoded_ziponly.bak.csv  (backup of the old one)
    data/interim/frs_pws_points.csv               (per-PWS facility coords used)

Usage: python src/geolocate_v2.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FRS = Path("data/raw/frs/NATIONAL_SINGLE.CSV")
DET = Path("data/interim/detection_matrix.csv")
GEO = Path("data/interim/locations_geocoded.csv")
GEO_BAK = Path("data/interim/locations_geocoded_ziponly.bak.csv")
PWS_PTS = Path("data/interim/frs_pws_points.csv")

CHUNK = 300_000
ACC_LIMIT = 1000.0          # metres
EARTH_R_KM = 6371.0088


def parse_pwsid(pgm: str) -> str | None:
    for tok in str(pgm).split(","):
        tok = tok.strip()
        if tok.startswith("SFDW:"):
            # "SFDW:AL0000001" or "SFDW:AL0000001 32664" -> first whitespace-split token
            return tok.split(":", 1)[1].strip().split()[0]
    return None


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def collect_frs_sfdw() -> pd.DataFrame:
    """Return best-accuracy facility coordinate per PWSID, plus spread stats."""
    print(f"1. Scanning {FRS} for SFDW facility points (chunked)...")
    parts = []
    it = pd.read_csv(
        FRS,
        usecols=["PGM_SYS_ACRNMS", "LATITUDE83", "LONGITUDE83", "ACCURACY_VALUE"],
        dtype=str, chunksize=CHUNK,
    )
    for i, ch in enumerate(it, 1):
        mask = ch["PGM_SYS_ACRNMS"].fillna("").str.contains("SFDW", na=False)
        sub = ch[mask]
        if sub.empty:
            continue
        pw = sub["PGM_SYS_ACRNMS"].apply(parse_pwsid)
        lat = pd.to_numeric(sub["LATITUDE83"], errors="coerce")
        lon = pd.to_numeric(sub["LONGITUDE83"], errors="coerce")
        acc = pd.to_numeric(sub["ACCURACY_VALUE"], errors="coerce")
        keep = pd.DataFrame({"PWSID": pw, "lat": lat.values, "lon": lon.values, "acc": acc.values})
        keep = keep.dropna(subset=["PWSID", "lat", "lon", "acc"])
        keep = keep[keep["acc"] <= ACC_LIMIT]
        if not keep.empty:
            parts.append(keep)
        if i % 5 == 0:
            print(f"   ...chunk {i}")
    fac = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["PWSID", "lat", "lon", "acc"])
    print(f"   usable SFDW facility points (acc<= {ACC_LIMIT:.0f} m): {len(fac):,}")
    print(f"   distinct PWSIDs with >=1 usable point: {fac['PWSID'].nunique():,}")

    # within-PWS spread: max distance from the best-accuracy point to the others
    print("2. Choosing best-accuracy facility per PWS + measuring within-PWS spread")
    fac = fac.sort_values(["PWSID", "acc"])
    best = fac.groupby("PWSID", as_index=False).first()  # lowest acc value = best
    # spread diagnostic
    spreads = []
    for pid, g in fac.groupby("PWSID"):
        if len(g) > 1:
            b = g.iloc[0]
            d = haversine_km(b["lat"], b["lon"], g["lat"].values, g["lon"].values)
            spreads.append(d.max())
    if spreads:
        s = np.array(spreads)
        print(f"   PWSs with multiple facilities: {len(s):,}")
        print(f"   within-PWS spread km: median={np.median(s):.2f}, "
              f"p90={np.percentile(s,90):.2f}, max={s.max():.2f}")
        print(f"   (spread small -> best-accuracy point well represents the PWS)")
    else:
        print("   no multi-facility PWSs in the usable set")
    best = best.rename(columns={"lat": "frs_lat", "lon": "frs_lon", "acc": "frs_acc"})
    return best[["PWSID", "frs_lat", "frs_lon", "frs_acc"]]


def main() -> None:
    best = collect_frs_sfdw()

    print("3. Loading existing ZIP-centroid geocode (fallback) + PWSID map")
    geo = pd.read_csv(GEO, dtype={"location_id": str, "PWSID": str})
    print(f"   existing geocoded rows: {len(geo):,} "
          f"(coord_source: {geo['coord_source'].value_counts().to_dict()})")

    # ensure we have PWSID on every location; if missing, pull from detection_matrix
    if "PWSID" not in geo.columns or geo["PWSID"].isna().any():
        det = pd.read_csv(DET, usecols=["location_id", "PWSID"], dtype=str)
        geo = geo.drop(columns=[c for c in ["PWSID"] if c in geo.columns]).merge(
            det, on="location_id", how="left")

    print("4. Broadcasting best-accuracy PWS coordinate to that PWS's locations")
    merged = geo.merge(best, on="PWSID", how="left")
    has_frs = merged["frs_lat"].notna() & merged["frs_lon"].notna()
    print(f"   locations receiving FRS facility coords: {int(has_frs.sum()):,} "
          f"/ {len(merged):,} ({100*has_frs.mean():.1f}%)")

    # hybrid coordinates + honest source label
    merged["lat_new"] = np.where(has_frs, merged["frs_lat"], merged["lat"])
    merged["lon_new"] = np.where(has_frs, merged["frs_lon"], merged["lon"])
    prev_source = merged["coord_source"].fillna("none")
    merged["coord_source_new"] = np.where(has_frs, "frs_facility", prev_source)

    # distance moved (ZIP -> facility) for the upgraded ones, as a sanity check
    up = merged[has_frs & merged["lat"].notna() & merged["lon"].notna()]
    if len(up):
        moved = haversine_km(up["lat"], up["lon"], up["lat_new"], up["lon_new"])
        print(f"   ZIP->facility shift km (upgraded rows): "
              f"median={np.median(moved):.2f}, p90={np.percentile(moved,90):.2f}, "
              f"max={moved.max():.2f}")
        print("   (large shifts = exactly the ZIP error we were worried about)")

    out = merged[["location_id", "PWSID", "State"]].copy()
    out["lat"] = merged["lat_new"]
    out["lon"] = merged["lon_new"]
    out["coord_source"] = merged["coord_source_new"]

    print("5. Backing up old ZIP-only geocode + writing hybrid geocode")
    if not GEO_BAK.exists():
        pd.read_csv(GEO, dtype=str).to_csv(GEO_BAK, index=False)
        print(f"   backup -> {GEO_BAK}")
    else:
        print(f"   backup already exists, not overwriting -> {GEO_BAK}")
    best.to_csv(PWS_PTS, index=False)
    out.to_csv(GEO, index=False)

    print(f"\n   new coord_source breakdown: {out['coord_source'].value_counts().to_dict()}")
    print(f"   wrote hybrid -> {GEO}  ({len(out):,} rows)")
    print("\nNEXT (no code changes needed, just re-run in order):")
    print("   python src\\spatial.py      # recompute 4 distances on new coords")
    print("   python src\\dataset.py      # rebuild model_table.csv")
    print("   python src\\robustness.py   # re-check decoupling delta")
    print("   Then the key sensitivity: re-run restricted to coord_source==")
    print("   'frs_facility' only -> if delta stays small there, the decoupling")
    print("   is NOT a geocoding artifact. That's the headline robustness check.")


if __name__ == "__main__":
    main()
