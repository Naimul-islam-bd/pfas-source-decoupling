"""
Module 3b: distance-to-source features (all four source layers).

For each geocoded UCMR 5 location, compute the distance to the nearest source
of each type, kept as SEPARATE features. The methodology requires the four
source types stay distinct -- merging them would erase exactly the signal the
decoupling test depends on.

This file builds all four layers:
    dist_industrial_km   nearest PFAS-relevant industrial facility (EPA FRS)
    dist_wwtp_km         nearest sewage treatment facility (NAICS 221320, FRS)
    dist_airport_km      nearest major scheduled-service airport (AFFF proxy)
    dist_military_km     nearest DoD installation or formerly-used defense site

The military layer combines DoD MIRTA installation points and USACE FUDS
property points, pulled from their public ArcGIS REST services (cached locally
after the first run).

AIRPORT-SOURCE NOTE (flagged deviation):
    The methodology names FAA Part 139 certificated airports as the airport
    source. The official Part 139 list, however, carries no coordinates. We
    therefore operationalise it with the OurAirports public-domain dataset,
    selecting US-and-territory airports of type large/medium with scheduled
    commercial service -- the AFFF-relevant set, which closely matches the ~519
    Part 139 airports. Because the feature is nearest-airport DISTANCE, the few
    airports where the two definitions differ have negligible effect. (A
    purely-federal alternative is the FAA NASR APT file; heavier to parse.)

ZERO-HALLUCINATION NOTE on the industrial NAICS list:
    The industrial NAICS codes are NOT hard-coded here. They are read at run
    time from EPA's published "PFAS Handling Industry Sectors" workbook -- the
    same list that powers EPA's ECHO PFAS Analytic Tools "industry sectors"
    layer. Reading the codes from the authoritative file (rather than
    transcribing them) removes any transcription/hallucination risk and keeps
    the pipeline fully reproducible. Salvatore et al. (2022) Table S-1 -- 38
    codes present on >=4 of 11 regulatory/academic lists -- is the documented
    alternative, retained as a sensitivity check.

Source files:
    EPA Facility Registry Service (FRS) national single-file CSV
        facilities with lat/lon + NAICS codes
        https://ordsext.epa.gov/FLA/www3/state_files/national_single.zip
    EPA PFAS Handling Industry Sectors (XLSX)
        https://echo.epa.gov/system/files/PFASHandlingIndustrySectors-Apr2023-Pub.xlsx
    OurAirports airports.csv (public domain)
        airport coordinates + type + scheduled-service flag
        https://davidmegginson.github.io/ourairports-data/airports.csv
    DoD MIRTA + USACE FUDS (public ArcGIS REST point layers)
        https://services7.arcgis.com/n1YM8pTrFmm7L4hs/ArcGIS/rest/services
            /mirta/FeatureServer/0  (installation points)
            /fuds/FeatureServer/1   (formerly-used defense property points)

Reads:  data/interim/locations_geocoded.csv
Writes: data/interim/source_distances.csv
        columns: location_id, dist_industrial_km, dist_wwtp_km,
                 dist_airport_km, dist_military_km, is_remote_outlier

Requires: pip install scikit-learn openpyxl   (see requirements.txt)
Usage:    python src/spatial.py
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

# --- paths -----------------------------------------------------------------
RAW_DIR = Path("data/raw/frs")
LOCATIONS = Path("data/interim/locations_geocoded.csv")
OUT = Path("data/interim/source_distances.csv")

# --- source downloads ------------------------------------------------------
FRS_ZIP_URL = "https://ordsext.epa.gov/FLA/www3/state_files/national_single.zip"
PFAS_NAICS_XLSX_URL = (
    "https://echo.epa.gov/system/files/PFASHandlingIndustrySectors-Apr2023-Pub.xlsx"
)
OURAIRPORTS_CSV_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"

# Airport layer (AFFF-relevant proxy for FAA Part 139). OurAirports tags
# territories with their own ISO codes, so filtering on "US" alone would drop
# Guam / Puerto Rico / etc. -- include the territory codes so they keep a real
# airport distance instead of becoming another trans-oceanic artefact.
US_ISO = {"US", "PR", "GU", "AS", "MP", "VI"}
AIRPORT_TYPES = {"large_airport", "medium_airport"}

# Military layer: DoD MIRTA installation points + USACE FUDS property points.
# Both are public point layers in the same USACE ArcGIS org (no token needed).
# Layer indices: MIRTA points = 0, FUDS property points = 1.
_USACE_REST = "https://services7.arcgis.com/n1YM8pTrFmm7L4hs/ArcGIS/rest/services"
MIRTA_QUERY = f"{_USACE_REST}/mirta/FeatureServer/0/query"
FUDS_QUERY = f"{_USACE_REST}/fuds/FeatureServer/1/query"
MILITARY_DIR = Path("data/raw/military")

# Wastewater is its own layer; keep NAICS 221320 OUT of the industrial set so
# the two source types remain mechanically distinct.
WWTP_NAICS = {"221320"}  # Sewage Treatment Facilities (U.S. Census NAICS 2022)

EARTH_RADIUS_KM = 6371.0088

# Geolocation-quality filter on SOURCE facilities, mirroring Salvatore et al.
# (2022): drop FRS facilities whose recorded accuracy is worse than this many
# metres, or missing. This is the fix for the spurious 0 km distances -- a
# centroid-accurate facility landing on a centroid-accurate location -- and it
# trims the low-quality coordinates that otherwise inflate the source count.
ACCURACY_MAX_M = 1000.0

# Valid-coordinate gates, used ONLY to discard broken coords (e.g. 0,0). The
# latitude floor now reaches American Samoa (~-14.3 deg) and the longitude test
# spans two bands so Guam / CNMI (~+145 deg) facilities are not silently dropped.
LAT_MIN, LAT_MAX = -15.0, 72.0
LON_BANDS = ((-180.0, -64.0), (143.0, 147.0))

# A location whose nearest source is beyond this is treated as having no local
# source at all (Pacific territories with no FRS coverage, etc.) and flagged so
# Module 4 can exclude or segment them. Set high enough (1000 km) that only
# trans-oceanic "no coverage" cases trip it -- genuinely remote-but-real rural
# distances (the informative far end of the source gradient) are kept.
REMOTE_OUTLIER_KM = 1000.0

# Some EPA endpoints reject the bare python-requests User-Agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (PFAS source-decoupling research pipeline)"}


def _lon_ok(lon: pd.Series) -> pd.Series:
    """True where longitude falls in any valid US band: the main CONUS/AK/HI/AS
    band, plus the separate Guam/CNMI band on the far side of the antimeridian."""
    ok = pd.Series(False, index=lon.index)
    for lo, hi in LON_BANDS:
        ok |= lon.between(lo, hi)
    return ok


def download(url: str, dest: Path) -> Path:
    """Stream a file to disk once, with a live progress read-out; skip if it is
    already present. Large federal files have no progress otherwise, which makes
    a slow download look indistinguishable from a hang.

    To download manually instead (e.g. via a browser for a progress bar /
    resumable transfer), just drop the file at `dest` and this step is skipped.
    """
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already have {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"  downloading {dest.name} ...")
    # (connect timeout, read timeout): a genuine stall now fails loudly instead
    # of sitting silently for minutes.
    with requests.get(url, stream=True, timeout=(30, 120), headers=HEADERS) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"    {done / 1e6:7.1f} / {total / 1e6:.1f} MB "
                          f"({done / total * 100:5.1f}%)", end="\r")
                else:
                    print(f"    {done / 1e6:7.1f} MB", end="\r")
        print()
    print(f"  saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def find_col(cols, *fragments: str, exclude: tuple[str, ...] = ()) -> str | None:
    """First column whose lower-cased name contains any fragment and none of the
    excluded substrings. The same defensive 'let the file reveal its schema'
    pattern used in the earlier modules -- we never assume a header we have not
    seen the file print."""
    for c in cols:
        low = str(c).lower()
        if any(f in low for f in fragments) and not any(x in low for x in exclude):
            return c
    return None


def load_industrial_naics(xlsx_path: Path) -> set[str]:
    """Read EPA's PFAS-handling NAICS codes from the published workbook.

    The workbook's internal layout is not assumed: every sheet is scanned, the
    NAICS column is located by name, and all 4-to-6-digit codes are pulled out
    by pattern. Whatever is found is printed so the selection is auditable.
    """
    xls = pd.ExcelFile(xlsx_path)
    print(f"  workbook sheets: {xls.sheet_names}")

    codes: set[str] = set()
    used_sheet = used_col = None
    for sheet in xls.sheet_names:
        frame = xls.parse(sheet, dtype=str)
        col = find_col(
            frame.columns, "naics", exclude=("desc", "descr", "title", "name")
        )
        if col is None:
            continue
        found = (
            frame[col]
            .dropna()
            .astype(str)
            .str.findall(r"\d{4,6}")
            .explode()
            .dropna()
            .unique()
        )
        if len(found):
            codes.update(map(str, found))
            used_sheet, used_col = sheet, col

    if not codes:
        raise SystemExit(
            "No NAICS codes found in the EPA workbook -- inspect the printed "
            "sheet/column names above and adjust find_col()."
        )

    print(f"  NAICS read from sheet '{used_sheet}', column '{used_col}'")
    print(f"  distinct PFAS-handling NAICS codes: {len(codes)}")
    return codes


def build_industrial_pattern(codes: set[str]) -> str | None:
    """Build one regex that matches a facility NAICS cell if any of its 6-digit
    codes is in the industrial set. Exact 6-digit codes match literally; shorter
    codes are treated as sector/subsector prefixes (e.g. '3329' matches any
    332-9xx). Wastewater (221320) is removed -- it is its own layer."""
    industrial = {c for c in codes if c not in WWTP_NAICS}
    exact6 = sorted(c for c in industrial if len(c) == 6)
    prefixes = sorted(c for c in industrial if 0 < len(c) < 6)
    print(
        f"  industrial set: {len(exact6)} six-digit codes, "
        f"{len(prefixes)} shorter prefixes (wwtp code 221320 excluded)"
    )
    if exact6 or prefixes:
        sample = (exact6 + prefixes)[:12]
        print(f"  sample codes: {', '.join(sample)}{' ...' if len(industrial) > 12 else ''}")

    parts = list(exact6) + [p + r"\d{" + str(6 - len(p)) + r"}" for p in prefixes]
    if not parts:
        return None
    # (?<!\d) ... (?!\d) anchors each match to a whole 6-digit token so we never
    # match a code that merely appears inside a longer run of digits.
    return r"(?<!\d)(?:" + "|".join(parts) + r")(?!\d)"


def extract_frs_csv(zip_path: Path, dest_dir: Path) -> Path:
    """Extract the national single-file CSV (the largest .csv in the zip)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise FileNotFoundError("No .csv inside the FRS national zip.")
        target = max(names, key=lambda n: zf.getinfo(n).file_size)
        zf.extract(target, dest_dir)
        size = zf.getinfo(target).file_size / 1e6
        print(f"  extracted {target} ({size:.1f} MB uncompressed)")
        return dest_dir / target


def harvest_sources(csv_path: Path, ind_pattern: str | None):
    """Stream the FRS national file in chunks; return (industrial_xy, wwtp_xy)
    as float arrays of [lat, lon]. Chunked + vectorised so the multi-million-row
    file never sits in memory all at once and the NAICS test stays fast.

    A facility is kept only if its coordinates are valid AND its geolocation
    accuracy is <= ACCURACY_MAX_M (the Salvatore-style quality gate that removes
    the spurious near-zero distances)."""
    cols = list(pd.read_csv(csv_path, nrows=0, encoding="latin-1").columns)
    print(f"  FRS columns ({len(cols)}): {', '.join(map(str, cols))}")
    lat_c = find_col(cols, "latitude", "lat")
    lon_c = find_col(cols, "longitude", "long", "lon")
    naics_c = find_col(cols, "naics", exclude=("desc", "descr"))
    acc_c = find_col(cols, "accuracy")
    print(f"  using -> lat='{lat_c}', lon='{lon_c}', naics='{naics_c}', "
          f"accuracy='{acc_c}'")
    if not all([lat_c, lon_c, naics_c]):
        raise SystemExit("Could not locate lat/lon/NAICS columns -- see list above.")
    if acc_c is None:
        print("  WARNING: no accuracy column found -- quality filter SKIPPED. "
              "Inspect the printed column list above.")

    wwtp_pattern = r"(?<!\d)221320(?!\d)"
    ind_parts, wwtp_parts = [], []
    scanned = geo_total = kept_total = 0

    usecols = [c for c in (lat_c, lon_c, naics_c, acc_c) if c]
    reader = pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=str,
        encoding="latin-1",
        chunksize=250_000,
        on_bad_lines="skip",
    )
    for chunk in reader:
        scanned += len(chunk)
        lat = pd.to_numeric(chunk[lat_c], errors="coerce")
        lon = pd.to_numeric(chunk[lon_c], errors="coerce")
        geo_ok = lat.between(LAT_MIN, LAT_MAX) & _lon_ok(lon)
        geo_total += int(geo_ok.sum())

        if acc_c is not None:
            acc = pd.to_numeric(chunk[acc_c], errors="coerce")
            keep = geo_ok & acc.notna() & (acc <= ACCURACY_MAX_M)
        else:
            keep = geo_ok
        kept_total += int(keep.sum())

        naics = chunk[naics_c].fillna("")
        wwtp_mask = keep & naics.str.contains(wwtp_pattern, regex=True)
        if wwtp_mask.any():
            wwtp_parts.append(np.column_stack([lat[wwtp_mask], lon[wwtp_mask]]))
        if ind_pattern is not None:
            ind_mask = keep & naics.str.contains(ind_pattern, regex=True)
            if ind_mask.any():
                ind_parts.append(np.column_stack([lat[ind_mask], lon[ind_mask]]))

        print(
            f"    scanned {scanned:,} | kept {kept_total:,} | "
            f"industrial {sum(len(p) for p in ind_parts):,} | "
            f"wwtp {sum(len(p) for p in wwtp_parts):,}",
            end="\r",
        )

    print()
    if acc_c is not None and geo_total:
        dropped = (1 - kept_total / geo_total) * 100
        print(f"  accuracy filter (<= {ACCURACY_MAX_M:.0f} m): kept {kept_total:,} "
              f"of {geo_total:,} geo-valid facilities ({dropped:.1f}% dropped)")
    ind_xy = np.vstack(ind_parts) if ind_parts else np.empty((0, 2))
    wwtp_xy = np.vstack(wwtp_parts) if wwtp_parts else np.empty((0, 2))
    return ind_xy, wwtp_xy


def load_airports(csv_path: Path) -> np.ndarray:
    """US-and-territory airports with scheduled commercial service, a
    reproducible stand-in for the FAA Part 139 set (whose official list carries
    no coordinates). Returns an [lat, lon] array. Schema discovered, not assumed.
    """
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8", on_bad_lines="skip")
    print(f"  OurAirports columns ({len(df.columns)}): "
          f"{', '.join(map(str, df.columns))}")
    lat_c = find_col(df.columns, "latitude", "lat")
    lon_c = find_col(df.columns, "longitude", "long", "lon")
    type_c = find_col(df.columns, "type")
    sched_c = find_col(df.columns, "scheduled")
    ctry_c = find_col(df.columns, "iso_country") or find_col(
        df.columns, "country", exclude=("name",)
    )
    print(f"  using -> lat='{lat_c}', lon='{lon_c}', type='{type_c}', "
          f"scheduled='{sched_c}', country='{ctry_c}'")
    if not all([lat_c, lon_c, type_c, sched_c, ctry_c]):
        raise SystemExit("Could not locate airport columns -- see list above.")

    sched_vals = df[sched_c].astype(str).str.strip().str.lower()
    is_sched = sched_vals.isin({"yes", "1", "true"})
    in_us = df[ctry_c].isin(US_ISO)
    mask = in_us & df[type_c].isin(AIRPORT_TYPES) & is_sched
    sub = df.loc[mask]
    lat = pd.to_numeric(sub[lat_c], errors="coerce")
    lon = pd.to_numeric(sub[lon_c], errors="coerce")
    ok = lat.notna() & lon.notna()
    print(f"  major scheduled-service airports (US + territories): {int(ok.sum()):,}")
    if ok.any():
        by_type = sub.loc[ok, type_c].value_counts()
        print("    " + by_type.to_string().replace("\n", "\n    "))
    # Reference only: Part 139 also covers smaller scheduled fields, so report
    # what including small_airport would add, to sanity-check against ~519.
    incl_small = in_us & df[type_c].isin(
        AIRPORT_TYPES | {"small_airport"}
    ) & is_sched
    print(f"  (reference, incl. scheduled small_airport: {int(incl_small.sum()):,})")
    return np.column_stack([lat[ok], lon[ok]])


def _coords_ok(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Numpy version of the valid-coordinate gate used for source files."""
    lon_ok = np.zeros(len(lon), dtype=bool)
    for lo, hi in LON_BANDS:
        lon_ok |= (lon >= lo) & (lon <= hi)
    return (lat >= LAT_MIN) & (lat <= LAT_MAX) & lon_ok


def _iter_coords(coords):
    """Yield [lon, lat] leaf pairs from arbitrarily nested GeoJSON/Esri coord
    lists (handles Point, LineString, Polygon, Multi* uniformly)."""
    if (
        isinstance(coords, (list, tuple))
        and len(coords) >= 2
        and all(isinstance(v, (int, float)) for v in coords[:2])
    ):
        yield coords[0], coords[1]
    elif isinstance(coords, (list, tuple)):
        for c in coords:
            yield from _iter_coords(c)


def _geojson_points(path: Path) -> np.ndarray:
    """Read [lat, lon] from a GeoJSON file. Point -> its coordinate; any area or
    line geometry -> centroid of its vertices."""
    import json

    with open(path, encoding="utf-8") as fh:
        gj = json.load(fh)
    rows = []
    for feat in gj.get("features", []):
        pts = list(_iter_coords((feat.get("geometry") or {}).get("coordinates")))
        if pts:
            a = np.asarray(pts, dtype=float)
            rows.append((a[:, 1].mean(), a[:, 0].mean()))  # lat, lon
    return np.asarray(rows, dtype=float) if rows else np.empty((0, 2))


def fetch_arcgis_points(query_url: str, name: str, label: str, page: int = 1000):
    """Return [lat, lon] for all features in an ArcGIS FeatureServer layer.

    Resolution order, designed so a single run succeeds without hand-holding:
      1. cached CSV from a previous run        (data/raw/military/<name>.csv)
      2. a GeoJSON the user downloaded manually (data/raw/military/<name>.geojson)
      3. live, paginated ArcGIS REST query      (cached to CSV afterwards)
    Point geometry is read directly; polygon/line geometry falls back to the
    vertex centroid, so a wrong layer index still yields usable coordinates.
    """
    MILITARY_DIR.mkdir(parents=True, exist_ok=True)
    cache = MILITARY_DIR / f"{name}.csv"
    geojson = MILITARY_DIR / f"{name}.geojson"

    if cache.exists() and cache.stat().st_size > 0:
        arr = pd.read_csv(cache).to_numpy(float)
        print(f"  {label}: {len(arr):,} points (cached)")
        return arr
    if geojson.exists() and geojson.stat().st_size > 0:
        arr = _geojson_points(geojson)
        pd.DataFrame(arr, columns=["lat", "lon"]).to_csv(cache, index=False)
        print(f"  {label}: {len(arr):,} points (from {geojson.name})")
        return arr

    print(f"  {label}: querying ArcGIS REST ...")
    lat_l: list[float] = []
    lon_l: list[float] = []
    offset = 0
    pages = 0
    try:
        while True:
            params = {
                "where": "1=1",
                "outFields": "OBJECTID",
                "returnGeometry": "true",
                "outSR": "4326",
                "f": "json",
                "resultOffset": offset,
                "resultRecordCount": page,
            }
            r = requests.get(query_url, params=params, headers=HEADERS,
                             timeout=(30, 120))
            r.raise_for_status()
            feats = r.json().get("features", [])
            if not feats:
                break
            for ft in feats:
                g = ft.get("geometry") or {}
                if g.get("x") is not None and g.get("y") is not None:
                    lon_l.append(g["x"])
                    lat_l.append(g["y"])
                else:
                    pts = list(_iter_coords(g.get("rings") or g.get("paths")))
                    if pts:
                        a = np.asarray(pts, dtype=float)
                        lon_l.append(a[:, 0].mean())
                        lat_l.append(a[:, 1].mean())
            print(f"    fetched {len(lat_l):,}", end="\r")
            pages += 1
            # Stop on a partial final page; the page cap is a backstop in case a
            # service ever ignores resultOffset (would otherwise loop forever).
            if len(feats) < page or pages >= 300:
                break
            offset += len(feats)
        print()
    except Exception as exc:  # network / service problem -> tell the user how to fix
        raise SystemExit(
            f"\n  Could not auto-download {label}: {exc}\n"
            f"  Manual fix: open the dataset's ArcGIS Hub page, choose\n"
            f"  Download -> GeoJSON, save the file as:\n      {geojson}\n"
            f"  then re-run -- the script will read it and continue."
        )

    arr = np.column_stack([lat_l, lon_l]) if lat_l else np.empty((0, 2))
    pd.DataFrame(arr, columns=["lat", "lon"]).to_csv(cache, index=False)
    print(f"  {label}: {len(arr):,} points")
    return arr


def load_military() -> np.ndarray:
    """Combined DoD MIRTA + USACE FUDS source points as an [lat, lon] array."""
    mirta = fetch_arcgis_points(MIRTA_QUERY, "mirta_points", "MIRTA installations")
    fuds = fetch_arcgis_points(FUDS_QUERY, "fuds_points", "FUDS properties")
    parts = [a for a in (mirta, fuds) if len(a)]
    mil = np.vstack(parts) if parts else np.empty((0, 2))
    if len(mil):
        mil = np.unique(mil, axis=0)  # drop any exact duplicate coordinates
        ok = _coords_ok(mil[:, 0], mil[:, 1])
        if int((~ok).sum()):
            print(f"  dropped {int((~ok).sum()):,} points with out-of-range coords")
        mil = mil[ok]
    print(f"  military sources (MIRTA + FUDS): {len(mil):,}")
    return mil


def nearest_km(source_xy: np.ndarray, loc_xy: np.ndarray) -> np.ndarray:
    """Great-circle distance (km) from each location to its nearest source, via a
    haversine BallTree on coordinates in radians."""
    if len(source_xy) == 0:
        return np.full(len(loc_xy), np.nan)
    tree = BallTree(np.radians(source_xy), metric="haversine")
    ang, _ = tree.query(np.radians(loc_xy), k=1)
    return ang[:, 0] * EARTH_RADIUS_KM


def summarise(name: str, d: np.ndarray) -> None:
    finite = d[np.isfinite(d)]
    if not len(finite):
        print(f"  {name}: no distances computed")
        return
    print(
        f"  {name}: min {finite.min():.1f} | median {np.median(finite):.1f} "
        f"| mean {finite.mean():.1f} | max {finite.max():.1f} km "
        f"({len(finite):,} located)"
    )


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Downloading source files")
    frs_zip = download(FRS_ZIP_URL, RAW_DIR / "frs_national_single.zip")
    naics_xlsx = download(PFAS_NAICS_XLSX_URL, RAW_DIR / "pfas_industry_sectors.xlsx")
    airports_csv = download(OURAIRPORTS_CSV_URL, RAW_DIR / "ourairports.csv")

    print("\n2. Building the PFAS-relevant industrial NAICS set (EPA workbook)")
    naics_codes = load_industrial_naics(naics_xlsx)
    ind_pattern = build_industrial_pattern(naics_codes)

    print("\n3. Extracting the FRS national single file")
    frs_csv = extract_frs_csv(frs_zip, RAW_DIR)

    print("\n4. Scanning FRS facilities for industrial + wastewater sources")
    ind_xy, wwtp_xy = harvest_sources(frs_csv, ind_pattern)
    print(f"  industrial source facilities: {len(ind_xy):,}")
    print(f"  wastewater source facilities: {len(wwtp_xy):,}")

    print("\n5. Loading airport sources (OurAirports)")
    airport_xy = load_airports(airports_csv)

    print("\n6. Loading military sources (DoD MIRTA + USACE FUDS)")
    military_xy = load_military()

    print("\n7. Loading geocoded UCMR 5 locations")
    loc = pd.read_csv(LOCATIONS, dtype={"location_id": str})
    placed = loc.dropna(subset=["lat", "lon"]).copy()
    print(f"  located: {len(placed):,} / {len(loc):,}")
    loc_xy = placed[["lat", "lon"]].to_numpy(float)

    print("\n8. Nearest-source distances (haversine BallTree)")
    placed["dist_industrial_km"] = nearest_km(ind_xy, loc_xy)
    placed["dist_wwtp_km"] = nearest_km(wwtp_xy, loc_xy)
    placed["dist_airport_km"] = nearest_km(airport_xy, loc_xy)
    placed["dist_military_km"] = nearest_km(military_xy, loc_xy)
    dist_cols = [
        "dist_industrial_km", "dist_wwtp_km", "dist_airport_km", "dist_military_km"
    ]
    for c in dist_cols:
        summarise(c, placed[c].to_numpy())

    # Diagnostic: how many locations still sit essentially on top of a source
    # after the accuracy filter? This should be a small fraction now, not the
    # artefact-driven pile-up the unfiltered run produced.
    near0 = int((placed["dist_industrial_km"] < 0.1).sum())
    print(f"  within 100 m of an industrial source: {near0:,} locations")

    print("\n9. Flagging remote outliers (no local source coverage)")
    # Worst layer per location: if a location's nearest source of ANY type is
    # absurdly far, its source picture is incomplete and gets flagged.
    far = placed[dist_cols].max(axis=1)
    print("  distance bands (worst layer per location):")
    for cut in (100, 500, 1000, 2000):
        print(f"    a source > {cut:>4} km away: {int((far > cut).sum()):,} locations")
    placed["is_remote_outlier"] = far > REMOTE_OUTLIER_KM
    n_out = int(placed["is_remote_outlier"].sum())
    print(f"  flagged is_remote_outlier (> {REMOTE_OUTLIER_KM:.0f} km): {n_out:,}")
    if n_out and "State" in placed.columns:
        breakdown = placed.loc[placed["is_remote_outlier"], "State"].value_counts()
        print("  by state/territory:")
        print("    " + breakdown.to_string().replace("\n", "\n    "))

    out = placed[["location_id", *dist_cols, "is_remote_outlier"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("\nsample:")
    print(out.head().to_string(index=False))
    print(f"\nsaved -> {OUT}  ({len(out):,} locations, 4 of 4 source layers)")
    print("Module 3b complete -- all four distance-to-source layers built.")


if __name__ == "__main__":
    main()
