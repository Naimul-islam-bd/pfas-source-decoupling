"""
Module 3a: geolocate every UCMR 5 sampling location.

Decision note: precise per-system coordinates via EPA's ECHO REST API proved
unavailable (the SDWA get_facility_info endpoint returns server errors). We
therefore geolocate each public water system to the centroid of the ZIP
code(s) it serves, using the UCMR5_ZIPCodes.txt file that ships with the
occurrence data plus offline ZIP-centroid lookups (pgeocode). Coarser than a
point coordinate, but reliable and fully reproducible; precise FRS coordinates
can be layered in later as a sensitivity check.

Reads:  data/interim/detection_matrix.csv
        data/raw/ucmr5/UCMR5_ZIPCodes.txt
Writes: data/interim/locations_geocoded.csv

Requires: pip install pgeocode
"""

from pathlib import Path

import pandas as pd
import pgeocode


MATRIX = Path("data/interim/detection_matrix.csv")
ZIPFILE = Path("data/raw/ucmr5/UCMR5_ZIPCodes.txt")
OUT = Path("data/interim/locations_geocoded.csv")


def find_col(cols, *fragments: str):
    for c in cols:
        if any(f in c.lower() for f in fragments):
            return c
    return None


def main() -> None:
    df = pd.read_csv(MATRIX, dtype={"PWSID": str})
    df["PWSID"] = df["PWSID"].str.strip()

    zips = pd.read_csv(ZIPFILE, sep="\t", dtype=str, encoding="latin-1")
    print("UCMR5_ZIPCodes columns:", list(zips.columns))

    pid = find_col(zips.columns, "pwsid", "pws")
    zc = find_col(zips.columns, "zip")
    if not (pid and zc):
        raise SystemExit("Could not find PWSID / ZIP columns -- see the list above.")
    print(f"using PWSID column = '{pid}', ZIP column = '{zc}'")

    zips = zips[[pid, zc]].rename(columns={pid: "PWSID", zc: "ZIP"})
    zips["PWSID"] = zips["PWSID"].str.strip()
    zips["ZIP"] = zips["ZIP"].str.extract(r"(\d{5})")[0]
    zips = zips.dropna(subset=["ZIP"]).drop_duplicates()
    print(f"PWSID-ZIP pairs: {len(zips):,}  |  "
          f"unique PWSIDs with a ZIP: {zips['PWSID'].nunique():,}")

    # Resolve every served ZIP to a centroid (offline lookup).
    nomi = pgeocode.Nominatim("us")
    uniq = sorted(zips["ZIP"].unique())
    cen = nomi.query_postal_code(uniq)[["postal_code", "latitude", "longitude"]]
    cen.columns = ["ZIP", "lat", "lon"]
    cen["ZIP"] = cen["ZIP"].astype(str).str.zfill(5)
    cen = cen.dropna(subset=["lat", "lon"])
    print(f"ZIPs resolved to coordinates: {len(cen):,} / {len(uniq):,}")

    # Each system -> mean centroid of the ZIP(s) it serves.
    paired = zips.merge(cen, on="ZIP", how="inner")
    sys_xy = paired.groupby("PWSID")[["lat", "lon"]].mean().reset_index()

    geo = df[["location_id", "PWSID", "State"]].merge(sys_xy, on="PWSID", how="left")
    geo["coord_source"] = geo["lat"].notna().map({True: "zip_centroid", False: "none"})

    placed = geo["lat"].notna()
    print(f"\nlocated: {placed.sum():,} / {len(geo):,} ({placed.mean()*100:.1f}%)")
    print("\ncoordinate source breakdown:")
    print(geo["coord_source"].value_counts().to_string())
    print("\nsample:")
    print(geo.dropna(subset=["lat"]).head().to_string(index=False))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    geo.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
