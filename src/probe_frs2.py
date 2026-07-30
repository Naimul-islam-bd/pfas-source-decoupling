"""
Probe v2: confirm the SFDW (drinking-water) program rows in FRS carry a PWSID
we can parse and usable coordinates -- and crucially, estimate how many of our
UCMR5 PWSIDs will actually match. This decides whether the geocoding upgrade is
worth it before we write the full pipeline.

Prints:
  * a few raw SFDW rows (so we SEE the PGM_SYS_ACRNMS / PWSID format)
  * how many SFDW rows have a parseable PWSID and non-null coordinates
  * ACCURACY_VALUE distribution (how many survive the <=1000 m Salvatore filter)
  * overlap with the UCMR5 PWSIDs in our detection_matrix.csv (the join rate)

Usage: python src/probe_frs2.py
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import pandas as pd

FRS = Path("data/raw/frs/NATIONAL_SINGLE.CSV")
DET = Path("data/interim/detection_matrix.csv")
CHUNK = 300_000
ACC_LIMIT = 1000.0  # metres, Salvatore-style filter (same as Module 3b)


def parse_pwsid(pgm: str) -> str | None:
    for tok in str(pgm).split(","):
        tok = tok.strip()
        if tok.startswith("SFDW:"):
            return tok.split(":", 1)[1].strip()
    return None


def main() -> None:
    # our UCMR5 PWSIDs
    det = pd.read_csv(DET, usecols=["PWSID"], dtype=str)
    ucmr_pwsids = set(det["PWSID"].dropna().str.strip())
    print(f"UCMR5 distinct PWSIDs: {len(ucmr_pwsids):,}")

    print(f"\nScanning {FRS} for SFDW rows...")
    examples = []
    n_sfdw = 0
    n_pwsid = 0
    n_coord = 0
    n_acc_ok = 0
    matched_pwsids = set()
    acc_vals = []

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
        n_sfdw += len(sub)

        pw = sub["PGM_SYS_ACRNMS"].apply(parse_pwsid)
        lat = pd.to_numeric(sub["LATITUDE83"], errors="coerce")
        lon = pd.to_numeric(sub["LONGITUDE83"], errors="coerce")
        acc = pd.to_numeric(sub["ACCURACY_VALUE"], errors="coerce")

        n_pwsid += int(pw.notna().sum())
        n_coord += int((lat.notna() & lon.notna()).sum())
        n_acc_ok += int((acc <= ACC_LIMIT).sum())
        acc_vals.extend(acc.dropna().tolist()[:5000])

        good = pw.notna() & lat.notna() & lon.notna()
        for p in pw[good]:
            if p in ucmr_pwsids:
                matched_pwsids.add(p)

        if len(examples) < 8:
            for _, row in sub.head(8 - len(examples)).iterrows():
                examples.append(row)

        if i % 5 == 0:
            print(f"  ...chunk {i}, SFDW so far {n_sfdw:,}")

    print(f"\nSFDW rows total:                 {n_sfdw:,}")
    print(f"  with parseable PWSID:          {n_pwsid:,}")
    print(f"  with non-null coordinates:     {n_coord:,}")
    print(f"  with ACCURACY_VALUE <= {ACC_LIMIT:.0f} m:  {n_acc_ok:,}")

    if acc_vals:
        a = np.array(acc_vals)
        print(f"\nACCURACY_VALUE sample (n={len(a):,}): "
              f"median={np.median(a):.0f} m, "
              f"pct<=1000m={100*np.mean(a<=1000):.1f}%, "
              f"pct<=100m={100*np.mean(a<=100):.1f}%")

    print(f"\nUCMR5 PWSIDs matched to an FRS SFDW facility w/ coords: "
          f"{len(matched_pwsids):,} / {len(ucmr_pwsids):,} "
          f"({100*len(matched_pwsids)/max(len(ucmr_pwsids),1):.1f}%)")

    print("\nExample SFDW rows (confirm PWSID format + coords):")
    ex = pd.DataFrame(examples)
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(ex.to_string(index=False))

    print("\n>> If match rate is high (say >70%) and many survive the accuracy")
    print("   filter, the facility-level re-geocode is worth it. If low, we keep")
    print("   ZIP centroids and report the geocoding limitation honestly instead.")


if __name__ == "__main__":
    main()
