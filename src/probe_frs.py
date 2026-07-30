"""
One-off probe: inventory the program-system acronyms in the FRS national file
so we know EXACTLY how SDWIS (public water system) rows are tagged before
writing the real geocoder. Prints the top acronyms, the SDWIS count, and a few
raw SDWIS rows so we can see the PWSID format inside PGM_SYS_ACRNMS.

Chunked read -> safe on the 2.67 GB file. Prints progress so it doesn't look
frozen.

Usage: python src/probe_frs.py
"""

from __future__ import annotations

import collections
from pathlib import Path

import pandas as pd

CSV = Path("data/raw/frs/NATIONAL_SINGLE.CSV")
CHUNK = 300_000


def main() -> None:
    print(f"Scanning {CSV} in {CHUNK:,}-row chunks (this takes a few minutes)...")
    acr = collections.Counter()
    n = 0
    sdwis_examples = []

    it = pd.read_csv(
        CSV,
        usecols=["PGM_SYS_ACRNMS", "LATITUDE83", "LONGITUDE83",
                 "ACCURACY_VALUE", "SITE_TYPE_NAME"],
        dtype=str, chunksize=CHUNK,
    )
    for i, ch in enumerate(it, 1):
        n += len(ch)
        pgm = ch["PGM_SYS_ACRNMS"].fillna("")
        for v in pgm:
            if not v:
                continue
            for tok in v.split(","):
                acr[tok.split(":")[0].strip()] += 1
        # collect a few example SDWIS rows (token before ':' == SDWIS)
        if len(sdwis_examples) < 10:
            mask = pgm.str.contains("SDWIS", na=False)
            for _, row in ch[mask].head(10 - len(sdwis_examples)).iterrows():
                sdwis_examples.append(row)
        if i % 5 == 0:
            print(f"  ...{n:,} rows scanned")

    print(f"\nTotal rows scanned: {n:,}")
    print(f"Rows/tokens with SDWIS acronym: {acr.get('SDWIS', 0):,}")
    print("\nTop 25 program acronyms in FRS:")
    for k, v in acr.most_common(25):
        print(f"  {k:20} {v:,}")

    print("\nExample SDWIS rows (to read the PWSID format inside PGM_SYS_ACRNMS):")
    if sdwis_examples:
        ex = pd.DataFrame(sdwis_examples)
        with pd.option_context("display.max_colwidth", 80, "display.width", 200):
            print(ex.to_string(index=False))
    else:
        print("  (none found -- SDWIS may be tagged differently; check the top list above)")


if __name__ == "__main__":
    main()
