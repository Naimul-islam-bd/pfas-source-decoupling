"""
Probe: confirm how CONCENTRATION is stored in the UCMR5 raw file, so we can
rebuild Module 2 to carry real concentrations (ppt) instead of only binary
detection. Verify-first: no assumptions about column names.

Prints:
  * the raw file's columns (so we see the exact concentration / sign / MRL /
    units / analyte field names)
  * a few sample rows
  * for the concentration-like column: how many rows are numeric vs blank,
    the value distribution, and the distinct 'sign' values (= vs <) that mark
    detects vs non-detects
  * distinct analyte names + how many rows each (to confirm the 29 PFAS)

This tells us whether a concentration rebuild is feasible and exactly which
columns to use. Nothing is written; this is read-only reconnaissance.

Adjust RAW if your UCMR5 file lives elsewhere / has a different name.

Usage: python src/probe_ucmr5.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Handoff says: data/raw/ucmr5/UCMR5_All.txt  (tab-delimited, ~309 MB)
RAW = Path("data/raw/ucmr5/UCMR5_All.txt")
CHUNK = 200_000


def main() -> None:
    if not RAW.exists():
        print(f"NOT FOUND: {RAW}")
        print("List what's actually in data/raw/ucmr5/ and tell me the filename.")
        p = Path("data/raw/ucmr5")
        if p.exists():
            for f in p.iterdir():
                print("   ", f.name, f"{f.stat().st_size/1e6:.1f} MB")
        return

    print(f"Reading header of {RAW}")
    head = pd.read_csv(RAW, sep="\t", dtype=str, nrows=8, encoding="latin-1")
    print(f"\nN COLS: {len(head.columns)}")
    print("COLUMNS:")
    for c in head.columns:
        print(f"   {c}")
    print("\nFIRST ROWS:")
    with pd.option_context("display.max_colwidth", 22, "display.width", 220):
        print(head.to_string(index=False))

    # Heuristic guesses for the key columns (we confirm, not assume)
    cols = list(head.columns)
    def find(cands):
        for cand in cands:
            for c in cols:
                if cand.lower() in c.lower():
                    return c
        return None

    col_val = find(["AnalyticalResultValue", "ResultValue", "Result", "Value", "Concentration"])
    col_sign = find(["Sign", "ResultsSign", "Qualifier"])
    col_mrl = find(["MRL", "MinimumReporting", "ReportingLevel"])
    col_unit = find(["Units", "Unit"])
    col_analyte = find(["Contaminant", "Analyte", "Chemical"])
    col_pws = find(["PWSID", "PWS_ID"])

    print("\nGUESSED KEY COLUMNS (confirm these look right):")
    for name, c in [("concentration value", col_val), ("sign (=/<)", col_sign),
                    ("MRL", col_mrl), ("units", col_unit),
                    ("analyte name", col_analyte), ("PWSID", col_pws)]:
        print(f"   {name:22} -> {c}")

    if not col_val or not col_analyte:
        print("\nCouldn't confidently find value/analyte columns -- paste the COLUMNS")
        print("list back and I'll map them exactly.")
        return

    print(f"\nScanning full file for '{col_val}' distribution + analyte counts...")
    usecols = [c for c in [col_val, col_sign, col_analyte, col_unit] if c]
    n = 0
    n_numeric = 0
    n_blank = 0
    vals = []
    analyte_counts = {}
    sign_counts = {}
    unit_counts = {}

    it = pd.read_csv(RAW, sep="\t", dtype=str, usecols=usecols, chunksize=CHUNK, encoding="latin-1")
    for i, ch in enumerate(it, 1):
        n += len(ch)
        v = pd.to_numeric(ch[col_val], errors="coerce")
        n_numeric += int(v.notna().sum())
        n_blank += int(ch[col_val].isna().sum() | (ch[col_val] == "").sum()
                       if False else v.isna().sum())
        vals.extend(v.dropna().tolist()[:2000])
        for a, cnt in ch[col_analyte].value_counts().items():
            analyte_counts[a] = analyte_counts.get(a, 0) + int(cnt)
        if col_sign:
            for s, cnt in ch[col_sign].fillna("(blank)").value_counts().items():
                sign_counts[s] = sign_counts.get(s, 0) + int(cnt)
        if col_unit:
            for u, cnt in ch[col_unit].fillna("(blank)").value_counts().items():
                unit_counts[u] = unit_counts.get(u, 0) + int(cnt)
        if i % 5 == 0:
            print(f"   ...{n:,} rows")

    print(f"\nTotal rows: {n:,}")
    print(f"  numeric '{col_val}': {n_numeric:,}  ({100*n_numeric/max(n,1):.1f}%)")
    print(f"  non-numeric/blank (non-detects): {n - n_numeric:,}")
    if vals:
        a = np.array(vals)
        print(f"  value sample (n={len(a):,}): min={a.min():.4g}, median={np.median(a):.4g}, "
              f"p95={np.percentile(a,95):.4g}, max={a.max():.4g}")

    if sign_counts:
        print(f"\n  '{col_sign}' values (detect vs non-detect marker):")
        for s, c in sorted(sign_counts.items(), key=lambda x: -x[1]):
            print(f"     {s!r:12} {c:,}")
    if unit_counts:
        print(f"\n  '{col_unit}' values:")
        for u, c in sorted(unit_counts.items(), key=lambda x: -x[1]):
            print(f"     {u!r:12} {c:,}")

    print(f"\n  distinct analytes in '{col_analyte}': {len(analyte_counts)}")
    print("  (rows per analyte, top 35):")
    for a, c in sorted(analyte_counts.items(), key=lambda x: -x[1])[:35]:
        print(f"     {str(a):<22} {c:,}")

    print("\n>> If there's a real numeric value column + a sign field (= / <) and")
    print("   units in ng/L or ug/L, a concentration rebuild is fully feasible.")
    print("   Send me this output and I'll write the Module-2 concentration build.")


if __name__ == "__main__":
    main()
