"""
Module 2 (concentration build): extract real PFAS concentrations (ug/L) from
the UCMR5 raw file, per location x analyte, and report whether a two-stage
(hurdle) concentration model is feasible.

CONTEXT: the detection matrix (Module 2 original) is binary. The raw file also
carries the numeric result (AnalyticalResultValue) with a sign flag
(AnalyticalResultsSign: '=' detect, '<' non-detect below MRL). Only ~2.9% of
measurements are detects, so concentration is heavily left-censored. The honest
plan is a HURDLE model:
    Stage 1 -- detection (where is PFAS present)  [already done]
    Stage 2 -- among DETECTED samples, what drives concentration
This script builds the Stage-2 material and, crucially, reports whether enough
detected samples are spread across enough locations/PWS for Stage 2 to have
statistical power. We decide the modelling approach AFTER seeing that.

What it does:
  * read UCMR5_All.txt (latin-1, tab, chunked) -- the micro-sign in 'ug/L'
    forces latin-1
  * drop lithium (not PFAS); keep the 29 PFAS analytes
  * location_id = PWSID + '_' + SamplePointID  (matches detection_matrix)
  * per location x analyte: max detected concentration (NaN if never detected);
    multiple sample events per location collapse to the max detect
  * total_pfas_conc = sum of per-analyte max detects at a location (a system's
    detected PFAS load)
  * VERIFY the constructed location_id overlaps the existing detection_matrix
    location_id (guards against a wrong id format silently breaking joins)
  * REPORT Stage-2 power: detects per analyte, locations/PWS with >=1 detect,
    total_pfas_conc distribution

Reads:
    data/raw/ucmr5/UCMR5_All.txt
    data/interim/detection_matrix.csv     (for location_id overlap check)
Writes:
    data/interim/concentration_matrix.csv   location_id + per-analyte max detect
                                             (ug/L) + total_pfas_conc + n_detected

Usage: python src/dataset_concentration.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/ucmr5/UCMR5_All.txt")
DET = Path("data/interim/detection_matrix.csv")
OUT = Path("data/interim/concentration_matrix.csv")
CHUNK = 200_000

VAL = "AnalyticalResultValue"
SIGN = "AnalyticalResultsSign"
ANALYTE = "Contaminant"
PWSID = "PWSID"
SPID = "SamplePointID"

NON_PFAS = {"lithium"}


def main() -> None:
    print("1. Streaming UCMR5 raw, keeping DETECTED PFAS rows only")
    usecols = [PWSID, SPID, ANALYTE, SIGN, VAL]
    detect_parts = []
    n_rows = 0
    n_detect = 0
    it = pd.read_csv(RAW, sep="\t", dtype=str, usecols=usecols,
                     chunksize=CHUNK, encoding="latin-1")
    for i, ch in enumerate(it, 1):
        n_rows += len(ch)
        ch = ch[~ch[ANALYTE].str.lower().isin(NON_PFAS)]
        det = ch[ch[SIGN] == "="].copy()
        det["val"] = pd.to_numeric(det[VAL], errors="coerce")
        det = det.dropna(subset=["val"])
        n_detect += len(det)
        det["location_id"] = det[PWSID].astype(str) + "_" + det[SPID].astype(str)
        detect_parts.append(det[["location_id", PWSID, ANALYTE, "val"]])
        if i % 5 == 0:
            print(f"   ...{n_rows:,} rows scanned, {n_detect:,} PFAS detects kept")

    det_all = pd.concat(detect_parts, ignore_index=True)
    print(f"   total rows: {n_rows:,} | PFAS detect rows: {len(det_all):,}")

    print("\n2. Collapsing to location x analyte (max detected concentration)")
    # multiple sample events / dates -> take the max detected value
    wide = (det_all.groupby(["location_id", PWSID, ANALYTE])["val"]
            .max().unstack(ANALYTE))
    wide = wide.reset_index()
    analyte_cols = [c for c in wide.columns if c not in ("location_id", PWSID)]
    wide["n_detected"] = wide[analyte_cols].notna().sum(axis=1).astype(int)
    wide["total_pfas_conc"] = wide[analyte_cols].sum(axis=1, skipna=True)
    print(f"   locations with >=1 PFAS detect: {len(wide):,}")
    print(f"   analytes present as columns: {len(analyte_cols)}")

    print("\n3. VERIFY location_id overlap with existing detection_matrix")
    det_mat = pd.read_csv(DET, usecols=["location_id"], dtype=str)
    dm_ids = set(det_mat["location_id"])
    conc_ids = set(wide["location_id"])
    inter = conc_ids & dm_ids
    print(f"   detection_matrix locations: {len(dm_ids):,}")
    print(f"   concentration locations:    {len(conc_ids):,}")
    print(f"   overlap: {len(inter):,} "
          f"({100*len(inter)/max(len(conc_ids),1):.1f}% of concentration locs "
          f"match a detection loc)")
    if len(inter) / max(len(conc_ids), 1) < 0.9:
        print("   !! LOW OVERLAP -- location_id format may be wrong. Inspect before")
        print("      trusting downstream joins. Sample concentration ids:")
        print("      ", list(conc_ids)[:5])
        print("      Sample detection ids:")
        print("      ", list(dm_ids)[:5])
    else:
        print("   OK: id format matches, safe to join to the model table later.")

    print("\n4. Writing concentration matrix")
    front = ["location_id", PWSID, "n_detected", "total_pfas_conc"]
    wide = wide[front + analyte_cols]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT, index=False)
    print(f"   saved -> {OUT}  ({wide.shape[0]:,} rows x {wide.shape[1]} cols)")

    print("\n5. STAGE-2 POWER REPORT (decides the modelling approach)")
    print(f"   locations with >=1 detect (Stage-2 n): {len(wide):,}")
    print(f"   distinct PWS with >=1 detect:          {wide[PWSID].nunique():,}")
    tp = wide["total_pfas_conc"]
    print(f"   total_pfas_conc (ug/L): median={tp.median():.4g}, "
          f"p90={tp.quantile(0.9):.4g}, max={tp.max():.4g}")
    print("\n   detects per analyte (locations where that analyte is detected):")
    per = wide[analyte_cols].notna().sum().sort_values(ascending=False)
    for name, cnt in per.items():
        print(f"     {str(name):<16} {int(cnt):,}")

    print("\n>> READ: if several key analytes (PFOA, PFOS, PFPeA, PFHxA, PFBA) each")
    print("   have >~1000 detected locations spread over many PWS, Stage-2")
    print("   concentration models (per analyte, or on total_pfas_conc) have")
    print("   real power. Thin analytes (few hundred detects) can still be")
    print("   modelled on total_pfas_conc pooled. Send this output and I'll write")
    print("   the Stage-2 concentration decoupling test to match the Stage-1 one.")


if __name__ == "__main__":
    main()
