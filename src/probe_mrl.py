"""
Audit probe: settle two data-integrity questions raised in the cross-check,
before the write-up.

Q1 (MRL heterogeneity): is the Minimum Reporting Level (MRL) uniform per
    analyte across the whole UCMR5 program, or does it vary by lab/system? If
    uniform, detection = (value > MRL) is a clean, comparable binary and the
    "MRL heterogeneity" concern is dismissed. If it varies, detection frequency
    partly reflects reporting sensitivity and must be caveated (or modelled).

Q2 (sampling-frequency confound): how many samples per location? If some
    systems are sampled far more than others, "ever detected" is mechanically
    higher for heavily-sampled systems. We quantify the spread so we know how
    big the concern is (and whether to add a sample-count covariate).

Read-only; writes nothing. Uses latin-1 (micro sign in ug/L).

Usage: python src/probe_mrl.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/ucmr5/UCMR5_All.txt")
CHUNK = 200_000

ANALYTE = "Contaminant"
MRL = "MRL"
UNITS = "Units"
PWSID = "PWSID"
SPID = "SamplePointID"
SIGN = "AnalyticalResultsSign"
NON_PFAS = {"lithium"}


def main() -> None:
    print(f"Scanning {RAW} for MRL uniformity + sampling frequency")
    mrl_by_analyte = defaultdict(set)
    units_by_analyte = defaultdict(set)
    samples_per_loc = defaultdict(int)      # (pwsid, spid, analyte) events -> count
    loc_sample_events = defaultdict(int)    # (pwsid, spid) -> number of analyte rows

    usecols = [ANALYTE, MRL, UNITS, PWSID, SPID, SIGN]
    n = 0
    it = pd.read_csv(RAW, sep="\t", dtype=str, usecols=usecols,
                     chunksize=CHUNK, encoding="latin-1")
    for i, ch in enumerate(it, 1):
        n += len(ch)
        ch = ch[~ch[ANALYTE].str.lower().isin(NON_PFAS)]
        for a, m in zip(ch[ANALYTE], ch[MRL]):
            mrl_by_analyte[a].add(m)
        for a, u in zip(ch[ANALYTE], ch[UNITS]):
            units_by_analyte[a].add(u)
        # sampling frequency: distinct sample events per location
        loc = ch[PWSID].astype(str) + "|" + ch[SPID].astype(str)
        for l in loc:
            loc_sample_events[l] += 1
        if i % 5 == 0:
            print(f"   ...{n:,} rows")

    print(f"\nTotal PFAS rows: {n:,}")

    print("\n=== Q1: MRL uniformity per analyte ===")
    n_uniform = 0
    n_vary = 0
    for a in sorted(mrl_by_analyte):
        vals = sorted(mrl_by_analyte[a])
        uni = len(vals) == 1
        n_uniform += uni
        n_vary += (not uni)
        tag = "uniform" if uni else f"VARIES ({len(vals)} values)"
        show = vals if len(vals) <= 6 else vals[:6] + ["..."]
        print(f"  {str(a):<16} {tag:<20} {show}")
    print(f"\n  analytes with a single MRL: {n_uniform} | with multiple: {n_vary}")
    if n_vary == 0:
        print("  >> CLEAN: MRL is uniform per analyte program-wide. The MRL-")
        print("     heterogeneity concern is dismissed; detection is comparable.")
    else:
        print("  >> MRL varies for some analytes. We'll caveat detection or")
        print("     restrict to the dominant MRL. Send this list and I'll handle it.")

    print("\n=== units check (should all be ug/L) ===")
    odd = {a: v for a, v in units_by_analyte.items() if v != {"\xb5g/L"} and v != {"ug/L"}}
    if not odd:
        print("  all analytes reported in a single consistent unit.")
    else:
        print("  NON-UNIFORM UNITS (inspect):")
        for a, v in list(odd.items())[:10]:
            print(f"     {a}: {v}")

    print("\n=== Q2: sampling frequency per location ===")
    counts = np.array(list(loc_sample_events.values()))
    # each location has ~29 analyte rows per sample event; convert to events
    # (approx): events ~ rows / (#analytes measured per event). Report raw rows too.
    print(f"  distinct locations: {len(counts):,}")
    print(f"  analyte-rows per location: median={int(np.median(counts))}, "
          f"p90={int(np.percentile(counts,90))}, max={int(counts.max())}")
    print("  (a single sampling event covers ~29 analytes, so divide by ~29 for")
    print("   sampling EVENTS. Large spread here => sampling-frequency covariate")
    print("   is worth adding; small/uniform => concern is minor.)")
    # crude event estimate
    ev = counts / 29.0
    print(f"  approx sampling EVENTS per location: median={np.median(ev):.1f}, "
          f"p90={np.percentile(ev,90):.1f}, max={ev.max():.1f}")


if __name__ == "__main__":
    main()
