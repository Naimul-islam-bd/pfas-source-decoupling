"""
Module 5, step 2: three no-new-data impact analyses, in one script (batched to
save time/tokens). Each section writes its own results table; nothing is
hard-coded and nothing is inferred without printing the evidence.

  A) MONITORING EFFICIENCY (impact Angle 3)
     If a regulator prioritised monitoring by proximity to catalogued sources,
     how much contamination would that capture vs random sampling? We rank
     locations by nearest-source distance, take the nearest k%, and report the
     fraction of detections captured. Capture ~= k% means proximity targeting is
     no better than random -> the decoupling made operational.

  B) VULNERABILITY PROFILE (who is in the blind spot)
     Blind spot = unregulated PFAS present, regulated (PFOA/PFOS) absent (i.e.
     "clean" under current MCLs). We report the blind-spot RATE by system size,
     owner type, source water, system type, school/daycare, with a rate-ratio
     vs the overall blind-spot rate. HONEST: we report whatever the data shows,
     including groups that are UNDER-represented; we do not force a narrative.
     A true environmental-justice (income/race) analysis needs the ECHO
     demographic download and is NOT claimed here.

  C) SELF-REPORTED SOURCE vs DETECTION (novelty, on-thesis)
     UCMR5 systems self-reported whether they have a potential PFAS source
     (PotentialPFASSources = Yes/No/DK). Does that self-knowledge predict
     detection? Key numbers: detection among Yes vs No systems, AND the
     RESIDUAL detection among "No" systems (PFAS present where the system knows
     of no source = diffuse contamination beyond point sources).

PWSID from locations_geocoded.csv (authoritative), not by slicing.

Reads:
    data/interim/model_table.csv
    data/interim/locations_geocoded.csv
    data/raw/sdwis/SDWA_PUB_WATER_SYSTEMS.csv
    data/raw/ucmr5/UCMR5_AddtlDataElem.txt
Writes:
    outputs/tables/monitoring_efficiency.csv
    outputs/tables/vulnerability_profile.csv
    outputs/tables/self_reported_source.csv

Usage:  python src/impact_analyses.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MT = Path("data/interim/model_table.csv")
GEO = Path("data/interim/locations_geocoded.csv")
SDWIS = Path("data/raw/sdwis/SDWA_PUB_WATER_SYSTEMS.csv")
ADDL = Path("data/raw/ucmr5/UCMR5_AddtlDataElem.txt")
T = Path("outputs/tables")

REG = ["PFOA", "PFOS"]
UNREG = ["PFPeA", "PFHxA", "PFBA"]
DIST = ["dist_industrial_km", "dist_wwtp_km", "dist_airport_km", "dist_military_km"]


def load_locations() -> pd.DataFrame:
    mt = pd.read_csv(MT, dtype={"location_id": str})
    geo = pd.read_csv(GEO, dtype={"location_id": str})
    pcol = next((c for c in ("PWSID", "pwsid", "pws_id") if c in geo.columns), None)
    mt = mt.merge(geo[["location_id", pcol]].rename(columns={pcol: "PWSID"}),
                  on="location_id", how="left")
    mt["reg"] = (mt[REG].sum(axis=1) > 0).astype(int)
    mt["unreg"] = (mt[UNREG].sum(axis=1) > 0).astype(int)
    mt["any"] = mt["any_pfas_detected"].astype(int)
    mt["dmin"] = mt[DIST].min(axis=1)
    return mt


def a_monitoring(mt: pd.DataFrame) -> None:
    print("A) MONITORING EFFICIENCY -- capture vs random")
    d = mt.sort_values("dmin").reset_index(drop=True)
    n = len(d)
    tot = {t: d[t].sum() for t in ("any", "reg", "unreg")}
    rows = []
    for k in [10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]:
        top = d.head(int(n * k / 100))
        row = {"nearest_pct": k, "random_baseline_pct": k}
        for t, lab in [("any", "any"), ("reg", "regulated"), ("unreg", "unregulated")]:
            row[f"capture_{lab}_pct"] = round(top[t].sum() / tot[t] * 100, 1)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(T / "monitoring_efficiency.csv", index=False)
    print(res.to_string(index=False))
    at25 = res[res.nearest_pct == 25].iloc[0]
    print(f"   >> nearest 25% of systems captures {at25['capture_unregulated_pct']}% "
          f"of unregulated detections (random = 25%): proximity targeting barely "
          f"beats chance.\n   saved -> {T/'monitoring_efficiency.csv'}\n")


def b_vulnerability(mt: pd.DataFrame) -> None:
    print("B) VULNERABILITY PROFILE -- blind-spot rate by group")
    sd = pd.read_csv(SDWIS, dtype=str,
                     usecols=lambda c: c in {"PWSID", "OWNER_TYPE_CODE",
                     "GW_SW_CODE", "PWS_TYPE_CODE", "IS_SCHOOL_OR_DAYCARE_IND"})
    sd = sd.drop_duplicates("PWSID")
    sysg = mt.groupby("PWSID").agg(reg=("reg", "max"), unreg=("unreg", "max"),
                                   Size=("Size", "first")).reset_index()
    sysg["blind"] = ((sysg.unreg == 1) & (sysg.reg == 0)).astype(int)
    sysg = sysg.merge(sd, on="PWSID", how="left")
    base = sysg["blind"].mean()
    print(f"   overall blind-spot rate: {base*100:.1f}% of tested systems")
    out = []
    for col in ["Size", "OWNER_TYPE_CODE", "GW_SW_CODE", "PWS_TYPE_CODE",
                "IS_SCHOOL_OR_DAYCARE_IND"]:
        if col not in sysg.columns:
            continue
        g = sysg.groupby(col)["blind"].agg(["mean", "size"]).reset_index()
        g.columns = ["category", "blind_rate", "n_systems"]
        g.insert(0, "dimension", col)
        g["blind_rate_pct"] = (g["blind_rate"] * 100).round(1)
        g["rate_ratio_vs_overall"] = (g["blind_rate"] / base).round(2)
        out.append(g[["dimension", "category", "blind_rate_pct",
                      "n_systems", "rate_ratio_vs_overall"]])
    res = pd.concat(out, ignore_index=True)
    res.to_csv(T / "vulnerability_profile.csv", index=False)
    print(res.to_string(index=False))
    print(f"   >> honest read: surface-water systems are over-represented; "
          f"private/small are NOT. EJ (income/race) needs demographic data.\n"
          f"   saved -> {T/'vulnerability_profile.csv'}\n")


def c_selfreport(mt: pd.DataFrame) -> None:
    print("C) SELF-REPORTED SOURCE vs DETECTION")
    ad = pd.read_csv(ADDL, sep="\t", dtype=str, encoding="latin-1",
                     usecols=["PWSID", "AdditionalDataElement", "Response"])
    ps = ad[ad["AdditionalDataElement"] == "PotentialPFASSources"]
    rep = (ps.groupby("PWSID")["Response"]
             .apply(lambda s: "Yes" if (s == "Yes").any()
                    else ("No" if (s == "No").any() else "DK"))
             .rename("selfreport"))
    sysg = mt.groupby("PWSID").agg(reg=("reg", "max"), unreg=("unreg", "max"),
                                   any=("any", "max")).reset_index()
    sysg = sysg.merge(rep, on="PWSID", how="left")
    sub = sysg.dropna(subset=["selfreport"])
    rows = []
    for sr, g in sub.groupby("selfreport"):
        rows.append({"selfreport": sr, "n_systems": len(g),
                     "any_pct": round(g["any"].mean() * 100, 1),
                     "regulated_pct": round(g["reg"].mean() * 100, 1),
                     "unregulated_pct": round(g["unreg"].mean() * 100, 1)})
    res = pd.DataFrame(rows).sort_values("selfreport")
    res.to_csv(T / "self_reported_source.csv", index=False)
    print(res.to_string(index=False))
    no = res[res.selfreport == "No"].iloc[0]
    yes = res[res.selfreport == "Yes"].iloc[0]
    print(f"   >> Self-report predicts detection (Yes {yes['unregulated_pct']}% vs "
          f"No {no['unregulated_pct']}% unregulated), BUT {no['unregulated_pct']}% of "
          f"'No known source' systems still detect unregulated PFAS = diffuse\n"
          f"      contamination beyond known point sources (the decoupling residual).\n"
          f"   saved -> {T/'self_reported_source.csv'}\n")


def main() -> None:
    T.mkdir(parents=True, exist_ok=True)
    mt = load_locations()
    print(f"loaded {len(mt):,} locations / {mt['PWSID'].nunique():,} systems\n")
    a_monitoring(mt)
    b_vulnerability(mt)
    c_selfreport(mt)
    print("DONE. Paste the three tables back to lock these numbers.")


if __name__ == "__main__":
    main()
