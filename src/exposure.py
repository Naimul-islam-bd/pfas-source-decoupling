"""
Module 5, step 1: POPULATION EXPOSURE (impact Angle 2).

Joins POPULATION_SERVED_COUNT from SDWIS (ECHO's SDWA_PUB_WATER_SYSTEMS.csv)
onto each tested PWS, keyed on PWSID, to report population served by systems
where PFAS was detected. This gives the population-level impact figure
that complements the detection analysis (cf. Tokranov et al., Science).

TWO HONEST CAVEATS, both built into the output (do not drop either):
  1. This is POPULATION SERVED, not unique individuals. A person served by two
     systems is counted twice. Report as "population served by affected
     systems", never "people exposed".
  2. WHOLESALE DOUBLE-COUNTING. ~1,233 tested systems are wholesalers that sell
     water to other systems; their served population is counted again in the
     retail systems they supply (~17% of the denominator). So every tier is
     reported BOTH ways: all systems, and excluding wholesalers (a conservative
     lower bound). Lead with system counts + percentages; treat the absolute
     population as a served-population range between the two columns.

DETECTION TIERS (system positive if ANY of its locations is positive):
  any_pfas / regulated(PFOA,PFOS) / unregulated(PFPeA,PFHxA,PFBA) /
  blind_spot = unregulated present AND regulated absent  <- THE HEADLINE.

PWSID is taken from locations_geocoded.csv (authoritative), not by slicing
location_id, consistent with robustness.py.

Reads:
    data/interim/model_table.csv
    data/interim/locations_geocoded.csv
    data/raw/sdwis/SDWA_PUB_WATER_SYSTEMS.csv
Writes:
    outputs/tables/exposure_population.csv
    outputs/tables/exposure_breakdown.csv

Usage:  python src/exposure.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MT = Path("data/interim/model_table.csv")
GEO = Path("data/interim/locations_geocoded.csv")
SDWIS = Path("data/raw/sdwis/SDWA_PUB_WATER_SYSTEMS.csv")
OUT_MAIN = Path("outputs/tables/exposure_population.csv")
OUT_BREAK = Path("outputs/tables/exposure_breakdown.csv")

REGULATED = ["PFOA", "PFOS"]
UNREGULATED = ["PFPeA", "PFHxA", "PFBA"]


def get_pwsid(mt: pd.DataFrame) -> np.ndarray:
    geo = pd.read_csv(GEO, dtype={"location_id": str})
    pcol = next((c for c in ("PWSID", "pwsid", "pws_id") if c in geo.columns), None)
    if pcol is None:
        raise SystemExit(f"No PWSID column in {GEO}; columns = {list(geo.columns)}")
    geo = geo[["location_id", pcol]].rename(columns={pcol: "PWSID"}).drop_duplicates("location_id")
    m = mt[["location_id"]].merge(geo, on="location_id", how="left")
    n_missing = int(m["PWSID"].isna().sum())
    if n_missing:
        fb = mt["location_id"].astype(str).str.split("_").str[0]
        m["PWSID"] = m["PWSID"].fillna(fb)
        print(f"  NOTE: {n_missing:,} location_ids lacked a geocoded PWSID; used fallback.")
    return m["PWSID"].values


def main() -> None:
    print("1. Per-location detection -> system-level tiers")
    mt = pd.read_csv(MT, dtype={"location_id": str})
    for grp in (REGULATED, UNREGULATED):
        miss = [c for c in grp if c not in mt.columns]
        if miss:
            raise SystemExit(f"analyte columns missing from model_table: {miss}")
    mt["PWSID"] = get_pwsid(mt)
    mt["reg"] = (mt[REGULATED].sum(axis=1) > 0).astype(int)
    mt["unreg"] = (mt[UNREGULATED].sum(axis=1) > 0).astype(int)
    mt["any"] = mt["any_pfas_detected"].astype(int)
    sysg = mt.groupby("PWSID").agg(any=("any", "max"), reg=("reg", "max"),
                                   unreg=("unreg", "max")).reset_index()
    sysg["blind_spot"] = ((sysg["unreg"] == 1) & (sysg["reg"] == 0)).astype(int)
    print(f"   unique tested PWS: {len(sysg):,}")

    print("\n2. SDWIS population (latest quarter per PWSID)")
    keep = ["SUBMISSIONYEARQUARTER", "PWSID", "POPULATION_SERVED_COUNT",
            "PWS_ACTIVITY_CODE", "IS_WHOLESALER_IND", "OWNER_TYPE_CODE",
            "PWS_TYPE_CODE", "GW_SW_CODE", "IS_SCHOOL_OR_DAYCARE_IND"]
    sd = pd.read_csv(SDWIS, dtype=str, usecols=lambda c: c in keep)
    sd = sd.sort_values("SUBMISSIONYEARQUARTER").groupby("PWSID", as_index=False).last()
    sd["pop_served"] = pd.to_numeric(sd["POPULATION_SERVED_COUNT"], errors="coerce")
    sd["is_wholesale"] = sd["IS_WHOLESALER_IND"].astype(str).str.upper().str.startswith("Y")
    print(f"   SDWIS PWS rows (deduped): {len(sd):,}")

    print("\n3. Joining population onto tested systems")
    df = sysg.merge(sd, on="PWSID", how="left", validate="one_to_one")
    n_nopop = int(df["pop_served"].isna().sum())
    if n_nopop:
        print(f"   NOTE: {n_nopop:,} tested PWS had no SDWIS population (set to 0).")
    df["pop_served"] = df["pop_served"].fillna(0)
    df["is_wholesale"] = df["is_wholesale"].fillna(False)

    n_inactive = int((df["PWS_ACTIVITY_CODE"].astype(str) != "A").sum())
    print(f"   (info) non-active systems among matches: {n_inactive:,} "
          f"(kept; flip to active-only as a sensitivity if desired)")

    total_all = df["pop_served"].sum()
    total_nw = df.loc[~df["is_wholesale"], "pop_served"].sum()

    def block(mask, label):
        sub = df[mask]
        p_all = sub["pop_served"].sum()
        p_nw = sub.loc[~sub["is_wholesale"], "pop_served"].sum()
        return {"tier": label, "n_systems": int(len(sub)),
                "pop_served_all": int(p_all),
                "pop_served_excl_wholesale": int(p_nw),
                "pct_tested_pop_all": round(100 * p_all / total_all, 1)}

    print("\n4. HEADLINE: population served by affected systems (two columns = range)")
    rows = [block(df["any"] == 1, "any_pfas_detected"),
            block(df["reg"] == 1, "regulated (PFOA/PFOS)"),
            block(df["unreg"] == 1, "unregulated (PFPeA/PFHxA/PFBA)"),
            block(df["blind_spot"] == 1, "BLIND SPOT: unregulated, no regulated")]
    res = pd.DataFrame(rows)
    res.loc[len(res)] = {"tier": "ALL tested systems (denominator)",
                         "n_systems": int(len(df)), "pop_served_all": int(total_all),
                         "pop_served_excl_wholesale": int(total_nw),
                         "pct_tested_pop_all": 100.0}
    OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_MAIN, index=False)
    print(res.to_string(index=False))
    print(f"   saved -> {OUT_MAIN}")
    print("   >> 'population served by affected systems' (NOT people exposed).")
    print("   >> Report a range: excl_wholesale (lower) .. all (upper).")

    print("\n5. NOVELTY BREAKDOWN: who is in the blind spot? (no extra data)")
    bs = df[df["blind_spot"] == 1]
    parts = []
    for col, pretty in [("OWNER_TYPE_CODE", "owner_type"),
                        ("PWS_TYPE_CODE", "system_type"),
                        ("GW_SW_CODE", "source_water"),
                        ("IS_SCHOOL_OR_DAYCARE_IND", "school_or_daycare")]:
        if col not in df.columns:
            continue
        g = (bs.groupby(col).agg(n_systems=("PWSID", "size"),
                                 pop_served=("pop_served", "sum")).reset_index()
               .rename(columns={col: "category"}))
        g.insert(0, "dimension", pretty)
        g["pop_served"] = g["pop_served"].astype(int)
        parts.append(g.sort_values("pop_served", ascending=False))
    if parts:
        bd = pd.concat(parts, ignore_index=True)
        bd.to_csv(OUT_BREAK, index=False)
        print(bd.to_string(index=False))
        print(f"   saved -> {OUT_BREAK}")

    print("\nDONE. Paste the two tables back to lock the numbers "
          "(no manuscript prose until locked).")


if __name__ == "__main__":
    main()
