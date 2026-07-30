"""
audit_numbers.py - print every number the manuscript cites, straight from the
current data and the saved results tables.

Why this exists: the pipeline was rebuilt after the facility-grade re-geocoding,
so the row count and prevalence in the manuscript may be stale. Rather than
guessing, this prints the authoritative values so the manuscript can be
corrected to match exactly what the code produces.

Reads:  data/interim/model_table.csv
        data/interim/concentration_matrix.csv   (if present)
        outputs/tables/*.csv                    (whatever exists)
Writes: nothing. It only prints.

Run:    python src/audit_numbers.py
"""
import sys
from pathlib import Path
import pandas as pd

MT = Path("data/interim/model_table.csv")
CONC = Path("data/interim/concentration_matrix.csv")
TABLES = Path("outputs/tables")

REGULATED = ["PFOA", "PFOS"]
SHORT_CHAIN = ["PFPeA", "PFHxA", "PFBA"]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 100)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    if not MT.exists():
        sys.exit(f"ERROR: {MT} not found.")

    df = pd.read_csv(MT)

    rule("1. HEADLINE COUNTS  (manuscript currently says 24,636 locations / 24.4%)")
    print(f"locations (rows)        : {len(df):,}")
    print(f"unique location_id      : {df['location_id'].nunique():,}")
    if "PWSID" in df.columns:
        print(f"unique PWSID            : {df['PWSID'].nunique():,}")
    else:
        pws = df["location_id"].astype(str).str.split("_").str[0]
        print(f"unique PWSID (derived)  : {pws.nunique():,}")
    if "any_pfas_detected" in df.columns:
        print(f"any-PFAS prevalence     : {df['any_pfas_detected'].mean()*100:.2f}%")
        print(f"locations with a detect : {int(df['any_pfas_detected'].sum()):,}")

    rule("2. PER-ANALYTE DETECTION PREVALENCE  (for Table 1 and Figure 1)")
    # analyte columns = 0/1 columns that are not the known non-analyte fields
    skip = {"any_pfas_detected", "n_pfas_detected", "n_facilities", "n_watertypes", "Region"}
    analytes = []
    for c in df.columns:
        if c in skip or df[c].dtype.kind not in "if":
            continue
        vals = set(pd.unique(df[c].dropna()))
        if vals.issubset({0, 1}) and len(vals) > 1:
            analytes.append(c)
    prev = (df[analytes].mean() * 100).sort_values(ascending=False)
    print(f"number of analyte columns detected: {len(analytes)}")
    print("\ntop 12 by prevalence:")
    for name, v in prev.head(12).items():
        tag = ""
        if name in REGULATED:
            tag = "   <- REGULATED"
        elif name in SHORT_CHAIN:
            tag = "   <- short-chain"
        print(f"   {name:<14} {v:6.2f}%{tag}")

    rule("3. PREVALENCE BY WATER-SOURCE TYPE  (for Table 1 and Figure 5)")
    if "FacilityWaterType" in df.columns and "any_pfas_detected" in df.columns:
        g = df.groupby("FacilityWaterType")["any_pfas_detected"].agg(["mean", "count"])
        g["prevalence_%"] = (g["mean"] * 100).round(2)
        print(g[["prevalence_%", "count"]].to_string())

    rule("4. CONCENTRATION STAGE  (manuscript says 6,184 locations / 3,539 systems, median 0.015)")
    if CONC.exists():
        c = pd.read_csv(CONC)
        print(f"concentration rows      : {len(c):,}")
        if "location_id" in c.columns:
            print(f"unique locations        : {c['location_id'].nunique():,}")
            pws = c["location_id"].astype(str).str.split("_").str[0]
            print(f"unique PWSID (derived)  : {pws.nunique():,}")
        tot = [x for x in c.columns if "total" in x.lower()]
        if tot:
            print(f"median {tot[0]}: {c[tot[0]].median():.4f}")
            print(f"max    {tot[0]}: {c[tot[0]].max():.4f}")
        print(f"\ncolumns: {list(c.columns)[:12]}")
    else:
        print("concentration_matrix.csv not found")

    rule("5. SAVED RESULTS TABLES  (these are what the manuscript quotes)")
    if not TABLES.exists():
        print("outputs/tables not found")
        return
    for f in sorted(TABLES.glob("*.csv")):
        print(f"\n--- {f.name} ---")
        try:
            t = pd.read_csv(f)
            print(t.to_string(index=False))
        except Exception as e:
            print(f"   (could not read: {e})")

    rule("DONE - send this whole output back so the manuscript numbers can be reconciled")


if __name__ == "__main__":
    main()
