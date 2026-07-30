"""
Module 2: clean and standardize UCMR 5 into a per-location detection matrix.

Locked decisions:
  - location unit : PWSID + SamplePointID  (one entry point per row)
  - detection rule: ever-detect (1 if AnalyticalResultsSign == '=' in any sample)
  - lithium dropped (not a PFAS; it rode along in the same monitoring rule)
  - NaN preserved where an analyte was never measured at a location

Reads:  data/raw/ucmr5/UCMR5_All.txt
Writes: data/interim/detection_matrix.csv
        outputs/tables/detection_frequency.csv
"""

from pathlib import Path

import pandas as pd

RAW = Path("data/raw/ucmr5/UCMR5_All.txt")
INTERIM = Path("data/interim/detection_matrix.csv")
TABLE = Path("outputs/tables/detection_frequency.csv")

DROP_CONTAMINANTS = {"lithium"}


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, encoding="latin-1", on_bad_lines="warn")


def main() -> None:
    df = load_raw(RAW)
    print(f"raw rows: {df.shape[0]:,}")

    # Confirm every sample point really is an entry point before we lean on it.
    print("\nSamplePointType:")
    print(df["SamplePointType"].value_counts(dropna=False).to_string())

    # 1. drop non-PFAS
    df = df[~df["Contaminant"].isin(DROP_CONTAMINANTS)].copy()

    # 2. binary detection. '=' means a concentration was reported (detected);
    #    '<' means below the reporting limit (non-detect).
    df["detect"] = (df["AnalyticalResultsSign"].str.strip() == "=").astype(int)

    # 3. location unit = entry point = PWSID + SamplePointID  (locked)
    df["SamplePointID"] = df["SamplePointID"].fillna("NA")
    df["location_id"] = df["PWSID"].str.strip() + "_" + df["SamplePointID"].str.strip()

    # 4. ever-detect collapse across all quarters: max of the 0/1 flag is 1 if
    #    the analyte ever registered at this location, else 0.
    pair = df.groupby(["location_id", "Contaminant"])["detect"].max().reset_index()

    # 5. long -> wide. Cells are 1, 0, or NaN (never measured here -- kept blank
    #    on purpose; a blank is NOT a non-detect).
    matrix = pair.pivot(index="location_id", columns="Contaminant", values="detect")
    matrix = matrix.astype("Int64")

    # 6. location-level attributes for the spatial step, plus integrity counts.
    attrs = df.groupby("location_id").agg(
        PWSID=("PWSID", "first"),
        State=("State", "first"),
        Region=("Region", "first"),
        Size=("Size", "first"),
        FacilityID=("FacilityID", "first"),
        FacilityWaterType=("FacilityWaterType", "first"),
        n_facilities=("FacilityID", "nunique"),
        n_watertypes=("FacilityWaterType", "nunique"),
    )

    out = attrs.join(matrix)
    INTERIM.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(INTERIM)

    # ---- diagnostics -----------------------------------------------------
    print(f"\nlocations: {matrix.shape[0]:,}")
    print(f"analytes : {matrix.shape[1]}")

    measured = matrix.notna().sum()
    detected = (matrix == 1).sum()
    freq = (
        pd.DataFrame({"n_measured": measured, "n_detected": detected})
        .assign(detection_rate=lambda d: (d.n_detected / d.n_measured).round(4))
        .sort_values("detection_rate", ascending=False)
    )
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    freq.to_csv(TABLE)

    print("\ndetection rate by analyte (top 12):")
    print(freq.head(12).to_string())

    bad = attrs[(attrs.n_facilities > 1) | (attrs.n_watertypes > 1)]
    print(f"\nlocations spanning >1 facility or water type: {len(bad)}")

    panel = matrix.notna().sum(axis=1)
    print("\nanalytes measured per location (count -> #locations):")
    print(panel.value_counts().sort_index().to_string())

    print(f"\nsaved matrix -> {INTERIM}")
    print(f"saved table  -> {TABLE}")


if __name__ == "__main__":
    main()
