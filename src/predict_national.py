"""
Module 6: NATIONAL SCREENING PREDICTION (impact Angle 4).

Trains a TRANSFERABLE PFAS-occurrence model on the UCMR5-tested systems using
only features available for EVERY U.S. public water system, then predicts
occurrence probability for the untested active community water systems (CWS)
-> a national screening / monitoring-prioritisation map.

WHY TRANSFERABLE FEATURES ONLY: the published model also used n_facilities /
n_watertypes, which are artefacts of UCMR5 sampling and do NOT exist for
untested systems. Using them would make the model impossible to apply
nationally. So we retrain on the features that DO exist for all systems:
    State, EPA region, size category (L/S), source water (GW/SW),
    + the 4 nearest-source distances (computed identically via spatial.py).

HONEST SCOPE (put this in the paper): state-grouped CV AUC is ~0.65 -- a
SCREENING / prioritisation signal, not a precise predictor. That coarseness is
itself consistent with the paper's central finding: occurrence is diffuse and
regional, so it is only coarsely predictable. Report this as "which untested
systems to test first", never as an exposure estimate.

Distances for untested systems reuse spatial.py's VERIFIED loaders and the same
source sets as Module 3b, so tested and untested distances are strictly
comparable (no re-implementation).

Reads:
    data/interim/model_table.csv
    data/interim/locations_geocoded.csv
    data/interim/frs_pws_points.csv
    data/raw/sdwis/SDWA_PUB_WATER_SYSTEMS.csv
    (via spatial.py) data/raw/frs/NATIONAL_SINGLE.CSV + airports + military
Writes:
    outputs/tables/national_predictions.csv     every untested CWS + risk
    outputs/tables/top_priority_untested.csv     top 500 highest-risk untested
    outputs/tables/national_model_auc.csv        transferable-model CV AUC
    outputs/figures/figure_national_prediction.(png|pdf)

Usage:  python src/predict_national.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spatial  # reuse verified Module-3b loaders + constants

MT = Path("data/interim/model_table.csv")
GEO = Path("data/interim/locations_geocoded.csv")
FRS_PTS = Path("data/interim/frs_pws_points.csv")
SDWIS = Path("data/raw/sdwis/SDWA_PUB_WATER_SYSTEMS.csv")
T = Path("outputs/tables"); F = Path("outputs/figures")

REG = ["PFOA", "PFOS"]; UNREG = ["PFPeA", "PFHxA", "PFBA"]
DIST = ["dist_industrial_km", "dist_wwtp_km", "dist_airport_km", "dist_military_km"]
CAT = ["State", "Region", "Size", "srcwater"]
FEATS = CAT + DIST
TARGETS = {"any": "any_pfas_detected_risk", "unreg": "unregulated_risk"}
WT_MAP = {"GW": "GW", "GU": "GW", "SW": "SW", "MX": "SW"}  # -> transferable GW/SW


def build_training() -> pd.DataFrame:
    mt = pd.read_csv(MT, dtype={"location_id": str})
    geo = pd.read_csv(GEO, dtype={"location_id": str})
    pcol = next(c for c in ("PWSID", "pwsid", "pws_id") if c in geo.columns)
    mt = mt.merge(geo[["location_id", pcol]].rename(columns={pcol: "PWSID"}),
                  on="location_id", how="left")
    mt["reg"] = (mt[REG].sum(axis=1) > 0).astype(int)
    mt["unreg"] = (mt[UNREG].sum(axis=1) > 0).astype(int)
    mt["any"] = mt["any_pfas_detected"].astype(int)
    g = mt.groupby("PWSID").agg(
        any=("any", "max"), unreg=("unreg", "max"),
        State=("State", "first"), Region=("Region", "first"),
        Size=("Size", "first"), FacilityWaterType=("FacilityWaterType", "first"),
        **{d: (d, "first") for d in DIST}).reset_index()
    g["srcwater"] = g["FacilityWaterType"].map(WT_MAP).fillna("GW")
    g["Region"] = pd.to_numeric(g["Region"], errors="coerce").astype("Int64")
    return g


def build_untested() -> pd.DataFrame:
    sd = pd.read_csv(SDWIS, dtype=str)
    sd = sd[(sd["PWS_ACTIVITY_CODE"] == "A") & (sd["PWS_TYPE_CODE"] == "CWS")].copy()
    tested = set(pd.read_csv(GEO, dtype=str)["PWSID"].dropna())
    sd = sd[~sd["PWSID"].isin(tested)].drop_duplicates("PWSID")
    sd["pop"] = pd.to_numeric(sd["POPULATION_SERVED_COUNT"], errors="coerce")
    sd["State"] = sd["STATE_CODE"]
    sd["Region"] = pd.to_numeric(sd["EPA_REGION"], errors="coerce").astype("Int64")
    sd["Size"] = np.where(sd["pop"] > 10000, "L", "S")
    sd["srcwater"] = sd["GW_SW_CODE"].map(WT_MAP).fillna("GW")
    return sd[["PWSID", "State", "Region", "Size", "srcwater", "pop", "ZIP_CODE"]]


def geocode_untested(un: pd.DataFrame) -> pd.DataFrame:
    """facility coords where available, else ZIP centroid (pgeocode)."""
    frs = pd.read_csv(FRS_PTS, dtype={"PWSID": str})
    un = un.merge(frs[["PWSID", "frs_lat", "frs_lon"]], on="PWSID", how="left")
    has = un["frs_lat"].notna()
    un["lat"] = un["frs_lat"]; un["lon"] = un["frs_lon"]
    un["coord_source"] = np.where(has, "frs_facility", "none")
    need = un[~has].copy()
    if len(need):
        try:
            import pgeocode
        except ImportError:
            raise SystemExit("pip install pgeocode  (needed for ZIP-centroid fallback)")
        nomi = pgeocode.Nominatim("us")
        z = need["ZIP_CODE"].astype(str).str.extract(r"(\d{5})")[0]
        cen = nomi.query_postal_code(z.fillna("00000").tolist())
        un.loc[~has, "lat"] = cen["latitude"].values
        un.loc[~has, "lon"] = cen["longitude"].values
        un.loc[~has & un["lat"].notna(), "coord_source"] = "zip_centroid"
    placed = un["lat"].notna() & un["lon"].notna()
    print(f"   geocoded untested: {int(placed.sum()):,}/{len(un):,} "
          f"({un['coord_source'].value_counts().to_dict()})")
    return un[placed].copy()


def distances_for(un: pd.DataFrame) -> pd.DataFrame:
    """Compute the 4 nearest-source distances for untested coords, reusing
    spatial.py's exact source loaders (same as Module 3b)."""
    spatial.RAW_DIR.mkdir(parents=True, exist_ok=True)
    frs_zip = spatial.download(spatial.FRS_ZIP_URL, spatial.RAW_DIR / "frs_national_single.zip")
    naics = spatial.download(spatial.PFAS_NAICS_XLSX_URL, spatial.RAW_DIR / "pfas_industry_sectors.xlsx")
    airports = spatial.download(spatial.OURAIRPORTS_CSV_URL, spatial.RAW_DIR / "ourairports.csv")
    ind_pat = spatial.build_industrial_pattern(spatial.load_industrial_naics(naics))
    frs_csv = spatial.extract_frs_csv(frs_zip, spatial.RAW_DIR)
    ind_xy, wwtp_xy = spatial.harvest_sources(frs_csv, ind_pat)
    air_xy = spatial.load_airports(airports)
    mil_xy = spatial.load_military()
    loc_xy = un[["lat", "lon"]].to_numpy(float)
    un["dist_industrial_km"] = spatial.nearest_km(ind_xy, loc_xy)
    un["dist_wwtp_km"] = spatial.nearest_km(wwtp_xy, loc_xy)
    un["dist_airport_km"] = spatial.nearest_km(air_xy, loc_xy)
    un["dist_military_km"] = spatial.nearest_km(mil_xy, loc_xy)
    return un


def align_categories(train: pd.DataFrame, score: pd.DataFrame):
    """Give train + score the SAME category sets so XGBoost encodes identically."""
    tr, sc = train.copy(), score.copy()
    for c in CAT:
        cats = pd.Index(sorted(set(tr[c].dropna().astype(str)) |
                               set(sc[c].dropna().astype(str))))
        tr[c] = pd.Categorical(tr[c].astype(str), categories=cats)
        sc[c] = pd.Categorical(sc[c].astype(str), categories=cats)
    return tr, sc


def new_model():
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
        eval_metric="auc", random_state=42)


def main() -> None:
    T.mkdir(parents=True, exist_ok=True); F.mkdir(parents=True, exist_ok=True)

    print("1. Building training set (tested systems, transferable features)")
    tr = build_training()
    print(f"   tested systems: {len(tr):,}")

    print("2. Building untested universe (active CWS not in UCMR5)")
    un = build_untested()
    print(f"   untested active CWS: {len(un):,}")

    print("3. Geocoding untested systems")
    un = geocode_untested(un)

    print("4. Computing 4 source distances for untested (via spatial.py)")
    un = distances_for(un)

    print("5. State-grouped CV AUC of the transferable model")
    auc_rows = []
    for key in ("any", "unreg"):
        Xtr, _ = align_categories(tr[FEATS], un[FEATS])
        y = tr[key]; groups = tr["State"].astype(str)
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for a, b in skf.split(Xtr, y, groups=groups):
            m = new_model(); m.fit(Xtr.iloc[a], y.iloc[a])
            aucs.append(roc_auc_score(y.iloc[b], m.predict_proba(Xtr.iloc[b])[:, 1]))
        auc_rows.append({"target": key, "cv_auc_mean": round(np.mean(aucs), 4),
                         "cv_auc_std": round(np.std(aucs, ddof=1), 4)})
        print(f"   {key:<6} AUC {np.mean(aucs):.3f} +/- {np.std(aucs, ddof=1):.3f}")
    pd.DataFrame(auc_rows).to_csv(T / "national_model_auc.csv", index=False)

    print("6. Fitting final models on ALL tested + predicting untested")
    out = un[["PWSID", "State", "Region", "Size", "srcwater", "pop",
              "coord_source"] + DIST].copy()
    for key, colname in TARGETS.items():
        Xtr, Xsc = align_categories(tr[FEATS], un[FEATS])
        m = new_model(); m.fit(Xtr, tr[key])
        out[colname] = m.predict_proba(Xsc)[:, 1].round(4)

    out.to_csv(T / "national_predictions.csv", index=False)
    print(f"   saved -> {T/'national_predictions.csv'}  ({len(out):,} systems)")

    top = out.sort_values("unregulated_risk", ascending=False).head(500)
    top.to_csv(T / "top_priority_untested.csv", index=False)
    print(f"   saved -> {T/'top_priority_untested.csv'}  (top 500 by predicted "
          f"unregulated risk)")

    print("7. Figure: mean predicted unregulated risk by state")
    st = (out.groupby("State")["unregulated_risk"].mean().sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(6.5, 8))
    ax.barh(st.index[::-1], st.values[::-1], color="#2E5A88", edgecolor="white")
    ax.set_xlabel("Mean predicted unregulated-PFAS probability (untested CWS)")
    ax.set_ylabel("State")
    ax.set_title("Predicted screening risk for untested community water systems",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(F / "figure_national_prediction.png", dpi=300, bbox_inches="tight")
    fig.savefig(F / "figure_national_prediction.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"   saved -> {F/'figure_national_prediction.png'}")

    print("\nDONE. Screening map + ranked priority list built. Paste the AUC line "
          "and top-of-list back to lock. (This is prioritisation, not exposure.)")


if __name__ == "__main__":
    main()
