"""
make_figures.py  —  main-text Figures 1 to 5 for the PFAS paper.

DESIGN PRINCIPLE: every number is read from data/interim/model_table.csv or from
outputs/tables/*.csv at run time. Nothing is hard-coded. If the pipeline is rebuilt,
the figures change with it, and they can never drift out of sync with the manuscript
again. If a required table is missing, the script stops with a clear error instead
of drawing something stale or blank.

ACS compliance built in:
  * no title inside the figure (the caption lives in the manuscript text)
  * all fonts >= 8 pt AFTER the figure is scaled to fit a column; widths are kept
    at or below 6.5 in so a double-column figure is not shrunk below 8 pt
  * 300 dpi PNG output
  * colour used only to carry meaning, with a legend

Reads:
  data/interim/model_table.csv
  outputs/tables/robustness_repeated_delta.csv
  outputs/tables/robustness_geocoding_quality.csv
  outputs/tables/stage2_concentration_delta.csv
Writes:
  outputs/figures/figure1_detection_prevalence.png (+ .pdf)  ... figure5_watertype.png

Run:  python src/make_figures.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

MT = Path("data/interim/model_table.csv")
TBL = Path("outputs/tables")
OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ---- ACS-safe style: 8 pt floor, widths <= 6.5 in so no shrink below 8 pt ----
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,           # base; nothing below 8 pt is used
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})
NAVY, BLUE, LTBLUE, ORANGE, GREY = "#1F3864", "#2E5A88", "#5B8BC7", "#C55A11", "#6B6B6B"

REGULATED = {"PFOA", "PFOS"}
UNREG_HL = {"PFPeA", "PFHxA", "PFBA"}   # the unregulated compounds the paper names


def need(path):
    if not path.exists():
        sys.exit(f"ERROR: required input {path} not found. Run the pipeline first.")
    return path


def save(fig, name):
    # keep bbox tight only vertically; width already <= 6.5 in by construction
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def load_targets_repeated():
    """Return dict of spatial-block and pws-grouped deltas + percentile CIs per target."""
    df = pd.read_csv(need(TBL / "robustness_repeated_delta.csv"))
    out = {}
    for _, r in df.iterrows():
        out[(r["target"], r["cv_scheme"])] = r
    return out


# ------------------------------------------------------------------ Figure 1
def figure1_prevalence():
    df = pd.read_csv(need(MT))
    # analyte columns = 0/1 columns excluding known non-analytes
    skip = {"any_pfas_detected", "n_pfas_detected", "n_facilities", "n_watertypes", "Region"}
    analytes = []
    for c in df.columns:
        if c in skip or df[c].dtype.kind not in "if":
            continue
        u = set(pd.unique(df[c].dropna()))
        if u.issubset({0, 1}) and len(u) > 1:
            analytes.append(c)
    prev = (df[analytes].mean() * 100).sort_values(ascending=False)
    top = prev.head(8)  # the compounds the paper discusses

    def colour(name):
        if name in REGULATED:
            return NAVY
        if name in UNREG_HL:
            return ORANGE
        return LTBLUE

    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    y = np.arange(len(top))[::-1]
    ax.barh(y, top.values, color=[colour(n) for n in top.index], edgecolor="white")
    for yi, (name, val) in zip(y, top.items()):
        ax.text(val + 0.2, yi, f"{val:.2f}%", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(top.index)
    ax.set_xlabel("Detection frequency (% of locations)")
    ax.set_xlim(0, max(top.values) * 1.15)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=ORANGE, label="Unregulated (PFPeA/PFHxA/PFBA)"),
        Patch(color=NAVY, label="Regulated (PFOA/PFOS)"),
        Patch(color=LTBLUE, label="Other detected PFAS"),
    ], loc="lower right", frameon=False, fontsize=8)
    save(fig, "figure1_detection_prevalence")


# ------------------------------------------------------------------ Figure 2
def figure2_forest():
    rep = load_targets_repeated()
    rows = [
        ("any_pfas_detected", "Any PFAS"),
        ("regulated_detected", "Regulated (PFOA/PFOS)"),
        ("unregulated_detected", "Unregulated (PFPeA/PFHxA/PFBA)"),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ylab, yc = [], []
    y = 0
    for key, label in rows:
        for scheme, colour, tag in [("spatial_block", NAVY, "spatial-block"),
                                    ("pws_grouped", ORANGE, "utility-grouped")]:
            r = rep[(key, scheme)]
            d = r["delta_mean"]
            lo, hi = r["pct_ci_lo"], r["pct_ci_hi"]   # conservative percentile CI
            ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=colour,
                        ms=5, lw=1.4, capsize=3)
            ax.text(hi + 0.001, y, f"{d:+.3f}", va="center", fontsize=8, color="#222222")
            ylab.append(f"{label}\n{tag}")
            yc.append(y)
            y += 1
        y += 0.6
    ax.axvline(0, color=GREY, ls="--", lw=1)
    ax.set_yticks(yc)
    ax.set_yticklabels(ylab)
    ax.invert_yaxis()
    ax.set_xlabel("Marginal lift of source-distance features over covariates (gain in AUC)")
    ax.margins(x=0.15)
    save(fig, "figure2_decoupling_forest")


# ------------------------------------------------------------------ Figure 3
def figure3_geocoding():
    g = pd.read_csv(need(TBL / "robustness_geocoding_quality.csv"))
    g = g[g["cv_scheme"] == "spatial_block"]
    targets = [("any_pfas_detected", "Any PFAS"),
               ("regulated_detected", "Regulated"),
               ("unregulated_detected", "Unregulated")]
    full = g[g["rowset"] == "full"].set_index("target")
    fac = g[g["rowset"] == "facility_only"].set_index("target")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(targets)); w = 0.38
    for i, (key, _) in enumerate(targets):
        rf, ra = full.loc[key], fac.loc[key]
        ax.bar(x[i] - w/2, rf["delta_mean"], w, color=LTBLUE, edgecolor="white",
               yerr=[[rf["delta_mean"] - rf["delta_ci95_low"]], [rf["delta_ci95_high"] - rf["delta_mean"]]],
               capsize=3, error_kw={"elinewidth": 0.8},
               label="Full sample" if i == 0 else None)
        ax.bar(x[i] + w/2, ra["delta_mean"], w, color=NAVY, edgecolor="white",
               yerr=[[ra["delta_mean"] - ra["delta_ci95_low"]], [ra["delta_ci95_high"] - ra["delta_mean"]]],
               capsize=3, error_kw={"elinewidth": 0.8},
               label="Facility-grade only" if i == 0 else None)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    n_full = int(full["n"].iloc[0]); n_fac = int(fac["n"].iloc[0])
    ax.set_xticklabels([t[1] for t in targets])
    ax.set_ylabel("Decoupling gain in AUC (spatial-block)")
    ax.legend(frameon=False, fontsize=8, title=f"Full n = {n_full:,}   Facility-grade n = {n_fac:,}",
              title_fontsize=8)
    save(fig, "figure3_geocoding_robustness")


# ------------------------------------------------------------------ Figure 4
def figure4_two_stage():
    rep = load_targets_repeated()
    s2 = pd.read_csv(need(TBL / "stage2_concentration_delta.csv"))
    s2 = s2[s2["cv_scheme"] == "spatial_block"].set_index("outcome")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.5, 3.6))

    # stage 1: detection deltas (spatial-block, percentile CI)
    labels = [("any_pfas_detected", "Any"), ("regulated_detected", "Regulated"),
              ("unregulated_detected", "Unregulated")]
    xs = np.arange(len(labels))
    for i, (key, _) in enumerate(labels):
        r = rep[(key, "spatial_block")]
        d, lo, hi = r["delta_mean"], r["pct_ci_lo"], r["pct_ci_hi"]
        axL.bar(i, d, 0.6, color=LTBLUE, edgecolor="white",
                yerr=[[d - lo], [hi - d]], capsize=3, error_kw={"elinewidth": 0.8})
        axL.text(i, hi + 0.001, f"{d:.3f}", ha="center", fontsize=8)
    axL.axhline(0, color="black", lw=0.7)
    axL.set_xticks(xs); axL.set_xticklabels([l[1] for l in labels])
    axL.set_ylabel("Gain in AUC")
    axL.set_title("Stage 1: detection", fontsize=10)

    # stage 2: concentration R2 lift
    order = [("log_total_pfas", "Total"), ("log_PFPeA", "PFPeA"), ("log_PFHxA", "PFHxA"),
             ("log_PFOS", "PFOS"), ("log_PFOA", "PFOA")]
    for i, (key, _) in enumerate(order):
        r = s2.loc[key]
        d, lo, hi = r["delta_mean"], r["delta_ci95_low"], r["delta_ci95_high"]
        col = NAVY if key in ("log_PFOS", "log_PFOA") else LTBLUE
        axR.bar(i, d, 0.6, color=col, edgecolor="white",
                yerr=[[d - lo], [hi - d]], capsize=3, error_kw={"elinewidth": 0.8})
        axR.text(i, hi + 0.002, f"{d:.3f}", ha="center", fontsize=8)
    axR.axhline(0, color="black", lw=0.7)
    axR.set_xticks(range(len(order))); axR.set_xticklabels([o[1] for o in order], rotation=0)
    axR.set_ylabel("R\u00b2 lift from source features")
    axR.set_title("Stage 2: concentration", fontsize=10)

    fig.tight_layout()
    save(fig, "figure4_two_stage")


# ------------------------------------------------------------------ Figure 5
def figure5_watertype():
    df = pd.read_csv(need(MT))
    order = ["GW", "GU", "MX", "SW"]
    names = {"GW": "Groundwater\n(GW)", "GU": "GW under\ninfluence (GU)",
             "MX": "Mixed\n(MX)", "SW": "Surface water\n(SW)"}
    z = 1.96
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    for i, wt in enumerate(order):
        g = df[df["FacilityWaterType"] == wt]
        n = len(g); p = g["any_pfas_detected"].mean()
        half = z * np.sqrt(p * (1 - p) / n) * 100
        col = NAVY if wt in ("SW", "MX") else LTBLUE
        ax.bar(i, p * 100, 0.62, color=col, edgecolor="white",
               yerr=half, capsize=4, error_kw={"elinewidth": 0.9})
        ax.text(i, p * 100 + half + 0.6, f"{p*100:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels([names[o] for o in order])
    ax.set_ylabel("Any-PFAS detection rate (%)")
    ax.set_ylim(0, 45)
    save(fig, "figure5_watertype")


def main():
    print("Building main-text figures from result tables (no hard-coded numbers)...")
    figure1_prevalence()
    figure2_forest()
    figure3_geocoding()
    figure4_two_stage()
    figure5_watertype()
    print("Done. All numbers read from data/interim and outputs/tables.")


if __name__ == "__main__":
    main()
