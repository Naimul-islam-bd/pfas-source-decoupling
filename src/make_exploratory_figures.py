"""
Exploratory analysis figures (Figures 6-8) for the PFAS paper, addressing
Prof. Serre's request for a visual exploratory layer BEFORE the modelling.

FIXED VERSION. The earlier script produced a BLANK Figure 7 because the source
count columns (n_industrial_5km etc.) do not live in model_table.csv; they are
written to a separate file, data/interim/source_counts.csv. The old code silently
hid every panel whose column was missing, so it saved an empty figure without an
error. This version:

  1. merges source_counts.csv onto model_table.csv by location_id,
  2. auto-detects whichever count columns actually exist (no hard-coded names),
  3. prints what it found, and
  4. raises a clear error instead of quietly saving a blank figure.

Reads:  data/interim/model_table.csv        (detection + distances + covariates)
        data/interim/source_counts.csv      (source counts within 5 km / 10 km)
Writes: outputs/figures/figure6_distance_binned.(png|pdf)
        outputs/figures/figure7_count_binned.(png|pdf)
        outputs/figures/figure8_covariate_assoc.(png|pdf)

Run:    python src/make_exploratory_figures.py

Wilson 95% confidence intervals are shown on each binned detection rate.
"""
import re
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_TABLE = Path("data/interim/model_table.csv")
SOURCE_COUNTS = Path("data/interim/source_counts.csv")
OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "figure.dpi": 300,
})
NAVY, BLUE, ORANGE, GREY = "#1F3864", "#3A6EA5", "#C55A11", "#7F7F7F"
TARGET = "any_pfas_detected"


# ---------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, max(centre - half, 0.0), min(centre + half, 1.0)


def binned_rate(df, col, target, edges, min_n=20):
    """Detection rate per bin with Wilson CI. Returns (mid, p, lo, hi, n)."""
    cats = pd.cut(df[col], bins=edges, include_lowest=True, duplicates="drop")
    out = []
    for interval, g in df.groupby(cats, observed=True):
        n = len(g)
        if n < min_n:
            continue
        p, lo, hi = wilson(int(g[target].sum()), n)
        out.append((interval.mid, p, lo, hi, n))
    out.sort(key=lambda r: r[0])
    return out


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def load_data():
    """Load model_table and merge the source counts if they are in a separate file."""
    if not MODEL_TABLE.exists():
        sys.exit(f"ERROR: {MODEL_TABLE} not found. Run the pipeline first.")
    df = pd.read_csv(MODEL_TABLE)
    print(f"model_table.csv: {len(df):,} rows, {df.shape[1]} columns")

    count_pat = re.compile(r"^n_.+_(5|10)km$")
    have = [c for c in df.columns if count_pat.match(c)]

    if not have and SOURCE_COUNTS.exists():
        sc = pd.read_csv(SOURCE_COUNTS)
        sc_counts = [c for c in sc.columns if count_pat.match(c)]
        if not sc_counts:
            print(f"  WARNING: {SOURCE_COUNTS} has no columns matching n_*_5km / n_*_10km")
            print(f"  its columns are: {list(sc.columns)}")
        else:
            key = "location_id" if "location_id" in sc.columns and "location_id" in df.columns else None
            if key is None:
                sys.exit("ERROR: cannot merge source_counts.csv (no shared 'location_id' column).")
            before = len(df)
            df = df.merge(sc[[key] + sc_counts], on=key, how="left")
            matched = df[sc_counts[0]].notna().sum()
            print(f"merged source_counts.csv on '{key}': {len(sc_counts)} count columns, "
                  f"{matched:,}/{before:,} rows matched")
            have = sc_counts
    elif have:
        print(f"count columns already present in model_table.csv: {have}")

    if not have:
        print("  NOTE: no source-count columns available; Figure 7 will be skipped.")
    return df, have


# ---------------------------------------------------------------- figures
def figure6(df, base):
    dist_specs = [("dist_industrial_km", "Industrial", 12),
                  ("dist_wwtp_km", "Wastewater", 25),
                  ("dist_military_km", "Military", 30),
                  ("dist_airport_km", "Airport", 80)]
    present = [(c, l, x) for c, l, x in dist_specs if c in df.columns]
    if not present:
        print("  SKIP figure 6: no dist_* columns found.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.4))
    drawn = 0
    for ax, (col, label, xmax) in zip(axes.ravel(), present):
        edges = np.linspace(0, xmax, 9)
        pts = binned_rate(df, col, TARGET, edges)
        if not pts:
            ax.set_visible(False)
            continue
        x = [p[0] for p in pts]
        y = [p[1] * 100 for p in pts]
        lo = [(p[1] - p[2]) * 100 for p in pts]
        hi = [(p[3] - p[1]) * 100 for p in pts]
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", color=BLUE, ms=5, lw=1.8, capsize=3)
        ax.axhline(base * 100, color=ORANGE, ls="--", lw=1.2, label="overall rate")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(f"Distance to nearest {label.lower()} source (km)")
        ax.set_ylabel("Detection rate (%)")
        ax.legend(frameon=False, fontsize=8)
        drawn += 1
    for ax in axes.ravel()[len(present):]:
        ax.set_visible(False)
    if drawn == 0:
        plt.close(fig)
        print("  SKIP figure 6: no bins had enough data.")
        return
    fig.tight_layout()
    save(fig, "figure6_distance_binned")


def figure7(df, base, count_cols):
    """Detection rate vs source count. Uses whichever count columns exist."""
    if not count_cols:
        print("  SKIP figure 7: no source-count columns available.")
        return

    def pretty(col):
        m = re.match(r"^n_(.+)_(5|10)km$", col)
        if not m:
            return col
        name = m.group(1).replace("_", " ")
        name = {"industrial": "Industrial", "wwtp": "Wastewater",
                "military": "Military", "airport": "Airport"}.get(name, name.title())
        return f"{name}, {m.group(2)} km"

    # prefer the four classic panels if present, else take the first four available
    preferred = ["n_industrial_5km", "n_industrial_10km", "n_wwtp_5km", "n_wwtp_10km"]
    chosen = [c for c in preferred if c in count_cols]
    for c in count_cols:
        if len(chosen) >= 4:
            break
        if c not in chosen:
            chosen.append(c)
    chosen = chosen[:4]
    print(f"  figure 7 panels: {chosen}")

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.4))
    drawn = 0
    for ax, col in zip(axes.ravel(), chosen):
        s = df[col].dropna()
        if s.empty:
            ax.set_visible(False)
            continue
        # quantile edges, fall back to a linear range if the counts are very sparse
        qs = np.unique(np.floor(s.quantile([0, .2, .4, .6, .8, 1.0]).to_numpy()))
        edges = qs if len(qs) >= 3 else np.linspace(s.min(), s.max() + 1, 6)
        pts = binned_rate(df.dropna(subset=[col]), col, TARGET, edges)
        if not pts:
            ax.set_visible(False)
            continue
        x = [p[0] for p in pts]
        y = [p[1] * 100 for p in pts]
        lo = [(p[1] - p[2]) * 100 for p in pts]
        hi = [(p[3] - p[1]) * 100 for p in pts]
        ax.errorbar(x, y, yerr=[lo, hi], fmt="s-", color=NAVY, ms=5, lw=1.8, capsize=3)
        ax.axhline(base * 100, color=ORANGE, ls="--", lw=1.2, label="overall rate")
        ax.set_title(pretty(col), fontsize=10)
        ax.set_xlabel("Number of sources within radius")
        ax.set_ylabel("Detection rate (%)")
        ax.legend(frameon=False, fontsize=8)
        drawn += 1
    for ax in axes.ravel()[len(chosen):]:
        ax.set_visible(False)

    if drawn == 0:
        plt.close(fig)
        raise SystemExit("ERROR: figure 7 would be blank (no panel could be drawn). "
                         "Check that source_counts.csv has usable count columns.")

    fig.tight_layout()
    save(fig, "figure7_count_binned")


def figure8(df, base):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.5, 3.4))
    if "FacilityWaterType" in df.columns:
        order = ["GW", "GU", "MX", "SW"]
        vals, los, his, labs = [], [], [], []
        for wt in order:
            g = df[df["FacilityWaterType"] == wt]
            if len(g) == 0:
                continue
            p, lo, hi = wilson(int(g[TARGET].sum()), len(g))
            vals.append(p * 100); los.append((p - lo) * 100); his.append((hi - p) * 100); labs.append(wt)
        axL.bar(labs, vals, yerr=[los, his], color=BLUE, capsize=4, edgecolor="white")
        axL.axhline(base * 100, color=ORANGE, ls="--", lw=1.2)
        axL.set_ylabel("Detection rate (%)")
        axL.set_title("By water-source type", fontsize=10)
    if "Region" in df.columns:
        reg = df.groupby("Region")[TARGET].agg(["mean", "count"]).reset_index().sort_values("Region")
        axR.bar(reg["Region"].astype(str), reg["mean"] * 100, color=NAVY, edgecolor="white")
        axR.axhline(base * 100, color=ORANGE, ls="--", lw=1.2, label="overall rate")
        axR.set_xlabel("EPA Region")
        axR.set_ylabel("Detection rate (%)")
        axR.set_title("By EPA region", fontsize=10)
        axR.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "figure8_covariate_assoc")


def main():
    df, count_cols = load_data()
    if TARGET not in df.columns:
        sys.exit(f"ERROR: target column '{TARGET}' not found in model_table.csv")
    base = df[TARGET].mean()
    print(f"overall detection rate: {base*100:.1f}%\n")

    figure6(df, base)
    figure7(df, base, count_cols)
    figure8(df, base)
    print("\nAll exploratory figures written to", OUT)


if __name__ == "__main__":
    main()
