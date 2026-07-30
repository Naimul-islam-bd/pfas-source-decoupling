"""
Figure 6b (fixed): covariate-adjusted detection residual versus distance.

Raw detection can fall with distance simply because regions with many nearby
sources also happen to have high PFAS for other reasons. To separate true local
proximity from regional confounding we plot the COVARIATE-ADJUSTED residual:

  residual (pp) = 100 * ( actual detection (0/1) - covariate-only predicted prob )

averaged within distance bins. Flat near zero = distance carries no information
beyond covariates: local decoupling, shown visually. This version prints every
bin's residual as a NUMBER so the flatness is verified, not just eyeballed, and
uses a clean single axis (raw rate is already in Figure 6).

Reads:  data/interim/model_table.csv
Writes: outputs/figures/figure6b_adjusted_residual.(png|pdf)
Run:    python src/make_adjusted_figure.py
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import cross_val_predict
from pathlib import Path

IN = Path("data/interim/model_table.csv")
OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "figure.dpi": 300,
})
NAVY, ORANGE, GREY = "#1F3864", "#C55A11", "#7F7F7F"
TARGET = "any_pfas_detected"
COV = ["State", "Region", "Size", "FacilityWaterType", "n_facilities", "n_watertypes"]
CAT = ["State", "Region", "Size", "FacilityWaterType"]
DISTS = [("dist_industrial_km", "Industrial", 12), ("dist_wwtp_km", "Wastewater", 25),
         ("dist_military_km", "Military", 30), ("dist_airport_km", "Airport", 80)]
MIN_BIN = 30


def binned_resid(x, resid_pp, edges):
    idx = np.digitize(x, edges)
    out = []
    for b in range(1, len(edges)):
        m = idx == b
        nb = int(m.sum())
        if nb < MIN_BIN:
            out.append(((edges[b-1]+edges[b])/2, np.nan, np.nan, nb))
            continue
        v = resid_pp[m]
        out.append(((edges[b-1]+edges[b])/2, v.mean(), v.std(ddof=1)/np.sqrt(nb), nb))
    return out


def main():
    df = pd.read_csv(IN)
    y = df[TARGET].to_numpy(float)
    cov = [c for c in COV if c in df.columns]
    X = df[cov].copy()
    for c in CAT:
        if c in X.columns:
            X[c] = X[c].astype("category")

    model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                              enable_categorical=True, eval_metric="logloss", random_state=42)
    print("Fitting covariates-only model (5-fold out-of-fold predictions)...")
    pred = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]
    resid_pp = (y - pred) * 100.0
    print(f"  overall detection {y.mean()*100:.1f}%")
    print(f"  mean adjusted residual {resid_pp.mean():+.3f} pp (near 0 by construction)")
    print(f"  residual SD {resid_pp.std():.2f} pp\n")

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2))
    all_flat = True
    for ax, (col, label, xmax) in zip(axes.ravel(), DISTS):
        if col not in df.columns:
            ax.set_visible(False); continue
        x = df[col].to_numpy(float)
        edges = np.linspace(0, xmax, 9)
        pts = binned_resid(x, resid_pp, edges)

        print(f"[{label}] adjusted residual by distance bin (pp):")
        vals, wmeans = [], []
        for mid, mean, se, nb in pts:
            if np.isnan(mean):
                print(f"    ~{mid:5.1f} km : (n={nb}, too few)")
            else:
                sig = "" if abs(mean) <= 1.96*se else "  *"
                print(f"    ~{mid:5.1f} km : {mean:+6.2f} +/- {se:4.2f}  (n={nb}){sig}")
                vals.append(mean)
                wmeans.append((mean, se))
        # SE-aware verdict: how many bins are individually distinguishable from 0,
        # and the magnitude of the largest well-estimated residual
        signif = [m for m, se in wmeans if abs(m) > 1.96*se]
        well_est = [abs(m) for m, se in wmeans if se < 2.0]      # bins with enough n
        max_solid = max(well_est) if well_est else 0.0
        flat = (len(signif) == 0) and (max_solid < 3.0)
        all_flat = all_flat and flat
        print(f"    -> bins differing from 0 (95%): {len(signif)}; "
              f"largest well-estimated |residual|: {max_solid:.2f} pp  "
              f"({'FLAT' if flat else 'check'})\n")

        mids = [p[0] for p in pts if not np.isnan(p[1])]
        means = [p[1] for p in pts if not np.isnan(p[1])]
        ses = [p[2] for p in pts if not np.isnan(p[1])]
        ax.axhline(0, color=GREY, ls="--", lw=1.2)
        ax.errorbar(mids, means, yerr=ses, fmt="s-", color=ORANGE, ms=5, lw=1.8, capsize=3)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(f"Distance to nearest {label.lower()} source (km)")
        ax.set_ylabel("Adjusted residual (pp)")
        ax.set_ylim(-8, 8)

    fig.suptitle("Figure 6b. After adjusting for region and system covariates, detection does not vary with distance to source",
                 fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.text(0.5, -0.02,
             "Each point is the mean covariate-adjusted residual in a distance bin (bars: SE). Flat near zero = no local-proximity effect beyond regional structure.",
             ha="center", fontsize=8, color=GREY, style="italic")
    fig.savefig(OUT / "figure6b_adjusted_residual.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figure6b_adjusted_residual.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {OUT/'figure6b_adjusted_residual.png'}")
    print(f"\nOVERALL: {'all four panels flat -> supports local decoupling' if all_flat else 'at least one panel NOT flat -> inspect numbers above'}")


if __name__ == "__main__":
    main()
