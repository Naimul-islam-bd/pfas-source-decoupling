"""
Figure 3: monitoring-efficiency capture curve (impact Angle 3).

Reads:  outputs/tables/monitoring_efficiency.csv   (from impact_analyses.py)
Writes: outputs/figures/figure3_monitoring_efficiency.(tif|png|pdf)

Shows the share of detections captured when systems are ranked by distance to
the nearest source and only the nearest fraction is sampled. A curve near the
diagonal means proximity targeting is no better than random selection.

Run:  python src/make_figure3_monitoring.py
"""
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path

IN = Path("outputs/tables/monitoring_efficiency.csv")
OUT = Path("outputs/figures"); OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial"],
  "font.size":9,"axes.titlesize":10,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,
  "legend.fontsize":8,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#444444",
  "figure.dpi":300,"savefig.dpi":300})
NAVY,BLUE,ORANGE,GREY="#1F3864","#2E5A88","#C55A11","#6B6B6B"

d = pd.read_csv(IN)
x = d["nearest_pct"].to_numpy()
fig, ax = plt.subplots(figsize=(5.6,4.4))
ax.plot([0,100],[0,100], ls="--", lw=1.3, color=GREY, label="Random selection")
ax.plot(x, d["capture_unregulated_pct"], "o-", color=ORANGE, lw=1.9, ms=5, label="Unregulated (PFPeA, PFHxA, PFBA)")
ax.plot(x, d["capture_any_pct"], "s-", color=BLUE, lw=1.7, ms=4, label="Any PFAS")
ax.plot(x, d["capture_regulated_pct"], "^-", color=NAVY, lw=1.5, ms=4, label="Regulated (PFOA, PFOS)")
ax.set_xlabel("Nearest fraction of systems sampled (%), ranked by distance to source")
ax.set_ylabel("Share of detections captured (%)")
ax.set_xlim(0,100); ax.set_ylim(0,100)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
for ext in ("png","pdf"):
    fig.savefig(OUT/f"figure3_monitoring_efficiency.{ext}", dpi=300, bbox_inches="tight")
# TIFF (journal upload)
fig.savefig(OUT/"figure3_monitoring_efficiency_raw.png", dpi=300, bbox_inches="tight")
from PIL import Image
Image.open(OUT/"figure3_monitoring_efficiency_raw.png").convert("RGB").save(
    OUT/"figure3_monitoring_efficiency.tif", format="TIFF", dpi=(300,300), compression="tiff_lzw")
plt.close(fig)
print("saved figure3_monitoring_efficiency.tif/.png/.pdf ->", OUT)
