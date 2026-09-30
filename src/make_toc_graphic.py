"""
make_toc_graphic.py - Table-of-Contents (TOC) / abstract graphic (ACS TOC-graphic specification).

Built to the ACS specification exactly (ACS "Guidelines for Table of Contents /
Abstract Graphics", updated 28 Feb 2024):

  * fits in an area no larger than 3.25 x 1.75 inches (8.25 x 4.45 cm)
  * sans serif type, preferably 8 pt, never smaller than 6 pt
  * TIFF at 300 dpi for colour  (PNG also written, for your own preview)
  * legible at thumbnail size; tells the story like a single slide
  * depicts an environmental outcome, not a synthesis schematic
    (ACS guidance favours outcome figures over synthesis schemes)

Paragon Plus validates the dimensions on upload, so the figure size is fixed and
bbox_inches='tight' is deliberately NOT used: that would crop and change the size.

The only quantitative content is your real headline result (gain in AUC of about
0.02 with 95% intervals); the left panel is a schematic and invents no data.

Output: outputs/figures/TOC_graphic.tif   <- upload this one
        outputs/figures/TOC_graphic.png   <- preview only
Run:    python src/make_toc_graphic.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from pathlib import Path

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ACS: sans serif, >= 6 pt, preferably 8 pt
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

NAVY, BLUE, ORANGE, GREY = "#1F3864", "#3A6EA5", "#C55A11", "#6B6B6B"

# EXACT ACS size. Do not change: Paragon Plus validates it.
FIG_W, FIG_H = 3.25, 1.75
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

# ---------------- LEFT: the question (schematic) ----------------
axL = fig.add_axes([0.005, 0.14, 0.42, 0.72])
axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis("off")

axL.add_patch(Circle((5.0, 5.0), 1.5, color=BLUE, zorder=5))
axL.text(5.0, 5.0, "PFAS", ha="center", va="center", color="white",
         fontsize=6.5, fontweight="bold", zorder=6)

for x, y, lbl in [(1.5, 8.5, "Ind."), (8.5, 8.5, "WWTP"),
                  (1.5, 1.5, "Air."), (8.5, 1.5, "Mil.")]:
    axL.add_patch(FancyBboxPatch((x - 1.15, y - 0.7), 2.3, 1.4,
                                 boxstyle="round,pad=0.05", color=GREY, zorder=4))
    axL.text(x, y, lbl, ha="center", va="center", color="white",
             fontsize=6.0, zorder=5)
    axL.plot([x, 5.0], [y, 5.0], color=GREY, lw=0.7, ls=(0, (2, 2)), zorder=2)

axL.text(5.0, 9.6, "Nearby sources?", ha="center", va="center",
         fontsize=7.0, color=NAVY, fontweight="bold")

arrow = FancyArrowPatch((10.4, 5.0), (12.2, 5.0), transform=axL.transData,
                        arrowstyle="-|>", mutation_scale=8, color=ORANGE,
                        lw=1.4, clip_on=False, zorder=10)
axL.add_patch(arrow)

# ---------------- RIGHT: the answer (real numbers) ----------------
axR = fig.add_axes([0.61, 0.26, 0.36, 0.52])
targets = ["Any", "Reg.", "Unreg."]
deltas = [0.018, 0.010, 0.023]
lo = [0.011, 0.013, 0.014]      # distance from bar top down to lower percentile bound
hi = [0.017, 0.026, 0.020]      # distance up to upper percentile bound
axR.bar(range(3), deltas, width=0.6, color=BLUE, edgecolor="white",
        yerr=[lo, hi], capsize=1.8,
        error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "#333333"})
axR.axhline(0, color="black", lw=0.6)
axR.set_xticks(range(3))
axR.set_xticklabels(targets, fontsize=6.0)
axR.set_ylim(-0.005, 0.05)
axR.set_yticks([0, 0.02, 0.04])
axR.set_yticklabels(["0", "0.02", "0.04"], fontsize=6.0)
axR.set_ylabel("Gain in AUC", fontsize=6.5, color=NAVY, labelpad=1.5)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)
axR.tick_params(length=1.6, width=0.5, pad=1.2)

fig.text(0.79, 0.90, "Source proximity adds\nalmost nothing", ha="center",
         va="center", fontsize=6.5, color=ORANGE, fontweight="bold",
         linespacing=1.15)

# ---------------- bottom line ----------------
fig.text(0.5, 0.035,
         "PFAS detection is regional, not local  ·  24,837 U.S. water locations",
         ha="center", va="bottom", fontsize=6.2, color=NAVY)

# No bbox_inches='tight': the size must stay exactly 3.25 x 1.75 in.
fig.savefig(OUT / "TOC_graphic.png", dpi=300, facecolor="white")
plt.close(fig)

# Flatten to RGB and write the TIFF. Matplotlib writes RGBA; a redundant alpha
# channel is a known nuisance in journal production workflows (it can render on a
# black background), so the alpha is composited onto white and dropped.
from PIL import Image
src = Image.open(OUT / "TOC_graphic.png")
if src.mode in ("RGBA", "LA"):
    bg = Image.new("RGB", src.size, "white")
    bg.paste(src, mask=src.split()[-1])
    flat = bg
else:
    flat = src.convert("RGB")
flat.save(OUT / "TOC_graphic.tif", format="TIFF", dpi=(300, 300), compression="tiff_lzw")
flat.save(OUT / "TOC_graphic.png", dpi=(300, 300))

for f in ("TOC_graphic.tif", "TOC_graphic.png"):
    im = Image.open(OUT / f)
    w, h = im.size
    ok = (w / 300 <= 3.2501) and (h / 300 <= 1.7501)
    print(f"{f}: {im.format}, mode {im.mode}, {w} x {h} px "
          f"-> {w/300:.2f} x {h/300:.2f} in at 300 dpi  [{'PASS' if ok else 'TOO LARGE'}]")
print("\nACS: area no larger than 3.25 x 1.75 in, sans serif >= 6 pt, TIFF at 300 dpi.")
print("Upload TOC_graphic.tif to Paragon Plus as the TOC/abstract graphic.")
