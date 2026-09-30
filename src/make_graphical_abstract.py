"""
Graphical abstract (Elsevier graphical-abstract specification).
EP spec: 531 x 1328 px (h x w) or proportionally more, sans-serif, TIFF preferred.
Only real headline numbers are used.
Writes: outputs/figures/graphical_abstract.(tif|png)
Run:    python src/make_graphical_abstract.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from pathlib import Path
OUT = Path("outputs/figures"); OUT.mkdir(parents=True, exist_ok=True)
matplotlib.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial"],
  "figure.dpi":300,"savefig.dpi":300})
NAVY,BLUE,ORANGE,GREY = "#1F3864","#3A6EA5","#C55A11","#6B6B6B"
DASH = u"\u2013"  # en dash for the range

W,H = 4.427, 1.77   # -> 1328 x 531 px at 300 dpi
fig = plt.figure(figsize=(W,H), facecolor="white")

# LEFT: sources -> PFAS schematic
axL = fig.add_axes([0.005,0.12,0.28,0.78]); axL.set_xlim(0,10); axL.set_ylim(0,10); axL.axis("off")
axL.add_patch(Circle((5,5),1.5,color=BLUE,zorder=5))
axL.text(5,5,"PFAS",ha="center",va="center",color="white",fontsize=6.2,fontweight="bold",zorder=6)
for x,y,l in [(1.5,8.4,"Ind."),(8.5,8.4,"WWTP"),(1.5,1.6,"Air."),(8.5,1.6,"Mil.")]:
    axL.add_patch(FancyBboxPatch((x-1.1,y-0.6),2.2,1.2,boxstyle="round,pad=0.05",color=GREY,zorder=4))
    axL.text(x,y,l,ha="center",va="center",color="white",fontsize=5.4,zorder=5)
    axL.plot([x,5],[y,5],color=GREY,lw=0.6,ls=(0,(2,2)),zorder=2)
axL.text(5,9.7,"Nearby sources?",ha="center",fontsize=6.2,color=NAVY,fontweight="bold")
axL.text(5,0.1,"proximity adds \u2248 0",ha="center",fontsize=5.8,color=ORANGE,fontweight="bold")

# MIDDLE: real decoupling deltas (Table 2, spatial-block)
axM = fig.add_axes([0.375,0.28,0.22,0.50])
axM.bar([0,1,2],[0.018,0.010,0.023],width=0.62,color=BLUE,edgecolor="white",
        yerr=[[0.011,0.013,0.014],[0.017,0.026,0.020]],capsize=1.5,
        error_kw={"elinewidth":0.7,"capthick":0.7,"ecolor":"#333"})
axM.axhline(0,color="black",lw=0.6); axM.set_xticks([0,1,2])
axM.set_xticklabels(["Any","Reg.","Unreg."],fontsize=5.4)
axM.set_ylim(-0.006,0.05); axM.set_yticks([0,0.02,0.04]); axM.set_yticklabels(["0","0.02","0.04"],fontsize=5.2)
axM.set_ylabel("Gain in AUC",fontsize=5.6,color=NAVY,labelpad=1.5)
for s in ("top","right"): axM.spines[s].set_visible(False)
axM.tick_params(length=1.4,width=0.5,pad=1.1)
axM.set_title("Source proximity\nadds almost nothing",fontsize=5.8,color=ORANGE,fontweight="bold",pad=3)

fig.patches.append(FancyArrowPatch((0.610,0.5),(0.655,0.5),transform=fig.transFigure,
    arrowstyle="-|>",mutation_scale=8,color=ORANGE,lw=1.3,clip_on=False))

# RIGHT: the impact (real numbers)
axR = fig.add_axes([0.665,0.12,0.325,0.78]); axR.axis("off")
axR.text(0.5,0.93,"A national regulatory\nblind spot",ha="center",va="top",fontsize=6.4,color=NAVY,fontweight="bold",linespacing=1.1)
axR.text(0.5,0.55,"1,497",ha="center",fontsize=14,color=ORANGE,fontweight="bold")
axR.text(0.5,0.38,"water systems carry only\nunregulated PFAS",ha="center",va="top",fontsize=5.4,color="#333",linespacing=1.1)
axR.text(0.5,0.10,"serving 44"+DASH+"65 million people",ha="center",fontsize=5.8,color=NAVY,fontweight="bold")

fig.text(0.5,0.03,"PFAS occurrence is diffuse and regional, not locally source-bound  \u00B7  24,837 UCMR5 locations",
         ha="center",va="bottom",fontsize=5.0,color=NAVY)

fig.savefig(OUT/"graphical_abstract.png",dpi=300,facecolor="white")
plt.close(fig)
from PIL import Image
im=Image.open(OUT/"graphical_abstract.png").convert("RGB")
im.save(OUT/"graphical_abstract.tif",format="TIFF",dpi=(300,300),compression="tiff_lzw")
w,h=im.size
print(f"graphical_abstract: {w} x {h} px -> {'OK' if w>=1328 and h>=531 else 'CHECK'}")
