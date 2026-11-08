"""Plot val AUROC vs footprint for the design grid."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).parent / "results" / "design"
rows = json.loads((D / "design_grid.json").read_text())
S = json.loads((D / "design_summary.json").read_text())
INK, MUTED, GRID, BLUE, AMBER = "#16205F", "#5B6275", "#D9DCE8", "#0F2DB3", "#B07810"
YMIN = 0.80

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(6.6, 4.1), dpi=300)
fig.patch.set_facecolor("#F0F2F9")
ax.set_facecolor("#F0F2F9")
shown = [r for r in rows if r["val_auroc"] >= YMIN]
ax.scatter([r["footprint_mb"] for r in shown], [r["val_auroc"] for r in shown], s=22, color="#B8BDCC",
           edgecolor="white", linewidth=0.6, zorder=2)

# frontier: best AUROC at or below each footprint
front, best = [], -1
for r in sorted(rows, key=lambda r: (r["footprint_mb"], -r["val_auroc"])):
    if r["val_auroc"] > best:
        best = r["val_auroc"]
        front.append(r)
ax.step([r["footprint_mb"] for r in front], [r["val_auroc"] for r in front], where="post", color=BLUE, lw=2, zorder=3)
ax.scatter([r["footprint_mb"] for r in front], [r["val_auroc"] for r in front], s=40, color=BLUE,
           edgecolor="white", linewidth=1.2, zorder=4)

named = [("heavy_reference", "WideResNet-50, 131 MB", (128, 0.86), "right"),
         ("current_default", "Our default, 13.5 MB", (24, 0.935), "left"),
         ("frugal_pick", "Smallest within 0.01\nof the best, 2.9 MB", (4.2, 0.885), "left")]
for k, label, xy, ha in named:
    r = S[k]
    ax.scatter([r["footprint_mb"]], [r["val_auroc"]], s=120, facecolor="none", edgecolor=AMBER, linewidth=2.2, zorder=5)
    ax.annotate(label, (r["footprint_mb"], r["val_auroc"]), xytext=xy, textcoords="data", ha=ha, va="center",
                fontsize=11.5, color=INK, fontweight="bold", zorder=6,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9, shrinkA=4, shrinkB=7))

ax.set_xscale("log")
ax.set_xticks([3, 10, 30, 100])
ax.set_xticklabels(["3 MB", "10 MB", "30 MB", "100 MB"])
ax.set_ylim(YMIN, 1.012)
ax.set_xlabel("Footprint: model weights used + memory bank (log scale)")
ax.set_ylabel("Validation AUROC")
ax.grid(True, color=GRID, lw=0.8, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
hidden = len(rows) - len(shown)
ax.text(0.99, 0.02, f"{len(shown)} of {len(rows)} designs shown; {hidden} below 0.80", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=10.5, color=MUTED)
fig.tight_layout()
fig.savefig(D / "design_space.png", facecolor=fig.get_facecolor())
print("saved", D / "design_space.png", "front", len(front))
