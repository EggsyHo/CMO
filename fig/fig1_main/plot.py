"""Fig 1 -- main result in three panels.

(a) final rel-L2 per arm, (b) mass residual on a log axis, (c) edge thickness
relative to the reference with the estimator floor drawn in. All five-seed
statistics; the paired numbers behind the text are not re-derived here.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figlib as F  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
ARMS = F.ORDER4
T95 = 2.776

EXPECT = {"cmo": (0.3845, 0.957), "fluxnet": (0.3615, 0.979),
          "nfv": (0.4338, 0.983), "fno": (0.6191, 0.624)}

F.apply()
fig = plt = F.plt.figure(figsize=F.figsize(164.6, 78))
gs = fig.add_gridspec(1, 3, wspace=0.62, left=0.085, right=0.985,
                      bottom=0.078, top=0.93)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

rel = {a: F.rel_l2_final(a) for a in ARMS}
res = {a: F.conservation(a) for a in ARMS}
shp = {a: F.sharpness(a) for a in ARMS}
assert all(abs(rel[a].mean() - EXPECT[a][0]) < 2e-3 for a in ARMS), "rel_l2 moved"
assert all(abs(shp[a]["ratio_median"].mean() - EXPECT[a][1]) < 2e-3 for a in ARMS), \
    "sharpness moved"

xs = np.arange(len(ARMS))


def seed_marks(ax, i, v, colour):
    ax.scatter(np.full(v.size, i) + np.linspace(-0.10, 0.10, v.size), v, s=4.5,
               facecolor="white", edgecolor=colour, linewidth=0.5, zorder=4)
    m = v.mean()
    ci = T95 * v.std(ddof=1) / np.sqrt(v.size)
    ax.plot([i - 0.24, i + 0.24], [m, m], color=colour, lw=1.0, zorder=5)
    for xx in (i - 0.24, i + 0.24):
        ax.plot([xx, xx], [m - ci, m + ci], color=colour, lw=0.6, zorder=5)
    return m


# ---- (a) accuracy
ax = axes[0]
for i, a in enumerate(ARMS):
    m = seed_marks(ax, i, rel[a], F.ARM[a][1])
    F.numeric(ax, f"{m:.2f}", i + 0.28, m, ha="left", va="center")
ax.set_xticks(xs, [F.ARM[a][0] for a in ARMS])
ax.set_xlim(-0.45, len(ARMS) - 0.02)
ax.set_ylim(0.29, 0.70)
ax.set_ylabel(r"relative $L_2$ error at $t=200$")
F.clean(ax)
F.letter(ax, "a")

# ---- (b) conservation: bars on a log axis, labels rotated clear of each other
ax = axes[1]
for i, a in enumerate(ARMS):
    v = res[a].mean()
    ax.bar(i, v - 1e-6, bottom=1e-6, width=0.46, color=F.ARM[a][1],
           edgecolor="none", zorder=3)
    lab = (rf"${v:.1e}$".replace("e-06", r"\times10^{-6}").replace("e+00", ""))
    ax.annotate(lab, (i, v), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", rotation=90,
                fontsize=F.SIZE["detail"], color=F.C["ink"])
ax.set_yscale("log")
ax.set_ylim(1e-6, 1e4)
ax.set_xticks(xs, [F.ARM[a][0] for a in ARMS])
ax.set_xlim(-0.62, len(ARMS) - 0.38)
ax.set_ylabel(r"mass residual $r(t=200)$")
F.clean(ax, grid_axis="y")
F.letter(ax, "b")

# ---- (c) edge thickness
ax = axes[2]
ref_w = float(np.mean([shp[a]["w_ref_median"].mean() for a in ARMS]))
floor_ratio = 1.1 / ref_w
ax.axhspan(0.55, floor_ratio, color=F.C["pale"], zorder=0)
ax.axhline(1.0, color=F.C["charcoal"], lw=0.6, ls=(0, (4, 2)), zorder=1)
ax.axhline(floor_ratio, color=F.C["charcoal"], lw=0.5, ls=(0, (1, 1.5)), zorder=1)
F.numeric(ax, "reference", -0.40, 1.0, ha="left", va="bottom", color=F.C["charcoal"])
F.numeric(ax, r"floor, $1.1$ cell", 3.38, floor_ratio, ha="right", va="bottom",
          color=F.C["charcoal"])
for i, a in enumerate(ARMS):
    m = seed_marks(ax, i, shp[a]["ratio_median"], F.ARM[a][1])
    F.numeric(ax, f"{m:.2f}", i + 0.28, m, ha="left", va="center")
ax.set_xticks(xs, [F.ARM[a][0] for a in ARMS])
ax.set_xlim(-0.45, len(ARMS) + 0.02)
ax.set_ylim(0.55, 1.06)
ax.set_ylabel("edge thickness / reference")
F.clean(ax)
F.letter(ax, "c")

for line in F.audit(fig, axes):
    print(" ", line)
for p in F.save(fig, OUT, "fig1_main"):
    print("wrote", os.path.basename(p))
