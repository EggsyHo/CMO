"""Fig 3 -- the 2 x 2 ablation inside one training file.

(a) the four cells with their 95% CI; (b) the same difference taken per seed, which
is the only form in which 'worth 3%' can be checked: neither arm's seed scatter is
small enough for the CI overlap to decide it on its own.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figlib as F  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
T95 = 2.776
CELLS = [("cmo", "cmo_nosep"), ("cmo_af", "cmo_af_nosep")]
MEAN = {"cmo": 0.3845, "cmo_nosep": 0.3974, "cmo_af": 0.3810, "cmo_af_nosep": 0.3923}
PCT = {"cmo": -3.23, "cmo_af": -2.89}
XLAB = ["form A\n(relative)", "form B\n(absolute, bounded)"]

F.apply()
fig = F.plt.figure(figsize=F.figsize(164.6, 78))
gs = fig.add_gridspec(1, 2, wspace=0.34, left=0.075, right=0.975,
                      bottom=0.105, top=0.93)
axes = [fig.add_subplot(gs[0, i]) for i in range(2)]

v = {a: F.rel_l2_final(a) for a in MEAN}
for a, m in MEAN.items():
    assert abs(v[a].mean() - m) < 2e-3, f"{a}: {v[a].mean():.4f} vs {m}"
    assert v[a].size == 5

# ---- (a) the four cells
ax = axes[0]
for gi, (sep, nos) in enumerate(CELLS):
    for off, a, filled in ((-0.15, nos, False), (0.15, sep, True)):
        x = gi + off
        m = v[a].mean()
        ci = T95 * v[a].std(ddof=1) / np.sqrt(v[a].size)
        colour = F.ARM[a][1]
        ax.plot([x, x], [m - ci, m + ci], color=colour, lw=0.9, zorder=3)
        ax.plot([x], [m], "o", ms=3.6, mfc=colour if filled else "white",
                mec=colour, mew=0.8, zorder=4)
        F.numeric(ax, f"{m:.3f}", x + 0.10, m + ci + 0.0015, ha="left",
                  va="bottom", color=F.C["charcoal"])
    ax.plot([gi - 0.15, gi + 0.15],
            [v[nos].mean(), v[sep].mean()], color=F.C["grid"], lw=0.8, zorder=1)
ax.set_xticks(range(len(CELLS)), XLAB)
ax.set_xlim(-0.45, 1.45)
ax.set_ylim(0.345, 0.437)
ax.set_ylabel(r"relative $L_2$ error")
F.clean(ax)
F.letter(ax, "a")
h1 = ax.plot([], [], "o", ms=3.6, mfc="white", mec=F.C["ink"], mew=0.8)[0]
h2 = ax.plot([], [], "o", ms=3.6, mfc=F.C["ink"], mec=F.C["ink"])[0]
ax.legend([h1, h2], ["no detail branch", "with detail branch"], loc="lower left",
          bbox_to_anchor=(0.14, 1.005), ncol=2, handletextpad=0.35)

# ---- (b) paired per-seed change
ax = axes[1]
gains = {}
for gi, (sep, nos) in enumerate(CELLS):
    d = v[nos] - v[sep]
    gains[sep] = d
    assert (d > 0).sum() == 5, f"{sep}: {d}"
    colour = F.ARM[sep][1]
    ax.scatter(np.full(5, gi) + np.linspace(-0.07, 0.07, 5), d, s=10,
               facecolor="white", edgecolor=colour, linewidth=0.7, zorder=4)
    ax.plot([gi - 0.16, gi + 0.16], [d.mean()] * 2, color=colour, lw=1.0, zorder=5)
    percent = 100 * (v[sep].mean() - v[nos].mean()) / v[nos].mean()
    assert abs(percent - PCT[sep]) < 0.05, f"{sep}: {percent:.2f} vs {PCT[sep]}"
    F.numeric(ax, f"${percent:.1f}\\%$", gi + 0.19, d.mean(), ha="left",
              va="center", color=colour)
ax.set_xticks([0, 1], XLAB)
ax.set_xlim(-0.45, 1.55)
ax.set_ylim(0.0, 0.037)
ax.set_ylabel("error without detail minus with detail")
F.clean(ax)
F.letter(ax, "b")
inter = gains["cmo_af"].mean() - gains["cmo"].mean()
assert abs(inter) < 0.002, inter
F.numeric(ax, f"interaction {inter:+.4f} (n.s.); 10/10 seeds same sign",
          -0.43, 0.0365, ha="left", va="top", color=F.C["charcoal"])

for line in F.audit(fig, axes):
    print(" ", line)
for p in F.save(fig, OUT, "fig3_ablation"):
    print("wrote", os.path.basename(p))
