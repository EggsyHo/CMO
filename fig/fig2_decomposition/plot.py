"""Fig 2 -- what the error is made of.

(a) per-arm dumbbell: total error and the part left once the best rigid shift is
removed; (b) that removable part as a per-trajectory CDF, which is where the
'common under-transport' claim actually lives.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figlib as F  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
ARMS = F.ORDER7
# share of the error removed by the shift, as printed in the text table
SHARE = {"fluxnet": 30.4, "cmo": 27.3, "cmo_nosep": 26.2, "cmo_af": 34.5,
         "cmo_af_nosep": 33.1, "nfv": 33.5, "fno": 15.7}
SHORT = {"cmo_nosep": "CMO $-$detail", "cmo_af": "Form B",
         "cmo_af_nosep": "Form B $-$detail"}

F.apply()
fig = F.plt.figure(figsize=F.figsize(164.6, 78))
gs = fig.add_gridspec(1, 2, wspace=0.42, left=0.105, right=0.975,
                      bottom=0.125, top=0.93)
axes = [fig.add_subplot(gs[0, i]) for i in range(2)]

tot = {a: F.lag(a)[0] for a in ARMS}
shp = {a: F.lag(a)[1] for a in ARMS}
for a in ARMS:
    share = 100 * (1 - shp[a].mean() / tot[a].mean())
    assert abs(share - SHARE[a]) < 0.15, f"{a}: {share:.1f} vs {SHARE[a]}"
    assert tot[a].size == 300


def label(a: str) -> str:
    return SHORT.get(a, F.ARM[a][0])


# ---- (a) dumbbell
ax = axes[0]
for i, a in enumerate(ARMS):
    y = len(ARMS) - 1 - i
    colour = F.ARM[a][1]
    m_s, m_t = shp[a].mean(), tot[a].mean()
    ax.plot([m_s, m_t], [y, y], color=colour, lw=1.3, solid_capstyle="butt", zorder=3)
    ax.plot([m_s], [y], "o", ms=2.6, mfc=colour, mec=colour, zorder=4)
    ax.plot([m_t], [y], "o", ms=2.9, mfc="white", mec=colour, mew=0.7, zorder=4)
    F.numeric(ax, f"{SHARE[a]:.0f}%", m_t + 0.008, y, ha="left", va="center",
              color=F.C["charcoal"])
ax.set_yticks(range(len(ARMS)),
              [label(a) for a in reversed(ARMS)])
ax.set_ylim(-0.6, len(ARMS) - 0.4)
ax.set_xlim(0.235, 0.685)
ax.set_xlabel(r"relative $L_2$ error")
F.clean(ax, grid_axis="x")
F.letter(ax, "a")
h1 = ax.plot([], [], "o", ms=2.9, mfc="white", mec=F.C["ink"], mew=0.7)[0]
h2 = ax.plot([], [], "o", ms=2.6, mfc=F.C["ink"], mec=F.C["ink"])[0]
ax.legend([h1, h2], ["total", r"after best shift"], loc="lower left",
          bbox_to_anchor=(0.14, 1.005), ncol=2, handletextpad=0.35)

# ---- (b) ECDF of the removable part
ax = axes[1]
handles = []
for a in ARMS:
    d = np.sort(tot[a] - shp[a])
    ln, = ax.step(d, np.arange(1, d.size + 1) / d.size, where="post",
                  color=F.ARM[a][1], lw=0.9, zorder=3, label=label(a))
    handles.append(ln)
ax.set_xlim(-0.005, 0.40)
ax.set_ylim(0, 1.0)
ax.set_xlabel("error removed by the best rigid shift")
ax.set_ylabel("fraction of trajectories")
F.clean(ax)
F.letter(ax, "b")
zero = {a: int((tot[a] - shp[a] == 0).sum()) for a in ARMS}
assert all(50 <= v <= 72 for v in zero.values()), zero
F.numeric(ax, f"no shift: {min(zero.values())}-{max(zero.values())} of 300",
          -0.003, 1.0, ha="left", va="top", color=F.C["charcoal"])
ax.legend(handles=handles, loc="lower right", ncol=2, columnspacing=0.8,
          handlelength=1.0, fontsize=F.SIZE["detail"])

for line in F.audit(fig, axes):
    print(" ", line)

# the legend sits inside (b); assert it sits on empty paper, not on a curve
lx0, ly0, lx1, ly1 = F.legend_box_data(axes[1])
for a in ARMS:
    d = np.sort(tot[a] - shp[a])
    cdf = np.arange(1, d.size + 1) / d.size
    assert np.interp(lx0, d, cdf) > ly1 + 0.02, f"legend covers {a}"
print(f"  legend (b) x {lx0:.2f}-{lx1:.2f} y {ly0:.2f}-{ly1:.2f}, lowest curve above it")
for p in F.save(fig, OUT, "fig2_decomposition"):
    print("wrote", os.path.basename(p))
