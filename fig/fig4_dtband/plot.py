"""Fig 4 -- the single-variable dt band.

Same 80 parameter vectors, two integration steps, nothing else changed. (a) per-seed
levels on both bands; (b) the scatter that comes with them; (c) the change per case,
which is what separates 'a tail got worse' from 'everything moved'.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figlib as F  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
CELLS = ["cmo", "fluxnet", "af_raw", "af_corr"]
REF_GAP = 0.0017
MEAN = {"cmo": (0.3864, 0.3876), "fluxnet": (0.3543, 0.2016),
        "af_raw": (0.3791, 0.4989), "af_corr": (0.4130, 0.3802)}
SD = {"cmo": (0.0067, 0.0067), "fluxnet": (0.0068, 0.0248),
      "af_raw": (0.0194, 0.1138), "af_corr": (0.0045, 0.0196)}
STYLE = {"af_corr": (0, (3, 1.6))}          # dashed: form B with the factor put back

F.apply()
fig = F.plt.figure(figsize=F.figsize(164.6, 78))
gs = fig.add_gridspec(1, 3, wspace=0.60, left=0.075, right=0.975,
                      bottom=0.145, top=0.88)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

band = {c: (F.dt_band(c, "coarse"), F.dt_band(c, "fine")) for c in CELLS}
for c in CELLS:
    assert band[c][0].shape == (5, 80)
    m = tuple(b.mean(axis=1) for b in band[c])
    for k, expect in enumerate(MEAN[c]):
        assert abs(m[k].mean() - expect) < 2e-3, f"{c} band{k}: {m[k].mean():.4f}"
        assert abs(m[k].std(ddof=1) - SD[c][k]) < 5e-3, f"{c} sd{k}"

# ---- (a) levels on both bands
ax = axes[0]
handles = []
for c in CELLS:
    co, fi = (b.mean(axis=1) for b in band[c])
    colour = F.ARM[c][1]
    ls = STYLE.get(c, "-")
    for k in range(5):
        ax.plot([0, 1], [co[k], fi[k]], color=colour, lw=0.5, alpha=0.55,
                ls=ls, zorder=3)
    ln, = ax.plot([0, 1], [co.mean(), fi.mean()], color=colour, lw=1.4, ls=ls,
                  marker="o", ms=2.6, mfc="white", mew=0.8, zorder=4,
                  label=F.ARM[c][0] + (r" $+\ dt$" if c == "af_corr" else ""))
    handles.append(ln)
ax.set_xticks([0, 1], [r"$dt = 0.100$", r"$dt = 0.016$"])
ax.set_xlim(-0.22, 1.22)
ax.set_ylim(0.15, 0.64)
for c in CELLS:                      # nothing may be silently clipped by the limit
    assert max(b.mean(axis=1).max() for b in band[c]) < ax.get_ylim()[1]
ax.set_ylabel(r"relative $L_2$ error, 80 cases")
F.clean(ax)
F.letter(ax, "a")

# ---- (b) scatter, as a dumbbell on a row per cell
ax = axes[1]
for i, c in enumerate(CELLS):
    y = len(CELLS) - 1 - i
    colour = F.ARM[c][1]
    a, b = SD[c]
    ax.plot([a, b], [y, y], color=colour, lw=1.2, zorder=3)
    ax.plot([a], [y], "o", ms=2.6, mfc=colour, mec=colour, zorder=4)
    ax.plot([b], [y], "o", ms=2.9, mfc="white", mec=colour, mew=0.8, zorder=4)
    F.numeric(ax, f"{b / a:.1f}$\\times$", max(a, b) + 0.006, y, ha="left",
              va="center", color=F.C["charcoal"])
ax.set_yticks(range(len(CELLS)), [F.ARM[c][0] for c in reversed(CELLS)])
ax.set_ylim(-0.6, len(CELLS) - 0.4)
ax.set_xlim(0, 0.155)
ax.set_xlabel("sd across the five seeds")
F.clean(ax, grid_axis="x")
F.letter(ax, "b")

# ---- (c) per-case change
ax = axes[2]
for c in ("cmo", "af_raw"):
    co, fi = band[c]
    d = (fi - co).ravel()
    d.sort()
    ax.step(d, np.arange(1, d.size + 1) / d.size, where="post",
            color=F.ARM[c][1], lw=1.1, zorder=3)
ax.axvline(REF_GAP, color=F.C["charcoal"], lw=0.6, ls=(0, (4, 2)), zorder=2)
ax.set_xlim(-0.03, 0.42)
ax.set_ylim(0, 1.0)
ax.set_xlabel(r"error at $0.016$ minus at $0.100$")
ax.set_ylabel("fraction of cases")
F.clean(ax)
F.letter(ax, "c")

frac_bad = float(((band["af_raw"][1] - band["af_raw"][0]) > 0).mean())
assert 0.68 < frac_bad < 0.75, frac_bad
F.numeric(ax, "CMO", 0.02, 0.90, ha="left", va="center", color=F.ARM["cmo"][1])
F.numeric(ax, "Form B", 0.22, 0.60, ha="left", va="center", color=F.ARM["af_raw"][1])
F.numeric(ax, "reference gap", REF_GAP + 0.006, 0.06, ha="left", va="center",
          color=F.C["charcoal"])

fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.995),
           handlelength=1.4, columnspacing=1.6)

for line in F.audit(fig, axes):
    print(" ", line)
for p in F.save(fig, OUT, "fig4_dtband"):
    print("wrote", os.path.basename(p))
