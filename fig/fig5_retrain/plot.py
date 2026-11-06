"""Fig 5 -- the retraining control.

Both forms are retrained from scratch on the same three-rate mixture, so the only
difference left is the parametrisation. (a) the three bands; (b) the scatter.
The middle band is the one that matters: it is in the training set.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figlib as F  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
CELLS = ["mix_cmo", "mix_af"]
BANDS = ["coarse", "mid", "fine"]
XT = [r"$0.100$", r"$0.050$", r"$0.016$"]
MEAN = {"mix_cmo": (0.3712, 0.3719, 0.3723), "mix_af": (0.4005, 0.4078, 0.4858)}
SD = {"mix_cmo": (0.0166, 0.0166, 0.0167), "mix_af": (0.0146, 0.0564, 0.1526)}

F.apply()
fig = F.plt.figure(figsize=F.figsize(164.6, 78))
gs = fig.add_gridspec(1, 2, wspace=0.34, left=0.075, right=0.975,
                      bottom=0.145, top=0.93)
axes = [fig.add_subplot(gs[0, i]) for i in range(2)]

vals = {c: np.array([F.dt_band(c, b).mean(axis=1) for b in BANDS]) for c in CELLS}
for c in CELLS:
    m = vals[c].mean(axis=1)
    s = vals[c].std(axis=1, ddof=1)
    for k in range(3):
        assert abs(m[k] - MEAN[c][k]) < 2e-3, f"{c} {BANDS[k]}: {m[k]:.4f}"
        assert abs(s[k] - SD[c][k]) < 3e-3, f"{c} sd {BANDS[k]}: {s[k]:.4f}"

# ---- (a) levels
ax = axes[0]
ax.axvspan(0.72, 1.28, color=F.C["pale"], zorder=0)
for c in CELLS:
    colour = F.ARM[c][1]
    for k in range(5):
        ax.plot(range(3), vals[c][:, k], color=colour, lw=0.5, alpha=0.55, zorder=3)
    ax.plot(range(3), vals[c].mean(axis=1), color=colour, lw=1.4, marker="o", ms=2.6,
            mfc="white", mew=0.8, zorder=4, label=F.ARM[c][0])
    # two lines, as in panel (b): at 7 pt the one-line form runs past the axes
    F.numeric(ax, F.ARM[c][0].replace(", ", "\n"), 2.06, vals[c].mean(axis=1)[2],
              ha="left", va="center", color=colour)
ax.set_xticks(range(3), XT)
ax.set_xlim(-0.2, 2.75)
ax.set_ylim(0.32, 0.80)
for c in CELLS:
    assert vals[c].max() < ax.get_ylim()[1]
ax.set_xlabel(r"$dt$ used to score")
ax.set_ylabel(r"relative $L_2$ error, 80 cases")
F.clean(ax)
F.letter(ax, "a")
gap = 100 * (vals["mix_af"].mean(axis=1)[1] / vals["mix_cmo"].mean(axis=1)[1] - 1)
assert abs(gap - 9.7) < 0.3, gap
F.numeric(ax, r"in training set", 1.0, 0.775, ha="center", va="top",
          color=F.C["charcoal"])
F.numeric(ax, f"{gap:.1f}% apart", 1.0, 0.745, ha="center", va="top",
          color=F.C["charcoal"])

# ---- (b) scatter across bands, log scale
ax = axes[1]
for c in CELLS:
    colour = F.ARM[c][1]
    ax.plot(range(3), vals[c].std(axis=1, ddof=1), color=colour, lw=1.1,
            marker="o", ms=2.6, mfc="white", mew=0.8, zorder=4)
ax.set_yscale("log")
ax.set_xticks(range(3), XT)
ax.set_xlim(-0.2, 2.95)
ax.set_ylim(0.01, 0.32)
ax.set_xlabel(r"$dt$ used to score")
ax.set_ylabel("sd across the five seeds")
F.clean(ax)
F.letter(ax, "b")
ratio = SD["mix_af"][2] / SD["mix_af"][0]
assert abs(ratio - 10.5) < 0.5, ratio
F.numeric(ax, f"Form B\n{ratio:.0f}$\\times$", 2.05, SD["mix_af"][2], ha="left",
          va="center", color=F.ARM["mix_af"][1])
F.numeric(ax, "Form A\nflat", 2.05, SD["mix_cmo"][2], ha="left", va="center",
          color=F.ARM["mix_cmo"][1])

for line in F.audit(fig, axes):
    print(" ", line)
for p in F.save(fig, OUT, "fig5_retrain"):
    print("wrote", os.path.basename(p))
