"""Supplementary figure S2 - extrapolating to a 20x anisotropy ratio.

`results/_ext80_<cell>_s{0..4}.json` holds 80 cases per seed, built as
`--regime ext,kappa20 --n 40,40`: cases 0:40 inside the training range (kappa = 1),
cases 40:80 at kappa = 20, never seen in training and never retrained on.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import figlib as F  # noqa: E402

F.apply()

OUT = os.path.dirname(os.path.abspath(__file__))
# drawn in increasing-error order so panel (b) reads top to bottom
CELLS = ["fluxnet", "af_corr", "cmo", "af_raw"]
LABEL = {"fluxnet": "FluxNet", "af_corr": "Form B + factor", "cmo": "CMO",
         "af_raw": "Form B"}
COL = {"fluxnet": F.ARM["fluxnet"][1], "af_corr": F.C["coral"], "cmo": F.ARM["cmo"][1],
       "af_raw": F.C["coral_light"]}

E = {c: F.ext80(c) for c in CELLS}
assert all(v.shape == (5, 80) for v in E.values())
IN = {c: E[c][:, :40] for c in CELLS}
K20 = {c: E[c][:, 40:] for c in CELLS}
MEAN_IN = {c: IN[c].mean() for c in CELLS}
MEAN_K20 = {c: K20[c].mean() for c in CELLS}

# --- numbers the panels claim ------------------------------------------------
ORDER_IN = sorted(CELLS, key=lambda c: MEAN_IN[c])
ORDER_20 = sorted(CELLS, key=lambda c: MEAN_K20[c])
assert ORDER_IN == ORDER_20 == CELLS, (ORDER_IN, ORDER_20)   # no crossover
RATIO = {c: MEAN_K20[c] / MEAN_IN[c] for c in CELLS}
assert abs(RATIO["cmo"] - 1.133) < 0.01, RATIO
assert abs(RATIO["fluxnet"] - 1.044) < 0.01, RATIO
WORSE = float((K20["cmo"] > K20["fluxnet"]).mean())

fig = F.plt.figure(figsize=F.figsize(90, 110))
gs = fig.add_gridspec(2, 1, hspace=0.42, left=0.155, right=0.968,
                      bottom=0.058, top=0.945)
axes = [fig.add_subplot(gs[i]) for i in range(2)]
rng = np.random.default_rng(11)


def strip(ax, values, x, color, alpha, width=0.26, ms=1.9):
    v = np.asarray(values, float)
    ax.plot(x + rng.uniform(-width / 2, width / 2, v.size), v, "o", ms=ms,
            mfc=color, mec="none", alpha=alpha, zorder=3)
    m = float(v.mean())
    ax.plot([x - width * 0.60, x + width * 0.60], [m, m], color=F.C["ink"], lw=1.1,
            solid_capstyle="butt", zorder=5)
    return m


# --- (a) every case, both halves --------------------------------------------
ax = axes[0]
for i, c in enumerate(CELLS):
    strip(ax, np.median(IN[c], axis=0), i - 0.17, COL[c], 0.32)
    strip(ax, np.median(K20[c], axis=0), i + 0.17, COL[c], 0.90)

ax.set_xticks(range(len(CELLS)))
ax.set_xticklabels([LABEL[c] for c in CELLS], rotation=20, ha="right",
                   rotation_mode="anchor")
ax.set_xlim(-0.6, len(CELLS) - 0.4)
ax.set_ylim(0.0, 1.10)
ax.set_ylabel(r"relative $L_2$ error")
F.clean(ax)
F.letter(ax, "a", x=-0.02, y=1.03)

ax.plot(ax.get_xlim()[0] + 0.06, 0.0, zorder=0)          # keep the limits
h = [F.plt.Line2D([], [], marker="o", ls="", ms=3.0, mfc=F.C["ink"], mec="none",
                  alpha=alp) for alp in (0.32, 0.90)]
ax.legend(h, ["in range", r"$\kappa$ = 20"], loc="upper left", ncol=2)

# --- (b) the same means, joined ---------------------------------------------
ax = axes[1]
handles = []
for c in CELLS:
    ln, = ax.plot([0, 1], [MEAN_IN[c], MEAN_K20[c]], "-o", color=COL[c], lw=1.0,
                  ms=3.0, zorder=3, label=f"{LABEL[c]}  {RATIO[c] * 100 - 100:+.0f}%")
    handles.append(ln)

ax.set_xticks([0, 1])
ax.set_xticklabels(["training range", r"$\kappa$ = 20"])
ax.set_xlim(-0.10, 1.10)
ax.set_ylim(0.36, 0.60)
ax.set_ylabel(r"mean relative $L_2$ error")
F.clean(ax)
F.letter(ax, "b", x=-0.02, y=1.03)
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 1.0),
          handlelength=1.6, labelspacing=0.25)

for line in F.audit(fig, axes):
    print(" ", line)
for p in F.save(fig, OUT, "figS2_extrap"):
    print(" ", os.path.basename(p), os.path.getsize(p) // 1024, "kB")
print("  a/b:", {c: (round(MEAN_IN[c], 3), round(MEAN_K20[c], 3),
                     round(RATIO[c], 3)) for c in CELLS}, "worse", round(WORSE, 3))
