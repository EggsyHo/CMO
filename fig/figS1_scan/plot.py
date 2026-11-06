"""Supplementary figure S1 - zero-shot on a 20x anisotropy ratio.

Two panels over the same 80 cases (`adr2d_kappa.h5`: 40 at kappa = 1, 40 at
kappa = 20, no retraining):
  a  the per-case difference between CMO and FluxNet,
  b  the per-case gain from the detail branch, for both forms.

Per-case errors are recomputed from the rollout npz with the metric the
evaluation used, because the summary json keeps only pooled statistics.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import figlib as F  # noqa: E402

F.apply()

OUT = os.path.dirname(os.path.abspath(__file__))
LEVELS = (1.0, 20.0)
MASKS = {k: F.kappa_mask(k) for k in LEVELS}
ERR = {a: F.zshot_cases(a) for a in
       ("cmo", "cmo_nosep", "cmo_af", "cmo_af_nosep", "fluxnet")}
for k, m in MASKS.items():
    assert m.sum() == 40, (k, m.sum())

# --- numbers the panels claim ------------------------------------------------
DIFF = {k: (ERR["cmo"][m] - ERR["fluxnet"][m]) / ERR["fluxnet"][m] * 100
        for k, m in MASKS.items()}
GAIN = {(f, k): (ERR[nos][m] - ERR[sep][m]) / ERR[sep][m] * 100
        for f, (sep, nos) in {"A": ("cmo", "cmo_nosep"),
                              "B": ("cmo_af", "cmo_af_nosep")}.items()
        for k, m in MASKS.items()}
WIN = {k: int((ERR["cmo"][m] < ERR["fluxnet"][m]).sum()) for k, m in MASKS.items()}

# the quoted percentage is the difference of the two pooled means (what the text
# uses); the dots show how that difference is distributed over the cases
POOL_DIFF = {k: (ERR["cmo"][m].mean() - ERR["fluxnet"][m].mean())
             / ERR["fluxnet"][m].mean() * 100 for k, m in MASKS.items()}
POOL_GAIN = {(f, k): (ERR[nos][m].mean() - ERR[sep][m].mean()) / ERR[sep][m].mean() * 100
             for f, (sep, nos) in {"A": ("cmo", "cmo_nosep"),
                                   "B": ("cmo_af", "cmo_af_nosep")}.items()
             for k, m in MASKS.items()}

assert abs(POOL_DIFF[1.0] - 8.3) < 0.15, POOL_DIFF[1.0]
assert abs(POOL_DIFF[20.0] - 10.5) < 0.15, POOL_DIFF[20.0]
assert (WIN[1.0], WIN[20.0]) == (13, 5), WIN
assert abs(POOL_GAIN[("A", 1.0)] - 5.70) < 0.1, POOL_GAIN[("A", 1.0)]
assert abs(POOL_GAIN[("A", 20.0)] - 6.56) < 0.1, POOL_GAIN[("A", 20.0)]
assert abs(POOL_GAIN[("B", 1.0)] - 3.64) < 0.1, POOL_GAIN[("B", 1.0)]
assert abs(POOL_GAIN[("B", 20.0)] - 3.48) < 0.1, POOL_GAIN[("B", 20.0)]
GAIN_WIN = {key: int((g > 0).sum()) for key, g in GAIN.items()}
assert GAIN_WIN == {("A", 1.0): 37, ("A", 20.0): 40,
                    ("B", 1.0): 39, ("B", 20.0): 40}, GAIN_WIN

fig = F.plt.figure(figsize=F.figsize(90, 110))
gs = fig.add_gridspec(2, 1, hspace=0.40, left=0.205, right=0.968,
                      bottom=0.050, top=0.945)
axes = [fig.add_subplot(gs[i]) for i in range(2)]
rng = np.random.default_rng(7)


def strip(ax, values, x, color, width=0.30, ms=2.0):
    v = np.asarray(values, float)
    jit = rng.uniform(-width / 2, width / 2, v.size)
    ax.plot(x + jit, v, "o", ms=ms, mfc=color, mec="none", alpha=0.75, zorder=3)
    med = float(np.median(v))
    ax.plot([x - width * 0.62, x + width * 0.62], [med, med], color=F.C["ink"],
            lw=1.1, solid_capstyle="butt", zorder=5)
    return med


# --- (a) CMO against FluxNet ------------------------------------------------
ax = axes[0]
for i, k in enumerate(LEVELS):
    strip(ax, DIFF[k], i, F.ARM["cmo"][1])

ax.axhline(0.0, color=F.C["charcoal"], lw=0.6, ls=(0, (3, 2)), zorder=1)
ax.set_xticks(range(len(LEVELS)))
ax.set_xticklabels([f"$\\kappa$ = {int(k)}" for k in LEVELS])
ax.set_xlim(-0.55, len(LEVELS) - 0.45)
ax.set_ylabel("CMO minus FluxNet error (%)")
F.clean(ax)
F.letter(ax, "a")

for i, k in enumerate(LEVELS):
    F.numeric(ax, f"{POOL_DIFF[k]:+.1f}%", i, DIFF[k].max() + 1.2, ha="center",
              va="bottom", color=F.ARM["cmo"][1])
    F.numeric(ax, f"{WIN[k]}/40 better", i, DIFF[k].min() - 1.2, ha="center",
              va="top", color=F.C["charcoal"])

lo = min(v.min() for v in DIFF.values()) - 7.0
hi = max(v.max() for v in DIFF.values()) + 7.0
ax.set_ylim(lo, hi)

# --- (b) the detail branch ---------------------------------------------------
ax = axes[1]
OFF = {"A": -0.17, "B": 0.17}
COL = {"A": F.ARM["cmo"][1], "B": F.ARM["cmo_af"][1]}
for i, k in enumerate(LEVELS):
    for f in ("A", "B"):
        strip(ax, GAIN[(f, k)], i + OFF[f], COL[f], width=0.24, ms=1.8)
        F.numeric(ax, f"{POOL_GAIN[(f, k)]:.1f}", i + OFF[f], GAIN[(f, k)].max() + 0.45,
                  ha="center", va="bottom", color=COL[f])

ax.set_xticks(range(len(LEVELS)))
ax.set_xticklabels([f"$\\kappa$ = {int(k)}" for k in LEVELS])
ax.set_xlim(-0.55, len(LEVELS) - 0.45)
ax.set_ylim(0.0, max(v.max() for v in GAIN.values()) * 1.30)
ax.set_ylabel("gain from the detail branch (%)")
F.clean(ax)
F.letter(ax, "b")

h = [F.plt.Line2D([], [], marker="o", ls="", ms=3.0, mfc=COL[f], mec="none")
     for f in ("A", "B")]
ax.legend(h, ["Form A", "Form B"], loc="upper left", ncol=2)
wins_b = GAIN_WIN
F.numeric(ax, f"A {wins_b['A', 1.0]}/40 $\\rightarrow$ {wins_b['A', 20.0]}/40, "
               f"B {wins_b['B', 1.0]}/40 $\\rightarrow$ {wins_b['B', 20.0]}/40",
          ax.get_xlim()[1] - 0.03, ax.get_ylim()[1] * 0.995, ha="right", va="top",
          color=F.C["charcoal"])

for line in F.audit(fig, axes):
    print(" ", line)
for p in F.save(fig, OUT, "figS1_scan"):
    print(" ", os.path.basename(p), os.path.getsize(p) // 1024, "kB")
print("  a:", {k: round(float(DIFF[k].mean()), 2) for k in LEVELS}, "wins", WIN)
print("  b:", {f"{f}k{int(k)}": round(float(v.mean()), 2)
               for (f, k), v in GAIN.items()}, "wins", wins_b)
