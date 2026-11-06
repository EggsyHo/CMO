"""Fig. 1 -- framework schematic (Sec. 3.1).

No data is plotted: every printed constant (12 modes, c_cfl <= 1/4, 1e-14) is a
design value documented in Secs. 3.3-3.5. The figure carries one argument, in the
two orange badges -- the coarse lane conserves because a divergence telescopes,
the detail lane conserves because it is projected in the mass inner product, and
neither fact involves a learned weight.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figlib as F  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))

BLUE = F.ARM["cmo"][1]
TEAL = F.ARM["fluxnet"][1]
ORANGE = F.ARM["nfv"][1]
INK = F.ARM["fno"][1]

F.apply()
fig = F.plt.figure(figsize=F.figsize(164.6, 75.7))
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_xlim(0, 100)
ax.set_ylim(3, 87)
ax.axis("off")

FS = 5.9
FS_SMALL = 5.0


def box(x, y, w, h, text, edge, fs=FS, fc="white", lw=0.7, ls="-", zorder=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=1.3",
                                linewidth=lw, edgecolor=edge, facecolor=fc,
                                linestyle=ls, zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, linespacing=1.45, zorder=zorder + 1)


def arrow(x0, y0, x1, y1, colour=INK, lw=0.65):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>,head_length=0.32,head_width=0.15",
                                lw=lw, color=colour, shrinkA=0, shrinkB=0),
                zorder=1)


# --- state ------------------------------------------------------------------
box(0.8, 66.0, 10.6, 16.0, "$\\mathbf{C}^{n}$\n$64\\times64$", INK)
box(13.6, 61.0, 13.6, 26.0,
    "$m^{n}=\\theta C^{n}$\n$+\\,\\rho_b\\,S(C^{n})$\n\ncomposite state", INK)

# --- coarse lane ------------------------------------------------------------
ya, ha = 64.0, 20.0
box(29.8, ya, 16.0, ha, "$\\Pi m^{n}$\n\n$32\\times32$\ncell means", BLUE)
box(47.8, ya, 16.0, ha,
    "spectral branch\n12 modes\n4 layers, width 16", BLUE)
box(65.8, ya, 16.0, ha,
    "face flux $\\hat{F}_{f}$\n\n$-\\Delta t\\,\\mathrm{div}\\,\\hat{F}_{f}$", BLUE)

# --- detail lane ------------------------------------------------------------
yb, hb = 36.0, 20.0
box(29.8, yb, 16.0, hb, "$(I-\\Pi)\\,m^{n}$\n\nzero mean\nper cell", TEAL)
box(47.8, yb, 16.0, hb,
    "db2 detail energy\n$\\hat{E}_{c}\\to\\sigma(\\beta(\\hat{E}_{c}-\\tau))$", TEAL)
box(65.8, yb, 16.0, hb,
    "$R_{\\phi}$ $3\\times3$, $c_\\text{cfl}\\tanh$\nmass-weighted\nprojection", TEAL)

# --- merge ------------------------------------------------------------------
box(84.0, 33.0, 15.4, 50.0,
    "$m^{n+1}=m^{n}$\n$+\\Delta m_{c}$\n$+\\Delta m_{d}$\n\n"
    "$C^{n+1}$\n$=S^{-1}(m^{n+1})$", INK, lw=0.9)

arrow(11.4, 74.0, 13.6, 74.0)
arrow(27.2, 78.5, 29.8, 74.0)
arrow(27.2, 69.5, 29.8, 46.0)
for x in (45.8, 63.8):
    arrow(x, 74.0, x + 2.0, 74.0, BLUE)
    arrow(x, 46.0, x + 2.0, 46.0, TEAL)
arrow(81.8, 74.0, 84.0, 74.0, BLUE)
arrow(81.8, 46.0, 84.0, 46.0, TEAL)

# --- the two equalities -----------------------------------------------------
ax.text(29.8, 60.0,
        "cell mass unchanged:  $\\sum_{C}\\Delta m_{c}=0$  because the divergence telescopes\n"
        "— holds for any flux values the network emits",
        ha="left", va="center", fontsize=FS_SMALL, color=ORANGE, linespacing=1.30)
ax.text(29.8, 32.0,
        "cell mass unchanged:  $\\sum_{C}\\Delta m_{d}=0$  because the increment is projected in $m$\n"
        "— holds for any gate value, any weight, any step size",
        ha="left", va="center", fontsize=FS_SMALL, color=ORANGE, linespacing=1.30)

# --- audit band -------------------------------------------------------------
ax.add_patch(FancyBboxPatch((0.8, 5.0), 98.6, 18.0,
                            boxstyle="round,pad=0,rounding_size=1.3",
                            linewidth=0.5, edgecolor=ORANGE, facecolor="white",
                            linestyle=(0, (4, 2)), zorder=2))
ax.text(50.0, 18.4,
        "audited budget on $M^{n}=\\sum_{i}m_{i}^{n}\\,\\Delta A$:   "
        "$r(t)=\\left|M^{n}-M^{0}-\\sum_{k}\\Delta t\\,(\\hat{S}^{k}-\\lambda M^{k})\\right|/\\,M^{0}$",
        ha="center", va="center", fontsize=FS, color=INK)
ax.text(50.0, 11.2,
        "$r$ is identical across the conserving arms ($6.1\\times10^{-6}$, the audit's own floor) and $O(1)$ without the flux form\n"
        "— the source cancels term by term, so $r(t)$ can respond to the operator and never to the gate",
        ha="center", va="center", fontsize=FS_SMALL, color=INK, linespacing=1.45)
arrow(91.7, 33.0, 91.7, 23.0, ORANGE)

for path in F.save(fig, OUT, "fig0_framework"):
    print(path)
