"""Shared style and data access for the paper figures.

Every constant here comes from the research[fig] spec files
(`layout-defaults.yaml`, `font-library.yaml`, `color-library.yaml`); nothing is
invented per figure. Sizes are in millimetres and converted at the point of use, so
a figure's physical size is exactly what its `figure.yaml` says.

One semantic colour per model, across all figures: CMO blue, FluxNet teal, NFV
orange, FNO charcoal, the flux-parametrised arm coral.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

MM = 1.0 / 25.4
FULL, SINGLE, GAP = 190.0, 90.0, 10.0
DPI = 500

FAMILY = "Arial"
# Elsevier's artwork rules ask for >= 7 pt on every glyph in the artwork. The tiers
# are read as point sizes on the canvas, so they only equal the printed size when
# the canvas is the printed width: the figures are now drawn at 165 mm (AWR
# \textwidth = 468.33 pt) and included at scale 1.0, so detail == printed ticks.
SIZE = {"detail": 7.0, "label": 7.5, "key": 8.5}

C = {
    "ink": "#20242A",
    "charcoal": "#74777B",
    "grid": "#D9DEE6",
    "pale": "#EEF1F5",
    "blue": "#526FB4",
    "blue_dark": "#304F8C",
    "blue_light": "#AFC0E1",
    "teal": "#3F929C",
    "teal_light": "#A8D0D3",
    "coral": "#D46B5F",
    "coral_light": "#E8B3AD",
    "orange": "#D4934B",
    "slate": "#71869C",
}

# name -> (display label, colour); identical in every figure
ARM = {
    "cmo": ("CMO", C["blue"]),
    "cmo_nosep": ("CMO, no detail", C["blue_light"]),
    "cmo_af": ("Form B", C["coral"]),
    "cmo_af_nosep": ("Form B, no detail", C["coral_light"]),
    "fluxnet": ("FluxNet", C["teal"]),
    "nfv": ("NFV", C["orange"]),
    "fno": ("FNO", C["charcoal"]),
    "af_raw": ("Form B", C["coral"]),
    "af_corr": ("Form B + factor", C["coral"]),
    "mix_cmo": ("Form A, retrained", C["blue"]),
    "mix_af": ("Form B, retrained", C["coral"]),
}
ORDER4 = ["cmo", "fluxnet", "nfv", "fno"]
# the seven rows of the decomposition table, in table order
ORDER7 = ["fluxnet", "cmo", "cmo_nosep", "cmo_af", "cmo_af_nosep", "nfv", "fno"]


def verify_fonts() -> None:
    """Fail loudly rather than silently rendering a substitute face."""
    path = fm.findfont(fm.FontProperties(family=FAMILY), fallback_to_default=False)
    if "arial" not in os.path.basename(path).lower():
        raise RuntimeError(f"{FAMILY} resolved to {path}, not the metric-compatible face")


def apply() -> None:
    verify_fonts()
    plt.rcParams.update({
        "font.family": FAMILY,
        # mathtext defaults to DejaVu, which lands a second typeface in the PDF
        # ($\kappa$, $dt$, $L_2$): every label must resolve to the same family
        "mathtext.fontset": "custom",
        "mathtext.rm": FAMILY,
        "mathtext.it": f"{FAMILY}:italic",
        "mathtext.bf": f"{FAMILY}:bold",
        "mathtext.sf": FAMILY,
        "mathtext.default": "it",
        "font.size": SIZE["detail"],
        "text.color": C["ink"],
        "axes.labelcolor": C["ink"],
        "axes.edgecolor": C["ink"],
        "xtick.color": C["ink"],
        "ytick.color": C["ink"],
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.35,
        "ytick.minor.width": 0.35,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        "xtick.labelsize": SIZE["detail"],
        "ytick.labelsize": SIZE["detail"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.labelsize": SIZE["key"],
        "axes.labelweight": "bold",
        "axes.titlesize": SIZE["key"],
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.fontsize": SIZE["detail"],
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.2,
        "lines.linewidth": 0.8,
        "lines.markersize": 2.4,
        "lines.markeredgewidth": 0.5,
        "patch.linewidth": 0.5,
        "errorbar.capsize": 1.2,
        "savefig.dpi": DPI,
        "savefig.bbox": "standard",       # the canvas is exactly figure.yaml's size
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "figure.dpi": 120,
    })


def figsize(width_mm: float, height_mm: float):
    return (width_mm * MM, height_mm * MM)


def letter(ax, text: str, x: float = -0.02, y: float = 1.02) -> None:
    """Panel letter in axes coordinates, bold, 8.5 pt."""
    ax.text(x, y, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=SIZE["key"], fontweight="bold", color=C["ink"])


def numeric(ax, s: str, x: float, y: float, ha: str = "center", va: str = "bottom",
            color: str | None = None, rotation: float = 0.0) -> None:
    """A data value printed on the canvas: 5.8 pt, regular, ink unless stated."""
    ax.text(x, y, s, ha=ha, va=va, fontsize=SIZE["detail"], color=color or C["ink"],
            rotation=rotation, zorder=6)


def legend_box_data(ax):
    """Legend extent in data coordinates -- a legend can be inside the axes and
    still sit on top of the marks, which no bbox-of-the-canvas check can see."""
    lg = ax.get_legend()
    if lg is None:
        return None
    ax.figure.canvas.draw()
    bb = lg.get_window_extent(ax.figure.canvas.get_renderer())
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return x0, y0, x1, y1


def clean(ax, grid_axis: str | None = "y") -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color=C["grid"], linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)


def save(fig, outdir: str, stem: str) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    made = []
    for ext in ("pdf", "svg", "png"):
        p = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(p, format=ext, dpi=DPI)
        made.append(p)
    plt.close(fig)
    return made


def audit(fig, axes=None) -> list[str]:
    """Geometric checks, because the figure itself is never inspected by eye.

    Reports (i) the physical size the caller asked for, (ii) every font family that
    actually appears in the canvas, (iii) text that falls outside the tight bbox,
    (iv) panel bboxes that overlap horizontally, (v) annotation text that collides
    with a neighbouring panel's data area.
    """
    problems = []
    w_mm, h_mm = fig.get_size_inches() * 25.4
    problems.append(f"size {w_mm:.1f} x {h_mm:.1f} mm")
    families = {t.get_fontfamily()[0] for t in fig.findobj(matplotlib.text.Text)
                if t.get_text().strip()}
    problems.append(f"font families {sorted(families)}")
    if any(f != FAMILY for f in families):
        problems.append("NON-COMPLIANT FONT FAMILY")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fb = fig.bbox                      # fixed canvas: nothing may leave it
    panels = axes if axes else fig.axes
    items = []
    for ax in panels:
        items += [ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
    clipped = []
    for t in items:
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(renderer=renderer)
        if bb.x0 < fb.x0 - 0.5 or bb.x1 > fb.x1 + 0.5 or \
                bb.y0 < fb.y0 - 0.5 or bb.y1 > fb.y1 + 0.5:
            clipped.append(t.get_text())
    if clipped:
        problems.append(f"CLIPPED BY CANVAS: {clipped}")

    if axes:
        boxes = [ax.get_tightbbox(renderer) for ax in axes]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ox = min(boxes[i].x1, boxes[j].x1) - max(boxes[i].x0, boxes[j].x0)
                oy = min(boxes[i].y1, boxes[j].y1) - max(boxes[i].y0, boxes[j].y0)
                if ox > 1 and oy > 1:      # stacked panels share x, that is fine
                    problems.append(f"PANEL OVERLAP between {i} and {j}")
        for i, ax in enumerate(axes):
            inner = ax.get_window_extent(renderer)
            for t in ax.texts:
                if t.get_transform() is ax.transAxes:      # panel letter: by design
                    continue
                bb = t.get_window_extent(renderer=renderer)
                if bb.width == 0 or bb.height == 0:
                    continue
                if bb.x0 < inner.x0 - 1 or bb.x1 > inner.x1 + 1 or \
                        bb.y0 < inner.y0 - 1 or bb.y1 > inner.y1 + 1:
                    problems.append(f"TEXT OUTSIDE AXES {i}: {t.get_text()!r}")
    return problems


# --------------------------------------------------------------------- data

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def _res(*parts: str) -> str:
    return os.path.normpath(os.path.join(RES, *parts))


def run_dirs(arm: str) -> list[str]:
    """Five seeds, as used by every table in the text. FluxNet's seed 0 predates
    the per-seed directories and lives under `s11/`."""
    if arm == "cmo":
        return [f"cmo_s{s}/cmo_adr2d_{s}" for s in range(5)]
    if arm == "nfv":
        return ["s11/nfv_adr2d_0"] + [f"nfv_s{s}/nfv_adr2d_{s}" for s in range(1, 5)]
    if arm == "fno":
        return ["s11/fno_adr2d_0"] + [f"fno_s{s}/fno_adr2d_{s}" for s in range(1, 5)]
    if arm == "fluxnet":
        return ["s11/fluxnet_adr2d_0"] + [
            f"fluxnet_s{s}/fluxnet_adr2d_{s}" for s in range(1, 5)]
    if arm in ("cmo_af", "cmo_nosep", "cmo_af_nosep"):
        return [f"{arm}_s{s}/cmo_adr2d_{s}_{arm}" for s in range(5)]
    if arm in ("mix_cmo", "mix_af"):
        stem = "cmo_af_dtmix" if arm == "mix_af" else "cmo_dtmix"
        return [f"{arm}_s{s}/cmo_adr2d_{s}_{stem}" for s in range(5)]
    raise KeyError(arm)


def arm_metrics(arm: str, field: str = "rel_l2") -> list[dict]:
    out = []
    for d in run_dirs(arm):
        with open(_res(d, "metrics_test.json")) as fh:
            out.append(json.load(fh)[field])
    return out


def rel_l2_final(arm: str) -> np.ndarray:
    return np.array([m["final"] for m in arm_metrics(arm, "rel_l2")])


def conservation(arm: str, key: str = "r_final_mean") -> np.ndarray:
    return np.array([m[key] for m in arm_metrics(arm, "conservation")])


def sharpness(arm: str) -> dict:
    """`sharpness_grad` from `results/_sharp/`: measured at the reference front,
    ratio < 1 means thinner than the reference at the same place and step."""
    rows = []
    for s in range(5):
        with open(_res("_sharp", f"{arm}_s{s}", "metrics_test.json")) as fh:
            rows.append(json.load(fh)["sharpness_grad"])
    return {k: np.array([r[k] for r in rows])
            for k in ("w_median", "w_ref_median", "ratio_median", "ratio_final")}


def dt_band(cell: str, band: str) -> np.ndarray:
    """80 paired cases (56 train + 24 holdout) for one cell on one `dt` band."""
    parts = []
    for s in range(5):
        with open(_res(f"_dtonly_{cell}_s{s}.json")) as fh:
            a = np.array(json.load(fh)[band])
        with open(_res(f"_dtonly_{cell}_s{s}_holdout.json")) as fh:
            a = np.concatenate([a, np.array(json.load(fh)[band])])
        parts.append(a)
    return np.array(parts)


def dt_band_meta(cell: str) -> dict:
    with open(_res(f"_dtonly_{cell}_s0.json")) as fh:
        j = json.load(fh)
    return {"cells": [b for b in ("coarse", "mid", "fine") if b in j],
            "ref_shift": float(j.get("ref_shift", float("nan")))}


def ext80(cell: str) -> np.ndarray:
    """Five seeds x 80 cases on the fine band. The file is built as
    `--regime ext,kappa20 --n 40,40`, so cases 0:40 are the training-range
    anisotropy and 40:80 are kappa = 20."""
    rows = []
    for s in range(5):
        with open(_res(f"_ext80_{cell}_s{s}.json")) as fh:
            rows.append(np.array(json.load(fh)["errs"]))
    return np.array(rows)


def scan() -> tuple[list[float], dict]:
    with open(_res("_bound_dt_scan.json")) as fh:
        j = json.load(fh)
    ks = sorted((float(k) for k in j), reverse=True)
    return ks, {float(k): v for k, v in j.items()}


def lag(arm: str) -> tuple[np.ndarray, np.ndarray]:
    with open(_res("_lag_decomposition.json")) as fh:
        j = json.load(fh)[arm]
    return np.array(j["E_total"]), np.array(j["E_shape"])


def rel_l2_series(arm: str) -> np.ndarray:
    """(5 seeds, 201 checkpoints) relative L2 over the rollout, for fig2b."""
    return np.array([np.array(m["series"], dtype=float)
                     for m in arm_metrics(arm, "rel_l2")])


# ------------------------------------------------------------------ zero-shot
# The kappa files store one rollout per case, so per-case numbers have to be
# recomputed from the npz with the same metric the evaluation used -- the summary
# json only keeps the pooled statistics.

ROOT = os.path.dirname(RES)


def _zshot_npz(arm: str):
    import numpy as np2
    p = _res("_zshot", f"kappa_{arm}", "rollout_test.npz")
    d = np2.load(p)
    return d["pred"][:, 0, -1, 0], d["ref"][:, 0, -1, 0]


def zshot_cases(arm: str) -> np.ndarray:
    """Per-case final rel-L2 for one arm on `adr2d_kappa.h5` (80 cases)."""
    import sys as _sys
    if ROOT not in _sys.path:
        _sys.path.insert(0, ROOT)
    from src.utils.metrics import rel_l2 as _rel
    pred, ref = _zshot_npz(arm)
    return np.array([float(_rel(pred[i], ref[i], axis=(-2, -1)))
                     for i in range(pred.shape[0])])


def kappa_mask(level: float) -> np.ndarray:
    """Which of the 80 cases sit at the given anisotropy ratio (column 2 of
    `PARAM_COLUMNS`); the file was built as `--regime kappa1,kappa20 --n 40,40`."""
    import h5py
    with h5py.File(os.path.join(ROOT, "data", "adr2d", "adr2d_kappa.h5"), "r") as fh:
        kap = fh["params"][:, 2]
    return np.isclose(kap, level)
