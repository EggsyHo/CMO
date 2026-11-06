"""Metric kernels (implementation.md §3.16).

Torch-free on purpose: the audit must be independent of AMP and of the model's
own dtype, so everything here is numpy in float64.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfcinv

EPS = 1e-30


# --------------------------------------------------------------------------- #
# error metrics
# --------------------------------------------------------------------------- #
def rel_l2(pred, true, axis=None):
    """||p - t||_2 / ||t||_2; per-sample array when `axis` is given.

    A zero reference returns **NaN**, not 0 or 1e15: the relative error is
    undefined there, and both alternatives lie. `1e-30` made the first two steps
    of every ADR test trajectory report exactly `0.0` -- at `t=0, dt` the plume is
    still empty, so `num == den == 0` -- which reads as a perfect prediction in
    the first row of a table. Callers that aggregate (`_nanmean`) then skip the
    step instead of averaging in a fiction.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    num = np.sqrt(np.sum((p - t) ** 2, axis=axis))
    den = np.sqrt(np.sum(t ** 2, axis=axis))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), np.nan)


# --------------------------------------------------------------------------- #
# mass partition (reimplemented here so the metric cannot inherit a torch bug)
# --------------------------------------------------------------------------- #
def _check_block(x: np.ndarray, block: int) -> None:
    if x.shape[-1] % block or x.shape[-2] % block:
        raise ValueError(f"grid {x.shape[-2:]} not divisible by block={block}")


def restrict(x, block: int = 2) -> np.ndarray:
    """Pi: exact block average."""
    arr = np.asarray(x, dtype=np.float64)
    _check_block(arr, block)
    h, w = arr.shape[-2] // block, arr.shape[-1] // block
    return arr.reshape(arr.shape[:-2] + (h, block, w, block)).mean(axis=(-3, -1))


def lift(xc, block: int = 2) -> np.ndarray:
    arr = np.asarray(xc, dtype=np.float64)
    return np.repeat(np.repeat(arr, block, axis=-2), block, axis=-1)


def rho_t(u, block: int = 2) -> float:
    """rho(t) = ||(I - Pi)u||^2 / ||u||^2 — share of energy in the detail band."""
    arr = np.asarray(u, dtype=np.float64)
    detail = arr - lift(restrict(arr, block), block)
    return float(np.sum(detail ** 2) / max(float(np.sum(arr ** 2)), EPS))


def rho_series(u_seq, block: int = 2) -> np.ndarray:
    arr = np.asarray(u_seq, dtype=np.float64)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return np.array([rho_t(arr[i], block) for i in range(arr.shape[0])])


def band_split(err, block: int = 2) -> tuple[float, float]:
    """(||Pi e||, ||(I-Pi) e||) — response variable of the interaction test."""
    e = np.asarray(err, dtype=np.float64)
    coarse = lift(restrict(e, block), block)
    return float(np.sqrt(np.sum(coarse ** 2))), float(np.sqrt(np.sum((e - coarse) ** 2)))


# --------------------------------------------------------------------------- #
# front sharpness
# --------------------------------------------------------------------------- #
def _bilinear(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Periodic bilinear sample at fractional index coordinates (x = col, y = row)."""
    h, w = field.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    x1, y1 = (x0 + 1) % w, (y0 + 1) % h
    x0, y0 = x0 % w, y0 % h
    return (
        field[y0, x0] * (1 - fx) * (1 - fy)
        + field[y0, x1] * fx * (1 - fy)
        + field[y1, x0] * (1 - fx) * fy
        + field[y1, x1] * fx * fy
    )


def _ray_profile(field: np.ndarray, origin, direction, step_cells: float = 0.25,
                 max_cells: float = 48.0) -> np.ndarray:
    s = np.arange(0.0, max_cells, step_cells)
    y = origin[0] + s * direction[0]
    x = origin[1] + s * direction[1]
    return _bilinear(field, x, y)


def _side_width(norm: np.ndarray, step_cells: float, lo: float, hi: float) -> float:
    """Distance from the `hi` crossing to the `lo` crossing, walking away from the peak."""
    i_max = int(np.argmax(norm))
    below_hi = norm < hi
    below_lo = norm < lo
    right = np.arange(norm.size) > i_max
    # right side
    r_hi = np.where(below_hi & right)[0]
    r_lo = np.where(below_lo & right)[0]
    # left side
    left = np.arange(norm.size) < i_max
    l_hi = np.where(below_hi & left)[0]
    l_lo = np.where(below_lo & left)[0]

    widths = []
    if r_hi.size and r_lo.size and r_lo[0] >= r_hi[0]:
        widths.append((r_lo[0] - r_hi[0]) * step_cells)
    if l_hi.size and l_lo.size and l_lo[-1] <= l_hi[-1]:
        widths.append((l_hi[-1] - l_lo[-1]) * step_cells)
    if not widths:
        return float("nan")
    return float(np.mean(widths))


def width_from_profile(prof: np.ndarray, step_cells: float = 0.25,
                       levels=(0.1, 0.9)) -> float:
    p = np.asarray(prof, dtype=np.float64)
    lo, hi = float(np.min(p)), float(np.max(p))
    if hi - lo <= 1e-12:
        return float("nan")
    norm = (p - lo) / (hi - lo)
    return _side_width(norm, step_cells, levels[0], levels[1])


def front_width(u2d, origin=None, levels=(0.1, 0.9), n_rays: int = 4,
                step_cells: float = 0.25) -> tuple[float, int]:
    """10-90% front width from up to four rays from `origin`.

    Returns (width, n_valid_rays); width is the median over valid rays, NaN if
    fewer than two rays qualify. The discarded count is returned, not hidden.
    """
    field = np.asarray(u2d, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError(f"front_width expects a 2D field, got shape {field.shape}")
    h, w = field.shape
    if origin is None:
        origin = (h / 2.0, w / 2.0)
    dirs = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071], [0.7071, -0.7071]])
    widths = []
    for d in dirs[:n_rays]:
        wid = width_from_profile(_ray_profile(field, origin, d, step_cells), step_cells, levels)
        if np.isfinite(wid) and wid > 0:
            widths.append(wid)
    if len(widths) < 2:
        return float("nan"), len(widths)
    return float(np.median(widths)), len(widths)


def interface_width(u2d, step_cells: float = 0.25,
                    half_window: float = 12.0, at=None) -> tuple[float, float]:
    """Edge thickness of the sharpest interface, in cells; gradient second moment.

    `front_width` above cannot answer "is this front sharper", because it normalises
    each ray by its own min/max: after the plume has decayed that span is set by the
    peak height, so a shorter peak reads as a wider front, and the absolute-level
    variant is undefined once the 10-90% crossings fall below the decay floor. This
    estimator drops the amplitude instead of the position:

      1. locate the steepest-gradient cell, `i,j = argmax |grad C|`, and take the
         local normal `n = grad/|grad|` there;
      2. sample `C` along that normal over `+-half_window` cells (cubic, periodic);
      3. keep only the *outer* half of the profile -- from the crest outward, the
         segment where the plume decays to background -- and return

             w = sqrt( sum_k a_k (s_k - sbar)^2 / sum_k a_k ),  a_k = |dC/ds|_k,

         the second moment of the edge gradient about its own centroid.

    Two things this buys, both of which the candidate replacements fail:
    *amplitude-free* (a_k is the weight, so any constant factor on the field cancels
    -- `front_width` is amplitude-blind only because it divides by the peak, which is
    what confuses peak height with width) and *size-free*: restricting to the outer
    edge means the number does not grow when the plume grows. `2||C||_1/TV`, the
    area-over-perimeter alternative, is amplitude-free but degenerates to `area /
    perimeter ~ R`, i.e. it measures plume radius and saturates against edge
    sharpness (measured: mean 2.93 cells on 64x64 fluxnet fields, indistinguishable
    from a plume radius).

    An error-function fit `base + A/2 erfc(-(s-s0)/w)` was tried first and abandoned:
    on this benchmark the plume edges are *curved*, so the profile along the normal is
    a bump rather than a half-space step, and the fit's residual gate rejected
    **60/60** real fields -- correct behaviour, useless column.

    `at=(i, j)` overrides the localisation and measures the edge *there*, along the
    direction of this field's own gradient at that cell. Use `interface_width_pair`
    rather than passing `at` by hand: a model whose front sits in the wrong place has
    its largest gradient on a different object, and then two `ratio` values are not
    comparing the same feature (measured: FNO's own steepest edge is 0.73 cells, below
    the estimator's floor, because its peak is 50% above the reference's and its
    curvature-to-gradient ratio is 1.5x the conserving arms').

    Returns `(w, frac)` with `frac = sum(a_k)/sum(all a_k)` the share of edge weight
    inside the window. `frac < 0.8` means the edge runs into the window and the width
    is a lower bound; `w` is `NaN` on a flat field. Scale-checked on an exact erfc:
    doubling the transition scale doubles `w` to 0.1%, and `x1e-3` on the amplitude
    changes it in the 14th digit (`tests/check_width.py`).
    """
    field = np.asarray(u2d, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError(f"interface_width expects a 2D field, got shape {field.shape}")
    gy, gx = np.gradient(field)
    gmag = np.hypot(gx, gy)
    if at is None:
        i, j = np.unravel_index(int(np.argmax(gmag)), gmag.shape)
        gmax = float(gmag[i, j])
    else:
        i, j = int(at[0]), int(at[1])
        gmax = float(gmag[i, j])
    if not np.isfinite(gmax) or gmax <= 0.0:
        return float("nan"), 0.0

    ny, nx = float(gy[i, j]) / gmax, float(gx[i, j]) / gmax
    return _edge_moment(field, i, j, ny, nx, step_cells, half_window)


def _edge_moment(field, i, j, ny, nx, step_cells, half_window):
    """Sample `field` along the line `(i,j) + s*n` and return (moment, frac)."""
    s = np.arange(-half_window, half_window + 1e-12, step_cells)
    from scipy.ndimage import map_coordinates
    prof = map_coordinates(field, np.vstack([i + s * ny, j + s * nx]),
                           order=3, mode="grid-wrap")
    a = np.abs(np.gradient(prof, step_cells))
    if not np.isfinite(a).all() or float(a.sum()) <= EPS:
        return float("nan"), 0.0

    # outer half only: `n` points uphill, so `s <= 0` is the downhill side, i.e. the
    # stretch where the plume decays to background. This is well posed for a monotone
    # step *and* for a bump, whereas "from the crest outward" is not: on a monotone
    # front there is no interior crest, `argmax` lands in the saturated plateau and
    # the segment collapses to one sample (measured: w=2 gave frac=0.000, w=4 gave
    # frac=1.000 for two fields of the same family).
    seg = np.where(s <= 0.0)[0]
    ss, aa = s[seg], a[seg]
    tot = float(aa.sum())
    if tot <= EPS:
        return float("nan"), 0.0
    sbar = float((aa * ss).sum() / tot)
    w = float(np.sqrt((aa * (ss - sbar) ** 2).sum() / tot))
    return w, tot / float(a.sum())


def interface_width_pair(pred, ref, step_cells: float = 0.25,
                         half_window: float = 12.0) -> tuple[float, float, float]:
    """Edge thickness of `pred` and `ref` measured **at the same place**, plus frac.

    The localisation and the normal come from the *reference*: `pred` is sampled along
    the reference's steepest-gradient normal through the reference's steepest cell.
    This is the comparison the sharpness claim needs -- "at the front the truth has,
    how thick is the model's edge" -- and it is not what two independent
    `interface_width` calls measure when the model's front is displaced or when its
    largest gradient sits on an artefact. Reported as `w_pred / w_ref`, so the
    constant of the estimator cancels and only a genuine thickening or thinning moves
    the ratio above or below 1.
    """
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    gy, gx = np.gradient(ref)
    gmag = np.hypot(gx, gy)
    i, j = np.unravel_index(int(np.argmax(gmag)), gmag.shape)
    if not np.isfinite(float(gmag[i, j])) or float(gmag[i, j]) <= 0.0:
        return float("nan"), float("nan"), 0.0

    def _at_own_edge(field, i0, j0, radius=4):
        """Re-centre on this field's own steepest cell within `radius` of the ref's.

        Needed for translation invariance. Measuring both fields through one fixed
        point is *not* shift invariant: the moment is taken over the downhill half
        window, so a front displaced by 2 cells falls in a differently truncated
        window and reads 35% thinner (measured: ratio 0.653 for a pure 2-cell shift).
        Every arm here lags the reference by 1-1.4 cells, so that artefact would be
        the whole result. The radius keeps the measurement on the reference's front
        while letting each field's edge centre itself.
        """
        g = np.hypot(*np.gradient(field))
        i_lo, i_hi = max(i0 - radius, 0), min(i0 + radius + 1, g.shape[0])
        j_lo, j_hi = max(j0 - radius, 0), min(j0 + radius + 1, g.shape[1])
        sub = g[i_lo:i_hi, j_lo:j_hi]
        di, dj = np.unravel_index(int(np.argmax(sub)), sub.shape)
        return i_lo + di, j_lo + dj, g

    out = []
    for f in (ref, pred):
        ii, jj, g = _at_own_edge(f, i, j)
        gm = float(g[ii, jj])
        if not np.isfinite(gm) or gm <= 0.0:
            return float("nan"), float("nan"), 0.0
        fy, fx = np.gradient(f)
        ny, nx = float(fy[ii, jj]) / gm, float(fx[ii, jj]) / gm
        w, fr = _edge_moment(f, ii, jj, ny, nx, step_cells, half_window)
        out.append((w, fr))
    (wr, _), (wp, frac) = out[0], out[1]
    return wp, wr, frac


def analytic_width(t: float, diff: float, levels=(0.1, 0.9)) -> float:
    """One-sided 10->90% decay width of a diffusing step, normalised to its min/max.

    The half-space solution is c = (C0/2) erfc(u), u = x / (2 sqrt(D t)), with
    c(-inf) = C0 and c(+inf) = 0. `front_width` normalises a ray by its own
    min/max, i.e. by C0, so the level it compares against is

        norm = c / C0 = 0.5 erfc(u)   =>   erfc(u) = 2 * norm

    not `erfc(u) = norm`. Hence
        w(t) = 2 sqrt(D t) [erfcinv(2 lo) - erfcinv(2 hi)]  ~= 3.6248 sqrt(D t)
    for (lo, hi) = (0.1, 0.9).

    The earlier 2.1485 came from dropping the factor of two in `2 * norm` -- it is
    the width of an `erfc` profile normalised by 2 rather than by its own peak, and
    reads 1.687x too small. Measured on an exact erfc patch with no solver at all,
    `front_width` returns exactly 3.6248 sqrt(D t), so this constant is the one the
    measurement actually inverts; see results/calibration/solver_check.json.

    `2 * hi` must stay <= 2, so this is only meaningful for hi < 1.
    """
    lo, hi = levels
    z = erfcinv(2.0 * lo) - erfcinv(2.0 * hi)
    return float(z * 2.0 * np.sqrt(diff * max(t, 0.0)))


def tv_ratio(pred, ref) -> float:
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)

    def tvd(a):
        return np.sum(np.abs(np.diff(a, axis=-1))) + np.sum(np.abs(np.diff(a, axis=-2)))

    return float(tvd(p) / max(tvd(r), EPS))


def error_growth_rate(err_seq) -> float:
    """Least-squares slope of log(mean error) vs step index."""
    e = np.asarray(err_seq, dtype=np.float64)
    mask = e > 0
    if mask.sum() < 2:
        return float("nan")
    x = np.arange(e.size)[mask]
    return float(np.polyfit(x, np.log(e[mask]), 1)[0])


# --------------------------------------------------------------------------- #
# transport-domain metrics
# --------------------------------------------------------------------------- #
def breakthrough(u_seq, outlet_col: int | None = None, dt: float = 1.0) -> dict:
    """Peak concentration, 50%-of-peak arrival time, tail mass after arrival."""
    arr = np.asarray(u_seq, dtype=np.float64)
    if arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(f"breakthrough expects one channel, got {arr.shape}")
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"breakthrough expects [T,H,W] or [T,1,H,W], got {arr.shape}")
    col = arr.shape[-1] // 2 if outlet_col is None else outlet_col
    series = arr[:, :, col].mean(axis=1)  # [T]
    peak = float(series.max())
    if peak <= 0:
        return {"peak": 0.0, "arrival": float("nan"), "tail": 0.0}
    idx = int(np.argmax(series >= 0.5 * peak))
    return {
        "peak": peak,
        "arrival": float(idx * dt),
        "tail": float(np.sum(series[idx:]) * dt),
    }


def bootstrap_ci(vals, n_boot: int = 10000, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval over *seeds* (Part 3 §2.2 asks for seed-level intervals)."""
    v = np.asarray(list(vals), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    if v.size == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(0)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def interaction_term(a_pp: float, a_mm: float, a_pm: float, a_mp: float) -> float:
    """(sep+,proj+) + (sep-,proj-) - (sep+,proj-) - (sep-,proj+)."""
    return float(a_pp + a_mm - a_pm - a_mp)
