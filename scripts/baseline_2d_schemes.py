"""Classical-scheme baseline on the ADR-2D benchmark (P0-2 for the CPC revision).

Question a CPC referee asks about the reference generator: *why donor-cell upwind
at all?*  This script answers with measurements rather than an assertion.

Physics comes from `data/gen_adr2d.py` -- same log-normal conductivity, same
Darcy velocity (with the FD projection that makes div q = 0 for the operator the
update uses), same dispersion tensor, same Langmuir partition, same injection
source, same domain and time step.  Only the *advective face value* changes:

    upwind   donor cell, first order (what the benchmark ships)
    muscl    van Leer slopes on central differences, second order
    weno5    Jiang--Shu WENO5 with SSP-RK3, fifth order where the field is smooth

Reported, all auditable offline:

  A. accuracy at the benchmark grid (N = 64, dt = 0.1, 200 steps) against a
     grid-refined WENO5 solution of the same physical case;
  B. observed order under joint refinement, physical parameters held fixed (so
     the cell Peclet number falls as the grid is refined -- the correct limit);
  C. mass residual and positivity-projection count for every scheme, because the
     manuscript's claim is that conservation and formal order are independent.

Consistency guard: with scheme="upwind" and an explicit-Euler stage this module
must reproduce `gen_adr2d.fvm_step` to round-off.  That check runs every time and
is written to the JSON, so the comparison cannot silently drift from the
generator it claims to compare against.

    python scripts/baseline_2d_schemes.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.special as special

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "data"))
import gen_adr2d as g  # noqa: E402

SCHEMES = ("upwind", "muscl", "weno5")
# temporal refinement exponent: dt ~ dx**(p/3) keeps the RK3 error below the
# spatial one.  p = 1 for the two second-order-or-lower schemes, 5 for WENO5.
DT_EXP = {"upwind": 1.0, "muscl": 1.0, "weno5": 5.0 / 3.0}
# --------------------------------------------------------------------------- #
# advective face reconstruction
# --------------------------------------------------------------------------- #
def _vanleer(a, b):
    ok = a * b > 0.0
    return np.where(ok, 2.0 * a * b / np.where(ok, a + b, 1.0), 0.0)


def _weno5(v0, v1, v2, v3, v4):
    """Jiang--Shu WENO5 face value from five cell values (v2 = upwind cell)."""
    b1 = 13 / 12 * (v0 - 2 * v1 + v2) ** 2 + 0.25 * (v0 - 4 * v1 + 3 * v2) ** 2
    b2 = 13 / 12 * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - v3) ** 2
    b3 = 13 / 12 * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (3 * v2 - 4 * v3 + v4) ** 2
    eps = 1e-6
    a1, a2, a3 = 0.1 / (eps + b1) ** 2, 0.6 / (eps + b2) ** 2, 0.3 / (eps + b3) ** 2
    p1 = v0 / 3 - 7 * v1 / 6 + 11 * v2 / 6
    p2 = -v1 / 6 + 5 * v2 / 6 + v3 / 3
    p3 = v2 / 3 + 5 * v3 / 6 - v4 / 6
    return (a1 * p1 + a2 * p2 + a3 * p3) / (a1 + a2 + a3)


def _faces(C, u_face, scheme, axis):
    """Face values at i+1/2 along `axis`, upwind-biased by the sign of u_face.

    The right-biased branch is the mirror of the left-biased one about the face,
    which keeps the two limiters identical and makes the scheme symmetric.
    """
    r = lambda k: np.roll(C, k, axis)              # noqa: E731
    flip = lambda k: np.roll(C, -k, axis)          # noqa: E731
    if scheme == "upwind":
        return np.where(u_face >= 0.0, C, flip(1))
    if scheme == "muscl":
        d = 0.5 * (flip(1) - r(1))
        left = C + 0.5 * _vanleer(d, np.roll(d, 1, axis))
        right = flip(1) - 0.5 * _vanleer(np.roll(d, -1, axis), d)
        return np.where(u_face >= 0.0, left, right)
    left = _weno5(r(2), r(1), C, flip(1), flip(2))
    right = _weno5(flip(3), flip(2), flip(1), C, r(1))
    return np.where(u_face >= 0.0, left, right)
# --------------------------------------------------------------------------- #
# spatial operator: identical to fvm_step except for the advective face value
# --------------------------------------------------------------------------- #
def _face_avg(a, axis):
    return 0.5 * (a + np.roll(a, -1, axis=axis))


def _flux_div(C, qx, qy, D, theta_b, dx, scheme):
    """div(F_adv + F_disp) with the same dispersion discretisation as fvm_step."""
    Dxx, Dxy, Dyy = D[:, 0], D[:, 1], D[:, 2]

    ux_f = 0.5 * (qx + np.roll(qx, -1, axis=-1))
    Fx_adv = ux_f * _faces(C, ux_f, scheme, -1)
    uy_f = 0.5 * (qy + np.roll(qy, -1, axis=-2))
    Fy_adv = uy_f * _faces(C, uy_f, scheme, -2)

    dCdx = (np.roll(C, -1, axis=-1) - C) / dx
    dCdy_c = (np.roll(C, -1, axis=-2) - np.roll(C, 1, axis=-2)) / (2.0 * dx)
    dCdy_f = 0.5 * (dCdy_c + np.roll(dCdy_c, -1, axis=-1))
    Fx_disp = -theta_b * (_face_avg(Dxx, -1) * dCdx + _face_avg(Dxy, -1) * dCdy_f)

    dCdy = (np.roll(C, -1, axis=-2) - C) / dx
    dCdx_c = (np.roll(C, -1, axis=-1) - np.roll(C, 1, axis=-1)) / (2.0 * dx)
    dCdx_f = 0.5 * (dCdx_c + np.roll(dCdx_c, -1, axis=-2))
    Fy_disp = -theta_b * (_face_avg(Dxy, -2) * dCdx_f + _face_avg(Dyy, -2) * dCdy)

    div = ((Fx_adv + Fx_disp) - np.roll(Fx_adv + Fx_disp, 1, axis=-1)) / dx
    div += ((Fy_adv + Fy_disp) - np.roll(Fy_adv + Fy_disp, 1, axis=-2)) / dx
    return div


def _project_positive(m):
    """Same "may not lose mass" fix as fvm_step: redistribute the deficit."""
    neg = m < 0.0
    n_neg = int(neg.sum())
    if n_neg:
        deficit = np.where(neg, -m, 0.0).sum(axis=(-2, -1), keepdims=True)
        m = np.where(neg, 0.0, m)
        total = m.sum(axis=(-2, -1), keepdims=True)
        spread = np.divide(deficit, total, out=np.zeros_like(total), where=total > 0.0)
        m = m * (1.0 - spread)
    return m, n_neg


def _src_exact(X, Y, x0, y0, sigma, Q, t_ramp, t, dx):
    """Exact cell average of the Gaussian well.

    The well is separable, so each cell average is a product of two error
    functions; the result satisfies sum(s)*dx^2 = Q(1-exp(-t/t_ramp)) for every
    dx, which is what the per-cell renormalisation of ``injection_source`` does
    not do.
    """
    amp = Q * (1.0 - np.exp(-t / t_ramp))
    h = 0.5 * dx / (np.sqrt(2.0) * sigma)
    fx = 0.5 * (special.erf((X - x0) / (np.sqrt(2.0) * sigma) + h)
                - special.erf((X - x0) / (np.sqrt(2.0) * sigma) - h))
    fy = 0.5 * (special.erf((Y - y0) / (np.sqrt(2.0) * sigma) + h)
                - special.erf((Y - y0) / (np.sqrt(2.0) * sigma) - h))
    return amp * fx * fy / dx ** 2


def _rhs_m(m, ctx, t):
    """d/dt of the composite mass, plus the source integral (for the mass budget).

    Integrating the *mass* rather than C keeps the scheme conservative by
    construction: every stage's divergence telescopes, so the only change in
    total mass is the source minus the decay term.
    """
    C = g.langmuir_inverse(m, ctx["th1"], ctx["rb1"], ctx["sm1"], ctx["kl1"])
    div = _flux_div(C, ctx["qx"], ctx["qy"], ctx["D"], ctx["theta_b"], ctx["dx"],
                    ctx["scheme"])
    if ctx["src_exact"]:
        src = _src_exact(ctx["X"], ctx["Y"], ctx["x0"], ctx["y0"], ctx["sigma"],
                         ctx["Q"], ctx["t_ramp"], t, ctx["dx"])
    else:
        src = g.injection_source(ctx["X"], ctx["Y"], ctx["x0"], ctx["y0"],
                                 ctx["sigma"], ctx["Q"], ctx["t_ramp"], t) \
            * ctx["src_scale"]
    S = float(src.sum() * ctx["dx"] ** 2)
    return -div + src - ctx["lam_b"] * m, S
# --------------------------------------------------------------------------- #
# time integration
# --------------------------------------------------------------------------- #
def _step_euler(m, Mp, ctx, t, dt):
    k1, S1 = _rhs_m(m, ctx, t)
    m, n = _project_positive(m + dt * k1)
    return m, Mp + dt * (S1 - ctx["lam"] * Mp), n


def _step_rk3(m, Mp, ctx, t, dt):
    """SSP-RK3 on the mass; the budget scalar takes the same convex combination."""
    k1, S1 = _rhs_m(m, ctx, t)
    m1 = m + dt * k1
    Mp1 = Mp + dt * (S1 - ctx["lam"] * Mp)
    m1p, n1 = _project_positive(m1)

    k2, S2 = _rhs_m(m1p, ctx, t + dt)
    m2 = 0.75 * m + 0.25 * (m1 + dt * k2)
    Mp2 = 0.75 * Mp + 0.25 * (Mp1 + dt * (S2 - ctx["lam"] * Mp1))
    m2p, n2 = _project_positive(m2)

    k3, S3 = _rhs_m(m2p, ctx, t + 0.5 * dt)
    mn = m / 3.0 + (2.0 / 3.0) * (m2 + dt * k3)
    Mpn = Mp / 3.0 + (2.0 / 3.0) * (Mp2 + dt * (S3 - ctx["lam"] * Mp2))
    mn, n3 = _project_positive(mn)
    return mn, Mpn, n1 + n2 + n3


def run_case(K, prm, idx, N, T, dt, scheme, integrator="rk3", src_mode="cell"):
    """One trajectory, one grid, one scheme.  K must already be on this grid.

    ``src_mode`` selects how the injection well is discretised: "cell" is the
    generator's per-cell renormalisation (regridded to carry the physical mass at
    every dx), "exact" is the analytic cell average.
    """
    dx = g.DOMAIN / N
    one = lambda k: np.asarray(prm[k], dtype=np.float64)[idx:idx + 1]  # noqa: E731

    qx, qy, div_res = g.darcy_velocity(K, one("gx"), one("gy"), dx=dx)
    th, rb, sm, kl = one("theta"), one("rho_b"), one("smax"), one("k_lang")
    D = g.dispersion_tensor(qx, qy, th, one("alpha_l"), one("alpha_t"))
    X, Y = g._grid(N, dx)

    ctx = {"qx": qx, "qy": qy, "D": D, "theta_b": th[:, None, None],
           "lam_b": one("lam")[:, None, None], "lam": float(one("lam")[0]),
           "dx": dx, "src_scale": 1.0 / dx ** 2,
           "src_exact": src_mode == "exact", "scheme": scheme, "X": X, "Y": Y,
           "x0": one("x0"), "y0": one("y0"), "sigma": one("sigma"),
           "Q": one("q_src"), "t_ramp": one("t_ramp"),
           "th1": th, "rb1": rb, "sm1": sm, "kl1": kl}

    m = np.zeros((1, N, N))
    Mp = 0.0
    clips = 0
    resid = 0.0
    scale = 1e-30
    step = _step_rk3 if integrator == "rk3" else _step_euler

    t0 = time.perf_counter()
    for k in range(T):
        m, Mp, n = step(m, Mp, ctx, k * dt, dt)
        clips += n
        Mnum = float(m.sum() * dx ** 2)
        scale = max(scale, abs(Mnum))
        resid = max(resid, abs(Mnum - Mp))
    wall = time.perf_counter() - t0

    C = g.langmuir_inverse(m, th, rb, sm, kl)
    return {"C": C[0], "wall_s": wall, "steps": T, "dx": dx,
            "mass_resid": float(resid / scale), "projection_cells": clips,
            "div_res": float(div_res), "M_final": float(m.sum() * dx ** 2)}
# --------------------------------------------------------------------------- #
# case set-up: one conductivity realization at the reference resolution, then
# restricted to every coarser grid, so all schemes see the *same* physical case
# --------------------------------------------------------------------------- #
def _restrict(a, f):
    """Block-mean (volume-average) restriction by a factor f on the last two axes."""
    if f == 1:
        return a
    N = a.shape[-1] // f
    return a.reshape(*a.shape[:-2], N, f, N, f).mean(axis=(-3, -1))


def make_cases(ncase, N_ref, seed=0, regime="train"):
    rng = np.random.default_rng(seed)
    prm = g.sample_parameters(rng, ncase, regime)
    K_ref = g.gen_log_normal_field(rng, ncase, N_ref, prm["corr_len"], prm["k_var"])
    return prm, K_ref


def dt_limit(qx, qy, D, dx, safety=0.9):
    """Largest stable step under the generator's own CFL gates."""
    adv = 0.4 * dx / max(float(np.max(np.sqrt(qx ** 2 + qy ** 2))), 1e-12)
    dif = 0.2 * dx ** 2 / max(float(np.max(D)), 1e-12)
    return safety * min(adv, dif)


def _grid_at(K_ref, N_ref, N):
    return _restrict(K_ref, N_ref // N)
# --------------------------------------------------------------------------- #
# A. consistency guard against the shipped generator
# --------------------------------------------------------------------------- #
def check_consistency(prm, K_ref, N_ref, N=32, dt=1e-3):
    dx = g.DOMAIN / N
    K = _grid_at(K_ref[0:1], N_ref, N)
    one = lambda k: np.asarray(prm[k], dtype=np.float64)[0:1]  # noqa: E731
    qx, qy, _ = g.darcy_velocity(K, one("gx"), one("gy"), dx=dx)
    th, rb, sm, kl = one("theta"), one("rho_b"), one("smax"), one("k_lang")
    D = g.dispersion_tensor(qx, qy, th, one("alpha_l"), one("alpha_t"))
    X, Y = g._grid(N, dx)

    xx = (X - 32.0) ** 2 + (Y - 32.0) ** 2
    C0 = np.exp(-xx / (2.0 * 4.0 ** 2))
    src = g.injection_source(X, Y, one("x0"), one("y0"), one("sigma"), one("q_src"),
                             one("t_ramp"), 0.5) / dx ** 2
    ref = g.fvm_step(C0, qx, qy, D, th, rb, sm, kl, one("lam"), src, dt, dx=dx)

    ctx = {"qx": qx, "qy": qy, "D": D, "theta_b": th[:, None, None],
           "lam_b": one("lam")[:, None, None], "lam": float(one("lam")[0]), "dx": dx,
           "src_scale": 1.0 / dx ** 2, "src_exact": False,
           "scheme": "upwind", "X": X, "Y": Y, "x0": one("x0"), "y0": one("y0"),
           "sigma": one("sigma"), "Q": one("q_src"), "t_ramp": one("t_ramp"),
           "th1": th, "rb1": rb, "sm1": sm, "kl1": kl}
    m0 = th[:, None, None] * C0 + rb[:, None, None] * g.langmuir(
        C0, sm[:, None, None], kl[:, None, None])
    m1, _, _ = _step_euler(m0, 0.0, ctx, 0.5, dt)
    got = g.langmuir_inverse(m1, th, rb, sm, kl)
    return float(np.abs(got - ref).max() / np.abs(ref).max())


def check_trajectory_consistency(prm, K_ref, N_ref, N=64, T=200, dt=0.1, idx=0):
    """Whole-trajectory identity: the generator's own loop against this operator.

    The one-step guard compares a single update; this compares all ``T`` of them,
    which is the claim the error budget rests on --- same flux, same well, same
    positivity projection, and no drift of a different order over 200 steps.
    """
    dx = g.DOMAIN / N
    K = _grid_at(K_ref[idx:idx + 1], N_ref, N)
    one = lambda k: np.asarray(prm[k], dtype=np.float64)[idx:idx + 1]  # noqa: E731
    qx, qy, _ = g.darcy_velocity(K, one("gx"), one("gy"), dx=dx)
    th, rb, sm, kl = one("theta"), one("rho_b"), one("smax"), one("k_lang")
    D = g.dispersion_tensor(qx, qy, th, one("alpha_l"), one("alpha_t"))
    X, Y = g._grid(N, dx)

    C = np.zeros((1, N, N))
    for t in range(T):
        src = g.injection_source(X, Y, one("x0"), one("y0"), one("sigma"),
                                 one("q_src"), one("t_ramp"), t * dt) / dx ** 2
        C = g.fvm_step(C, qx, qy, D, th, rb, sm, kl, one("lam"), src, dt, dx=dx)
    ours = run_case(K, prm, idx, N, T, dt, "upwind", "euler")["C"]
    return float(np.linalg.norm(ours - C[0]) / np.linalg.norm(C[0]))


# --------------------------------------------------------------------------- #
# B / C. accuracy, cost and mass audit
# --------------------------------------------------------------------------- #
def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _rel_l1(a, b):
    return float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-30))


def _core_cells(C):
    """Cells above half the peak: the resolution of the plume core."""
    return int((C > 0.5 * C.max()).sum())


def _moments(C, dx):
    """Mass-weighted centroid and rms radius, for the bulk (converged) part."""
    N = C.shape[-1]
    x = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(x, x)
    m = C.sum() * dx ** 2
    cx, cy = (X * C).sum() * dx ** 2 / m, (Y * C).sum() * dx ** 2 / m
    r2 = (((X - cx) ** 2 + (Y - cy) ** 2) * C).sum() * dx ** 2 / m
    return float(cx), float(cy), float(np.sqrt(r2))


def run_reference(prm, K_ref, N_hi, N_ref, t_end, dt0=0.1, verbose=True):
    """WENO5 at the reference resolution, time step refined with dx**(5/3).

    Returns the stacked fields, one WENO5 run per case (each case has its own
    velocity field, so it has its own stability limit).  `K_ref` lives on N_hi;
    the reference grid sees its block-mean restriction, so the reference and the
    scheme runs are restrictions of one physical realisation.
    """
    ncase = K_ref.shape[0]
    out = np.empty((ncase, N_ref, N_ref))
    info = []
    for i in range(ncase):
        K = _grid_at(K_ref[i:i + 1], N_hi, N_ref)
        dt, T = _dt_for(K, prm, i, N_ref, "weno5", dt0, 1.0, t_end)
        r = run_case(K, prm, i, N_ref, T, dt, "weno5", "euler")
        out[i] = r["C"]
        info.append({"case": i, "dt": dt, "steps": T, "wall_s": r["wall_s"],
                     "mass_resid": r["mass_resid"],
                     "projection_cells": r["projection_cells"]})
        if verbose:
            print(f"[ref] case {i} N={N_ref} dt={dt:.4f} steps={T} "
                  f"wall={r['wall_s']:.1f}s mass={r['mass_resid']:.2e}", flush=True)
    return out, info
def _dt_for(K, prm, idx, N, scheme, dt0=0.1, dx0=1.0, t_end=None):
    """Refine dt with dx**(p/3), then clamp at the generator's own stability gate."""
    dx = g.DOMAIN / N
    one = lambda k: np.asarray(prm[k], dtype=np.float64)[idx:idx + 1]  # noqa: E731
    qx, qy, _ = g.darcy_velocity(K, one("gx"), one("gy"), dx=dx)
    D = g.dispersion_tensor(qx, qy, one("theta"), one("alpha_l"), one("alpha_t"))
    dt = min(dt0 * (dx / dx0) ** DT_EXP[scheme], dt_limit(qx, qy, D, dx))
    if t_end is None:
        return dt, 0
    T = int(np.ceil(t_end / dt))
    return t_end / T, T


def study_benchmark_grid(prm, K_ref, N_hi, ref_C, ref_N, ncase, N=64, dt=0.1, T=200):
    """Table A: the shipped configuration, scored against the refined solution."""
    rows = []
    for scheme in SCHEMES:
        errs, l1s, walls, masses, projs, cores = [], [], [], [], [], []
        for i in range(ncase):
            K = _grid_at(K_ref[i:i + 1], N_hi, N)
            r = run_case(K, prm, i, N, T, dt, scheme, "euler")
            ref = _restrict(ref_C[i], ref_N // N)
            e = _rel_l2(r["C"], ref)
            errs.append(e)
            l1s.append(_rel_l1(r["C"], ref))
            walls.append(r["wall_s"])
            masses.append(r["mass_resid"])
            projs.append(r["projection_cells"])
            cores.append(_core_cells(r["C"]))
            print(f"[A] {scheme:6s} case {i} N={N} T={T} errL2={e:.3e} "
                  f"errL1={l1s[-1]:.3f} core={cores[-1]} "
                  f"mass={r['mass_resid']:.1e} wall={r['wall_s']:.1f}s", flush=True)
        rows.append({"scheme": scheme, "N": N, "T": T, "dt": dt,
                     "err_rel_l2": errs, "err_rel_l2_mean": float(np.mean(errs)),
                     "err_rel_l1": l1s, "err_rel_l1_mean": float(np.mean(l1s)),
                     "core_cells": cores, "core_cells_ref": _core_cells(ref_C[0]),
                     "wall_s": walls, "wall_mean": float(np.mean(walls)),
                     "mass_resid": max(masses), "projection_cells": int(sum(projs))})
    return rows


def study_order(prm, K_ref, N_hi, ref_C0, ref_N, grids, t_end, dt0=0.1):
    """Table B: joint refinement, physical parameters fixed, error vs the reference."""
    out = {}
    for scheme in SCHEMES:
        rows = []
        for N in grids:
            K = _grid_at(K_ref[0:1], N_hi, N)
            dt, T = _dt_for(K, prm, 0, N, scheme, dt0, 1.0, t_end)
            r = run_case(K, prm, 0, N, T, dt, scheme, "euler")
            e = _rel_l2(r["C"], _restrict(ref_C0, ref_N // N))
            rows.append({"N": N, "dx": g.DOMAIN / N, "dt": dt, "steps": T,
                         "err_rel_l2": e, "mass_resid": r["mass_resid"],
                         "wall_s": r["wall_s"],
                         "projection_cells": r["projection_cells"]})
            print(f"[B] {scheme:6s} N={N:3d} dt={dt:.4f} steps={T:4d} "
                  f"err={e:.3e} wall={r['wall_s']:.1f}s", flush=True)
        e = np.array([r["err_rel_l2"] for r in rows])
        d = np.array([r["dx"] for r in rows])
        order_fit = float(np.polyfit(np.log(d), np.log(e), 1)[0])
        out[scheme] = {"rows": rows,
                       "order": (np.log(e[:-1] / e[1:]) / np.log(d[:-1] / d[1:])).tolist(),
                       "order_fit": order_fit,
                       # the reference is not exact: its own error is bounded by
                       # the finest-grid error divided by the measured rate
                       "ref_err_bound": float(e[-1] / 2.0 ** max(order_fit, 0.5))}
    return out
# --------------------------------------------------------------------------- #
# C. 2D implementation check: formal order on a smooth analytic solution
# --------------------------------------------------------------------------- #
_DERIV2 = {2: ((-1, 1.0), (0, -2.0), (1, 1.0)),
           4: ((-2, -1.0), (-1, 16.0), (0, -30.0), (1, 16.0), (2, -1.0)),
           6: ((-3, 2.0), (-2, -27.0), (-1, 270.0), (0, -490.0),
               (1, 270.0), (2, -27.0), (3, 2.0))}


def _deriv2_axis(C, dx, order, axis):
    """Second derivative along one axis, at the stencil order matching the scheme.

    An O(dx^2) diffusion floor would cap WENO5 at second order and the measured
    order would say nothing about the advective reconstruction.
    """
    st = _DERIV2[order]
    norm = {2: 1.0, 4: 12.0, 6: 180.0}[order]
    out = np.zeros_like(C)
    for k, w in st:
        out += w * (C if k == 0 else np.roll(C, -k, axis=axis))
    return out / (norm * dx ** 2)


def smooth2d_run(N, scheme, cfl=0.25, t_end=2.0, v=1.0, Dm=0.5,
                 s0=4.0, L=64.0, n_ref=64):
    """Advected-diffused 2D Gaussian, periodic, constant velocity and D.

    Exact: C = (s0^2/s^2) exp(-r^2/2s^2), s^2 = s0^2 + 2 Dm t.  The domain is 64 m
    and the pulse stays within 20 m of its start, so the periodic images sit at
    exp(-50) and can be dropped.
    """
    dx = L / N
    x = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(x, x)
    x0 = y0 = 24.0
    C = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * s0 ** 2))[None]
    p = 5.0 / 3.0 if scheme == "weno5" else 1.0
    dt = min(cfl * dx / v * (dx / (L / n_ref)) ** (p - 1.0), 0.2 * dx ** 2 / Dm)
    steps = max(1, int(np.ceil(t_end / dt)))
    dt = t_end / steps
    o2 = 2 if scheme == "upwind" else (4 if scheme == "muscl" else 6)

    def rhs(Cs):
        ux = np.full_like(Cs, v)
        Fx = ux * _faces(Cs, ux, scheme, -1)
        uy = np.full_like(Cs, v)
        Fy = uy * _faces(Cs, uy, scheme, -2)
        div = (Fx - np.roll(Fx, 1, axis=-1)) / dx + (Fy - np.roll(Fy, 1, axis=-2)) / dx
        return -div + Dm * (_deriv2_axis(Cs, dx, o2, -1) + _deriv2_axis(Cs, dx, o2, -2))

    for _ in range(steps):
        k1 = rhs(C)
        k2 = rhs(C + dt * k1)
        k3 = rhs(C + 0.25 * dt * k1 + 0.25 * dt * k2)
        C = C + dt * (k1 / 6 + k2 / 6 + 2 * k3 / 3)

    s = np.sqrt(s0 ** 2 + 2 * Dm * t_end)
    ex = (s0 / s) ** 2 * np.exp(-((X - x0 - v * t_end) ** 2
                                  + (Y - y0 - v * t_end) ** 2) / (2 * s ** 2))[None]
    err = float(np.linalg.norm(C - ex) / np.linalg.norm(ex))
    return {"N": N, "dx": dx, "dt": dt, "steps": steps, "err_rel_l2": err}


def study_smooth2d(n_list):
    out = {}
    for scheme in SCHEMES:
        rows = [smooth2d_run(N, scheme) for N in n_list]
        e = np.array([r["err_rel_l2"] for r in rows])
        d = np.array([r["dx"] for r in rows])
        out[scheme] = {"rows": rows,
                       "order": (np.log(e[:-1] / e[1:]) / np.log(d[:-1] / d[1:])).tolist(),
                       "order_fit": float(np.polyfit(np.log(d), np.log(e), 1)[0])}
        print(f"[C] {scheme:6s} order_fit={out[scheme]['order_fit']:+.2f} "
              f"err={['%.2e' % x for x in e]}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# D. where the benchmark-grid error comes from: scheme, step or resolution
# --------------------------------------------------------------------------- #
def study_decomposition(prm, K_ref, N_hi, ref_field, ref_N, N=64, dt=0.1, T=200,
                        t_end=20.0):
    """Relative-L2 budget at the benchmark grid, all terms against the same case."""
    def run(n, steps, step_dt, scheme):
        return run_case(_grid_at(K_ref[0:1], N_hi, n), prm, 0, n, steps, step_dt,
                        scheme, "euler")["C"]

    u64 = run(N, T, dt, "upwind")
    w64 = run(N, T, dt, "weno5")
    m64 = run(N, T, dt, "muscl")
    u64f = run(N, 2 * T, dt / 2, "upwind")
    # the reference step of the shipped archive (Sec. 5.5), same case, same solver
    t_fine = int(round(t_end / 0.016))
    u64f6 = run(N, t_fine, 0.016, "upwind")
    rdt, rsteps = _dt_for(_grid_at(K_ref[0:1], N_hi, ref_N), prm, 0, ref_N, "weno5",
                          dt, 1.0, t_end)
    u_fine = run(ref_N, rsteps, rdt, "upwind")
    f = ref_N // N
    u64_ref = _restrict(u_fine, f)
    dec = {
        "scheme_upwind_vs_muscl": _rel_l2(u64, m64),
        "scheme_upwind_vs_weno5": _rel_l2(u64, w64),
        "scheme_upwind_vs_weno5_at_fine": _rel_l2(u_fine, ref_field),
        "timestep_dt_halved": _rel_l2(u64, u64f),
        "timestep_dt_fine": _rel_l2(u64, u64f6),
        "timestep_dt_fine_value": 0.016,
        "timestep_dt_fine_steps": t_fine,
        "grid_64_vs_fine_L2": _rel_l2(u64, u64_ref),
        "grid_64_vs_fine_L1": _rel_l1(u64, u64_ref),
        "peak_N64": float(u64.max()), "peak_N64_weno5": float(w64.max()),
        "peak_fine_upwind": float(u_fine.max()),
        "peak_fine_weno5": float(ref_field.max()),
        "core_cells_N64": _core_cells(u64), "core_cells_fine": _core_cells(u_fine),
        "centroid_N64": _moments(u64, g.DOMAIN / N),
        "centroid_fine": _moments(u_fine, g.DOMAIN / ref_N),
        "err_L1_weno5_vs_ref_at_64": _rel_l1(w64, u64_ref),
    }
    print(f"[D] scheme  up-vs-muscl={dec['scheme_upwind_vs_muscl']:.3f} "
          f"up-vs-weno5={dec['scheme_upwind_vs_weno5']:.3f} "
          f"| dt halved={dec['timestep_dt_halved']:.1e} "
          f"dt=0.016={dec['timestep_dt_fine']:.1e} "
          f"| grid 64-vs-{ref_N} L1={dec['grid_64_vs_fine_L1']:.3f} "
          f"L2={dec['grid_64_vs_fine_L2']:.3f} "
          f"core={dec['core_cells_N64']}->{dec['core_cells_fine']} cells", flush=True)
    return dec


def study_source(prm, K_ref, N_hi, ref_N, t_end, grids=(32, 64), dt0=0.1):
    """Table E: does the injection well, not the scheme, set the observed order?

    The generator normalises the well profile by its discrete sum, which makes
    the injected mass grid-dependent and the profile only second-order accurate.
    WENO5 is run twice on the same case: with that profile ("cell") and with the
    analytic cell average of the same Gaussian ("exact").  If the observed order
    jumps to the formal one only in the second case, the well is the floor.
    """
    out = {}
    for mode in ("cell", "exact"):
        rows = []
        for N in list(grids) + [ref_N]:
            K = _grid_at(K_ref[0:1], N_hi, N)
            dt, T = _dt_for(K, prm, 0, N, "weno5", dt0, 1.0, t_end)
            r = run_case(K, prm, 0, N, T, dt, "weno5", "rk3", src_mode=mode)
            rows.append({"N": N, "dx": g.DOMAIN / N, "dt": dt, "steps": T,
                         "C": r["C"], "mass_resid": r["mass_resid"],
                         "injected": r["M_final"]})
            print(f"[E] {mode:5s} N={N:3d} steps={T:4d} mass_resid="
                  f"{r['mass_resid']:.1e}", flush=True)
        ref = rows[-1]["C"]
        for row in rows:
            row["err_rel_l2"] = _rel_l2(row["C"], _restrict(ref, ref_N // row["N"]))
        e = np.array([r["err_rel_l2"] for r in rows[:-1]])
        d = np.array([r["dx"] for r in rows[:-1]])
        rate = (np.log(e[:-1] / e[1:]) / np.log(d[:-1] / d[1:])).tolist()
        out[mode] = {"rows": [{k: v for k, v in r.items() if k != "C"} for r in rows],
                     "order": rate,
                     "order_fit": float(np.polyfit(np.log(d), np.log(e), 1)[0])}
        print(f"[E] {mode:5s} rate={out[mode]['order_fit']:+.2f} "
              f"per_grid={['%.2f' % o for o in rate]}", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="2D classical-scheme baseline (P0-2)")
    ap.add_argument("--ncase", type=int, default=2)
    ap.add_argument("--nref", type=int, default=128, help="reference grid")
    ap.add_argument("--nref-order", type=int, default=0,
                    help="reference grid for the refinement study; 0 = reuse --nref")
    ap.add_argument("--grids", default="16,32,64")
    ap.add_argument("--smooth-grids", default="64,128,256")
    ap.add_argument("--T", type=int, default=200, help="steps at the benchmark grid")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--t-end", type=float, default=None, help="default: T * dt")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/verification/baseline_2d.json")
    args = ap.parse_args(argv)

    if args.quick:
        args.ncase, args.nref, args.nref_order, args.grids = 1, 128, 0, "32,64"
        args.smooth_grids = "64,128"
    if not args.nref_order:
        args.nref_order = args.nref
    n_hi = max(args.nref, args.nref_order)
    grids = [int(x) for x in str(args.grids).split(",") if x.strip()]
    smooth_grids = [int(x) for x in str(args.smooth_grids).split(",") if x.strip()]
    t_end = args.t_end if args.t_end is not None else args.T * args.dt
    t0 = time.time()

    prm, K_ref = make_cases(args.ncase, n_hi, seed=args.seed)
    print(f"case set: ncase={args.ncase} N_hi={n_hi} t_end={t_end} "
          f"Pe={np.round(prm['Pe'], 2).tolist()} "
          f"Da={np.round(prm['Da'], 2).tolist()} "
          f"R0={np.round(prm['R0'], 2).tolist()}", flush=True)

    cons = check_consistency(prm, K_ref, n_hi)
    print(f"[guard] |upwind(euler) - fvm_step| / max = {cons:.2e}", flush=True)
    traj = check_trajectory_consistency(prm, K_ref, n_hi, N=64, T=args.T, dt=args.dt)
    print(f"[guard] |upwind(euler) - fvm_step| over {args.T} steps = {traj:.2e}",
          flush=True)

    ref_C, ref_info = run_reference(prm, K_ref, n_hi, args.nref, t_end, args.dt)
    bench = study_benchmark_grid(prm, K_ref, n_hi, ref_C, args.nref, args.ncase,
                                 N=64, dt=args.dt, T=args.T)
    if args.nref_order == args.nref:
        ref_B, ref_B_info = ref_C[:1], ref_info[:1]
    else:
        ref_B, ref_B_info = run_reference(prm, K_ref[:1], n_hi, args.nref_order,
                                          t_end, args.dt)
    refine = study_order(prm, K_ref, n_hi, ref_B[0], args.nref_order, grids, t_end,
                         args.dt)
    smooth = study_smooth2d(smooth_grids)
    dec = study_decomposition(prm, K_ref, n_hi, ref_B[0], args.nref_order, t_end=t_end)
    # E needs its own high-resolution conductivity so that 32/64/128 all see one
    # physical realisation and 256 can serve as the reference.
    _, K_src = make_cases(1, 512, seed=args.seed + 7)
    srcE = study_source(prm, K_src, 512, 256, t_end, grids=(32, 64, 128))

    print("\n[A] benchmark grid (N=64, dt=%.3f, %d steps), error vs N=%d WENO5"
          % (args.dt, args.T, args.nref))
    for r in bench:
        print(f"    {r['scheme']:6s} errL2={r['err_rel_l2_mean']:.3f} "
              f"errL1={r['err_rel_l1_mean']:.3f}  "
              f"mass={r['mass_resid']:.1e}  wall={r['wall_mean']:.2f}s")
    print("\n[B] refinement at the benchmark regime (error vs the refined solution)")
    for s in SCHEMES:
        r = refine[s]
        print(f"    {s:6s} rate={r['order_fit']:+.2f}  "
              f"per_grid={['%.2f' % o for o in r['order']]}  "
              f"err={['%.2e' % x['err_rel_l2'] for x in r['rows']]}")
    print("\n[C] smooth 2D Gaussian: formal order of each reconstruction")
    for s in SCHEMES:
        r = smooth[s]
        print(f"    {s:6s} order_fit={r['order_fit']:+.2f}  "
              f"per_grid={['%.2f' % o for o in r['order']]}")
    print("\n[E] WENO5 order with a per-cell vs a cell-averaged injection well")
    for m in ("cell", "exact"):
        print(f"    {m:5s} order_fit={srcE[m]['order_fit']:+.2f}  "
              f"per_grid={['%.2f' % o for o in srcE[m]['order']]}")

    report = {"ncase": args.ncase, "n_hi": n_hi, "n_ref": args.nref,
              "n_ref_order": args.nref_order, "t_end": t_end,
              "benchmark_grid": {"N": 64, "dt": args.dt, "steps": args.T},
              "consistency_upwind_vs_fvm_step": cons,
              "consistency_full_trajectory": traj,
              "reference": ref_info, "reference_order": ref_B_info,
              "table_a_benchmark": bench, "table_b_regime_refinement": refine,
              "table_c_smooth_order": smooth, "table_d_decomposition": dec,
              "table_e_source_order": srcE,
              "wall_clock_s": time.time() - t0}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}  ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
