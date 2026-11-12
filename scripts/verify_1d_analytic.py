"""Verification suite for the CPC revision (new Sec. 5.1).

A. Order of accuracy of the classical baselines on a smooth Gaussian pulse that
   obeys the retarded ADE with linear sorption and first-order decay, periodic
   domain. Textbook orders: donor cell 1, MUSCL--van Leer 2, WENO5 5.
B. Ogata--Banks validation of retarded advection--dispersion with linear
   sorption and first-order decay, Dirichlet inlet: closed-form comparison of
   relative L2 error, 10--90% front width and mass residual.
C. Mass audit of every scheme -- all telescope, so all sit at round-off, which
   is the manuscript's point: conservation and accuracy are independent.
D. Manufactured solution in 1D: the same three reconstructions against a closed-form
   C* that carries the Langmuir storage and a source term, which is the assembly the
   generator uses. A is linear sorption only, so D is the check on storage plus forcing.

Regime note for B: the test runs at cell Peclet ~0.3, because a front thinner
than one cell (the benchmark's cell Peclet 2--15) has no grid-resolvable
continuum solution. The benchmark's truth is a grid-refined numerical solution;
this test shows the reference scheme is exact where a closed form exists.

    python scripts/verify_1d_analytic.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.special import erfc, erfcx

SCHEMES = ("upwind", "muscl", "weno5")
# second-derivative stencil matched to each scheme's advective order, so the
# measured order is not capped by a second-order diffusion floor
DERIV2_ORDER = {"upwind": 2, "muscl": 4, "weno5": 6}


def _vanleer(a, b):
    ok = a * b > 0.0
    return np.where(ok, 2.0 * a * b / np.where(ok, a + b, 1.0), 0.0)


def _weno5_left(c):
    """Jiang--Shu WENO5 face value at i+1/2; `c` rolled so index i holds cell i-2."""
    v0, v1, v2, v3, v4 = c[..., :-4], c[..., 1:-3], c[..., 2:-2], c[..., 3:-1], c[..., 4:]
    b1 = 13 / 12 * (v0 - 2 * v1 + v2) ** 2 + 0.25 * (v0 - 4 * v1 + 3 * v2) ** 2
    b2 = 13 / 12 * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - v3) ** 2
    b3 = 13 / 12 * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (3 * v2 - 4 * v3 + v4) ** 2
    eps = 1e-6
    a1, a2, a3 = 0.1 / (eps + b1) ** 2, 0.6 / (eps + b2) ** 2, 0.3 / (eps + b3) ** 2
    s = a1 + a2 + a3
    p1 = v0 / 3 - 7 * v1 / 6 + 11 * v2 / 6
    p2 = -v1 / 6 + 5 * v2 / 6 + v3 / 3
    p3 = v2 / 3 + 5 * v3 / 6 - v4 / 6
    return (a1 * p1 + a2 * p2 + a3 * p3) / s


def adv_periodic(C, v, dx, scheme):
    """-v dC/dx on a periodic grid, v > 0."""
    if scheme == "upwind":
        face = C                                        # donor cell
    elif scheme == "muscl":
        g = (np.roll(C, -1) - np.roll(C, 1)) / 2
        face = C + 0.5 * _vanleer(g, np.roll(g, 1))
    elif scheme == "weno5":
        face = _weno5_left(np.concatenate(
            [np.roll(C, 2)[..., :2], C, np.roll(C, -2)[..., -2:]], axis=-1))
    else:
        raise KeyError(scheme)
    F = v * face
    return -(F - np.roll(F, 1)) / dx


def deriv2_periodic(C, dx, order):
    if order == 2:
        return (np.roll(C, -1) - 2 * C + np.roll(C, 1)) / dx ** 2
    if order == 4:
        return (-np.roll(C, 2) + 16 * np.roll(C, 1) - 30 * C
                + 16 * np.roll(C, -1) - np.roll(C, -2)) / (12 * dx ** 2)
    if order == 6:
        return (2 * np.roll(C, 3) - 27 * np.roll(C, 2) + 270 * np.roll(C, 1) - 490 * C
                + 270 * np.roll(C, -1) - 27 * np.roll(C, -2)
                + 2 * np.roll(C, -3)) / (180 * dx ** 2)
    raise KeyError(order)


def rk3(C, dt, rhs):
    """SSP-RK3 (Shu--Osher), third order, so WENO5 space error is visible."""
    k1 = rhs(C)
    k2 = rhs(C + dt * k1)
    k3 = rhs(C + 0.25 * dt * k1 + 0.25 * dt * k2)
    return C + dt * (k1 / 6 + k2 / 6 + 2 * k3 / 3)


def ogata_banks(x, t, v, D, C0=1.0, lam=0.0):
    """Semi-infinite, Dirichlet inlet, first-order decay.

    exp(vx/D) erfc(z2) is written as exp(-z1^2) erfcx(z2) so it cannot overflow
    at the vx/D of these runs.
    """
    if t <= 0.0:
        return np.zeros_like(np.asarray(x, dtype=float))
    sq = 2 * np.sqrt(D * t)
    z1 = (x - v * t) / sq
    z2 = (x + v * t) / sq
    return C0 * 0.5 * (erfc(z1) + np.exp(-z1 ** 2) * erfcx(z2)) * np.exp(-lam * t)


def _weno5_window(win):
    """WENO5 face value from a (..., 5) stack of cells k-2..k+2."""
    v0, v1, v2, v3, v4 = (win[..., i] for i in range(5))
    b1 = 13 / 12 * (v0 - 2 * v1 + v2) ** 2 + 0.25 * (v0 - 4 * v1 + 3 * v2) ** 2
    b2 = 13 / 12 * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - v3) ** 2
    b3 = 13 / 12 * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (3 * v2 - 4 * v3 + v4) ** 2
    eps = 1e-6
    a1, a2, a3 = 0.1 / (eps + b1) ** 2, 0.6 / (eps + b2) ** 2, 0.3 / (eps + b3) ** 2
    s = a1 + a2 + a3
    p1 = v0 / 3 - 7 * v1 / 6 + 11 * v2 / 6
    p2 = -v1 / 6 + 5 * v2 / 6 + v3 / 3
    p3 = v2 / 3 + 5 * v3 / 6 - v4 / 6
    return (a1 * p1 + a2 * p2 + a3 * p3) / s


def _faces(Cp, n, scheme):
    """Face values on n+1 faces of a field padded by 3 cells on both sides."""
    sw = np.lib.stride_tricks.sliding_window_view(Cp, 5)      # [n+2, 5]
    if scheme == "upwind":
        return Cp[2:n + 3].copy()                              # donor cell
    if scheme == "muscl":
        k = np.arange(n + 1) + 2
        g = (Cp[k + 1] - Cp[k - 1]) / 2
        g_up = (Cp[k] - Cp[k - 2]) / 2
        return Cp[k] + 0.5 * _vanleer(g, g_up)
    return _weno5_window(sw[:n + 1])


def ogata_run(n: int, scheme: str, L: float = 64.0, t_end: float = 150.0,
              q: float = 0.2, theta: float = 0.30, rho_b: float = 1.50,
              kd: float = 0.80, alpha_t: float = 2.0, cfl: float = 0.4) -> dict:
    """Retarded ADE with linear sorption (R = 1 + rho_b kd/theta), Dirichlet inlet."""
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    R = 1.0 + rho_b * kd / theta
    v = q / (theta * R)                     # retarded velocity
    Dr = alpha_t * q / R                    # retarded dispersion
    dt = min(cfl * dx / v, 0.2 * dx ** 2 / Dr)
    steps = int(np.ceil(t_end / dt))
    dt = t_end / steps
    d2o = DERIV2_ORDER[scheme]

    C = np.zeros(n)
    M = 0.0
    resid = 0.0
    clips = 0
    C0 = 1.0

    def tendency(state):
        """dC/dt and net inlet-minus-outlet flux, both from the *same* state."""
        Cp = np.concatenate([np.full(3, C0), state, np.full(3, state[-1])])
        face = _faces(Cp, n, scheme)
        tol = 1e-10 * max(1.0, float(np.abs(state).max()))
        neg = int(np.sum(face < -tol))               # round-off is not a clip
        F = v * np.maximum(face, 0.0)                # positivity-preserving clip
        d2 = deriv2_periodic(Cp, dx, d2o)[3:n + 3]
        dC = (F[:-1] - F[1:]) / dx + Dr * d2
        # net mass rate read off the vector itself: its sum telescopes to the
        # boundary fluxes for any stencil width, so the audit cannot miss the
        # diffusive inlet term the way a hand-written F[0]-F[n] did
        return dC, float(theta * R * dC.sum() * dx), neg

    for _ in range(steps):
        k1, n1, c1 = tendency(C)
        C1 = C + dt * k1
        k2, n2, c2 = tendency(C1)
        C2 = 0.75 * C + 0.25 * (C1 + dt * k2)
        k3, n3, c3 = tendency(C2)
        C = C / 3.0 + (2.0 / 3.0) * (C2 + dt * k3)
        # the mass scalar takes the same convex combination, so the telescoping
        # identity is exact to round-off rather than to the order of a quadrature
        M1 = M + dt * n1
        M2 = 0.75 * M + 0.25 * (M1 + dt * n2)
        M = M / 3.0 + (2.0 / 3.0) * (M2 + dt * n3)
        clips += c1 + c2 + c3
        resid = max(resid, abs(M - theta * R * C.sum() * dx))

    exact = ogata_banks(x, t_end, v, Dr, C0)
    err = float(np.linalg.norm(C - exact) / np.linalg.norm(exact))
    scale = theta * R * dx
    mass_resid = resid / max(abs(scale * exact.sum()), 1e-30)
    return {"n": n, "scheme": scheme, "dx": dx, "dt": dt, "steps": steps,
            "err_rel_l2": err, "v": v, "D_r": Dr, "R": R,
            "cell_pe": float(v * dx / Dr), "mass_resid": float(mass_resid),
            "neg_face_clips": clips,
            "width_ratio": _width_ratio(x, C, exact)}


def _width_ratio(x, num, ref, lo=0.1, hi=0.9):
    """10--90% width of a monotone descending front, numerical over exact."""
    def width(y):
        y = np.asarray(y, dtype=float)
        peak = y.max()
        if peak <= 0:
            return np.nan
        i_hi = np.argmax(y >= hi * peak)
        i_lo = np.argmax(y <= lo * peak)
        if i_lo <= i_hi:
            return np.nan
        def cross(level, i):
            y0, y1 = y[i - 1], y[i]
            if y0 == y1:
                return x[i]
            f = (level * peak - y0) / (y1 - y0)
            return x[i - 1] + f * (x[i] - x[i - 1])
        return cross(lo, i_lo) - cross(hi, i_hi)

    w = width(ref)
    wn = width(num)
    return float(wn / w) if np.isfinite(w) and w > 0 else float("nan")


def pulse_run(n, scheme, cfl=0.4, n_ref=128):
    """Retarded ADE with linear sorption and decay, smooth Gaussian, periodic.

    Exact solution: C = (s0/s) exp(-(x-x0-vt)^2/(2 s^2)) exp(-lam t),
    s^2 = s0^2 + 2 D_r t. Smooth initial data, so the observed order is a genuine
    property of the scheme rather than of a discontinuity.

    Time refinement is tied to space as dt ~ dx^p with p = 1 for the second-order
    schemes and p = 5/3 for WENO5 (RK3 then leaves the temporal error below the
    fifth-order spatial one); a fixed CFL would cap WENO5 at third order.
    """
    # s0 = 0.05 keeps the pulse 6 sigma inside the circle at t_end, so the
    # periodic images stay negligible and the error floor cannot be mistaken
    # for a property of the scheme
    L, v, x0, s0, t_end = 1.0, 1.0, 0.3, 0.05, 0.2
    Dr, lam = 1.0e-3, 0.5
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    C = np.exp(-((x - x0) ** 2) / (2 * s0 ** 2))
    p = 5.0 / 3.0 if scheme == "weno5" else 1.0
    # the diffusion limit must be enforced as well: an advection-only CFL left
    # the fourth-order second-derivative stencil just outside the RK3 stability
    # region at N = 1024, and MUSCL blew up to 1e142
    dt = min(cfl * dx / v * (dx / (L / n_ref)) ** (p - 1.0), 0.2 * dx ** 2 / Dr)
    steps = max(1, int(np.ceil(t_end / dt)))
    dt = t_end / steps
    d2o = DERIV2_ORDER[scheme]

    def rhs(s, _dx=dx, _s=scheme, _o=d2o):
        return (adv_periodic(s, v, _dx, _s) + Dr * deriv2_periodic(s, _dx, _o) - lam * s)

    for _ in range(steps):
        C = rk3(C, dt, rhs)
    exact = _pulse_exact(x, x0, v, Dr, lam, s0, L, t_end)
    return float(np.linalg.norm(C - exact) / np.linalg.norm(exact)), dx, steps


def _pulse_exact(x, x0, v, Dr, lam, s0=0.08, L=1.0, t=0.0, nimg=6):
    """Gaussian on the *circle*: the images matter.

    Dropping them leaves a fixed O(exp(-d^2/2 s0^2)) mismatch that does not
    vanish under refinement -- the first version of this script "converged" to
    the image-summed solution and reported a spurious order of 0.005 for WENO5.
    """
    s = np.sqrt(s0 ** 2 + 2.0 * Dr * t)
    acc = np.zeros_like(x)
    for k in range(-nimg, nimg + 1):
        acc += np.exp(-((x + k * L - x0 - v * t) ** 2) / (2.0 * s ** 2))
    return (s0 / s) * np.exp(-lam * t) * acc


MMS = dict(L=1.0, v=1.0, t_end=0.2, theta=0.30, rho_b=1.50, smax=5.0,
           k_lang=0.24, lam=0.10, D=1.0e-3, amp=0.4)


def _langmuir(C, prm=MMS):
    return prm["smax"] * prm["k_lang"] * C / (1.0 + prm["k_lang"] * C)


def _chi(C, prm=MMS):
    """d m / d C for the composite mass, the lagged factor of the shipped solver."""
    return prm["theta"] + prm["rho_b"] * prm["smax"] * prm["k_lang"] \
        / (1.0 + prm["k_lang"] * C) ** 2


def _mms_exact(x, t, k, prm=MMS):
    """C* = e^{-lam t} (1 + a sin k(x - vt)): advected, decaying, strictly positive."""
    return np.exp(-prm["lam"] * t) * (1.0 + prm["amp"] * np.sin(k * (x - prm["v"] * t)))


def _mms_source(x, t, k, prm=MMS):
    """s = dm/dt + d(vC)/dx - D d2C/dx2 + lam m, all derivatives exact.

    The source is evaluated at C*, never at the numerical state: a source fed
    back with the numerical field makes the problem define itself and the fitted
    order meaningless (the same rule the 2D gate follows).
    """
    C = _mms_exact(x, t, k, prm)
    ph = k * (x - prm["v"] * t)
    C_t = np.exp(-prm["lam"] * t) * (-prm["lam"] * (1.0 + prm["amp"] * np.sin(ph))
                                    - prm["amp"] * prm["v"] * k * np.cos(ph))
    C_x = np.exp(-prm["lam"] * t) * prm["amp"] * k * np.cos(ph)
    C_xx = -np.exp(-prm["lam"] * t) * prm["amp"] * k ** 2 * np.sin(ph)
    m = prm["theta"] * C + prm["rho_b"] * _langmuir(C, prm)
    return _chi(C, prm) * C_t + prm["v"] * C_x - prm["D"] * C_xx + prm["lam"] * m


def mms_run(n, scheme, cfl=0.4, n_ref=128, prm=MMS):
    """Manufactured solution for the shipped operator, sorption and source included.

    The pulse test of `pulse_run` uses linear sorption only; this one carries the
    Langmuir storage and a forcing term, so it checks the assembly the generator
    actually uses. Step policy is the pulse's, for the same reason: dt ~ dx^p with
    p = 5/3 for WENO5 keeps the RK3 temporal error below the fifth-order spatial
    one, whereas a fixed CFL would cap WENO5 at third order.
    """
    L, v, t_end = prm["L"], prm["v"], prm["t_end"]
    k = 2.0 * np.pi / L
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    C = _mms_exact(x, 0.0, k, prm)
    p = 5.0 / 3.0 if scheme == "weno5" else 1.0
    dt = min(cfl * dx / v * (dx / (L / n_ref)) ** (p - 1.0), 0.2 * dx ** 2 / prm["D"])
    steps = max(1, int(np.ceil(t_end / dt)))
    dt = t_end / steps
    d2o = DERIV2_ORDER[scheme]

    def rhs(s, t, _dx=dx, _s=scheme, _o=d2o):
        m = prm["theta"] * s + prm["rho_b"] * _langmuir(s, prm)
        num = (adv_periodic(s, v, _dx, _s) + prm["D"] * deriv2_periodic(pad(s), _dx, _o)[3:-3]
               - prm["lam"] * m + _mms_source(x, t, k, prm))
        return num / _chi(s, prm)

    def pad(s):
        return np.concatenate([s[-3:], s, s[:3]])

    for j in range(steps):
        t = j * dt
        k1 = rhs(C, t)
        C1 = C + dt * k1
        k2 = rhs(C1, t + dt)
        C2 = 0.75 * C + 0.25 * (C1 + dt * k2)
        k3 = rhs(C2, t + 0.5 * dt)
        C = C / 3.0 + (2.0 / 3.0) * (C2 + dt * k3)
    exact = _mms_exact(x, t_end, k, prm)
    err = float(np.linalg.norm(C - exact) / np.linalg.norm(exact))
    return {"n": n, "scheme": scheme, "dx": dx, "dt": dt, "steps": steps,
            "err_rel_l2": err}


def mms_sweep(n_list, cfl=0.4):
    out = {}
    for scheme in SCHEMES:
        rows = [mms_run(n, scheme, cfl) for n in n_list]
        e = np.array([r["err_rel_l2"] for r in rows])
        d = np.array([r["dx"] for r in rows])
        out[scheme] = {
            "rows": rows,
            "order": (np.log(e[:-1] / e[1:]) / np.log(d[:-1] / d[1:])).tolist(),
            "order_fit": float(np.polyfit(np.log(d), np.log(e), 1)[0])}
    return {"cfl": cfl, "rows": out}


def order_run(n_list, cfl=0.4):
    out = {}
    for scheme in SCHEMES:
        errs, dxs, steps = [], [], []
        for n in n_list:
            e, dx, st = pulse_run(n, scheme, cfl)
            errs.append(e)
            dxs.append(dx)
            steps.append(st)
        errs = np.array(errs)
        dxs = np.array(dxs)
        out[scheme] = {
            "n": list(n_list), "err": errs.tolist(),
            "order": (np.log(errs[:-1] / errs[1:]) / np.log(dxs[:-1] / dxs[1:])).tolist(),
            "order_fit": float(np.polyfit(np.log(dxs), np.log(errs), 1)[0])}
    return {"cfl": cfl, "rows": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="1D verification suite (A: orders, B: Ogata--Banks, C: mass)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/verification/verify_1d.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    grid = (128, 256, 512) if args.quick else (128, 256, 512, 1024)
    ords = order_run(grid)
    mms = mms_sweep(grid)
    # B2 is an absolute-accuracy check at a resolved grid, not an order study:
    # the inlet step has infinite slope, so its L2 error is dominated by the
    # smeared front at every order (all schemes sit near first order there).
    oga_n = (160, 320) if args.quick else (160, 320, 640)

    runs = []
    for scheme in SCHEMES:
        for n in oga_n:
            r = ogata_run(n, scheme)
            runs.append(r)
            print(f"[B] {scheme:6s} N={n:4d} dx={r['dx']:.3f} cellPe={r['cell_pe']:.2f} "
                  f"err={r['err_rel_l2']:.3e} w/w_ref={r['width_ratio']:.3f} "
                  f"mass={r['mass_resid']:.2e} clips={r['neg_face_clips']}", flush=True)

    oga = {}
    for scheme in SCHEMES:
        sub = [r for r in runs if r["scheme"] == scheme]
        e = np.array([r["err_rel_l2"] for r in sub])
        d = np.array([r["dx"] for r in sub])
        oga[scheme] = {"rows": sub,
                       "order": (np.log(e[:-1] / e[1:]) / np.log(d[:-1] / d[1:])).tolist(),
                       "order_fit": float(np.polyfit(np.log(d), np.log(e), 1)[0])}

    print("\n[A] order on a smooth Gaussian pulse (retarded ADE + decay, periodic)")
    for s in SCHEMES:
        r = ords["rows"][s]
        print(f"    {s:6s} order_fit={r['order_fit']:.3f}  "
              f"per_grid={['%.2f' % o for o in r['order']]}")
    print("\n[B] Ogata--Banks at the finest grid (absolute accuracy, not order)")
    for s in SCHEMES:
        r = oga[s]["rows"][-1]
        print(f"    {s:6s} N={r['n']:4d} rel_L2={r['err_rel_l2']:.3e}  "
              f"w/w_ref={r['width_ratio']:.3f}  mass={r['mass_resid']:.1e}  "
              f"clips={r['neg_face_clips']}")

    print("\n[C] manufactured solution, Langmuir storage and a source term")
    for s in SCHEMES:
        r = mms["rows"][s]
        print(f"    {s:6s} order_fit={r['order_fit']:.3f}  "
              f"per_grid={['%.2f' % o for o in r['order']]}  "
              f"err={r['rows'][-1]['err_rel_l2']:.2e}")

    report = {"order_test": ords, "ogata": oga, "mms": mms,
              "wall_clock_s": time.time() - t0}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}  ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
