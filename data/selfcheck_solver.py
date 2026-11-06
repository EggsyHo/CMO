"""Two gates that must pass before any training data is written (implementation.md §3.2).

    python data/selfcheck_solver.py

Gate 1 (manufactured solution) catches a sign error in a flux that the mass gate
cannot see: a wrong-signed diffusive flux still conserves mass exactly, because
it telescopes. Gate 2 calibrates `front_width`; every later w number inherits it.
"""
from __future__ import annotations

import json
import os
import platform
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gen_adr2d import fvm_step, langmuir, mass_balance_series  # noqa: E402
from src.utils import metrics  # noqa: E402

L_DOMAIN = 64.0
# A plausible point inside the training bands, so the sorption nonlinearity is
# actually exercised rather than nearly linear.
PRM = dict(theta=0.30, rho_b=1.50, smax=5.0, k_lang=0.24, lam=0.10)


def _mms_exact(X, Y, t, a, k):
    """C* = e^-t (1 + a sin kx sin ky).

    Deliberately NOT the `e^-t prod sin` of §3.2: that field is negative over half
    the domain and `langmuir_inverse` clamps negative roots to zero, so the clamp
    would absorb exactly the error this gate exists to measure.
    """
    return np.exp(-t) * (1.0 + a * np.sin(k * X) * np.sin(k * Y))


def _mms_source(C, t, a, k, D, prm):
    """s = dm/dt + div(qC - theta D grad C) + lam m, with q = 0, D isotropic."""
    th, rb = prm["theta"], prm["rho_b"]
    sm, kl, lam = prm["smax"], prm["k_lang"], prm["lam"]
    lap = -2.0 * k * k * (C - np.exp(-t))          # laplacian of the perturbation
    dsorb = sm * kl / (1.0 + kl * C) ** 2
    # `langmuir` is already full-shape once smax broadcasts against C; indexing
    # the result took row 0 and broadcast it over every row, an O(1) error that
    # showed up only through the lam*m term (hence only when lam, rho_b and k_lang
    # are all non-zero) and never shrank with the grid.
    m = th * C + rb * langmuir(C, sm, kl)
    return -(th + rb * dsorb) * C + (-th * D * lap) + lam * m


def manufactured_solution_check(N: int = 64, dt: float = 0.02, steps: int = 20,
                                a: float = 0.4, D: float = 0.5,
                                refine_time: bool = True) -> dict:
    """20 steps at N/2, N, 2N; returns the fitted order and the mass residual.

    `refine_time` scales dt with dx^2 so the time error falls at the same rate as
    the space error. With dt pinned, the header's literal "20 steps" makes the
    explicit-Euler temporal error a fixed floor (~7.4e-3 here) and the measured
    order collapses to 0.00 -- the space error is already below it at N=32. That
    is a property of the experiment, not of the scheme, so both are reported.
    """
    grids = [N // 2, N, 2 * N]
    k = 2.0 * np.pi / L_DOMAIN
    th, rb = np.array([PRM["theta"]]), np.array([PRM["rho_b"]])
    sm, kl, lam = np.array([PRM["smax"]]), np.array([PRM["k_lang"]]), np.array([PRM["lam"]])
    t_final = dt * steps

    errs, dxs, r_ref, rho_ref = [], [], 0.0, 0.0
    for n in grids:
        dx = L_DOMAIN / n
        dt_n = dt * (dx / (L_DOMAIN / N)) ** 2 if refine_time else dt
        steps_n = max(int(round(t_final / dt_n)), 1)
        dt_n = t_final / steps_n
        x = (np.arange(n, dtype=np.float64) + 0.5) * dx
        X, Y = np.meshgrid(x, x)
        C = _mms_exact(X, Y, 0.0, a, k)
        Dfield = np.stack([np.full((1, n, n), D), np.zeros((1, n, n)),
                           np.full((1, n, n), D)], axis=1)
        zero = np.zeros((1, n, n))
        traj, srcs = [C.copy()], []
        for j in range(steps_n):
            # The source must be L[C*] at the *exact* solution, never at the
            # numerical one: feeding the numerical state back into the source
            # makes the problem define itself and the fitted order meaningless.
            src = _mms_source(_mms_exact(X, Y, j * dt_n, a, k), j * dt_n, a, k, D, PRM)
            srcs.append(src.copy())
            C = fvm_step(C[None], zero, zero, Dfield, th, rb, sm, kl, lam,
                         src[None], dt_n, dx=dx)[0]
            traj.append(C.copy())
        exact = _mms_exact(X, Y, t_final, a, k)
        errs.append(float(np.linalg.norm(C - exact) / np.linalg.norm(exact)))
        dxs.append(dx)
        if n == grids[-1]:
            r_ref = mass_balance_series(np.stack(traj)[None], np.stack(srcs)[None],
                                        th, rb, sm, kl, lam, dt_n, dx=dx)
            rho_ref = float(np.nanmax(metrics.rho_series(np.stack(traj), 2)))

    order = float(np.log(errs[-2] / errs[-1]) / np.log(dxs[-2] / dxs[-1]))
    fit = float(np.polyfit(np.log(dxs), np.log(errs), 1)[0])
    return {"order": order, "order_fit": fit, "err_l2": errs[-1], "grids": grids,
            "errs": errs, "r_ref_max": float(r_ref), "rho_ref_max": rho_ref,
            "dt": dt, "steps": steps, "t_final": t_final,
            "refine_time": refine_time, "D": D,
            "passed": bool(order >= 0.9 and r_ref <= 1e-13)}


def analytic_width_check(D: float = 2.0, t_final: float = 4.0, n: int = 8,
                         N: int = 64, band_half: float = 12.0,
                         smooth: float = 0.25) -> dict:
    """Diffuse a square patch; compare `front_width` with 3.6248 sqrt(Dt).

    Geometry, not physics, decides whether this gate can work:
    - A 1-D step gives `front_width` exactly one valid ray. `+x` walks down from the
      peak, `-x` walks *up* from the level-0.5 midpoint and never crosses below 0.1,
      so with `n_rays=4` the median is over too few rays and returns NaN.
    - A vertical *band* fixes the ray count but corrupts the number: the diagonal
      rays cross a front that varies only in x, so they measure w/cos(45) = 1.414 w.
      Measured on an exact erfc profile with no solver at all, a band gives
      ratio 2.39 instead of 1.0 -- enough to fail the gate on geometry alone.
    A square patch with `n_rays=2` measures +x and +y, both through an edge
    midpoint, where the local solution is the half-space erfc the constant assumes.

    Sorption is switched **off** here. With it on, the front moves under the
    retarded diffusivity D_eff = theta D / (theta + rho_b S'(C)), which at the
    training bands is D/5.2; comparing against the bare-D erfc width then reports
    ratio 0.43 and the gate fails for a physical reason, not a broken measurement.
    The gate exists to invert the *measurement* on a known profile, so the profile
    is made a pure diffusion problem with an exact closed form.
    """
    dx = L_DOMAIN / N
    x = (np.arange(N, dtype=np.float64) + 0.5) * dx
    X, Y = np.meshgrid(x, x)
    r = np.maximum(np.abs(X - L_DOMAIN / 2.0), np.abs(Y - L_DOMAIN / 2.0))
    C = 0.5 * (1.0 - np.tanh((r - band_half) / smooth))

    th, rb = np.array([1.0]), np.array([0.0])
    sm, kl, lam = np.array([PRM["smax"]]), np.array([PRM["k_lang"]]), np.array([0.0])
    Dfield = np.stack([np.full((1, N, N), D), np.zeros((1, N, N)),
                       np.full((1, N, N), D)], axis=1)
    zero = np.zeros((1, N, N))

    dt = 0.2 * dx ** 2 / D                     # the generator's own diffusive CFL
    n_steps = int(round(t_final / dt))
    dt = t_final / n_steps
    sample = set(np.unique(np.linspace(1, n_steps, n).round().astype(int)).tolist())

    origin = (L_DOMAIN / 2.0 / dx - 0.5, L_DOMAIN / 2.0 / dx - 0.5)
    rows = []
    for j in range(1, n_steps + 1):
        C = fvm_step(C[None], zero, zero, Dfield, th, rb, sm, kl, lam,
                     np.zeros((1, N, N)), dt, dx=dx)[0]
        if j in sample:
            w, nv = metrics.front_width(C, origin, n_rays=2)
            t = j * dt
            rows.append({"t": float(t), "w": float(w),
                         "analytic": float(metrics.analytic_width(t, D)),
                         "valid_rays": int(nv)})

    ok = [r for r in rows if np.isfinite(r["w"]) and r["analytic"] > 0]
    if not ok:
        return {"ratio": float("nan"), "bias": float("nan"), "slope": float("nan"),
                "rows": rows, "passed": False, "reason": "no measurable width"}
    ratio = np.array([r["w"] / r["analytic"] for r in ok])
    sq = np.sqrt(np.array([D * r["t"] for r in ok]))
    slope = float(np.polyfit(sq, np.array([r["w"] for r in ok]), 1)[0])
    med = float(np.median(ratio))
    # single source of truth for the constant: ask metrics, never re-type 3.6248
    target = metrics.analytic_width(1.0, 1.0)
    return {"ratio": med, "bias": float(np.mean(ratio) - 1.0), "slope": slope,
            "slope_target": target, "ratio_all": ratio.tolist(), "rows": rows,
            "passed": bool(0.85 <= med <= 1.15 and abs(slope / target - 1.0) <= 0.10)}


def main(argv=None) -> int:
    out_dir = os.path.join(ROOT, "results", "calibration")
    os.makedirs(out_dir, exist_ok=True)

    msg = manufactured_solution_check()
    wid = analytic_width_check()

    report = {"manufactured": msg, "width": wid,
              "environment": {"python": platform.python_version(),
                              "numpy": np.__version__, "platform": platform.platform()},
              "passed": bool(msg["passed"] and wid["passed"])}
    path = os.path.join(out_dir, "solver_check.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)

    print(f"[selfcheck] order={msg['order']:.3f} (fit {msg['order_fit']:.3f}) "
          f"err_l2={msg['err_l2']:.3e} r_ref_max={msg['r_ref_max']:.3e} "
          f"rho_ref_max={msg['rho_ref_max']:.3e} -> {'PASS' if msg['passed'] else 'FAIL'}")
    print(f"[selfcheck] width ratio={wid['ratio']:.4f} bias={wid['bias']:+.4f} "
          f"slope={wid['slope']:.4f} (target {wid['slope_target']:.4f}) -> "
          f"{'PASS' if wid['passed'] else 'FAIL'}")
    for r in wid["rows"]:
        print(f"           t={r['t']:.3f} w={r['w']:.4f} analytic={r['analytic']:.4f} "
              f"rays={r['valid_rays']}")
    print(f"[selfcheck] wrote {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
