"""ADR-2D reference data generator: vectorised finite-volume solver (§3.1).

numpy + scipy only — no torch, so the generator runs without the model package.  The
medium is "uniform" by default (K == 1, the archived benchmark); "--conductivity
lognormal" draws a unit-mean correlated field per trajectory and solves the Darcy head
with a divergence-free projection built from the same difference operator as the update.

Governing equation (periodic on [0,L]^2, dx = 1 m):

    d/dt[theta C + rho_b S(C)] + div(q C - theta D grad C) = s - lam [theta C + rho_b S(C)]
    S(C) = smax K_L C / (1 + K_L C)

Unit system (self-consistent, so the sorbed phase is comparable to the aqueous one):
    C [mg/L], S [mg/kg], rho_b [kg/L], theta [-], K_L [L/mg], lam [1/s].

Dimensionless controls sampled per trajectory (the design's bands, expressed in
quantities that stay observable at L = 64 m):
    retardation   R0 = 1 + rho_b smax K_L / theta   train [1.5, 8]
    cell Peclet   Pe = dx / alpha_T                 train [2, 15]
    Damkohler     Da = lam L theta / |q|            train [0, 3]
    anisotropy    kappa = alpha_L / alpha_T          train [1, 12]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import h5py
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

DX = 1.0
DOMAIN = 64.0
GENERATOR_VERSION = "1.0"

BANDS = {
    "train": dict(q=(0.05, 0.35), pe=(2.0, 15.0), r0=(1.5, 8.0), da=(0.0, 3.0),
                  kappa=(1.0, 12.0), theta=(0.25, 0.40), rho_b=(1.2, 1.8), smax=(1.0, 10.0)),
    "ext": dict(q=(0.40, 0.60), pe=(1.0, 1.8), r0=(9.0, 12.0), da=(3.5, 5.0),
                kappa=(14.0, 20.0), theta=(0.25, 0.40), rho_b=(1.2, 1.8), smax=(1.0, 10.0)),
    "kappa1": dict(q=(0.18, 0.22), pe=(2.0, 15.0), r0=(4.0, 5.0), da=(1.4, 1.6),
                   kappa=(1.0, 1.0), theta=(0.30, 0.34), rho_b=(1.4, 1.6), smax=(4.0, 6.0)),
    "kappa20": dict(q=(0.18, 0.22), pe=(2.0, 15.0), r0=(4.0, 5.0), da=(1.4, 1.6),
                    kappa=(20.0, 20.0), theta=(0.30, 0.34), rho_b=(1.4, 1.6), smax=(4.0, 6.0)),
}

PARAM_COLUMNS = ("Pe", "Da", "kappa", "theta", "rho_b", "smax", "k_lang", "lam", "dt")


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
def sample_parameters(rng: np.random.Generator, n: int, regime: str) -> dict:
    if regime not in BANDS:
        raise KeyError(f"unknown regime '{regime}', expected one of {sorted(BANDS)}")
    b = BANDS[regime]

    def u(key):
        lo, hi = b[key]
        return rng.uniform(lo, hi, n) if hi > lo else np.full(n, lo)

    q_mag = u("q")
    pe_target = u("pe")
    r0 = u("r0")
    da = u("da")
    kappa = u("kappa")
    theta = u("theta")
    rho_b = u("rho_b")
    smax = u("smax")

    alpha_t = DX / pe_target                       # Pe = dx / alpha_T
    alpha_l = kappa * alpha_t
    q_ang = rng.uniform(0.0, 2 * np.pi, n)
    lam = da * q_mag / (theta * DOMAIN)            # Da = lam L theta / |q|
    k_lang = (r0 - 1.0) * theta / np.maximum(rho_b * smax, 1e-9)

    return dict(
        q_mag=q_mag, q_ang=q_ang,
        gx=q_mag * np.cos(q_ang), gy=q_mag * np.sin(q_ang),
        Pe=pe_target, Da=da, kappa=kappa, lam=lam,
        alpha_l=alpha_l, alpha_t=alpha_t, theta=theta, rho_b=rho_b,
        smax=smax, k_lang=k_lang, R0=r0,
        sigma=rng.uniform(1.5, 3.0, n),
        q_src=rng.uniform(0.5, 2.0, n),
        t_ramp=rng.uniform(0.5, 1.5, n),
        x0=rng.uniform(16.0, 48.0, n), y0=rng.uniform(16.0, 48.0, n),
        corr_len=rng.uniform(5.0, 20.0, n),
        k_var=rng.uniform(0.3, 1.0, n),
    )


def parse_regimes(regime, n) -> tuple[list[str], list[int]]:
    """Accept `--regime kappa1,kappa20 --n 40,40` (or a single regime with one n).

    Needed because §3.3 puts both kappa families in **one** file (`adr2d_kappa.h5`,
    2x40 trajectories). Two separate CLI calls would each open the file in "w"
    mode and the second would silently delete the first 40 trajectories.
    """
    regimes = ([r.strip() for r in regime.split(",")] if isinstance(regime, str)
               else [str(r) for r in regime])
    if not regimes or any(r not in BANDS for r in regimes):
        raise KeyError(f"unknown regime(s) {regimes}, expected a subset of {sorted(BANDS)}")
    if isinstance(n, (list, tuple, np.ndarray)):
        counts = [int(x) for x in n]
    else:
        counts = [int(n) // len(regimes)] * len(regimes)
        counts[-1] += int(n) - sum(counts)
    if len(counts) != len(regimes) or min(counts) <= 0:
        raise ValueError(f"regimes {regimes} and counts {counts} do not line up")
    return regimes, counts


def sample_parameters_multi(rng: np.random.Generator, regimes: list[str],
                           counts: list[int]) -> dict:
    """Concatenate per-regime parameter tables; one rng stream, so the seed still
    reproduces the whole file."""
    parts = [sample_parameters(rng, c, r) for r, c in zip(regimes, counts)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# --------------------------------------------------------------------------- #
# random fields and Darcy velocity
# --------------------------------------------------------------------------- #
def gen_log_normal_field(rng: np.random.Generator, n: int, N: int, corr_len: np.ndarray,
                         k_var: np.ndarray | None = None) -> np.ndarray:
    """[n,N,N] unit-mean log-normal field by spectral synthesis.

    ``corr_len`` is a physical length in metres and must therefore be divided by
    the domain size before it multiplies an integer wavenumber: the Gaussian
    correlation exp(-r^2/2ell^2) has spectrum exp(-2 pi^2 ell^2 k^2/L^2).  Omitting
    the /L^2 makes the filter's width of order one grid cell (every k>0 is damped
    to zero), which returns a constant field rather than a random one.
    """
    kx = np.fft.fftfreq(N, d=1.0 / N)[None, :]
    ky = np.fft.fftfreq(N, d=1.0 / N)[:, None]
    k2 = kx ** 2 + ky ** 2
    out = np.empty((n, N, N), dtype=np.float64)
    sigma_ln = np.asarray(k_var if k_var is not None else np.full(n, 0.5), dtype=np.float64)
    for i in range(n):
        ell = float(corr_len[i])
        filt = np.exp(-2.0 * np.pi ** 2 * ell ** 2 * k2 / DOMAIN ** 2)
        white = rng.standard_normal((N, N))
        field = np.real(np.fft.ifft2(np.fft.fft2(white) * filt))
        field = (field - field.mean()) / (field.std() + 1e-12)
        out[i] = np.exp(sigma_ln[i] * field - 0.5 * sigma_ln[i] ** 2)
    out = out / out.mean(axis=(1, 2), keepdims=True)
    return np.clip(out, 0.1, 10.0)


def _fd_div(qx: np.ndarray, qy: np.ndarray, dx: float = DX) -> np.ndarray:
    """Backward-difference divergence (the operator the FVM update telescopes with)."""
    return (qx - np.roll(qx, 1, axis=-1)) / dx + (qy - np.roll(qy, 1, axis=-2)) / dx


def _fd_grad(phi: np.ndarray, dx: float = DX) -> tuple[np.ndarray, np.ndarray]:
    """Forward-difference gradient: div_fd(grad_fd phi) == 5-point Laplacian."""
    gx = (np.roll(phi, -1, axis=-1) - phi) / dx
    gy = (np.roll(phi, -1, axis=-2) - phi) / dx
    return gx, gy


def _solve_poisson_fd(rhs: np.ndarray, dx: float = DX) -> np.ndarray:
    """Solve the periodic 5-point Laplacian with the FFT eigenbasis (exact, discrete)."""
    n = rhs.shape[-1]
    kx = np.fft.fftfreq(n, d=1.0 / n)[None, :]
    ky = np.fft.fftfreq(n, d=1.0 / n)[:, None]
    lam = -4.0 * (np.sin(np.pi * kx / n) ** 2 + np.sin(np.pi * ky / n) ** 2) / dx ** 2
    lam[0, 0] = 1.0  # zero mode: arbitrary constant, rhs has zero mean
    r_hat = np.fft.fft2(rhs) / lam
    r_hat[..., 0, 0] = 0.0
    return np.real(np.fft.ifft2(r_hat))


def _solve_head_hetero(K: np.ndarray, rhs: np.ndarray, dx: float) -> np.ndarray:
    """Solve div(K grad phi) = rhs on the periodic cell, to CG tolerance.

    The operator is built from the *same* finite-difference pair as the update
    (backward divergence of a forward gradient), so the head it returns makes the
    flux divergence-free for the discretisation rather than for the continuum.
    A fixed-point/Picard iteration with a constant-coefficient preconditioner ---
    the obvious alternative --- is only conditionally convergent in the contrast,
    which is why the heterogeneous branch is solved rather than iterated.
    """
    n, N, _ = K.shape
    out = np.empty((n, N, N), dtype=np.float64)
    for i in range(n):
        Ki = K[i]

        def matvec(p, Ki=Ki):
            p2 = p.reshape(N, N)
            gxp = (np.roll(p2, -1, axis=1) - p2) / dx
            gyp = (np.roll(p2, -1, axis=0) - p2) / dx
            qx, qy = Ki * gxp, Ki * gyp
            return ((qx - np.roll(qx, 1, axis=1)) / dx
                    + (qy - np.roll(qy, 1, axis=0)) / dx).ravel()

        op = LinearOperator((N * N, N * N), matvec=matvec, dtype=np.float64)
        sol, info = cg(op, rhs[i].ravel(), rtol=1e-13, atol=0.0, maxiter=4000)
        if info != 0:
            raise RuntimeError(f"variable-coefficient head solve did not converge ({info})")
        out[i] = sol.reshape(N, N)
    return out


def darcy_velocity(K: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                   n_iter: int = 60, dx: float = DX) -> tuple[np.ndarray, np.ndarray, float]:
    """Divergence-free Darcy velocity q = -K(grad h - g), then FD projection.

    Returns (qx, qy, max|div q| / (|q|/dx)). The projection step is what makes the
    stored velocity divergence-free *for the finite-difference operator the update
    uses*, which is the property the mass audit relies on. ``n_iter`` is retained
    for call compatibility and is unused: the head is now solved, not iterated.
    For K == const the projection is the identity (div q* == 0 to round-off), so
    the uniform branch returns q = K g exactly as before.
    """
    n, N, _ = K.shape
    gx = np.asarray(gx, dtype=np.float64).reshape(n, 1, 1)
    gy = np.asarray(gy, dtype=np.float64).reshape(n, 1, 1)

    qx = K * gx
    qy = K * gy
    div = _fd_div(qx, qy, dx)
    phi = _solve_head_hetero(K, div, dx)
    px, py = _fd_grad(phi, dx)
    qx, qy = qx - K * px, qy - K * py
    div = _fd_div(qx, qy, dx)
    scale = np.maximum(np.sqrt(qx ** 2 + qy ** 2).mean(axis=(1, 2)), 1e-12) / dx
    residual = float(np.max(np.abs(div)) / np.max(scale))
    return qx, qy, residual


def dispersion_tensor(qx, qy, theta, alpha_l, alpha_t, d_mol: float = 1e-6) -> np.ndarray:
    """[n,3,N,N] = (Dxx, Dxy, Dyy); D = aT|q|I + (aL-aT) q q^T/|q| + d_mol/theta I."""
    qm = np.maximum(np.sqrt(qx ** 2 + qy ** 2), 1e-12)
    aL = alpha_l.reshape(-1, 1, 1)
    aT = alpha_t.reshape(-1, 1, 1)
    th = theta.reshape(-1, 1, 1)
    dxx = aT * qm + (aL - aT) * qx * qx / qm + d_mol / th
    dxy = (aL - aT) * qx * qy / qm
    dyy = aT * qm + (aL - aT) * qy * qy / qm + d_mol / th
    return np.stack([dxx, dxy, dyy], axis=1)
# --------------------------------------------------------------------------- #
# sorption (numpy mirror of src/models/balance.py; cross-checked in selfcheck)
# --------------------------------------------------------------------------- #
def langmuir(C, smax, k_lang):
    kc = np.maximum(k_lang, 1e-12) * np.maximum(C, 0.0)
    while np.ndim(kc) < np.ndim(C):
        kc = kc[..., None]
    return smax * kc / (1.0 + kc)


def langmuir_inverse(m, theta, rho_b, smax, k_lang):
    """Closed-form non-negative root of theta K C^2 + b C - m = 0."""
    shape = np.shape(m)
    th = np.reshape(theta, (-1,) + (1,) * (m.ndim - 1)) if np.ndim(theta) == 1 else theta
    rb = np.reshape(rho_b, np.shape(th)) if np.ndim(rho_b) == 1 else rho_b
    sm = np.reshape(smax, np.shape(th)) if np.ndim(smax) == 1 else smax
    kl = np.reshape(k_lang, np.shape(th)) if np.ndim(k_lang) == 1 else k_lang
    a = rb * sm * kl
    b = th + a - m * kl
    disc = np.maximum(b * b + 4.0 * th * kl * m, 0.0)
    root = np.sqrt(disc)
    # Each algebraic form is unstable on one side of b = 0 -- switch at the sign
    # (Numerical Recipes' stable quadratic), which keeps float32 round-trips of
    # the inverse exact in both the weak-sorption and the saturated regime.
    with np.errstate(divide="ignore", invalid="ignore"):
        C_hi = (-b + root) / np.maximum(2.0 * th * kl, 1e-300)
        C_lo = 2.0 * m / np.maximum(b + root, 1e-300)
    C = np.where(b <= 0.0, C_hi, C_lo)
    C = np.where(np.abs(th * kl) < 1e-12, m / np.maximum(th, 1e-12), C)
    return np.maximum(C, 0.0).reshape(shape)


def injection_source(X, Y, x0, y0, sigma, Q, t_ramp, t):
    """Gaussian well, ramped as Q(1 - exp(-t/t_ramp)); [n,N,N]."""
    sx = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-6)[:, None, None]
    g = np.exp(-((X - np.asarray(x0)[:, None, None]) ** 2
                 + (Y - np.asarray(y0)[:, None, None]) ** 2) / (2.0 * sx ** 2))
    g = g / g.sum(axis=(-2, -1), keepdims=True)
    amp = (np.asarray(Q) * (1.0 - np.exp(-max(float(t), 0.0) / np.maximum(np.asarray(t_ramp), 1e-6))))
    return amp[:, None, None] * g


# --------------------------------------------------------------------------- #
# one conservative finite-volume step
# --------------------------------------------------------------------------- #
def _face_avg(a):
    return 0.5 * (a + np.roll(a, -1, axis=-1))


def _face_avg_y(a):
    return 0.5 * (a + np.roll(a, -1, axis=-2))


def fvm_step(C, qx, qy, D, theta, rho_b, smax, k_lang, lam, src, dt: float,
             dx: float | None = None) -> np.ndarray:
    """Advance the composite mass by one conservative step, return C^{n+1}.

    `dx` defaults to the module constant so every existing call is unchanged; the
    convergence study in `selfcheck_solver.py` passes `L/N` explicitly, which is
    the only way the same *physical* problem can be posed on three grids.
    """
    dx = DX if dx is None else float(dx)
    theta_b = theta[:, None, None]
    rho_b_b = rho_b[:, None, None]
    lam_b = lam[:, None, None]
    smax_b = smax[:, None, None]
    kl_b = k_lang[:, None, None]
    Dxx, Dxy, Dyy = D[:, 0], D[:, 1], D[:, 2]

    # --- advective fluxes (first-order upwind, donor cell) ---
    ux_f = 0.5 * (qx + np.roll(qx, -1, axis=-1))
    Cx_up = np.where(ux_f >= 0.0, C, np.roll(C, -1, axis=-1))
    Fx_adv = ux_f * Cx_up
    uy_f = 0.5 * (qy + np.roll(qy, -1, axis=-2))
    Cy_up = np.where(uy_f >= 0.0, C, np.roll(C, -1, axis=-2))
    Fy_adv = uy_f * Cy_up

    # --- dispersive fluxes (central differences at faces) ---
    dCdx = (np.roll(C, -1, axis=-1) - C) / dx
    dCdy_c = (np.roll(C, -1, axis=-2) - np.roll(C, 1, axis=-2)) / (2.0 * dx)
    dCdy_f = 0.5 * (dCdy_c + np.roll(dCdy_c, -1, axis=-1))
    Fx_disp = -theta_b * (_face_avg(Dxx) * dCdx + _face_avg(Dxy) * dCdy_f)

    dCdy = (np.roll(C, -1, axis=-2) - C) / dx
    dCdx_c = (np.roll(C, -1, axis=-1) - np.roll(C, 1, axis=-1)) / (2.0 * dx)
    dCdx_f = 0.5 * (dCdx_c + np.roll(dCdx_c, -1, axis=-2))
    Fy_disp = -theta_b * (_face_avg_y(Dxy) * dCdx_f + _face_avg_y(Dyy) * dCdy)

    div = ((Fx_adv + Fx_disp) - np.roll(Fx_adv + Fx_disp, 1, axis=-1)) / dx
    div += ((Fy_adv + Fy_disp) - np.roll(Fy_adv + Fy_disp, 1, axis=-2)) / dx

    m = theta_b * C + rho_b_b * langmuir(C, smax_b, kl_b)
    m_new = m - dt * div + dt * (src - lam_b * m)

    # Positivity limiter that may not lose mass. Central differences at a sharp
    # front with strong anisotropy push individual cells a few 1e-7 *below* zero;
    # `langmuir_inverse` then clamps C to 0, and that clipped mass is gone for
    # good. Measured on the training bands this accumulates ~1.7e-5 over 200 steps
    # on the worst trajectory (12.7% of cells go negative in a typical step), which
    # is 8 orders above the 1e-13 reference-solver gate of §3.2.
    #
    # Sign matters: zeroing a negative cell *raises* the total, so the deficit has
    # to be taken back from the surviving mass (1 - deficit/total), not handed to
    # it. Getting this backwards doubles the drift (3.46e-5, measured) instead of
    # removing it, because the total then moves by 2 * deficit.
    neg = m_new < 0.0
    if neg.any():
        deficit = np.where(neg, -m_new, 0.0).sum(axis=(-2, -1), keepdims=True)
        m_new = np.where(neg, 0.0, m_new)
        total = m_new.sum(axis=(-2, -1), keepdims=True)
        # zero division only for a member whose entire field went negative, which
        # cannot happen with s >= 0; that member keeps its (all-zero) field.
        spread = np.divide(deficit, total, out=np.zeros_like(total), where=total > 0.0)
        m_new = m_new * (1.0 - spread)

    return langmuir_inverse(m_new, theta, rho_b, smax, k_lang)


def check_cfl(qx, qy, D, dt: float, dx: float = DX) -> dict:
    adv = float(np.max(np.sqrt(qx ** 2 + qy ** 2)) * dt / DX)
    diff = float(np.max(D) * dt / DX ** 2)
    if adv > 0.4:
        raise RuntimeError(f"CFL (advection) violated: {adv:.3f} > 0.4; reduce dt by {adv / 0.4:.2f}x")
    if diff > 0.2:
        raise RuntimeError(f"CFL (dispersion) violated: {diff:.3f} > 0.2; reduce dt by {diff / 0.2:.2f}x")
    return {"cfl_adv": adv, "cfl_diff": diff}


def kahan_cumsum(a: np.ndarray) -> np.ndarray:
    """Compensated cumulative sum along axis 1.

    A plain `np.cumsum` over the source budget is a sequential sum, so it drifts
    like n*eps: 1250 steps give 1.4e-13, i.e. the audit itself would be the thing
    that fails the 1e-13 gate. `np.longdouble` is a trap here -- on Windows it is
    still 52-bit (eps 2.2e-16), so the compensated sum is the only portable fix.
    """
    out = np.empty_like(a, dtype=np.float64)
    s = np.zeros(a.shape[0])
    c = np.zeros(a.shape[0])
    for t in range(a.shape[1]):
        y = a[:, t] - c
        tot = s + y
        c = (tot - s) - y
        s = tot
        out[:, t] = s
    return out


def mass_balance_series(C_seq, src_seq, theta, rho_b, smax, k_lang, lam, dt: float,
                        dx: float | None = None) -> float:
    """max_k |M_k - M_0 - sum_{j<k} dt (S_j - lam M_j)| / M_0 for one trajectory."""
    dx = DX if dx is None else float(dx)
    n = np.shape(C_seq)[0]
    th = np.asarray(theta, dtype=np.float64).reshape(n, 1, 1, 1)
    rb = np.asarray(rho_b, dtype=np.float64).reshape(n, 1, 1, 1)
    sm = np.asarray(smax, dtype=np.float64).reshape(n, 1, 1, 1)
    kl = np.asarray(k_lang, dtype=np.float64).reshape(n, 1, 1, 1)
    lm = np.asarray(lam, dtype=np.float64).reshape(n, 1)
    m = th * C_seq + rb * langmuir(C_seq, sm, kl)
    M = m.sum(axis=(-2, -1)) * dx ** 2            # [n, T+1]
    inj = src_seq.sum(axis=(-2, -1)) * dx ** 2   # [n, T]
    expected = M[:, :1] + kahan_cumsum(dt * (inj - lm * M[:, :-1]))
    # Normalise by the trajectory peak, not by M_0: the continuation segment of
    # §3.3 starts at t = T, where decay has already cut the mass by e^-Da, so an
    # M_0 denominator reports 7.9e-8 for a solver whose absolute drift is 1e-13.
    scale = np.maximum(np.abs(M).max(axis=1, keepdims=True), 1e-12)
    resid = np.abs(M[:, 1:] - expected) / scale
    return float(resid.max())
# --------------------------------------------------------------------------- #
# trajectory blocks and HDF5 writing (streamed, so 400 x 100 x 64^2 never sits in RAM)
# --------------------------------------------------------------------------- #
def _grid(N: int, dx: float):
    x = (np.arange(N, dtype=np.float64) + 0.5) * dx
    X, Y = np.meshgrid(x, x)
    return X[None], Y[None]


def _run_block(prm: dict, idx: np.ndarray, N: int, T: int, dt: float,
               rng: np.random.Generator, background: float = 0.0,
               cond: str = "uniform") -> dict:
    """Integrate one block of trajectories; returns float32 fields + QC numbers.

    ``cond`` selects the conductivity of the medium: "uniform" (K == 1, the
    archived benchmark, every trajectory a stationary homogeneous medium) or
    "lognormal" (a unit-mean correlated field per trajectory, seeded by the same
    stream so the file stays reproducible).
    """
    nb = idx.size
    dx = DOMAIN / N
    sub = {k: (np.asarray(v)[idx] if isinstance(v, np.ndarray) and v.ndim == 1 else v)
           for k, v in prm.items()}
    if cond == "lognormal":
        K = gen_log_normal_field(rng, nb, N, sub["corr_len"], sub["k_var"])
    elif cond == "uniform":
        K = np.ones((nb, N, N), dtype=np.float64)
    else:
        raise ValueError(f"unknown conductivity mode {cond!r}")
    qx, qy, div_res = darcy_velocity(K, sub["gx"], sub["gy"], dx=dx)
    D = dispersion_tensor(qx, qy, sub["theta"], sub["alpha_l"], sub["alpha_t"])
    cfl = check_cfl(qx, qy, D, dt, dx)

    X, Y = _grid(N, dx)
    C = np.full((nb, N, N), background, dtype=np.float64)
    Cseq = np.empty((nb, T + 1, N, N), dtype=np.float32)
    Srces = np.empty((nb, T, N, N), dtype=np.float32)
    Cseq[:, 0] = C

    # float64 mass bookkeeping, accumulated in the loop (no float32 round-trip)
    th4 = sub["theta"].reshape(nb, 1, 1)
    rb4 = sub["rho_b"].reshape(nb, 1, 1)
    sm4 = sub["smax"].reshape(nb, 1, 1)
    kl4 = sub["k_lang"].reshape(nb, 1, 1)
    lm1 = sub["lam"].reshape(nb, 1)
    M_hist = np.empty((nb, T + 1), dtype=np.float64)
    M_hist[:, 0] = ((th4 * C + rb4 * langmuir(C, sm4, kl4)).sum(axis=(-2, -1))) * dx ** 2
    inj_hist = np.empty((nb, T), dtype=np.float64)

    for t in range(T):
        src = injection_source(X, Y, sub["x0"], sub["y0"], sub["sigma"], sub["q_src"],
                               sub["t_ramp"], t * dt)
        Srces[:, t] = src
        inj_hist[:, t] = src.sum(axis=(-2, -1)) * dx ** 2
        C = fvm_step(C, qx, qy, D, sub["theta"], sub["rho_b"], sub["smax"], sub["k_lang"],
                     sub["lam"], src, dt)
        Cseq[:, t + 1] = C
        M_hist[:, t + 1] = ((th4 * C + rb4 * langmuir(C, sm4, kl4)).sum(axis=(-2, -1))) * dx ** 2

    # see mass_balance_series: the audit accumulation must not drift like n*eps
    expected = M_hist[:, :1] + kahan_cumsum(dt * (inj_hist - lm1 * M_hist[:, :-1]))
    scale = np.maximum(np.abs(M_hist).max(axis=1, keepdims=True), 1e-12)
    per = (np.abs(M_hist[:, 1:] - expected) / scale).max(axis=1)
    qc = float(per.max())
    return {"C": Cseq, "src": Srces, "qx": qx.astype(np.float32), "qy": qy.astype(np.float32),
            "D": D.astype(np.float32), "params": sub, "qc_per": per,
            "qc": {**cfl, "div_res": div_res, "mass_resid": qc}}


def _write_traj(fh: h5py.File, name: str, blk: dict, prm: dict) -> None:
    g = fh.require_group(name)
    if "C" in g:
        del g["C"]
    g.create_dataset("C", data=blk["C"], compression="gzip", compression_opts=4)
    for key in ("src", "qx", "qy", "D"):
        if key in blk:
            if key in g:
                del g[key]
            g.create_dataset(key, data=blk[key], compression="gzip", compression_opts=4)
    for key in ("theta", "rho_b", "smax", "k_lang", "lam", "Pe", "Da", "kappa", "R0"):
        g.attrs[key] = float(prm[key])
    g.attrs["x0"] = float(prm["x0"])
    g.attrs["y0"] = float(prm["y0"])
    if "qc" in blk:
        for k, v in blk["qc"].items():
            g.attrs[f"qc_{k}"] = float(v)


def generate_to_file(path: str, regime: str, n: int, T: int, N: int = 64, dt: float = 0.2,
                     seed: int = 0, splits: dict | None = None, batch: int = 25,
                     long_n: int = 0, long_T: int = 0, ref128: int = 0,
                     verbose: bool = True, cond: str = "uniform") -> dict:
    rng = np.random.default_rng(seed)
    regimes, counts = parse_regimes(regime, n)
    n = int(sum(counts))
    prm = sample_parameters_multi(rng, regimes, counts)
    prm["dt"] = np.full(n, dt)
    if splits is None:
        ids = np.arange(n)
        splits = {"train": ids[:n], "val": np.array([], dtype=int), "test": np.array([], dtype=int)}
    meta = {"N": N, "T": T, "dt": dt, "dx": DOMAIN / N, "regime": "+".join(regimes), "seed": seed,
            "generator_version": GENERATOR_VERSION, "n_traj": n, "conductivity": cond,
            "regime_blocks": json.dumps(list(zip(regimes, counts))),
            "bands": json.dumps([{"regime": r, "n": c,
                                  "bands": {k: list(v) if isinstance(v, tuple) else v
                                            for k, v in BANDS[r].items()}}
                                 for r, c in zip(regimes, counts)])}

    ref_store: dict[int, tuple] = {}
    if ref128:
        sub_idx = np.arange(min(ref128, n))
        blk = _run_block(prm, sub_idx, 2 * N, T, dt, rng, cond=cond)
        for j, tid in enumerate(sub_idx):
            ref_store[int(tid)] = (blk["C"][j], {k: np.asarray(v)[tid] if isinstance(v, np.ndarray) and v.ndim == 1 else v
                                                 for k, v in prm.items()})

    t0 = time.time()
    worst = {"mass_resid": 0.0, "div_res": 0.0, "cfl_adv": 0.0, "cfl_diff": 0.0}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with h5py.File(path, "w") as fh:
        mgroup = fh.require_group("/meta")
        for key, val in meta.items():
            mgroup.attrs[key] = val
        arr = np.stack([np.asarray(prm[c], dtype=np.float64) for c in PARAM_COLUMNS], axis=1)
        fh.create_dataset("/params", data=arr)
        fh["/params"].attrs["columns"] = np.array(PARAM_COLUMNS, dtype="S8")
        for name, ids in splits.items():
            fh.create_dataset(f"/splits/{name}", data=np.asarray(ids, dtype=np.int32))

        for start in range(0, n, batch):
            idx = np.arange(start, min(start + batch, n))
            blk = _run_block(prm, idx, N, T, dt, rng, cond=cond)
            for k in worst:
                if k in blk["qc"]:
                    worst[k] = max(worst[k], blk["qc"][k])
            for j, tid in enumerate(idx):
                sub_prm = {k: (np.asarray(v)[tid] if isinstance(v, np.ndarray) and v.ndim == 1 else v)
                           for k, v in prm.items()}
                one = {k: blk[k][j] for k in ("C", "src", "qx", "qy", "D")}
                one["qc"] = blk["qc"]
                _write_traj(fh, f"traj_{tid:04d}", one, sub_prm)
            if verbose:
                print(f"  [{regime}] {min(start + batch, n)}/{n} traj, "
                      f"mass_resid={worst['mass_resid']:.2e}, {time.time() - t0:.1f}s", flush=True)

        if long_T:
            # §3.3 continues the same cases, so --long-n defaults to the whole file;
            # requiring it explicitly silently produced a 200-step file with no
            # continuation at all.
            idx = np.arange(min(int(long_n) or n, n))
            C_main = np.stack([fh[f"/traj_{tid:04d}/C"][:] for tid in idx], 0)      # [k, T+1]
            src_main = np.stack([fh[f"/traj_{tid:04d}/src"][:] for tid in idx], 0)  # [k, T]
            C0 = C_main[:, -1].astype(np.float64).copy()
            qx = np.stack([fh[f"/traj_{tid:04d}/qx"][:] for tid in idx], 0).astype(np.float64)
            qy = np.stack([fh[f"/traj_{tid:04d}/qy"][:] for tid in idx], 0).astype(np.float64)
            D = np.stack([fh[f"/traj_{tid:04d}/D"][:] for tid in idx], 0).astype(np.float64)
            sub = {k: np.asarray(v)[idx] for k, v in prm.items() if isinstance(v, np.ndarray) and v.ndim == 1}
            X, Y = _grid(N, DOMAIN / N)
            # float64 internally: auditing a float32 copy caps the residual at 6e-8
            # and hides what the solver does. Narrowed to float32 only on write.
            Cseq = np.empty((idx.size, long_T + 1, N, N), dtype=np.float64)
            Srces = np.empty((idx.size, long_T, N, N), dtype=np.float64)
            Cseq[:, 0] = C0
            for t in range(long_T):
                src = injection_source(X, Y, sub["x0"], sub["y0"], sub["sigma"], sub["q_src"],
                                       sub["t_ramp"], (T + t) * dt)
                Srces[:, t] = src
                C0 = fvm_step(C0, qx, qy, D, sub["theta"], sub["rho_b"], sub["smax"],
                              sub["k_lang"], sub["lam"], src, dt)
                Cseq[:, t + 1] = C0
            worst["long_mass_resid"] = mass_balance_series(
                Cseq, Srces, sub["theta"], sub["rho_b"], sub["smax"], sub["k_lang"], sub["lam"], dt)
            # Store the *whole* run. Writing the continuation under the same
            # `traj_NNNN` name used to overwrite the first T steps, so the file kept
            # only t > T and the initial condition was gone; §3.3 asks for "the same
            # 20 cases continued", not a detached 200-step segment.
            for j, tid in enumerate(idx):
                one = {"C": np.concatenate([C_main[j], Cseq[j, 1:]], axis=0).astype(np.float32),
                       "src": np.concatenate([src_main[j], Srces[j]], axis=0).astype(np.float32),
                       "qx": qx[j].astype(np.float32), "qy": qy[j].astype(np.float32),
                       "D": D[j].astype(np.float32),
                       "qc": {"mass_resid": float(worst["long_mass_resid"])}}
                sub_prm = {k: np.asarray(v)[tid] if isinstance(v, np.ndarray) and v.ndim == 1 else v
                           for k, v in prm.items()}
                _write_traj(fh, f"traj_{tid:04d}", one, sub_prm)
            fh["/meta"].attrs["T"] = int(T + long_T)

        for tid, (C128, sub_prm) in ref_store.items():
            g = fh.require_group(f"/ref128/traj_{tid:04d}")
            g.create_dataset("C", data=C128, compression="gzip", compression_opts=4)
    return {"path": path, "n": n, "T": T, "qc": worst, "wall_clock_s": time.time() - t0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="generate ADR-2D HDF5 reference data")
    ap.add_argument("--regime", default="train",
                    help="one regime, or a comma list (e.g. kappa1,kappa20) sharing one file")
    ap.add_argument("--n", default="400", help="trajectory count, or a comma list matching --regime")
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/adr2d/adr2d.h5")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--long-n", type=int, default=0)
    ap.add_argument("--long-T", type=int, default=0)
    ap.add_argument("--ref128", type=int, default=0)
    ap.add_argument("--conductivity", default="uniform", choices=("uniform", "lognormal"),
                    help="medium: 'uniform' (K==1, the archived benchmark) or "
                         "'lognormal' (unit-mean correlated field per trajectory)")
    args = ap.parse_args(argv)

    n_arg = [int(x) for x in str(args.n).split(",")] if "," in str(args.n) else int(args.n)
    regimes, counts = parse_regimes(args.regime, n_arg)
    n_tot = int(sum(counts))
    ids = np.arange(n_tot)
    if regimes == ["train"]:
        splits = {"train": ids[: max(int(0.7 * n_tot), 1)],
                  "val": ids[int(0.7 * n_tot): int(0.85 * n_tot)],
                  "test": ids[int(0.85 * n_tot):]}
    else:
        splits = {"train": np.array([], dtype=np.int32), "val": ids, "test": ids}
    info = generate_to_file(args.out, args.regime, n_arg, args.T, N=args.N, dt=args.dt,
                            seed=args.seed, splits=splits, batch=args.batch,
                            long_n=args.long_n, long_T=args.long_T, ref128=args.ref128,
                            cond=args.conductivity)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


