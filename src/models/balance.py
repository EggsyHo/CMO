"""Sorption law, composite mass and the analytic source/decay step (§3.7).

One implementation of the storage law, imported by the generator, the model and
the audit, so those three cannot disagree about what "mass" means.
"""
from __future__ import annotations

import torch

EPS = 1e-30


def langmuir(C: torch.Tensor, smax: torch.Tensor, k_lang: torch.Tensor) -> torch.Tensor:
    """Sorbed concentration S = smax * K C / (1 + K C)  (mg/kg)."""
    kc = k_lang * C.clamp_min(0.0)
    return smax * kc / (1.0 + kc)


def langmuir_derivative(C: torch.Tensor, smax: torch.Tensor, k_lang: torch.Tensor) -> torch.Tensor:
    """dS/dC = smax * K / (1 + K C)^2  (dimensionless)."""
    kc = k_lang * C.clamp_min(0.0)
    return smax * k_lang / (1.0 + kc) ** 2


def composite_mass(C: torch.Tensor, theta, rho_b, smax, k_lang) -> torch.Tensor:
    """m = theta * C + rho_b * S(C).  This is the quantity the audit conserves."""
    return theta * C + rho_b * langmuir(C, smax, k_lang)


def langmuir_inverse(m_c: torch.Tensor, theta, rho_b, smax, k_lang,
                     clamp: bool = True) -> tuple[torch.Tensor, int]:
    """Invert m = theta C + rho_b S(C) for C >= 0.

    theta K C^2 + (theta + a - m K) C - m = 0,  a = rho_b smax K.
    Returns (C, n_clamped); n_clamped counts samples where the analytic root was
    negative (physical overshoot) and had to be clipped to zero.
    """
    theta, rho_b, smax, k_lang = (
        torch.as_tensor(v, dtype=m_c.dtype, device=m_c.device)
        for v in (theta, rho_b, smax, k_lang)
    )
    a = rho_b * smax * k_lang
    thK = theta * k_lang
    b = theta + a - m_c * k_lang
    disc = (b * b + 4.0 * thK * m_c).clamp_min(0.0)
    root = torch.sqrt(disc)
    # Both algebraic forms are exact; each is unstable on one side of b = 0.
    #   b <= 0 (m well above the sorption capacity): (-b + root)/(2 theta K) adds
    #     two positive numbers -- use it.
    #   b >  0 (weak sorption / b >> root): (-b + root) cancels; 2m/(b + root)
    #     does not, and stays defined at K = 0 where it returns m/theta.
    C_hi = (-b + root) / (2.0 * thK).clamp_min(EPS)
    C_lo = 2.0 * m_c / (b + root).clamp_min(EPS)
    C = torch.where(b <= 0.0, C_hi, C_lo)
    n_clamped = 0
    if clamp:
        # a relative tolerance, so the -1e-30 that float summation leaves behind
        # is not reported as a physical overshoot; the clip itself is harmless
        tol = 1e-12 * C.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        bad = C < -tol
        n_clamped = int(bad.sum().item())
        C = C.clamp_min(0.0)
    return C, n_clamped


def total_mass(C: torch.Tensor, theta, rho_b, smax, k_lang, dx: float) -> torch.Tensor:
    """Composite mass per sample, [B]."""
    m = composite_mass(C, theta, rho_b, smax, k_lang)
    return m.sum(dim=(-2, -1)) * (dx ** 2)


def aqueous_mass(C: torch.Tensor, theta, dx: float) -> torch.Tensor:
    """theta * C only — the *negative control* budget of Part 3 §2.2."""
    return (theta * C).sum(dim=(-2, -1)) * (dx ** 2)


def analytic_step(C: torch.Tensor, flux_div: torch.Tensor, src: torch.Tensor, lam,
                  theta, rho_b, smax, k_lang, dt: float) -> tuple[torch.Tensor, int]:
    """m^{n+1} = m^n - dt*div(F) + dt*(s - lam*m^n), then C = S^{-1}(m).

    The decay is lagged (explicit), deliberately: the audit must test the same
    discretisation the model applies, not a better one.
    """
    m = composite_mass(C, theta, rho_b, smax, k_lang)
    m_new = m - dt * flux_div + dt * (src - lam * m)
    C_new, n_clamped = langmuir_inverse(m_new, theta, rho_b, smax, k_lang)
    return C_new, n_clamped
