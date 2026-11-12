"""How much would a heterogeneous conductivity change the archived benchmark?

The shipped trajectories carry a spatially uniform conductivity, because the
spectral synthesis in ``gen_log_normal_field`` applies its correlation kernel
with the wavenumber scaled to cycles-per-domain but the length scaled in metres,
so every non-zero wavenumber is annihilated and the field collapses to its mean.
``corr_len`` and ``k_var`` are still sampled for every trajectory, so the intent
was clearly a log-normal K.

This probe regenerates the intended field (kernel divided by L^2) for a single
parameter set and compares the resulting plume against the archived neutral case
at the same grid, so the cost of *not* fixing the data can be quantified.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "data"))
sys.path.insert(0, _HERE)
import baseline_2d_schemes as b  # noqa: E402
import gen_adr2d as g  # noqa: E402


def log_normal_fixed(rng, N, corr_len, k_var, domain=g.DOMAIN):
    """Unit-mean log-normal K with the kernel scaled by the domain length."""
    kx = np.fft.fftfreq(N, d=1.0 / N)[None, :]
    ky = np.fft.fftfreq(N, d=1.0 / N)[:, None]
    filt = np.exp(-np.pi ** 2 * corr_len ** 2 * (kx ** 2 + ky ** 2) / (2.0 * domain ** 2))
    white = rng.standard_normal((N, N))
    f = np.real(np.fft.ifft2(np.fft.fft2(white) * filt))
    f = (f - f.mean()) / (f.std() + 1e-12)
    K = np.exp(k_var * f - 0.5 * k_var ** 2)
    return K / K.mean()


def run(K, prm, N, t_end=20.0):
    dt, T = b._dt_for(K, prm, 0, N, "weno5", 0.1, 1.0, t_end)
    r = b.run_case(K, prm, 0, N, T, dt, "weno5", "rk3")
    dx = g.DOMAIN / N
    mom = b._moments(r["C"], dx)
    return {
        "peak_C": float(r["C"].max()),
        "core_cells": int(b._core_cells(r["C"])),
        "centroid_x": float(mom[0]), "centroid_y": float(mom[1]),
        "rms_radius": float(mom[2]),
        "mass_resid": float(r["mass_resid"]),
        "C": r["C"],
    }


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="heterogeneity impact probe")
    ap.add_argument("--n", type=int, default=3, help="parameter sets")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--out", default="results/verification/heterogeneity_probe.json")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(0)
    prm = g.sample_parameters(rng, args.n, "train")
    rows = []
    for i in range(args.n):
        K = log_normal_fixed(rng, args.N, float(prm["corr_len"][i]),
                             float(prm["k_var"][i]))[None]
        Ku = np.ones((1, args.N, args.N))
        qh, qv, _ = g.darcy_velocity(K, prm["gx"][i:i + 1], prm["gy"][i:i + 1],
                                     dx=g.DOMAIN / args.N)
        qm = np.hypot(qh, qv)
        rh, ru = run(K, prm, args.N), run(Ku, prm, args.N)
        Ch, Cu = rh.pop("C"), ru.pop("C")
        rows.append({
            "corr_len": float(prm["corr_len"][i]), "k_var": float(prm["k_var"][i]),
            "K_std_over_mean": float(K.std() / K.mean()),
            "K_min": float(K.min()), "K_max": float(K.max()),
            "q_std_over_mean": float(qm.std() / qm.mean()),
            "hetero": rh,
            "uniform": ru,
            "rel_l2_hetero_vs_uniform": float(np.linalg.norm(Ch - Cu)
                                              / np.linalg.norm(Cu)),
        })
        h, u = rows[-1]["hetero"], rows[-1]["uniform"]
        print(f"[{i}] ell={rows[-1]['corr_len']:.1f} s={rows[-1]['k_var']:.2f} "
              f"K={rows[-1]['K_min']:.2f}..{rows[-1]['K_max']:.2f} "
              f"|q|spread={rows[-1]['q_std_over_mean']:.2f}")
        print(f"    peak {h['peak_C']:.4f} vs {u['peak_C']:.4f}   "
              f"core {h['core_cells']} vs {u['core_cells']}   "
              f"rms {h['rms_radius']:.2f} vs {u['rms_radius']:.2f}   "
              f"mass {h['mass_resid']:.1e}/{u['mass_resid']:.1e}")
        print(f"    centroid ({h['centroid_x']:.1f},{h['centroid_y']:.1f}) vs "
              f"({u['centroid_x']:.1f},{u['centroid_y']:.1f})   "
              f"rel L2 diff {rows[-1]['rel_l2_hetero_vs_uniform']:.2f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"N": args.N, "rows": rows}, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
