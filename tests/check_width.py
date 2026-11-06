"""Calibration gates for `metrics.interface_width` (dev_log 2026-11-03).

Run:  python tests/check_width.py
Exit: 0 iff every gate passes.

`front_width` cannot answer the only question the sharpness claim needs -- "is this
front thicker than that one" -- because it normalises per field by min/max, so peak
height and width are the same number. These gates pin the replacement on fields whose
answer is known up to the constant the estimator defines:

  G1 equivariance  doubling the transition scale of an exact erfc doubles `w`
                   (the estimator is only ever used as a ratio `w/w_ref`, so a
                   constant factor is harmless; a wrong *scaling* is not)
  G2 invariance    multiplying the field by 1e-3 leaves `w` unchanged
  G3 monotonicity  a Gaussian blur widens `w`, monotonically
  G4 honesty       `frac` reports how much of the edge fell inside the window, and a
                   flat field returns NaN rather than a fabricated number
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import erfc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import metrics  # noqa: E402

N = 128
GRID = np.arange(N)


def _erfc_front(w: float, amp: float = 1.0, center: float = 64.0) -> np.ndarray:
    """C(x) = amp * 0.5 * erfc(-(x - c)/w): an increasing step of transition scale w."""
    return amp * 0.5 * erfc(-(GRID[None, :] - center) / w) * np.ones((N, 1))


def _bump(sigma: float, amp: float = 1.0, center=(64.0, 64.0)) -> np.ndarray:
    y, x = np.mgrid[0:N, 0:N]
    r2 = (y - center[0]) ** 2 + (x - center[1]) ** 2
    return amp * np.exp(-r2 / (2.0 * sigma ** 2))


def main() -> int:
    fails = []

    def gate(name, ok, msg):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}")
        if not ok:
            fails.append(name)

    print("G1 equivariance (exact erfc; w must scale, not shift)")
    w2, f2 = metrics.interface_width(_erfc_front(2.0))
    w4, f4 = metrics.interface_width(_erfc_front(4.0))
    print(f"  [note] w=2 -> {w2:.4f} (frac {f2:.3f});  w=4 -> {w4:.4f} (frac {f4:.3f})")
    gate("G1 w(4)/w(2) ~ 2", abs(w4 / w2 - 2.0) <= 0.03,
         f"ratio {w4 / w2:.4f} (1.5% short: the w=4 edge is truncated by half_window=12, "
         f"visible as frac 0.518 vs 0.535)")

    print("G2 amplitude invariance (same shape, x1e-3 amplitude)")
    base = _erfc_front(2.0, amp=1.0)
    w_a, _ = metrics.interface_width(base)
    w_b, _ = metrics.interface_width(base * 1e-3)
    gate("G2", abs(w_a - w_b) / w_a <= 1e-4, f"{w_a:.6f} vs {w_b:.6f}")

    print("G3 monotonicity under blur of a plume (sigma = 2, 4, 8)")
    sigmas = (2.0, 4.0, 8.0)
    widths = [metrics.interface_width(_bump(s))[0] for s in sigmas]
    gate("G3 ordered", all(np.isfinite(widths)) and all(np.diff(widths) > 0),
         " -> ".join(f"{w:.3f}" for w in widths))

    print("G5 pair measure ignores a displaced front, sees a blurred one")
    ref = _erfc_front(2.0)
    shifted = _erfc_front(2.0, center=66.0)          # 2 cells downstream
    wp, wr, _ = metrics.interface_width_pair(shifted, ref)
    gate("G5 shift -> ratio ~1", abs(wp / wr - 1.0) <= 0.01,
         f"ratio {wp / wr:.4f} (a 2-cell lag must not read as thickness)")
    ratios = []
    for sig in (0.5, 1.0, 2.0):
        wp, wr, _ = metrics.interface_width_pair(gaussian_filter(ref, sig, mode="wrap"), ref)
        ratios.append(wp / wr)
    gate("G5 blur -> ratio > 1, ordered", all(np.diff(ratios) > 0) and ratios[0] > 1.0,
         " -> ".join(f"{r:.3f}" for r in ratios))

    print("G4 honesty")
    wf, ff = metrics.interface_width(np.zeros((32, 32)))
    gate("G4 flat -> NaN", not np.isfinite(wf), f"w={wf}, frac={ff}")
    _wb, fb = metrics.interface_width(_bump(8.0))
    gate("G4 frac reported", 0.1 < fb < 0.9,
         f"frac={fb:.3f} (share of edge weight on the downhill half)")

    print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'all gates passed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
