"""Fast unit checks for the operator's load-bearing identities (§3.6).

Run:  python tests/check_units.py [--h5 data/adr2d/adr2d.h5] [--steps 100]
Exit: 0 iff every gate passes.

Every check prints the measured number, not just PASS: these numbers are what the
paper quotes in its conservation table, so they must be reproducible by hand.

Three of them are the *whole* conservation argument and used to be printed without
being gated -- `sum(div) == 0`, `restrict` preserving mass, the Nyquist guard -- so
a regression in any of them would still have exited 0.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.balance import (analytic_step, composite_mass,  # noqa: E402
                                langmuir, langmuir_inverse, total_mass)
from src.models.blocks import (MassPartition, SpectralConv2d, detail_ratio,  # noqa: E402
                               divergence, positivity_limiter, zero_mean_project)
from src.models.cmo import CMO  # noqa: E402
from src.utils.config import load_config  # noqa: E402

_FAILED: list[str] = []


def gate(ok: bool, name: str, detail: str) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        _FAILED.append(name)
    return bool(ok)


def _rollout(batch: dict, model: CMO, steps: int, dx: float):
    """Returns (final state, max relative mass residual) for the analytic audit.

    The mass is summed in float64 even when the model runs in float32: adding 4096
    float32 cells costs ~sqrt(4096)*eps = 6e-6 on its own, which would be printed
    as if it were the model's drift (§3.6).
    """
    S = batch["s"].double().sum(dim=(-3, -2, -1)).flatten()
    lam = batch["lam"].double().flatten()
    sc = {k: batch[k] for k in ("theta", "rho_b", "smax", "k_lang")}
    sc64 = {k: v.double() for k, v in sc.items()}
    u = batch["u"].clone()
    M = total_mass(u.double(), **sc64, dx=dx).flatten()
    M0, cum = M.clone(), torch.zeros_like(M)
    clamps = 0
    for _ in range(steps):
        with torch.no_grad():
            u, diag = model(dict(batch, u=u), return_diag=True)
        clamps += diag["clamp_count"]
        cum = cum + float(batch["dt"]) * (S - lam * M)
        M = total_mass(u.double(), **sc64, dx=dx).flatten()
    return u, ((M - M0 - cum).abs() / M0).amax().item(), clamps


def check_projection(n: int = 64) -> None:
    """Pi must move no mass and leave no block mean."""
    part = MassPartition(n, 2)
    x = torch.rand(2, 1, n, n, dtype=torch.float64) + 1.0

    err = abs(float(part.restrict(x).sum()) * part.m ** 2 - float(x.sum())) / float(x.sum())
    gate(err < 1e-12, "restrict preserves mass", f"|sum(restrict x)*m^2 - sum(x)|/sum(x) = {err:.2e}")

    # avg_pool2d must be an exact cell average, else lift(restrict x) is not the
    # block average and the projection shifts the front's mean level quietly
    xr = x.reshape(2, 1, n // part.m, part.m, n // part.m, part.m).mean(dim=(3, 5))
    mean_err = float((part.restrict(x) - xr).abs().max()) / float(x.abs().max())
    gate(mean_err < 1e-15, "restrict == exact cell mean", f"max relative difference {mean_err:.2e}")

    proj_err = abs(float(part.forward(x).sum()) - float(x.sum())) / float(x.sum())
    gate(proj_err < 1e-12, "Pi = lift . restrict preserves mass",
         f"|sum(Pi x) - sum(x)|/sum(x) = {proj_err:.2e}")

    r = part.residual(x)
    blk = float(part.restrict(r).abs().amax()) / float(part.restrict(x).abs().amax())
    gate(blk < 1e-14, "residual has zero block mean", f"max|block mean| ratio = {blk:.2e}")
    gate(abs(float(r.sum())) / float(x.sum()) < 1e-12, "residual is zero-sum",
         f"|sum(x - Pi x)|/sum(x) = {abs(float(r.sum())) / float(x.sum()):.2e}")

    # an increment 1e3 times the state, adversarial signs: still zero block mean
    # and still zero-sum -- that is what lets the detail branch be mass-neutral
    d = torch.randn(2, 1, n, n, dtype=torch.float64) * 1e3
    try:
        dm = zero_mean_project(d, part, check=True)
        note = "internal 1e-6 check passed"
    except AssertionError as exc:  # pragma: no cover - the failure being gated
        dm, note = zero_mean_project(d, part), f"FAILED: {exc}"
    gate(not note.startswith("FAILED"), "zero_mean_project block mean < 1e-6", note)
    tot = abs(float(dm.sum())) / float(d.abs().sum())
    gate(tot < 1e-12, "projected increment is zero-sum", f"|sum(dm)|/sum|d| = {tot:.2e}")


def check_divergence(n_c: int = 32) -> None:
    """The telescope: every face appears twice with opposite signs."""
    F = torch.randn(2, 2, n_c, n_c, dtype=torch.float64)
    s = abs(float(divergence(F, 1.0).sum())) / float(F.abs().sum())
    gate(s < 1e-12, "sum(div F) == 0 (float64)", f"|sum(div F)|/sum|F| = {s:.2e}")

    # float32 is what the model runs in, so report its floor: torch's pairwise
    # reduction over M^2 cells, not a leak. The fp64 number is the claim.
    F32 = F.float()
    s32 = abs(float(divergence(F32, 1.0).sum())) / float(F32.abs().sum())
    gate(s32 < 1e-5, "sum(div F) at float32 stays at the summation floor",
         f"|sum(div F)|/sum|F| = {s32:.2e} on {n_c}x{n_c} (n*eps floor)")

    # non-vacuity: a spectral contraction alone is NOT conservative. If it were,
    # the zero above would be a property of the grid, not of the telescope.
    torch.manual_seed(0)
    layer = SpectralConv2d(2, 2, modes=max(n_c // 4, 1)).double()
    x = torch.randn(1, 2, n_c, n_c, dtype=torch.float64)
    with torch.no_grad():
        y = layer(x)
    leak = abs(float(y.sum()) - float(x.sum())) / float(x.abs().sum())
    gate(leak > 1e-3, "contrast: a bare spectral conv does not conserve",
         f"|sum(FNO x) - sum(x)|/sum|x| = {leak:.2e} (so the telescope is load-bearing)")


def check_guards(n: int = 64) -> None:
    """Both construction guards must raise, not silently degrade."""
    try:
        MassPartition(n, 3)
        gate(False, "MassPartition rejects a non-divisible grid", f"{n} % 3 != 0 did not raise")
    except ValueError as exc:
        gate(True, "MassPartition rejects a non-divisible grid", str(exc))

    limit = n // 2
    x = torch.randn(1, 2, n, n)
    try:
        with torch.no_grad():
            out = SpectralConv2d(2, 2, modes=limit)(x)
        gate(tuple(out.shape) == tuple(x.shape), "SpectralConv2d runs at the Nyquist limit",
             f"modes={limit} on {n}x{n} -> {tuple(out.shape)}")
    except Exception as exc:  # noqa: BLE001 - it must not raise
        gate(False, "SpectralConv2d runs at the Nyquist limit", f"raised {type(exc).__name__}: {exc}")

    try:
        with torch.no_grad():
            SpectralConv2d(2, 2, modes=limit + 1)(x)
        gate(False, "SpectralConv2d rejects modes > Nyquist",
             f"modes={limit + 1} on {n}x{n} did not raise")
    except ValueError as exc:
        gate("Nyquist" in str(exc), "SpectralConv2d rejects modes > Nyquist", str(exc))
    except Exception as exc:  # noqa: BLE001
        gate(False, "SpectralConv2d rejects modes > Nyquist",
             f"raised {type(exc).__name__} instead of ValueError: {exc}")


def check_detail_ratio(n: int = 64) -> None:
    """`detail_ratio` is the diagnostic every arm reports; check it against its
    own definition -- the energy share below the coarse grid."""
    x = torch.rand(2, 1, n, n, dtype=torch.float64)
    coarse = MassPartition(n, 2).forward(x)
    checker = ((torch.arange(n).view(1, 1, 1, n) + torch.arange(n).view(1, 1, n, 1)) % 2)
    r_coarse = detail_ratio(coarse, 2)
    r_check = detail_ratio(checker.double().expand(2, 1, n, n), 2)
    gate(r_coarse < 1e-12, "detail_ratio(Pi x) == 0", f"rho = {r_coarse:.2e} (block-constant field)")
    # a 2x2-aligned checkerboard has an exact analytic value: each block mean is
    # 0.5 and the residual is +-0.5, so rho = 0.25/0.5 = 0.5 exactly. Anything
    # else means Pi is not the block average it claims to be.
    gate(abs(r_check - 0.5) < 1e-12, "detail_ratio(checkerboard) == 0.5 (analytic)",
         f"rho = {r_check:.12f}")


def check_limiter(n: int = 64) -> None:
    """A flux-divergence increment must pass untouched (alpha = 1).

    `dm = c (m_{i-1} - m_i)` is what an upwind flux divergence produces: zero-sum,
    and non-negative wherever the cell is empty, so a conservative scheme never
    needs limiting. The guard exists for the learned flux, which is not upwind.
    """
    m = torch.rand(1, 1, n, n, dtype=torch.float64)
    m[0, 0, :, : n // 2] = 0.0                      # half the domain is empty
    dm = 0.3 * (torch.roll(m, 1, dims=-1) - m)
    out = positivity_limiter(dm, m)
    scale = float(dm.abs().sum()) + 1e-30
    gate(float((m + out).min()) >= 0.0, "limiter: upwind dm stays non-negative",
         f"min(m + out) = {float((m + out).min()):.2e}")
    gate(float((out - dm).abs().max()) == 0.0, "limiter: upwind dm passes untouched",
         f"max|out - dm| = {float((out - dm).abs().max()):.2e}")
    gate(abs(float(out.sum()) - float(dm.sum())) / scale < 1e-12, "limiter: upwind dm keeps its sum",
         f"|sum(out) - sum(dm)|/sum|dm| = {abs(float(out.sum()) - float(dm.sum())) / scale:.2e}")

    # the adversarial case: a zero-sum increment that tries to drain empty cells
    part = MassPartition(n, 2)
    dm2 = zero_mean_project(-0.3 * m, part)
    out2 = positivity_limiter(dm2, m)
    scale2 = float(dm2.abs().sum()) + 1e-30
    gate(float((m + out2).min()) >= 0.0, "limiter: hostile dm is made non-negative",
         f"min(m + out) = {float((m + out2).min()):.2e}")
    gate(abs(float(out2.sum()) - float(dm2.sum())) / scale2 < 1e-12, "limiter: hostile dm keeps its sum",
         f"|sum(out) - sum(dm)|/sum|dm| = {abs(float(out2.sum()) - float(dm2.sum())) / scale2:.2e}")


def check_inverse() -> None:
    for kl, tag in ((50.0, "strong"), (1e-6, "weak")):
        C = torch.rand(4, 1, 16, 16, dtype=torch.float64) * 3.0
        m = composite_mass(C, 0.3, 1.6, 1e-3, kl)
        back, _ = langmuir_inverse(m, 0.3, 1.6, 1e-3, kl)
        err = float((back - C).abs().max())
        gate(err < 1e-14, f"langmuir_inverse round trip ({tag} binding)",
             f"max|S^-1(m(C)) - C| = {err:.2e} mg/L for C in [0,3]")


def check_conservation(cfg_path: str, h5: str, steps: int) -> None:
    from src.data.adr_dataset import ADRDataset, collate

    from src.utils.config import get

    cfg = load_config(cfg_path)
    dx = float(get(cfg, "data.dx", 1.0))
    ds = ADRDataset(h5, "train", K=1)
    norm = ds.fit_normalizer(min(8, len(ds)))
    # late windows: step 0 of every trajectory is all-zero, and a zero state is
    # insensitive to every weight, which would fake a passing gradient test
    idx = list(range(len(ds) - 2, len(ds)))
    batch, _ = collate([ds[i] for i in idx])
    for dtype in (torch.float64, torch.float32):
        for proj in (True, False):
            torch.manual_seed(0)
            model = CMO(cfg, norm).to(dtype)
            model.use_projection = proj
            # the detail head is zero-initialised (that is why the model is
            # trainable); seed it so the unprojected ablation has something to
            # drift *with*. Same seed for both arms, so the only difference is
            # the projection.
            with torch.no_grad():
                model.wave.conv[-1].weight.normal_(0.0, 0.02)
                model.wave.conv[-1].bias.zero_()
            b = {k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
                 for k, v in batch.items()}
            _, r, clamps = _rollout(b, model, steps, dx)
            tag = f"{str(dtype).split('.')[-1]} proj={int(proj)}"
            measured = f"r = {r:.3e} over {steps} steps, clamps={clamps}"
            # the ablation is the negative control *in both precisions*: it must
            # fail, otherwise the audit is measuring nothing
            if not proj:
                gate(r > 1e-3, f"mass audit ({tag}) is the negative control", measured)
            elif dtype is torch.float64:
                gate(r < 1e-13, f"mass audit ({tag})", measured)
            else:
                gate(r < 1e-4, f"mass audit ({tag}) at float32", measured + " (n*eps floor)")


def check_gradients(cfg_path: str, h5: str) -> None:
    from src.data.adr_dataset import ADRDataset, collate

    cfg = load_config(cfg_path)
    ds = ADRDataset(h5, "train", K=1)
    norm = ds.fit_normalizer(4)
    model = CMO(cfg, norm)
    batch, _ = collate([ds[len(ds) - 2]])
    out = model(dict(batch))

    # the mean of the output is conserved, so d(mean)/d(everything) must vanish:
    # a free non-vacuity test that the projection really is wired in
    out.mean().backward(retain_graph=True)
    mean_g = model.log_flux_scale.grad.abs().sum().item()
    model.zero_grad(set_to_none=True)

    target = torch.rand_like(out)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    g_scale = float(model.log_flux_scale.grad.abs().sum())
    g_coarse = sum(float(p.grad.abs().sum()) for p in model.coarse.parameters() if p.grad is not None)
    g_wave = sum(float(p.grad.abs().sum()) for p in model.wave.parameters() if p.grad is not None)
    gate(mean_g < 1e-6, "d(mean out)/d(log_flux_scale) == 0",
         f"{mean_g:.2e} (the mean is conserved, so its derivative must vanish)")
    gate(g_coarse > 0.0 and g_wave > 0.0, "both branches receive gradient",
         f"|g(coarse)| = {g_coarse:.2e}, |g(wave)| = {g_wave:.2e}, "
         f"|g(log_flux_scale)| = {g_scale:.2e}")


def check_arms(h5: str) -> None:
    """One step of every arm: parameter count, **transport** mass defect, `m=0` map.

    The audit is not a unit test of the operator alone: the paper's claim is
    comparative, so the same measurement must be shown for the arms that are
    supposed to fail it. Two details keep the comparison honest:

    * the analytic source is subtracted -- it injects mass on purpose, and a
      budget that included it would report the physics as an error;
    * the `u = 0` probe is run with `s = 0`, otherwise the source alone fills the
      domain and every arm "creates mass" at exactly the same rate.
    """
    from src.data.adr_dataset import ADRDataset, collate
    from src.models import build_model
    from src.models.balance import total_mass

    ds = ADRDataset(h5, "train", K=1)
    norm = ds.fit_normalizer(min(8, len(ds)))
    batch, _ = collate([ds[len(ds) - 1]])
    sc = {k: batch[k] for k in ("theta", "rho_b", "smax", "k_lang")}
    S = batch["s"].double().sum(dim=(-3, -2, -1)).flatten()
    lam = batch["lam"].double().flatten()
    dt = float(batch["dt"])
    m0 = total_mass(batch["u"], **sc, dx=1.0).double()

    def defect(model) -> float:
        with torch.no_grad():
            out = model(batch)
        m1 = total_mass(out, **sc, dx=1.0).double()
        d = m1 - m0 - dt * (S - lam * m0)
        return float((d.abs() / m0.clamp_min(1e-30)).max())

    for name in ("cmo", "nfv", "fluxnet", "fno"):
        cfg_path = "configs/default.yaml" if name == "cmo" else f"configs/baseline_{name}.yaml"
        model = build_model(load_config(cfg_path), norm, verbose=False)
        with torch.no_grad():
            _, diag = model(batch, return_diag=True)
            zero = model(dict(batch, u=torch.zeros_like(batch["u"]),
                              s=torch.zeros_like(batch["s"])))
        zero_max = float(zero.abs().max())
        gate(zero_max == 0.0, f"{name}: empty state stays empty",
             f"max|out(m=0, s=0)| = {zero_max:.3e} (a flux is carried by solute, so u=0 -> u=0)")
        print(f"  -     {name:8s} n={model.n_params():>8,}  init transport dm/M = "
              f"{defect(model):.3e}  rho = {diag['rho']:.3f}")
        if name == "fno":
            # `du` is zero-initialised, so the init defect is 0 for a trivial
            # reason. Seed the head: a model that *cannot* drift would still read
            # 0 here, so this is the non-vacuity probe for the whole arm.
            with torch.no_grad():
                model.fc2.weight.normal_(0.0, 1e-3)
            seeded = defect(model)
            gate(seeded > 1e-6, "fno: the zero init defect is a trivial pass",
                 f"dm/M = {seeded:.3e} with fc2 seeded (so the arm *can* fail the audit)")


def check_langmuir_monotone() -> None:
    C = torch.linspace(0, 5, 2001, dtype=torch.float64)
    S = langmuir(C, 1e-3, 3.0)
    dS = S[1:] - S[:-1]
    m = composite_mass(C, 0.3, 1.6, 1e-3, 3.0)
    dm = (m[1:] - m[:-1]) / (C[1] - C[0])
    gate(float(dS.min()) >= 0.0, "Langmuir isotherm is monotone",
         f"min dS = {float(dS.min()):.2e} over C in [0,5] mg/L (must be >= 0)")
    gate(float(dm.min()) > 0.0, "composite mass is strictly increasing in C",
         f"d m/dC in [{float(dm.min()):.3f}, {float(dm.max()):.3f}] "
         "(the retarded front needs this to be single-valued)")


def check_source_step() -> None:
    """Analytic step alone must reproduce M + dt(S - lambda M) to machine precision."""
    B, N, dt = 2, 32, 0.05
    C = torch.rand(B, 1, N, N, dtype=torch.float64)
    src = torch.rand(B, 1, N, N, dtype=torch.float64) * 0.01
    sc = dict(theta=0.3, rho_b=1.6, smax=1e-3, k_lang=3.0)
    lam = 0.02
    C2, _ = analytic_step(C, torch.zeros_like(C), src, lam, dt=dt, **sc)
    M = total_mass(C, **sc, dx=1.0)
    M2 = total_mass(C2, **sc, dx=1.0)
    pred = M + dt * (src.sum(dim=(-3, -2, -1)).reshape_as(M) - lam * M)
    err = float(((M2 - pred).abs() / M).max())
    gate(err < 1e-13, "analytic source step matches its own budget",
         f"max|M2 - (M + dt(S - lam M))|/M = {err:.2e}")


def check_param_budget() -> None:
    """The four arms of the main table must sit in the same parameter band (§3.10).

    `nfv` and `fluxnet` sat at 4,547 / 4,545 parameters against CMO's 308,407 for
    two steps, because both read `model.coarse_width` -- which in `default.yaml` is
    CMO's *spectral* width (16), not a face-MLP width. A win at 1/68th of the
    parameters is a statement about the budget, not about the mechanism, and it is
    the direction that gets a paper rejected. Gated rather than printed: the
    builder already warned on stderr and the warning was ignored twice.
    """
    from src.models import PARAM_BUDGET, build_model
    from src.utils.config import get, load_config

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lo, hi = PARAM_BUDGET
    counts = {}
    for cfg_name in ("default.yaml", "baseline_fno.yaml",
                     "baseline_nfv.yaml", "baseline_fluxnet.yaml"):
        cfg = load_config(os.path.join(root, "configs", cfg_name))
        model = build_model(cfg, None, verbose=False)
        counts[str(get(cfg, "model.name"))] = model.n_params()

    cmo = counts.get("cmo", 0)
    out = ", ".join(f"{k} {v:,} ({v / cmo:.2f}x)" for k, v in counts.items())
    bad = [k for k, v in counts.items() if not (lo <= v <= hi)]
    gate(not bad, "every arm sits in the parameter band [150k, 600k]", out)
    # a band alone is not enough: two arms could both be in-band at 0.3x and 3x.
    # The ratio to CMO is what the reader divides, so it is what gets a bound.
    ratios = {k: v / cmo for k, v in counts.items() if k != "cmo"}
    ok = all(0.5 <= r <= 2.0 for r in ratios.values())
    gate(ok, "no arm differs from CMO by more than 2x in parameters",
         ", ".join(f"{k} {r:.2f}x" for k, r in ratios.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--h5", default="data/adr2d/adr2d.h5")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--allow-skip", action="store_true",
                    help="exit 0 even if --h5 is missing (default: a missing file fails)")
    args = ap.parse_args()

    check_projection()
    check_divergence()
    check_guards()
    check_detail_ratio()
    check_limiter()
    check_inverse()
    check_langmuir_monotone()
    check_source_step()
    check_param_budget()
    if os.path.exists(args.h5):
        check_conservation(args.config, args.h5, args.steps)
        check_gradients(args.config, args.h5)
        check_arms(args.h5)
    elif args.allow_skip:
        print(f"  SKIP  data-dependent checks: {args.h5} not found (--allow-skip)")
    else:
        # the trap of step 6: a data check that silently skips and exits 0 is not
        # a gate. A missing file is a failure unless the caller asks otherwise.
        gate(False, "data-dependent checks ran", f"{args.h5} not found; pass --allow-skip to waive")

    print(f"\n{len(_FAILED)} failed gate(s)" + ("" if _FAILED else " -- all pass"))
    return 1 if _FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
