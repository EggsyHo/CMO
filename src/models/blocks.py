"""Building blocks of the proposed operator (implementation.md §3.6).

Three of these carry the paper's claims, so each is written so that the claim can
be *checked* rather than asserted:

* `MassPartition`         -- the projection Pi; `avg_pool2d` is an exact cell
                             average, so `restrict` conserves mass exactly.
* `CoarseFluxPropagator`  -- predicts face fluxes, never states.
* `divergence`            -- the telescope that makes `sum(div(F)) == 0`
                             identically. Nothing else in the model has to be
                             correct for the composite budget to close; that is
                             the design, and §3.6 calls it "the single line".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# projection Pi
# --------------------------------------------------------------------------- #


class MassPartition(nn.Module):
    """Pi = lift . restrict, with `restrict` the exact m x m cell average.

    Non-trainable and resolution agnostic: the 128^2 transfer test rebuilds the
    same object with `n_fine=128, block=2 -> M=64` and touches no weight.
    """

    def __init__(self, n_fine: int = 64, block: int = 2) -> None:
        super().__init__()
        if n_fine % block != 0:
            raise ValueError(f"n_fine={n_fine} is not divisible by block={block}")
        self.m = int(block)
        self.M = int(n_fine // block)

    def restrict(self, x: torch.Tensor) -> torch.Tensor:
        """[B,C,N,N] -> [B,C,M,M] by exact cell averaging (equal weights)."""
        return F.avg_pool2d(x, self.m)

    def lift(self, xc: torch.Tensor) -> torch.Tensor:
        """[B,C,M,M] -> [B,C,N,N] by replicating each coarse cell m x m times."""
        return xc.repeat_interleave(self.m, dim=-2).repeat_interleave(self.m, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lift(self.restrict(x))

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """x - Pi x: zero block mean in every coarse cell, by construction."""
        return x - self.forward(x)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"block={self.m}, M={self.M}"


def zero_mean_project(delta: torch.Tensor, partition: MassPartition,
                      tol: float = 1e-6, check: bool = False) -> torch.Tensor:
    """Remove the block mean of an *increment* (never of the state).

    Projecting the state would erase the front; projecting the increment is the
    trick of Part 2 §3.5 -- the coarse branch keeps ownership of the cell mean,
    the fine branch only redistributes inside a cell, so total mass is untouched.
    """
    out = partition.residual(delta)
    if check:
        block_mean = partition.restrict(out).abs().amax()
        scale = partition.restrict(delta).abs().amax().clamp_min(1e-30)
        rel = float(block_mean / scale)
        if rel > tol:
            raise AssertionError(f"zero_mean_project left a block mean of {rel:.3e} (relative)")
    return out


# --------------------------------------------------------------------------- #
# spectral convolution (FNO contraction, applied on the coarse grid)
# --------------------------------------------------------------------------- #
class SpectralConv2d(nn.Module):
    """Truncated 2D Fourier layer, with an explicit Nyquist guard."""

    def __init__(self, in_ch: int, out_ch: int, modes: int) -> None:
        super().__init__()
        self.in_ch, self.out_ch, self.modes = int(in_ch), int(out_ch), int(modes)
        scale = 1.0 / max(self.in_ch, 1)
        self.w = nn.Parameter(scale * torch.rand(self.in_ch, self.out_ch, modes, modes, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2], x.shape[-1]
        limit = min(H, W) // 2
        if self.modes > limit:
            raise ValueError(
                f"modes={self.modes} exceeds the Nyquist limit {limit} of a {H}x{W} grid; "
                f"pass --set model.modes=<={limit} (implementation.md §0 decision 1)"
            )
        m = self.modes
        # This layer must run in fp32/complex64 even under fp16 autocast: the
        # `einsum` on a ComplexHalf operand has no CUDA kernel at all
        # ("baddbmm_cuda not implemented for 'ComplexHalf'", i.e. the whole model
        # dies on GPU with amp on), and `torch.fft` promotes half to float
        # anyway. `enabled=False` is scoped to this module only, so the spatial
        # branches keep their fp16 speed; the cast back at the end keeps the
        # block's activations in one dtype.
        in_dtype = x.dtype
        # fp32 is the *floor* of the compute dtype, not a fixed one: clamping to
        # float() here broke every float64 caller (the fft of an fp32 input is
        # complex64 while `view_as_complex` of an fp64 weight is complex128, and
        # einsum then raises "expected scalar type ComplexFloat but found
        # ComplexDouble"). `promote_types` keeps half/bfloat16 at fp32 -- which is
        # what the autocast note above is about -- and leaves fp64 at fp64.
        compute = torch.promote_types(x.dtype, torch.float32)
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = torch.fft.rfft2(x.to(compute), dim=(-2, -1))
            wk = torch.view_as_complex(self.w.contiguous())
            out = torch.zeros(x.shape[0], self.out_ch, H, W // 2 + 1,
                              dtype=xf.dtype, device=xf.device)
            out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], wk)
            out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], wk)
            y = torch.fft.irfft2(out, s=(H, W), dim=(-2, -1))
        return y.to(in_dtype)


# --------------------------------------------------------------------------- #
# coarse flux branch
# --------------------------------------------------------------------------- #
class CoarseFluxPropagator(nn.Module):
    """Spectral stack on the coarse grid, emitting (x-face, y-face) fluxes.

    The 12 input channels are a contract, not a choice:
        u_c (1) + q_c (2) + D_c (3) + s_c (1) + theta, rho_b, smax, k_lang, lam (5)
    Layer widths are small on purpose: the parameter budget of Part 2 §3.6 is
    0.2-0.5 M *including* the residual branch.
    """

    PARAM_CH = ((1,), (2,), (3,), (1,), (5,))
    IN_CH = 12

    def __init__(self, in_ch: int = 12, width: int = 16, modes: int = 12,
                 n_layers: int = 4) -> None:
        super().__init__()
        if in_ch != self.IN_CH:
            raise ValueError(f"CoarseFluxPropagator expects in_ch={self.IN_CH}, got {in_ch}")
        self.width, self.modes, self.n_layers = int(width), int(modes), int(n_layers)
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spec = nn.ModuleList(SpectralConv2d(width, width, modes) for _ in range(n_layers))
        self.bypass = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(n_layers))
        self.head = nn.Conv2d(width, 2, 1)
        self.act = nn.GELU()

    def forward(self, u_c: torch.Tensor, params_c: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """`u_c` [B,1,M,M] normalized; params [B,c,M,M] or [B,1,1,1] -> (Fhat, features)."""
        B, _, M, _ = u_c.shape
        parts = [u_c]
        for key in ("q", "D", "s"):
            v = params_c[key]
            if v.shape[-1] != M or v.shape[-2] != M:
                v = F.interpolate(v, size=(M, M), mode="area")
            parts.append(v)
        for key in ("theta", "rho_b", "smax", "k_lang", "lam"):
            parts.append(params_c[key].reshape(B, 1, 1, 1).expand(B, 1, M, M))
        h = self.act(self.lift(torch.cat(parts, 1)))
        for sp, bp in zip(self.spec, self.bypass):
            h = self.act(sp(h) + bp(h))
        return self.head(h), h


def mass_weighted_project(rate: torch.Tensor, m_ref: torch.Tensor,
                          partition: MassPartition, eps: float = 1e-30) -> torch.Tensor:
    """Remove the block mean of a *dimensionless rate*, weighted by the mass.

    Returns ``dm = m (rate - rbar)`` with ``rbar = <rate>_m`` the mass-weighted
    block mean, so

    * ``sum(dm) == 0`` per block -- exactly, because rbar is *defined* as the
      mass-weighted average, not as an unweighted one;
    * ``|dm| <= 2 |rate|_inf m`` -- and with ``|rate| <= 1/4`` the increment can
      never drive the composite mass negative, so no clipping and no dead
      gradients in cells where the plume has not arrived.

    An unweighted projection (subtract the block mean of the field) fails the
    second property: inside a block that straddles a front, the mean is large
    where the local mass is near zero, and the correction drains those cells.
    """
    mblk = partition.restrict(m_ref)
    rblk = partition.restrict(m_ref * rate) / mblk.clamp_min(eps)
    return m_ref * (rate - partition.lift(rblk))


def positivity_limiter(dm: torch.Tensor, m_ref: torch.Tensor,
                       frac: float = 0.5, eps: float = 1e-12) -> torch.Tensor:
    """Enforce `m_ref + dm >= 0` **without** changing `sum(dm)`.

    Per-cell clipping alone injects mass (that is exactly the failure the
    untrained-model audit is built to catch), and a single global rescale dies as
    soon as one empty cell needs limiting -- which is always, because most of the
    domain holds no solute. So: clip the negative part per cell, then take the
    mass that clipping gave back (``repair >= 0``) out of the *positive* part in
    proportion to its size. Since the increment is zero-sum, ``repair`` never
    exceeds the positive mass, so the positive part is only ever shrunk -- the
    result stays non-negative and its sum is unchanged to float rounding.

    Gradient note: clipping is active only in cells that would have gone
    negative, so a mass-proportional flux (the normal case) passes untouched.
    """
    lower = -float(frac) * m_ref.clamp_min(0.0)
    clipped = torch.maximum(dm, lower)
    dims = (-3, -2, -1)
    repair = (clipped - dm).sum(dim=dims, keepdim=True)          # >= 0
    pos = clipped.clamp_min(0.0)
    pos_sum = pos.sum(dim=dims, keepdim=True)
    # Hard gate on "did anything get clipped at all". Without it, `repair = 0`
    # and `pos_sum = 0` simultaneously (an increment that touches nothing), and
    # the quotient's *derivative* is 1/eps = 1e12 even though its value is 0 --
    # measured as a 1e3 gradient on a 1e-3 loss. The mask is exact (inactive
    # means `out == clipped` anyway) and keeps the pass-through branch clean.
    active = (repair > 0).to(dm.dtype)
    share = active * repair / pos_sum.clamp_min(eps)
    return clipped - share.clamp(0.0, 1.0) * pos


def divergence(Fhat: torch.Tensor, dx: float) -> torch.Tensor:
    """Telescoping divergence of face fluxes per unit face area -> [B,1,H,W].

    Every interior face appears once with each sign, so the grid sum is zero up
    to float summation order: `sum(div) == 0` exactly. This is the conservation
    guarantee, and the unit check asserts it to 1e-12 relative.
    """
    Fx, Fy = Fhat[:, 0:1], Fhat[:, 1:2]
    d = (Fx - torch.roll(Fx, 1, dims=-1)) + (Fy - torch.roll(Fy, 1, dims=-2))
    return d / float(dx)


# --------------------------------------------------------------------------- #
# wavelet residual branch (fixed db2 filter bank, GPU-friendly)
# --------------------------------------------------------------------------- #
# db2 (Daubechies-2 / D4) orthonormal low-pass taps; high-pass is the QMF
# g[n] = (-1)^n h[3-n]. Baked in rather than imported from pywt so the branch
# runs inside autocast on the GPU with `requires_grad=False` weights.
DB2_LO = (0.4829629131445341, 0.8365163037378079,
          0.2241438680420134, -0.1294095225512604)
DB2_HI = (DB2_LO[3], -DB2_LO[2], DB2_LO[1], -DB2_LO[0])


class WaveletResidualBranch(nn.Module):
    """Detail-energy gate input + fine-grid detail compensation (Part 2 §3.5).

    9 input channels: r_n (1) + the three level-1 detail subbands (3, upsampled
    from M x M to N x N) + q (2) + D (3).
    """

    IN_CH = 9

    def __init__(self, width: int = 32, in_ch: int = 9, wavelet: str = "db2",
                 levels: int = 2) -> None:
        super().__init__()
        if wavelet != "db2":
            raise ValueError("only the db2 filter bank is baked in; use wavelet: db2")
        if levels != 2:
            raise ValueError("levels=2 is assumed (64->32->16); change both code and config")
        if in_ch != self.IN_CH:
            raise ValueError(f"WaveletResidualBranch expects in_ch={self.IN_CH}, got {in_ch}")
        self.levels = int(levels)
        self.register_buffer("lo", torch.tensor(DB2_LO).view(1, 1, 1, 4), persistent=False)
        self.register_buffer("hi", torch.tensor(DB2_HI).view(1, 1, 1, 4), persistent=False)
        self.conv = nn.Sequential(
            nn.Conv2d(self.IN_CH, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, 1, 3, padding=1),
        )
        # zero the head: a randomly initialised conv emits |out| ~ 10, tanh
        # saturates, and the detail branch then has a *literally* zero gradient
        # (d tanh(x)/dx = 1e-14 at |x| = 16). At zero the branch starts as the
        # identity -- the model is conservative *and* trainable at step 0.
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    # ---- fixed analysis (periodic, stride 2) -----------------------------
    def _along(self, x: torch.Tensor, k: torch.Tensor, dim: int) -> torch.Tensor:
        C = x.shape[1]
        if dim == -1:
            w = k.expand(C, 1, 1, 4)
            xp = F.pad(x, (2, 1, 0, 0), mode="circular")
            return F.conv2d(xp, w, stride=(1, 2), groups=C)
        w = k.transpose(-1, -2).expand(C, 1, 4, 1)
        xp = F.pad(x, (0, 0, 2, 1), mode="circular")
        return F.conv2d(xp, w, stride=(2, 1), groups=C)

    def _split(self, x: torch.Tensor):
        """(LL, LH, HL, HH): L/H = low/high pass along width first, then height."""
        a, b = self._along(x, self.lo, -1), self._along(x, self.hi, -1)
        return (self._along(a, self.lo, -2), self._along(a, self.hi, -2),
                self._along(b, self.lo, -2), self._along(b, self.hi, -2))

    def wavelet_features(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(detail_energy [B,1,M,M], detail subbands [B,3,M,M], LL2 [B,1,M/2,M/2])."""
        ll1, lh1, hl1, hh1 = self._split(r)             # 64 -> 32 = M
        energy = (lh1 ** 2 + hl1 ** 2 + hh1 ** 2).sum(dim=1, keepdim=True)
        bands = torch.cat([lh1, hl1, hh1], 1)           # [B,3,M,M]
        ll2, _, _, _ = self._split(ll1)                 # 32 -> 16
        return energy, bands, ll2

    def forward(self, r_n: torch.Tensor, q: torch.Tensor, D: torch.Tensor,
                beta: torch.Tensor, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        E, bands, _ = self.wavelet_features(r_n)
        Ehat = E / (E.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        g = torch.sigmoid(beta * (Ehat - tau))
        N = r_n.shape[-1]
        x = torch.cat([r_n, F.interpolate(bands, size=(N, N), mode="nearest"), q, D], 1)
        return self.conv(x), g, Ehat


class SoftGate(nn.Module):
    """Learnable sharpness gate; conservation holds for any (beta, tau)."""

    def __init__(self, beta_init: float = 4.0, tau_init: float = 0.1) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))

    def forward(self, Ehat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.beta * (Ehat - self.tau))


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def detail_ratio(x: torch.Tensor, block: int = 2) -> float:
    """`||(I-Pi)x||^2 / ||x||^2`: the share of the state that lives below the
    coarse grid. Reported for every model (it is the diagnostic that the detail
    branch is supposed to move), so it is defined once and imported, not copied.
    """
    part = MassPartition(x.shape[-1], block)
    r = part.residual(x)
    denom = (x ** 2).sum().clamp_min(1e-30)
    with torch.no_grad():
        return float((r ** 2).sum() / denom)
