"""Re-evaluate existing checkpoints with the sharpness estimator (no retraining).

Roughly one minute per run on one GPU; five arms x five seeds ~ 25 min.

Writes to `results/_sharp/<arm>_s<seed>/` so the primary `metrics_*.json` files the
paper table was built from are never overwritten -- the point is to add the
`sharpness_grad` block beside the old `sharpness` block, not to replace it.

    python scripts/recompute_sharpness.py                 # the four arms of the table
    python scripts/recompute_sharpness.py cmo_nosep cmo_af_nosep
"""
import os
import subprocess
import sys

# the launcher's cwd is not this directory, and every path below is relative to it
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable


def ckpt(arm, seed):
    """Checkpoint path per arm. Seed 0 predates the per-seed `*_s<n>` directories for
    the three baselines, so it is special-cased rather than symlinked."""
    if arm in ("fluxnet", "nfv", "fno"):
        return (f"results/s11/{arm}_adr2d_0/best.pth" if seed == 0
                else f"results/{arm}_s{seed}/{arm}_adr2d_{seed}/best.pth")
    if arm == "cmo":
        return f"results/cmo_s{seed}/cmo_adr2d_{seed}/best.pth"
    if arm == "cmo_af":
        return f"results/cmo_af_s{seed}/cmo_adr2d_{seed}_cmo_af/best.pth"
    if arm == "cmo_nosep":
        return f"results/cmo_nosep_s{seed}/cmo_adr2d_{seed}_cmo_nosep/best.pth"
    if arm == "cmo_af_nosep":
        return f"results/cmo_af_nosep_s{seed}/cmo_adr2d_{seed}_cmo_af_nosep/best.pth"
    raise KeyError(arm)


ARMS = sys.argv[1:] or ["fluxnet", "nfv", "fno", "cmo", "cmo_af"]
for arm in ARMS:
    for s in range(5):
        c = ckpt(arm, s)
        if not os.path.exists(c):
            print(f"[skip] {c} missing", flush=True)
            continue
        out = f"results/_sharp/{arm}_s{s}"
        print(f"[start] {arm} s{s}", flush=True)
        r = subprocess.run([PY, "-u", "src/evaluate.py", "--ckpt", c, "--modes", "test",
                            "--steps", "200", "--device", "cuda", "--out", out],
                           capture_output=True, text=True)
        tail = [ln for ln in r.stdout.splitlines() if "rel_l2_final" in ln]
        print(f"[done] {arm} s{s} -> {tail[-1].strip() if tail else 'NO OUTPUT'}", flush=True)
        if r.returncode != 0:
            print(r.stderr[-800:], flush=True)
print("ALL DONE", flush=True)
