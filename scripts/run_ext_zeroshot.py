"""Zero-shot family 2: the long-horizon band (`ext`, T=1250, Pe -> 1).

`kappa` asks whether the claims survive a change in the coefficients; this asks
whether they survive a change in *duration*. It is the sharper test of the paper's
current mechanism claim: if every arm under-transports at a fixed rate, then at 1250
steps the displacement must accumulate to something like 6x the 1.2-1.4 cells seen at
200 -- and if it does not, the "deterministic lag" reading is wrong rather than merely
incomplete.

Deliberately a driver rather than a flag: one process per arm keeps VRAM flat (the
1250-step rollout keeps the whole series on device) and makes an OOM on one arm
recoverable instead of losing the batch. Each arm writes to its own `--out`.

25 min/arm was the measured worst case at 200 steps x 60; 1250 x 80 is ~8x that, so
budget ~85 min for seven arms. Run it in the background.
"""
import os
import subprocess
import sys
import time

# Every path below is relative to `code/`, but a background launch does not inherit
# the caller's working directory -- it starts wherever the parent process happened to
# be, and the first attempt reported all seven checkpoints as "missing". Anchor to the
# script's own location instead of trusting the cwd.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PY = sys.executable
CK = {
    "cmo": "results/cmo_s0/cmo_adr2d_0/best.pth",
    "cmo_nosep": "results/cmo_nosep_s0/cmo_adr2d_0_cmo_nosep/best.pth",
    "cmo_af": "results/cmo_af_s0/cmo_adr2d_0_cmo_af/best.pth",
    "cmo_af_nosep": "results/cmo_af_nosep_s0/cmo_adr2d_0_cmo_af_nosep/best.pth",
    "fluxnet": "results/s11/fluxnet_adr2d_0/best.pth",
    "nfv": "results/s11/nfv_adr2d_0/best.pth",
    "fno": "results/s11/fno_adr2d_0/best.pth",
}
STEPS = 1250

for arm, ckpt in CK.items():
    out = f"results/_zshot/ext_{arm}"
    if os.path.exists(f"{out}/metrics_test.json"):
        print(f"[skip] {arm}: already done", flush=True)
        continue
    if not os.path.exists(ckpt):
        print(f"[miss] {arm}: {ckpt}", flush=True)
        continue
    t0 = time.time()
    print(f"[run ] {arm} ({STEPS} steps)", flush=True)
    r = subprocess.run([PY, "-u", "src/evaluate.py", "--ckpt", ckpt,
                        "--modes", "ext", "--steps", str(STEPS),
                        "--device", "cuda", "--out", out],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = [ln for ln in r.stdout.splitlines() if ln.startswith("[evaluate]")]
    print(f"[{'ok  ' if r.returncode == 0 else 'fail'}] {arm} "
          f"{tail[-1] if tail else ''}  ({time.time() - t0:.0f}s)", flush=True)
    if r.returncode != 0:
        print(r.stderr[-1500:], flush=True)
print("ALL DONE", flush=True)
