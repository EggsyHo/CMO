#!/usr/bin/env bash
# Regenerate every ADR-2D data file (§3.3, §3.19).
#
# The solver self-check runs first and aborts the whole pipeline: writing training
# data with an unverified kernel is exactly the failure mode §3.2 exists to prevent.
#
# dt / T are not cosmetic. §3.1 gates `max D dt/dx^2 <= 0.2`, and measured on the
# sampled bands that needs dt <= 0.172 (train), <= 0.121 (kappa20) and <= 0.019
# (ext, where Pe drops to 1 and the dispersion tensor reaches |D| ~ 9.5). Every
# file therefore keeps t_final = 20 s and pays for it in step count, so the four
# sets share one physical time axis and stay comparable.
#
# Usage:  bash code/scripts/gen_data.sh        (git bash / WSL on Windows)
#         PY=/path/to/python bash code/scripts/gen_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "${PY:-}" ]; then :
elif [ -x "../.venv/Scripts/python.exe" ]; then PY="../.venv/Scripts/python.exe"
elif [ -x "../.venv/bin/python" ]; then PY="../.venv/bin/python"
else PY="python"
fi
echo "python: $PY"

echo "== gate: solver self-check (MMS order + analytic width + mass) =="
"$PY" data/selfcheck_solver.py > results/calibration/selfcheck.log 2>&1 \
  || { tail -n 30 results/calibration/selfcheck.log; echo "self-check FAILED"; exit 1; }

gen() {  # gen <out> <expected traj count> <args...>
  local out="$1" want="$2"; shift 2
  echo "== $out (expect $want trajectories) =="
  "$PY" data/gen_adr2d.py --out "$out" "$@" 2>&1 | tail -n 4
  "$PY" - "$out" "$want" <<'PYEOF'
import sys, h5py
path, want = sys.argv[1], int(sys.argv[2])
with h5py.File(path) as f:
    trajs = [k for k in f if k.startswith("traj_")]
    n = int(f["/meta"].attrs["n_traj"])
    ids = {int(i) for s in ("train", "val", "test") for i in f[f"/splits/{s}"][:]}
    sizes = "/".join(str(len(f[f"/splits/{s}"][:])) for s in ("train", "val", "test"))
    resid = max(float(f[k].attrs.get("qc_mass_resid", 0.0)) for k in trajs)
    print(f"   -> {len(trajs)} traj, n_traj={n}, splits {sizes} (unique {len(ids)}), "
          f"max mass_resid={resid:.2e}")
    # unique, not summed: a zero-shot file exposes its one set under both val and test
    assert n == want and len(trajs) == want and ids == set(range(want)), "inconsistent HDF5"
PYEOF
}

gen data/adr2d/adr2d.h5 400 --regime train --n 400 --T 200 --dt 0.1 --seed 0
gen data/adr2d/adr2d_ext.h5 80 --regime ext --n 80 --T 1250 --dt 0.016 --seed 0
gen data/adr2d/adr2d_kappa.h5 80 --regime kappa1,kappa20 --n 40,40 --T 200 --dt 0.1 --seed 0
gen data/adr2d/adr2d_long.h5 20 --regime train --n 20 --T 200 --dt 0.1 --long-T 200 --seed 0

echo "== gate: data contract (keys, shapes, index completeness, stats.npz) =="
"$PY" tests/check_data.py || { echo "data contract FAILED"; exit 1; }

echo "== done =="
ls -lh data/adr2d/*.h5
