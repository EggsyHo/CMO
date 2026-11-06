"""Run one 80-case cell over seeds 0-4, sequentially, as four concurrent drivers.

`cmd /c "for %s in ..."` inside a quoted command line is not a reliable launcher here
(it fails with "文件名、目录名或卷标语法不正确" before Python ever starts), so the
loop lives in Python and each seed is a subprocess of `ext_seeds_full.py`, which
carries the resume check. One driver per cell, four drivers at once.
"""
import subprocess
import sys
import os

cell = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
for s in range(5):
    r = subprocess.run([sys.executable, "-u", os.path.join(here, "ext_seeds_full.py"),
                        "--cell", cell, "--seed", str(s)])
    if r.returncode != 0:
        print(f"[fail] {cell} seed {s} rc={r.returncode}", flush=True)
print(f"[driver done] {cell}", flush=True)
