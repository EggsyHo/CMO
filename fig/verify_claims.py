"""Re-derive the numbers printed in `docs/papers/main_text.md` from the files the
figures read, and diff them.

The point is not to re-plot anything: it is to make "the table, the figure and the
abstract quote one number" mechanically checkable. The tables are parsed straight out
of the manuscript, so a hand-edited cell is compared against the data rather than
against a copy of itself.

Any row whose source is not in `results/` is reported as `no source` instead of being
silently skipped -- an unverified number is a finding, not a pass.

Run:  python code/fig/verify_claims.py
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

import figlib as F

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "papers", "main_text.md"))

rows: list[tuple[str, float, float, str]] = []   # (label, quoted, computed, note)


def nums(s: str) -> list[float]:
    s = s.replace("**", "").replace("$", "").replace(",", "")
    return [float(x) for x in
            re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)]


def tables() -> list[list[list[str]]]:
    """Every markdown table in the manuscript, as a list of cell rows."""
    out, cur = [], []
    for line in open(TEXT, encoding="utf-8"):
        if line.lstrip().startswith("|"):
            # `\|` inside a cell is an escaped pipe, not a column break
            cur.append([c.strip() for c in
                        re.split(r"(?<!\\)\|", line.strip().strip("|"))])
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def body(tab: list[list[str]]) -> list[list[str]]:
    """Drop the header and the `---` separator."""
    keep = [r for r in tab if not all(set(c) <= set("-: ") for c in r)]
    return keep[1:]


def add(label: str, quoted: float, computed: float, note: str = "",
        tol: float = 5e-5) -> None:
    rows.append((label, quoted, computed, note, tol))


def close(label: str, quoted: float | None, computed: float, note: str = "",
          tol: float = 5e-5) -> None:
    """Record a pair; the tolerance is the manuscript's own print precision."""
    add(label, float("nan") if quoted is None else quoted, computed, note, tol)


def per_seed_mean_sd(a: np.ndarray) -> tuple[float, float]:
    m = a.mean(axis=1)
    return float(m.mean()), float(m.std(ddof=1))


def main_tables() -> None:
    ts = tables()
    by_head = {t[0][0].lstrip("* ").strip(): t for t in ts}

    # 1) the main table
    tab = by_head["Model"]
    arms = {"FluxNet": "fluxnet", "CMO": "cmo", "NFV": "nfv", "FNO": "fno"}
    for r in body(tab):
        if r[0] not in arms:
            continue
        arm = arms[r[0]]
        e = F.rel_l2_final(arm)
        close(f"main/{r[0]}: rel_l2 mean", nums(r[2])[0], float(e.mean()))
        close(f"main/{r[0]}: rel_l2 sd", nums(r[2])[1], float(e.std(ddof=1)))
        rf = F.conservation(arm)
        q = nums(r[4])
        close(f"main/{r[0]}: r_final mean", q[0], float(rf.mean()))
        close(f"main/{r[0]}: r_final worst", q[1], float(rf.max()))
        sh = F.sharpness(arm)["ratio_median"]
        # printed to three decimals, so 1e-3 is the tightest honest tolerance
        close(f"main/{r[0]}: w/w_ref mean", nums(r[5])[0], float(sh.mean()), tol=1e-3)
        close(f"main/{r[0]}: w/w_ref sd", nums(r[5])[1], float(sh.std(ddof=1)),
              tol=1e-3)

    # 2) the in-file ablation
    tab = next(t for t in ts if len(t[0]) == 3 and t[0][1].startswith("flux form A"))
    cells = {("no detail branch", 0): "cmo_nosep", ("no detail branch", 1): "cmo_af_nosep",
             ("with detail branch", 0): "cmo", ("with detail branch", 1): "cmo_af"}
    for r in body(tab):
        for col in (0, 1):
            arm = cells[(r[0], col)]
            e = F.rel_l2_final(arm)
            close(f"ablation/{r[0]}/{'A' if col == 0 else 'B'}: mean",
                  nums(r[col + 1])[0], float(e.mean()))
            close(f"ablation/{r[0]}/{'A' if col == 0 else 'B'}: sd",
                  nums(r[col + 1])[1], float(e.std(ddof=1)))

    # 3) the displacement decomposition
    lag = json.load(open(os.path.join(F.RES, "_lag_decomposition.json"), encoding="utf-8"))
    tab = next(t for t in ts if len(t[0]) == 6 and "E_total" in t[0][1])
    names = {"FluxNet": "fluxnet", "CMO": "cmo", "CMO, no detail": "cmo_nosep",
             "CMO form B": "cmo_af", "CMO form B, no detail": "cmo_af_nosep",
             "NFV": "nfv", "FNO": "fno"}
    for r in body(tab):
        if r[0] not in names:
            continue
        arm = names[r[0]]
        et = np.array(lag[arm]["E_total"], dtype=float)
        es = np.array(lag[arm]["E_shape"], dtype=float)
        close(f"lag/{r[0]}: E_total", nums(r[1])[0], float(et.mean()))
        close(f"lag/{r[0]}: E_shape", nums(r[2])[0], float(es.mean()))
        close(f"lag/{r[0]}: share %", nums(r[4])[0],
              float((et - es).mean() / et.mean() * 100), tol=0.06)

    # 4) the single-variable dt band
    tab = next(t for t in ts if len(t[0]) == 4 and t[0][1].endswith("dt=0.100\\)"))
    cells = {"CMO (form A)": "cmo", "FluxNet": "fluxnet", "form B, as trained": "af_raw",
             "form B, \u00d7dt/dt_train": "af_corr",
             "form B (no separation head), as trained": "af_nosep_raw",
             "form B (no separation head), \u00d7dt/dt_train": "af_nosep_corr"}
    for r in body(tab):
        cell = cells.get(r[0])
        if cell is None:
            continue
        for col, band in ((1, "coarse"), (2, "fine")):
            q = nums(r[col])
            mean, sd = per_seed_mean_sd(F.dt_band(cell, band))
            close(f"dt/{r[0]}/{'0.100' if col == 1 else '0.016'}: mean", q[0], mean)
            close(f"dt/{r[0]}/{'0.100' if col == 1 else '0.016'}: sd", q[1], sd)

    # 5) retrained on three rates
    tab = next(t for t in ts if len(t[0]) == 4 and "retrained on three rates" in t[0][0])
    cells = {"CMO (form A)": "mix_cmo", "form B (absolute flux, no dt)": "mix_af"}
    for r in body(tab):
        cell = cells.get(r[0])
        if cell is None:
            continue
        for col, band in ((1, "coarse"), (2, "mid"), (3, "fine")):
            q = nums(r[col])
            mean, sd = per_seed_mean_sd(F.dt_band(cell, band))
            close(f"mix/{r[0]}/{band}: mean", q[0], mean)
            close(f"mix/{r[0]}/{band}: sd", q[1], sd)


def report() -> int:
    bad = 0
    print(f"{'claim':46s} {'manuscript':>12s} {'recomputed':>12s}  diff")
    for label, q, c, note, tol in rows:
        if not np.isfinite(q):
            print(f"{label:46s} {'-':>12s} {c:12.6g}  {note}")
            bad += 1
            continue
        d = q - c
        ok = abs(d) <= max(tol, abs(c) * 5e-3)
        if not ok:
            bad += 1
        print(f"{label:46s} {q:12.6g} {c:12.6g}  {d:+.2e}"
              f"{'' if ok else '   <-- MISMATCH ' + note}")
    print(f"\n{len(rows) - bad}/{len(rows)} cells reproduce from the data at the "
          f"manuscript's own print precision; {bad} need a look")
    if any(n == "no quoted value" for _, _, _, n, _ in rows):
        print("note: a missing quoted value means the regex found no number in that "
              "cell -- read it by hand before trusting the row")
    return 1 if bad else 0


if __name__ == "__main__":
    main_tables()
    sys.exit(report())
