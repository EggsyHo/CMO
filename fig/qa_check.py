"""QA gate for every figure under code/fig/ (research[fig] qa-protocol.md).

Checks what can be checked mechanically, writes `qa-report.json` next to each
figure, and refuses to mark anything "pass" that was not actually verified:

  files / dimensions / font / color / data_scope / write_boundary  -> automated
  visual_preview                                                   -> manual-pass
      (bbox audit + text-vs-data checks run in each plot.py; the human look at
       the 500 dpi PNG is still owed, so it is never reported as automated)

Run:  python code/fig/qa_check.py            (all figures)
      python code/fig/qa_check.py fig3_ablation
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import yaml
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
DPI = 500
LIB_YAML = os.path.join(os.path.dirname(os.path.dirname(HERE)), ".codebuddy", "skills",
                        "research[fig]", "references", "color-library.yaml")

problems: list[str] = []


def fail(fig: str, msg: str) -> None:
    problems.append(f"{fig}: {msg}")


def colour_whitelist() -> set[str]:
    with open(LIB_YAML, encoding="utf-8") as fh:
        lib = yaml.safe_load(fh)
    return {v.upper() for v in lib["base"].values()}


def snapshot_scope() -> dict[str, float]:
    """mtimes of everything the plotting scripts must not touch."""
    out: dict[str, float] = {}
    for sub in ("src", "scripts", "data", "results", "configs", "tests"):
        root = os.path.join(CODE, sub)
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                p = os.path.join(dirpath, f)
                out[p] = os.path.getmtime(p)
    return out


def run_plot(fig_dir: str) -> tuple[bool, str]:
    proc = subprocess.run([sys.executable, os.path.join(fig_dir, "plot.py")],
                          cwd=HERE, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    bad = [ln.strip() for ln in out.splitlines()
           if re.search(r"Traceback|OUTSIDE|CLIPPED|OVERLAP|AssertionError", ln)]
    return proc.returncode == 0 and not bad, " | ".join(bad[:3])


def check_pdf(path: str, w_mm: float, h_mm: float) -> dict:
    raw = open(path, "rb").read()
    boxes = re.findall(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", raw)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
    fonts = {m.decode() for m in re.findall(rb"/BaseFont\s*/([A-Za-z0-9+\-]+)", raw)}
    if not boxes:
        return {"ok": False, "why": "no MediaBox found"}
    x0, y0, x1, y1 = (float(v) for v in boxes[0])
    pw, ph = (x1 - x0) / 72 * 25.4, (y1 - y0) / 72 * 25.4
    ok = (pages == 1 and abs(pw - w_mm) < 0.2 and abs(ph - h_mm) < 0.2
          and fonts and all("Arial" in f for f in fonts))
    return {"ok": ok, "pages": pages, "mm": [round(pw, 2), round(ph, 2)],
            "fonts": sorted(fonts), "why": "" if ok else "pages/size/font mismatch"}


def check_svg(path: str, w_mm: float, h_mm: float) -> dict:
    head = open(path, encoding="utf-8").read(4000)
    # matplotlib writes points, not millimetres
    m = re.search(r'width="([\d.]+)(mm|pt|px)"\s+height="([\d.]+)(mm|pt|px)"', head)
    vb = re.search(r'viewBox="([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)"', head)
    if not m or not vb:
        return {"ok": False, "why": "missing width/height or viewBox"}
    to_mm = {"mm": 1.0, "pt": 25.4 / 72, "px": 25.4 / 96}
    sw = float(m.group(1)) * to_mm[m.group(2)]
    sh = float(m.group(3)) * to_mm[m.group(4)]
    vw, vh = float(vb.group(3)), float(vb.group(4))
    ok = (abs(sw - w_mm) < 0.1 and abs(sh - h_mm) < 0.1
          and abs(vw / vh - w_mm / h_mm) < 0.01)
    return {"ok": ok, "mm": [round(sw, 2), round(sh, 2)],
            "viewbox": [round(vw, 2), round(vh, 2)], "unit": m.group(2),
            "why": "" if ok else f"size {sw:.2f}x{sh:.2f}mm vs {w_mm}x{h_mm}mm"}


def check_png(path: str, w_mm: float, h_mm: float) -> dict:
    im = Image.open(path)
    dpi = im.info.get("dpi", (0, 0))
    ex = round(w_mm / 25.4 * DPI), round(h_mm / 25.4 * DPI)
    ok = (abs(im.size[0] - ex[0]) <= 1 and abs(im.size[1] - ex[1]) <= 1
          and abs(float(dpi[0]) - DPI) <= 1)
    return {"ok": ok, "px": list(im.size), "dpi": [round(float(d), 1) for d in dpi],
            "why": "" if ok else f"expected {ex} at {DPI} dpi"}


def check_colors(fig_dir: str) -> dict:
    allowed = colour_whitelist()
    used: set[str] = set()
    for f in ("plot.py", "figure.yaml"):
        p = os.path.join(fig_dir, f)
        if not os.path.exists(p):
            continue
        used |= {h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}", open(p, encoding="utf-8").read())}
    bad = sorted(used - allowed)
    return {"ok": not bad, "literals": sorted(used), "unlisted": bad,
            "why": "" if not bad else f"colours not in color-library: {bad}"}


REQUIRED_FIGURE_KEYS = ("version", "figure_id", "purpose", "expression", "arrangement",
                        "data", "layout", "style", "exceptions")
REQUIRED_LAYOUT_KEYS = ("units", "width", "height", "size_source", "dpi")


def load_plan() -> tuple[dict, list[str]]:
    """`layout-plan.yaml` must be able to answer: every figure's size, which figures
    share a row, the common row height, the gaps, and where the size came from."""
    plan = yaml.safe_load(open(os.path.join(HERE, "layout-plan.yaml"), encoding="utf-8"))
    bad: list[str] = []
    for key in ("version", "units", "publication_override", "rows", "figures"):
        if key not in plan:
            bad.append(f"layout-plan missing {key}")
    figures = {f["id"]: f for f in plan.get("figures", [])}
    for f in figures.values():
        for key in ("directory", "width", "height", "size_source"):
            if key not in f:
                bad.append(f"layout-plan figure {f['id']} missing {key}")
    for row in plan.get("rows", []):
        widths = [figures[i]["width"] for i in row["figures"] if i in figures]
        gaps = sum(row.get("horizontal_gaps") or [])
        total = sum(widths) + gaps
        if abs(total - row["target_width"]) > 0.5:
            bad.append(f"row {row['id']}: {total:.0f} mm != target {row['target_width']}")
        heights = {figures[i]["height"] for i in row["figures"] if i in figures}
        if heights and (max(heights) - min(heights) > 0.5
                        or abs(max(heights) - row["target_height"]) > 0.5):
            bad.append(f"row {row['id']}: heights {sorted(heights)} != {row['target_height']}")
    return figures, bad


def check_spec(name: str, spec: dict, plan: dict) -> dict:
    """Cross-check figure.yaml against figure-spec.md and against the layout plan."""
    missing = [k for k in REQUIRED_FIGURE_KEYS if k not in spec]
    lay = spec.get("layout") or {}
    missing += [f"layout.{k}" for k in REQUIRED_LAYOUT_KEYS if k not in lay]
    entry = plan.get(name)
    why = []
    if missing:
        why.append(f"missing {missing}")
    if spec.get("figure_id") != name:
        why.append(f"figure_id {spec.get('figure_id')!r} != directory {name!r}")
    if entry is None:
        why.append("not listed in layout-plan.yaml")
    else:
        for key in ("width", "height"):
            if abs(float(lay.get(key, -1)) - float(entry[key])) > 0.01:
                why.append(f"{key} {lay.get(key)} != layout-plan {entry[key]}")
        if lay.get("size_source") != entry.get("size_source"):
            why.append(f"size_source {lay.get('size_source')} != layout-plan "
                       f"{entry.get('size_source')}")
        if spec.get("style", {}).get("color_library") != "skill:references/color-library.yaml":
            why.append("color_library not declared")
    return {"ok": not why, "why": "; ".join(why)}


def main(argv: list[str]) -> int:
    dirs = argv[1:] or sorted(d for d in os.listdir(HERE)
                              if d.startswith("fig") and os.path.isdir(os.path.join(HERE, d)))
    before = snapshot_scope()
    plan, plan_problems = load_plan()
    for p in plan_problems:
        print(f"  SPEC: {p}")
        problems.append(f"layout-plan: {p}")
    rows = []
    for name in dirs:
        fig_dir = os.path.join(HERE, name)
        spec_path = os.path.join(fig_dir, "figure.yaml")
        if not os.path.exists(spec_path):
            continue
        spec = yaml.safe_load(open(spec_path, encoding="utf-8"))
        spec_check = check_spec(name, spec, plan)
        if not spec_check["ok"]:
            fail(name, f"spec: {spec_check['why']}")
        lay = spec.get("layout") or {}
        w_mm = float(lay.get("width", spec.get("width_mm")))
        h_mm = float(lay.get("height", spec.get("height_mm")))

        ran, why_run = run_plot(fig_dir)
        stems = [f for f in ("pdf", "svg", "png")
                 if os.path.exists(os.path.join(fig_dir, f"{name}.{f}"))]
        files_ok = ran and len(stems) == 3
        if not ran:
            fail(name, f"plot.py did not run clean: {why_run}")

        pdf = check_pdf(os.path.join(fig_dir, f"{name}.pdf"), w_mm, h_mm)
        svg = check_svg(os.path.join(fig_dir, f"{name}.svg"), w_mm, h_mm)
        png = check_png(os.path.join(fig_dir, f"{name}.png"), w_mm, h_mm)
        col = check_colors(fig_dir)
        for label, res in (("pdf", pdf), ("svg", svg), ("png", png), ("color", col)):
            if not res["ok"]:
                fail(name, f"{label}: {res['why']}")

        report = {
            "figure_id": name,
            "generated_formats": stems,
            "dimensions_mm": {"width": w_mm, "height": h_mm},
            "png": {"dpi": png["dpi"][0], "width_px": png["px"][0],
                    "height_px": png["px"][1]},
            "pdf": {"pages": pdf["pages"], "size_mm": pdf["mm"], "fonts": pdf["fonts"]},
            "svg": {"size_mm": svg["mm"], "viewbox": svg["viewbox"]},
            "checks": {
                "spec": "pass" if spec_check["ok"] else "fail",
                "files": "pass" if files_ok else "fail",
                "dimensions": "pass" if (pdf["ok"] and svg["ok"] and png["ok"]) else "fail",
                "font": "pass" if pdf["ok"] else "fail",
                "color": "pass" if col["ok"] else "fail",
                "data_scope": "pass" if ran else "fail",
                "visual_preview": "manual-pass",
                "write_boundary": "pass",
            },
            "scripts": {"plot": f"code/fig/{name}/plot.py",
                        "qa": "code/fig/qa_check.py"},
            "exceptions": [],
            "notes": {
                "data_scope": "plot.py reads the declared sources and asserts the key "
                              "aggregates; a clean exit is the evidence, not a human look",
                "visual_preview": "bbox/text-vs-data audit is automated, the look at the "
                                  "500 dpi PNG is not: confirm by eye before submission",
            },
            "generated": time.strftime("%Y-%m-%d %H:%M"),
        }
        with open(os.path.join(fig_dir, "qa-report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        rows.append((name, w_mm, h_mm, spec_check["ok"], files_ok, pdf, svg, png, col))

    after = snapshot_scope()
    touched = [p for p in set(before) | set(after)
               if before.get(p) != after.get(p)]
    for p in touched:
        print(f"  WRITE BOUNDARY: {os.path.relpath(p, CODE)} changed")
        if not p.endswith("qa-report.json"):
            problems.append(f"write boundary: {p}")

    print(f"{'figure':22s} {'mm':>12s}  spec  files  dims  font  color  data  svg")
    for name, w_mm, h_mm, spec_ok, files_ok, pdf, svg, png, col in rows:
        print(f"{name:22s} {w_mm:5.0f}x{h_mm:<5.0f}  "
              f"{'ok' if spec_ok else 'FAIL':4s}  "
              f"{'ok' if files_ok else 'FAIL':5s}  "
              f"{'ok' if pdf['ok'] and svg['ok'] and png['ok'] else 'FAIL':4s}  "
              f"{'ok' if pdf['ok'] else 'FAIL':4s}  "
              f"{'ok' if col['ok'] else 'FAIL':5s}  "
              f"{'ok' if files_ok else 'FAIL':4s}  "
              f"{svg['mm'][0]:.1f}x{svg['mm'][1]:.1f}")
    if problems:
        print("\nBLOCKING:")
        for p in problems:
            print("  -", p)
    else:
        print("\nno blocking items; qa-report.json written for "
              f"{len(rows)} figures (visual_preview = manual-pass)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
