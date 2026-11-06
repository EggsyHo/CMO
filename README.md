# Conservation-Exact Multiscale Operator Learning for Sharp-Front Transport Fields

Code and figure sources for the manuscript of the same title (under review, 2026).

The model learns the advection--diffusion--reaction operator on a coarse grid, carries a
mass-exact partition of unity over an overset of translated windows, and reaches
time-step-independent transport by predicting a *relative* (per-step) flux rather than an
absolute one. The repository contains everything needed to regenerate the datasets,
retrain every arm, recompute the reported metrics, and redraw every figure.

## Layout

```
code/
├── src/              model, training loop, evaluation, solvers
│   ├── models/       conservation-exact operator (windows, partition, flux heads)
│   ├── train.py      training entry point (config + CLI overrides)
│   ├── evaluate.py   rel-L2 / mass metrics on test and extrapolation splits
│   └── ...
├── configs/          one YAML per arm; `default.yaml` is the full model
├── data/
│   ├── gen_adr2d.py  ADR2D dataset generator (finite-volume reference solver)
│   ├── selfcheck_solver.py  solver convergence self-check (Table 1)
│   └── check_data.py        dataset sanity checks
├── scripts/          experiment drivers (one-off analyses, seed sweeps, transfer, Δt band)
├── tests/            numerical unit tests (run directly, no pytest needed)
├── fig/              figure sources: figlib.py, one directory per figure, QA gate
└── requirements.txt
```

## Install

Python 3.11 was used for the paper. Install a torch build that matches your CUDA runtime
first, then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Everything runs on a single GPU; a training run peaks at ~192 MB of VRAM.

## 1. Data

```bash
bash scripts/gen_data.sh          # writes data/adr2d/*.h5 (~4.4 GB in total)
```

The script runs `data/selfcheck_solver.py` first and aborts if the reference solver is not
converged. Generation is solver-bound and takes a while; it is a one-off.

The Δt pair used for the timestep band (Table 3, Fig. 5) is deliberately *not* part of
`gen_data.sh`, because both files must share the same 80 parameters and differ only in
`dt`. The reference solutions agree to 0.17% at t=20, so any error change is attributable
to the model:

```bash
python data/gen_adr2d.py --regime train --n 80 --T 200  --dt 0.100 --seed 0 --out data/adr2d/adr2d_dtcoarse.h5
python data/gen_adr2d.py --regime train --n 80 --T 1250 --dt 0.016 --seed 0 --out data/adr2d/adr2d_dtfine.h5
```

## 2. Train

```bash
python src/train.py --config configs/default.yaml --set run.seed=0
```

One run takes ~14 min on one GPU (308k parameters) and writes
`results/<tag>_s<seed>/<run_id>/`: the resolved config, `train_info.json` (wall clock, peak
VRAM, parameter count), and `best.pth` (~1.2 MB). The paper uses seeds 0--4 for every arm:

```bash
for arm in default cmo_nosep cmo_absflux cmo_af_nosep; do
  for s in 0 1 2 3 4; do
    python src/train.py --config configs/$arm.yaml --set run.seed=$s
  done
done
```

`configs/default.yaml` is the full model (relative flux, detail branch). The other arms are
single-axis deletions of it; `configs/baseline_*.yaml` hold the comparison operators and
`configs/ablation_*.yaml` the earlier four-way ablation.

## 3. Evaluate

```bash
python src/evaluate.py --ckpt results/<run_id>/best.pth --modes test ext
```

Metrics land in the run directory as JSON; the analysis scripts in `scripts/` aggregate
them into the JSON/CSV files the figures read.

## 4. Figures and tables

Figures 1--4, 6 and S2 are drawn from the committed metrics and can be redrawn without
retraining or without the datasets:

```bash
python fig/fig1_main/plot.py        # any figure: fig/fig<N>_*/plot.py
python fig/qa_check.py              # gate: size, single page, fonts, data sandbox
```

Figure S1 is the exception: it recomputes per-case errors from a stored rollout
(`results/_zshot/kappa_*/rollout_test.npz`), which is not committed; regenerate it with
`scripts/kappa_zeroshot.py` after training the four arms. The scripts that touch weights or
rollouts are marked by reading `*.pth` / `*.npz`.

Each figure directory holds `plot.py`, `layout-plan.yaml`, `analyze.json` (or an
equivalent expectation file), and the rendered `*.pdf` / `*.png` / `*.svg`.

Note the numbering: the directories keep the number the figure had before the framework
panel was added, so `figN_*` is the manuscript's Figure N+1.

| Manuscript | Data comes from | Directory |
|---|---|---|
| Fig. 1 framework | schematic, no data | `fig/fig0_framework` |
| Fig. 2, Table 1 | `configs/default.yaml`, `configs/baseline_*.yaml` | `fig/fig1_main` |
| Fig. 3 error decomposition | `scripts/lag_decomposition.py`, `scripts/centroid_drift.py` | `fig/fig2_decomposition` |
| Fig. 4, Table 2 (2×2) | `default`, `cmo_nosep`, `cmo_absflux`, `cmo_af_nosep` | `fig/fig3_ablation` |
| Fig. 5, Table 3 (Δt band) | Δt pair above + `scripts/dt_only_band.py`, `scripts/bound_dt_*.py` | `fig/fig4_dtband` |
| Fig. 6, Table 4 (retrain) | `scripts/ext_seeds_*.py`, `scripts/dt_only_band.py` | `fig/fig5_retrain` |
| Fig. S1 (κ=20, zero-shot) | `scripts/kappa_zeroshot.py` | `fig/figS1_scan` |
| Fig. S2 (extrapolation) | `scripts/ext_zeroshot.py`, `scripts/ext_seeds_full.py` | `fig/figS2_extrap` |
| Table 5 (resolution transfer) | `scripts/resolution_transfer.py` on `adr2d_128test.h5` | — |

## Tests

```bash
python tests/check_units.py     # conservation, partition, flux algebra
python tests/check_width.py     # window/overlap geometry
python tests/check_train.py     # one-step training smoke test
python tests/check_data.py      # dataset invariants
```

## What is in the repository, and what is not

Committed: source, configs, data generators, figure code and rendered figures, the per-run
metrics (`results/**/*.json`, `*.csv`, `*.yaml`) that the figures and tables are drawn from,
and the `best.pth` checkpoint of every run (50 files, 1.2 MB each, 308k parameters). Every
reported number can therefore be re-derived, and every figure that is metric-only redrawn,
without retraining; the checkpoints cover the runs that need a rollout (Fig. S1).

Not committed, because it is regenerable and large: the datasets (11 files, ~10 GB, archived
separately with a DOI in the paper), the rollout field dumps (~38 GB), and run logs.
`scripts/gen_data.sh` and the training loop above reproduce all three.

## Reproducibility notes

* Every reported metric is aggregated by the figure/analysis scripts from the committed
  per-run JSON, not transcribed by hand.
* Paired comparisons in the paper are per-seed differences over the ten matched pairs of
  the 2×2 design (five seeds per cell).
* The Δt band and the retraining table use the same 80 cases as the in-band evaluation;
  no split overlaps training.

## License

MIT, see `LICENSE`.

## Contact

Chenze He, hecz@mail.ustc.edu.cn · Boya Zhou, zby0529@mail.ustc.edu.cn
