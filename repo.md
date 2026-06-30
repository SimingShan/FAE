# WFAE — repository layout (proposed)

**Thesis in one line:** whatever pixel-space methods do directly, the **FAE** does in *latent space from
sparse observations* — ONE frozen encoder, THREE jobs (**forecast · reconstruct · invert**), across 4 datasets
(NS · RBC · SW 2D, **adv3d** 3D).

## Design principles
1. **One entry, config-driven.** `python main.py <config.yaml>` — no per-script argparse. Every knob lives in a
   config; the config names the *stage* and the rest is data. A reviewer reads one YAML to understand a run.
2. **Library vs experiments vs runs.** `src/wfae/` is the importable library (pure, tested). `configs/` are the
   experiments (declarative). `results/`, `data/`, `external/` are gitignored (regenerated).
3. **Ours vs competitors are visibly separated.** `models/` = the FAE & its heads; `baselines/` = MAE/JEPA/FNO/
   DeepONet. No reviewer should wonder which is the contribution.
4. **The three jobs are first-class** — each downstream job is one stage + one config, sharing the frozen encoder.

```
WFAE/
├── main.py                     # SINGLE entry: reads a config, dispatches a stage. No argparse sprawl.
├── pyproject.toml              # deps + `pip install -e .` (exposes `wfae`)
├── README.md                   # what it is + how to run the 4 datasets × 3 jobs
├── CLAUDE.md                   # agent working brief (kept)
│
├── configs/                    # THE experiment definitions — one file == one run
│   ├── <ds>/                   #   ns · rbc · sw · adv3d
│   │   ├── base.yaml           #     dataset spec (res, channels, dt_max, label names)
│   │   ├── fae/                #     encoder PRETRAIN: default.yaml (+ ablations/)
│   │   ├── mae/  jepa/         #     baseline encoder pretrain
│   │   └── downstream/         #     forecast.yaml · reconstruct.yaml · invert.yaml (frozen-encoder jobs)
│   └── _archive/               #   dropped methods (dino …) — inactive, kept for provenance
│
├── src/wfae/                   # the importable library
│   ├── models/                 #   OURS
│   │   ├── fae.py              #     functional encoder + coordinate decoder + dt-predictor
│   │   ├── decoder.py          #     LatentDecoder — unified query decoder (reads ANY encoder's tokens)
│   │   └── operators.py        #     TokenPredictor (latent operator: attn + linear/Koopman)
│   ├── baselines/              #   COMPETITORS (own arch + own training)
│   │   ├── mae.py  jepa.py     #     SSL encoders
│   │   └── operators.py        #     FNO · pixel-DeepONet · grid-CAE (L-DeepONet)
│   ├── data/
│   │   ├── dataset.py          #     PDEDataset — 2D & 3D, reads data/<ds>/, normalizes
│   │   ├── prep.py             #     materialize Well/PDE-Arena (ns·shear·sw·rbc)
│   │   ├── generate.py         #     APEBench on-the-fly generation (adv3d)
│   │   └── coords.py           #     make_coords (2D/3D) + fields_to_tokens
│   ├── stages/                 #   the pipeline steps main.py dispatches to
│   │   ├── pretrain.py         #     SSL pretrain (FAE or a baseline)
│   │   ├── forecast.py         #     train latent operator + field-space error vs horizon/#sensors
│   │   ├── reconstruct.py      #     sparse-input reconstruction vs #sensors
│   │   └── invert.py           #     parameter probe (frozen encoder)
│   ├── eval/
│   │   ├── probe.py            #     THE probe: full-grid, train→test, RidgeCV (never in-log)
│   │   ├── metrics.py          #     relL2 · participation ratio · effective rank
│   │   └── figures.py          #     paper figures (sparse curve · ablation · cube viz)
│   └── utils/
│       └── seed.py  log.py  plot.py
│
├── results/                    # gitignored — checkpoints/ · figures/ · probes/  (regenerate)
├── data/                       # gitignored — materialized datasets data/<ds>/
├── external/                   # gitignored — latent-deeponet · the_well · apebench venv
├── arxiv/                      # archived earlier directions (REPA gen · flowbench · one-offs)
└── docs/
    └── RESULTS.md              # live canonical scorecard (the only numbers that count)
```

## The single-entry pattern
```yaml
# configs/ns/fae/downstream/reconstruct.yaml
stage: reconstruct                 # -> wfae.stages.reconstruct
dataset: ns                        # base merged from configs/ns/base.yaml
encoder: ns/fae/default            # frozen checkpoint to load
decoder: ns/fae/default            # trained recon decoder
sensors: [64, 128, 256, 512, 1024, dense]
```
`python main.py configs/ns/fae/downstream/reconstruct.yaml` → load frozen encoder+decoder → sweep sensors →
write `results/figures/ns/sparse_recon.png`. Identical shape for every dataset × job.

## Migration map (current → proposed)
| current | →  proposed |
|---|---|
| `scripts/train/train_fae.py`, `train_baseline.py` | `stages/pretrain.py` (config-selected model) |
| `scripts/downstream/{train_operator,eval_forecast}.py` | `stages/forecast.py` |
| `scripts/downstream/{train_decoder,eval_reconstruct}.py` | `stages/reconstruct.py` |
| `scripts/eval/reprobe_canonical.py` (drop the rest) | `eval/probe.py` |
| `scripts/eval/fig*_*.py`, `viz_*` (24 files) | `eval/figures.py` (keep ~5, archive one-offs) |
| `scripts/L_forecast/{train_fno,train_pixel_deeponet,train_grid_ae}.py` | `baselines/operators.py` + `stages/forecast.py` |
| `scripts/L_forecast/viz_*` (7 files) | `arxiv/` |
| `src/{fno,deeponet,grid_ae,latent_op}.py` (loose root) | `models/operators.py` + `baselines/operators.py` |
| `src/{encoders,eval,sparse,config,plotstyle}.py` (loose root) | `models/decoder.py` / `eval/` / `utils/` |
| `benchmarks/{mae,jepa}` → `src/wfae/baselines/` ; `benchmarks/dino` | `arxiv/` |
| `results/figs` **and** `results/figures` (dup) | single `results/figures/` |
| `summary.md` · `mentor_requirements.md` · `request_devgpu.md` · `slurm-*.out` | `arxiv/notes/` or delete |
| `requirements.txt` + `requirements_cluster.txt` + `pyproject.toml` | `pyproject.toml` only |

## Safe deletions (regenerable or dead)
- `slurm-2036330.out`, `slurm-2037900.out` (orphan logs at repo root)
- `benchmarks/dino/`, `configs/*/dino/` → `_archive` (dropped: invariance-SSL ill-posed for PDE fields)
- the ~20 one-off `scripts/eval/` probes/viz superseded by `eval/probe.py` + `eval/figures.py`

## Migration status
**DONE (2026-06-30, this branch — safe, no behavior change, all scripts verified to parse + import):**
- ✅ **Archived dead code** → `arxiv/dead_*`: `scripts/eval/` 24→2 (kept `reprobe_canonical`), `L_forecast/`
  12→5 (kept the 4 operator trainers), `benchmarks/dino` + `configs/*/dino`, dead `src/sparse.py`, stray
  top-level docs (`summary.md`, `mentor_requirements.md`, …) + orphan `slurm-*.out`.
- ✅ **Grouped `src/` root** 9 loose → 2: operators → `src/baselines/` (fno·deeponet·grid_ae·latent_op);
  `plotstyle`+`config` → `src/utils/`; only `encoders.py`+`eval.py` (active interfaces) remain at root.
- ✅ **Dedup** `results/figs` + `results/figures` → one `results/figures/`.

**PENDING (the higher-churn rename — do AFTER current experiments settle so nothing in-flight breaks):**
1. **package + entry** — `src/` → `src/wfae/`, add `main.py` + `pyproject.toml` exposing `wfae`.
2. **stages** — wrap the 6 trainers/evals (`train_fae`, `train_baseline`, the 4 downstream) as `stages/*`
   callables `main.py` dispatches to; keep old scripts as thin shims first.
3. **configs** — lift every argparse default into `configs/<ds>/.../*.yaml`; delete the shims.
4. **fold baselines** — `benchmarks/{mae,jepa}` → `src/wfae/baselines/` (the `build_model` factory is the one
   wide dependency — defer until the rename pass).
