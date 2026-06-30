# Repo structure (post cleanup, 2026-06-27)

The paper: **whatever pixel methods do directly, FAE does in latent space from sparse observations**
— one frozen encoder (FAE vs MAE vs JEPA), evaluated by the figures in `scripts/eval/`.

## scripts/
```
gpu.slurm                         generic launcher: sbatch ... scripts/gpu.slurm python <script> <args>

train/                            PRETRAINING (the encoders)
  train_fae.py                    FAE — modes: recon | recon_both | twoview_present | twoview (default)
  train_baseline.py               MAE / I-JEPA (matched ~ViT; native-aspect for shear)
  run.py                          config-driven harness: configs/<ds>/<method>/<setup>.yaml -> command
  run_config.slurm

eval/                             PRETRAINING EVAL (these figures = all encoder-side eval)
  probe_all.py                    canonical linear probe (src.eval: mean-pool, RidgeCV, valid->test, PR, floor)
  sweep_sensors.py                producer: probe-R2 + ICC vs #sensors (endpoint == dense probe, asserted)
  fig1_recon  fig2_training  fig3_probe_sensors  fig4_icc_sensors  fig5_ablation
  icc_encoders (FAE/MAE/JEPA ICC)  fig_interp_input (what the ViTs ingest)

L_forecast/                       DOWNSTREAM forecast job (L-DeepONet extension; separate phase)
  train_forecast · train_grid_ae · train_pixel_deeponet · train_fno · viz_* · gen_sw · merge_sw · viz_sw*

data/                             download_ns · prep_typhoon · download_*.slurm
```

## src/ · benchmarks/
```
src/   models/fae.py · data/{well2d,ns,typhoon,flowbench}.py · metrics/probes.py · eval.py · plotstyle.py
       (L-DeepONet modules: deeponet.py · fno.py · grid_ae.py · latent_op.py)
benchmarks/ mae/mae.py · jepa/ijepa2d.py   (rectangular-aware; MAE/JEPA both square + 128x256)
```

## Rules (each cost real time)
1. **Probe = `src.eval`** (one canonical procedure); never reimplement. Trivial floor FIRST. PR = collapse guard.
2. **Nothing hardcoded** — every knob flows config -> run.py cmd -> trainer.
3. **Sensor sweep**: full-grid endpoint MUST equal the dense probe (asserted); test split shares fit stats.
4. **Param/geometry matched** by construction; shear is native 128x256 for ALL encoders (fair).

`results/`, checkpoints, data, figures are gitignored (regenerate). `arxiv/superseded/` = cleaned-out
scripts (reimplemented probes, old viz, broken REPA evals); `arxiv/{pre_repa_pivot,repa_generation}/` = prior phases.
