# WFAE — methods, benchmarks, and results

**Thesis.** Whatever pixel-space methods do directly, the Functional AutoEncoder (FAE) does in
**latent space from sparse / irregular observations**. One frozen encoder, evaluated by a linear probe
of physical properties (the *invert* job), plus observation-invariance (ICC) and reconstruction — all
as a function of **sensor count**. The differentiator is the **sparse regime**, where fixed-grid ViTs
must interpolate to their grid and the coordinate encoder does not.

---

## 1. Encoders (one frozen encoder; decoder/predictor discarded at eval)

| encoder | paradigm | params | where |
|---|---|---|---|
| **FAE** (ours) | functional / coordinate (Perceiver-Senseiver) | **7.49M** | `src/models/fae.py`, `scripts/train/train_fae.py` |
| **MAE** | masked reconstruction (Kaiming) | 8.93M | `benchmarks/mae/mae.py`, `scripts/train/train_baseline.py` |
| **I-JEPA** | latent prediction (Assran et al.) | 8.36M | `benchmarks/jepa/ijepa2d.py` |

Param note: the ViTs are *larger* than the FAE (depth-6 vs FAE depth-5), so any FAE tie/win is
**conservative** (achieved with fewer parameters).

### FAE architecture
Perceiver-style set encoder. Input = K sensors as `(value ⊕ Fourier(coord))` tokens. Cross-attention
into **128 learnable latents**, then `num_iter=4` iterations of (cross + `depth_per_iter=5` self-attn
blocks); `emb_dim=320`, 4 cross-heads / 8 self-heads, `n_freq=max_freq=32`. Decode = cross-attention
from query-coordinate tokens to the latents → field values at arbitrary coordinates (resolution-free).
`TokenPredictor` (a small transformer) advances the latent by a normalized horizon `dt` (temporal head).

### MAE / I-JEPA
Standard ViTs, **rectangular-aware** (pos-embed + patch(un)patchify from a `(gh, gw)` grid), so shear is
ingested at native **128×256** (16×32 = 512 patch tokens), not squished to square. Probe input = the
**dense field** (MAE `mask_ratio=0`; JEPA target encoder), GAP over patch tokens. Masking/context are
pretraining-only.

---

## 2. FAE objective cells (the ablation)

| cell | mode | loss |
|---|---|---|
| **Senseiver** | `recon` | present recon: encode sparse x_t → decode x_t |
| **+temporal** | `recon_both` | + future recon: predictor(dt) → decode x_{t+dt} (non-collapsible) |
| **+dual-view** | `twoview_present` | recon + a 2nd sparsity view sharing the x_t target (invariance) |
| **full FAE** | `twoview` (default) | dual-view + temporal |

---

## 3. Datasets / data structure

| dataset | channels | probe target | resolution | source |
|---|---|---|---|---|
| **shear_flow** | 4 — tracer, pressure, vx, vy | logRe, logSc | **128×256** (native 1:2) | The Well |
| **NS-2D-conditioned** | 3 — smoke, vx, vy | buoyancy | 128² | PDE-Arena (SSLForPDEs) |
| **typhoon** | 1 — IR | wind, pressure | 128² | Digital Typhoon |

Datasets yield clips `(B, C, T, H, W)`; `fields_to_tokens(field, sensor_idx)` gathers K sensor values
at flat indices, paired with `make_coords_2d_hw` coordinates. Splits: **shear** train→valid (valid is too
small — 4 traj/file — for an internal split); **typhoon/NS** valid→test.

---

## 4. Probe / eval protocol (`scripts/eval/`)

- **Canonical linear probe** (`src/eval.py`, `probe_all.py`): frozen encoder → **mean-pool** tokens →
  StandardScaler + **RidgeCV** on **standardized** labels, train→test split. **Trivial floor first**
  (channel mean+std). **Participation ratio (PR)** = collapse guard.
- **Sensor sweep** (`sweep_sensors.py`): probe-R² and ICC vs #sensors `[64,256,1024,4096,16384(,32768)]`.
  FAE ingests K sensors natively; ViTs get them **linear-interpolated to the grid** (`src/sparse.py`).
  **Invariant**: the full-grid endpoint *equals* the dense probe (asserted), with the test split sharing
  the fit split's normalization stats.
- **Cross-encoder ICC** (`icc_encoders.py`): encode each field with two sensor draws; ICC = between-field
  / (between + within-draw) variance — observation-invariance.

---

## 5. Results

### 5a. Dense probe — R² on standardized labels (best fair split)
| encoder | shear logRe / logSc | typhoon wind / pressure | NS buoyancy |
|---|---|---|---|
| FLOOR | 0.073 / 0.061 | 0.007 / 0.019 | −0.423 |
| FAE · Senseiver | 0.447 / 0.283 | **0.602 / 0.553** | 0.876 |
| FAE · +temporal | 0.476 / 0.267 | 0.512 / 0.433 | **0.935** |
| FAE · +dual-view | **0.486 / 0.317** | 0.567 / 0.510 | 0.884 |
| FAE · full | 0.480 / 0.313 | 0.522 / 0.444 | **0.935** |
| **MAE** | 0.489 / 0.320 | 0.587 / 0.537 | 0.892 |
| **JEPA** | 0.283 / 0.174 | 0.589 / 0.523 | 0.724 |

**Read:** dense probe is a **tie** on shear/typhoon (FAE ≈ MAE; on shear MAE even edges it once given the
native aspect) and an **FAE win on NS**. JEPA is weakest (esp. shear). *Which* FAE cell wins is
dataset-dependent: +dual (shear), Senseiver (typhoon), +temporal/full (NS) — temporal helps where the
dynamics are rich (NS) and *hurts* where they are weak (typhoon).

### 5b. Sparse regime — probe R² vs #sensors (endpoints verified == dense)
| dataset | @64 sensors | full grid | FAE sparse advantage |
|---|---|---|---|
| **NS** (sharp smoke+velocity) | FAE **0.71** ≫ MAE 0.10, JEPA −0.18 | 0.94 / 0.89 / 0.59 | **strong** |
| **shear** (turbulent 128×256) | FAE **0.45** > MAE 0.35 | 0.48 / 0.49 (MAE crosses ~1024) | mild |
| **typhoon** (smooth IR) | FAE 0.37 < MAE 0.39 | 0.51 / 0.59 | none / reversed |

This is the headline: the FAE’s **sensor-efficiency** advantage scales with how badly the field defeats
interpolation — strong on sharp NS, mild on shear, absent on smooth typhoon IR (where griddata works
fine and the grid ViT is not handicapped).

### 5c. Observation-invariance ICC (FAE / MAE / JEPA), @64 sensors → full grid
| dataset | FAE | MAE | JEPA |
|---|---|---|---|
| NS | **0.965** → 1.0 | 0.826 | 0.847 |
| shear | **0.980** → 1.0 | 0.967 | 0.835 |
| typhoon | 0.811 → 1.0 | 0.863 | 0.820 |

FAE most observation-invariant on NS/shear; comparable on typhoon; JEPA worst.

### 5d. Reconstruction (fig1, relL2 — frozen encoder → decode)
shear FAE 0.131 (256 scattered sensors = 1.6% of pixels) vs MAE 0.074 (75%-masked patches = 25%);
NS FAE 0.209 vs MAE 0.200; typhoon FAE 0.519 vs MAE 0.403. The FAE reconstructs from ~16× fewer,
*arbitrarily-placed* observations.

### 5e. Forecasting (L-DeepONet extension, NS — `scripts/L_forecast/`)
FNO dense full-grid oracle ceilings at **relL2 ≈ 0.265** (modes 12→20, 2.8× params, no change) and beats
persistence (0.665) by ~60%. Latent forecasting (FAE encoder → operator → decode) is the sparse/irregular
extension; the grid-CAE + FlatOperator + FNO are the architecture-axis baselines.

---

## 6. Figures (per dataset, `results/figs/<ds>/`)
fig1 recon · fig2 training (loss + in-loop probe) · fig3 probe-vs-sensors · fig4 ICC (cells +
cross-encoder) · fig5 ablation · fig_interp_input (what the ViTs ingest). All endpoint-/structure-verified.

## 7. Standardized experiment setup
128 resolution / patch-8 ViTs / depth-5 FAE (7.49M) / 200 epochs; shear at native 128×256 for **all**
encoders (fair-by-construction). Config-driven (`configs/<ds>/<method>/<setup>.yaml` → `run.py` → trainer);
nothing hardcoded.
