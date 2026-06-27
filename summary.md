# WFAE — Functional AutoEncoder for PDE Representation Learning: Status & Open Problem

> Self-contained summary for an outside reader. The honest open question is in §6 — that's the part
> we most want help thinking about. Everything before it is context.

---

## 1. What we're building, and the question we're stuck on

**FAE (Functional AutoEncoder)** is a coordinate-native, set-latent encoder for 2D PDE/physical
fields. It was first introduced in the **FunDiff** paper (Wang et al., for *generation*); we want a
**separate paper that defines FAE as a representation-learning method** and shows where it is
*necessary* (not merely competitive) vs grid-based encoders (ViTs).

**The unresolved question (the whole point of this doc): _why does anyone need representation
learning for PDEs at all?_** We have strong technical results showing FAE is a good encoder, but we
cannot yet articulate a compelling, honest motivation for the paradigm. See §6.

---

## 2. Architecture

| encoder | latent | paradigm | params |
|---|---|---|---|
| **FAE (ours)** | **128-token SET** (Perceiver: learned latents cross-attend to N input sensors); a **coordinate decoder** queries arbitrary (x,y) coords that cross-attend the latents → field value | functional / coordinate; resolution-free; ingests *scattered* points | ~7.5M |
| **MAE** | 256 patch tokens on a 16×16 **grid** | masked reconstruction | ~7.5M |
| **I-JEPA** | 256 patch tokens on a grid | latent prediction (no pixel decoder) | ~7.5M |

All matched: emb_dim 320, res 128, 200 epochs, 3 seeds, same data budget.

**FAE's training objective ("twoview"):** dual sparse views (observation-invariance) + a latent
temporal predictor (set→set, Δt-conditioned) + dual reconstruction target (present + future field).

**Key architectural fact:** FAE's latent is a **permutation-style set with consistent slot identity
but NO spatial-grid index** (slot-5 is a fixed learned query, not "the top-left patch"). This is its
strength (resolution-free, sparse, invariant) and the source of all the generation friction below.

**Data:** NS-2D-buoyancy (PDE-Arena; 3 channels smoke/vx/vy; probe target = buoyancy). shear_flow
(The Well; 4 channels; probe logRe/logSc). Real data downloaded: **Digital Typhoon** (58 GB real IR
satellite, 1099 typhoons, 512×512).

---

## 3. Trustworthy results (representation quality)

### 3.1 Linear probe (frozen encoder → RidgeCV → physical parameter), NS-buoyancy, 3 seeds
**FAE 0.929 ± 0.005 > MAE 0.905 ± 0.009 > JEPA 0.656 ± 0.057** (R²; trivial floor −0.423).
A small but statistically real FAE win over a *strong, properly-trained* MAE. (Earlier confounded
runs had MAE at 0.53; fixing resolution/data/schedule jumped it to 0.91 — every earlier comparison
was confounded.)

### 3.2 Ablation — what actually drives FAE (2×2: dual-view × temporal predictor)
On the **full-grid probe**: Senseiver (−dual −temporal) 0.871 · +temporal 0.928 · +dual 0.878 ·
full FAE 0.929. **Temporal prediction drives accuracy (+0.06); dual-view barely moves it (+0.007).**
So `Senseiver + temporal` ≈ full FAE at ~half the training cost.

### 3.3 Sensor sweep (eval-time, frozen FAE, vary # sensors fed in)
FAE probe R²: 64→0.706, 256→0.876, **512→0.906 (≈ MAE's full-grid 0.905)**, 1024→0.919, full→0.929.
**FAE matches a full-grid ViT using ~3% of the points and beats it at ~6%. ViTs cannot ingest fewer
than the full grid at all** — they appear as single points. This is FAE's clearest "necessity" axis.

### 3.4 Observation-invariance (ICC = between-field var / (between + within-view var))
Same field, 8 partial views, does the latent encode the *field* or the *sampling*?
- ICC: **FAE 1.000**, JEPA 0.920, MAE 0.840.
- Mean-**centered** cosine (rules out collapse): FAE same-field 0.999 / different-field **0.028**
  (near-orthogonal → fields cleanly separated, NOT collapsed; raw 0.944 was a shared-offset artifact).
- Across budgets: FAE ICC stays ~0.95–1.0 down to 64 sensors; MAE/JEPA collapse (≈0.45/0.63 @1024).

**BUT — important caveat from the ablation:** a **bare Senseiver already has ICC ≈ 0.93–1.0**.
So **observation-invariance is a property of the coordinate ARCHITECTURE, not of FAE's twoview
objective.** Dual-view adds essentially nothing to invariance. (Temporal slightly helps it.)

---

## 4. Generation attempts and what we learned (mostly negative, but clarifying)

### 4.1 REPA (align a pixel-space DiT's per-patch tokens to a frozen encoder's per-patch features)
**REPA structurally does not fit FAE.** REPA's loss is a *position-wise* cosine: SiT-patch-i ↔
encoder-token-i must be the *same spatial location*. MAE/JEPA tokens ARE grid patches → fine. FAE's
slots have **no spatial index** → no correspondence. To force it, we decode FAE at patch-center
coords — but that decoder readout is **one linear layer from the pixels** (near-pixel, not a semantic
representation).

Measured consequence: FAE's SiT aligns to its near-pixel target at **cosine ≈ 0.95 (near-trivial → ≈
pixel-DiT)**; MAE's at **≈ 0.58 (real semantic guidance)**. Result: **MAE-REPA now beats FAE-REPA**.
(An *earlier* run showed FAE winning — but that was with an *undertrained* MAE (probe 0.53) at res 64;
aligning to a bad encoder *hurts*, so MAE fell below even pixel. The "FAE wins REPA" was a
baseline-quality artifact, now overturned.)

### 4.2 RAE / FunDiff (diffuse IN the encoder latent, then decode)
We initially thought a set latent was awkward to diffuse — **wrong.** FunDiff (FAE's origin paper)
diffuses the Perceiver set latent directly: treat the 128 tokens as a **sequence**, add a **1D
positional embedding** over the latent indices, run a standard **DiT (self-attn + AdaLN)**, decode via
the coordinate decoder. **Diffusion is latent-internal — it only needs consistent slot identity, not a
spatial grid.** (REPA fails because it's a *cross-model spatial* alignment; FunDiff works because it's
*latent-internal generation*. Different requirements.) **So latent diffusion IS FAE's native
generative mode — but it's already published (FunDiff), so it can't be our novelty.**

---

## 5. Key conceptual insights (the through-line)

1. **FAE is set-native; REPA & grid-RAE are grid-native.** The set vs grid mismatch breaks REPA
   (spatial alignment) but NOT latent diffusion (FunDiff). Geometry lives in FAE's *decoder*
   (coordinate queries), not its latent — that's why it's resolution-free.
2. **The probe is the only *fair* set-vs-grid comparison.** Freeze + linear-probe is type-agnostic.
   Generation comparisons across set-vs-grid are confounded (different diffusion machinery + decoders),
   so "FAE generates better" is structurally hard to claim cleanly.
3. **FAE's headline advantages are ENCODER properties** (sparse-ingestion, observation-invariance,
   resolution-free) — and observation-invariance is the *architecture*, not the objective.
4. **FAE is necessary (not just better) only in the sparse / irregular / heterogeneous regime** —
   where ViTs cannot operate at all.

---

## 6. THE OPEN PROBLEM — why representation learning for PDEs? (please think about this)

This is where we're genuinely stuck. The standard motivation for representation learning
(label-scarcity + many downstream tasks + structure you can't write down) **largely fails for clean
simulated PDEs**, because:
- **you usually have the governing equations** (you don't need to *learn* the physics),
- **you can simulate** (labels are not scarce — generate them), and
- **direct supervised neural operators (FNO/DeepONet) exist** (why pretrain a representation?).

**Worse, our own probe is artificial:** it predicts *buoyancy* from a simulated field — but buoyancy
is a **known input to the simulation**. We're "inferring" something nobody needs to infer. The 0.929
is a fine *proxy for representation quality* but does **not** answer "why."

Representation learning for PDEs is only genuinely motivated where those three break:
- **(a) Real/observational data, unknown/unclosed physics** (turbulence closure, climate, experiment) —
  can't simulate the answer.
- **(b) Sparse, heterogeneous, multi-source observations** (satellite + buoy + radar, mixed
  resolutions/geometries) — no solver makes that distribution, no fixed grid ingests it. **This is the
  only regime where FAE's necessity and the "why" coincide.**
- **(c) Expensive high-fidelity sim → label/compute scarcity** → pretrain on cheap/unlabeled, transfer
  few-shot.

**Candidate honest "why" + the experiment that would prove it:**
> Representation learning is needed for PDEs where the toolchain breaks — real, sparse, heterogeneous
> observation — because there's no full-grid sim to supervise an operator, no grid for a ViT, and the
> true state/parameters are scarce or unavailable. A self-supervised coordinate-native encoder turns
> incomplete measurements into a representation that is **label-efficient** and **observation-invariant**.

Proof = a **label-efficiency curve in the sparse regime** (x = # labeled examples, y = downstream
accuracy): FAE-SSL-pretrained → few-shot probe should **dominate supervised-from-scratch in the
low-label regime** (the representation-learning value) **while grid baselines can't even run** on
sparse input (the FAE necessity). Our current probe sits at the far-right (abundant labels) where the
honest answer is "you don't need it."

**A possibly-more-honest task than probing:** **forecasting from sparse observations** — predict the
*future* field from a handful of scattered sensors. The future is genuinely unknown (can't "type it
into the solver"), it's FAE-necessary (grid models need full frames), and it's not FunDiff
(conditional, not unconditional generation).

---

## 7. Strategic state & options

- **Generation is taken**: FunDiff (latent diffusion), GeoFunFlow (geometry). REPA doesn't fit FAE.
- So the new paper must be **FAE as a representation method**, on a task where FAE is *necessary*.
- Candidate applications (representation/discriminative, not generation):
  1. **Cross-observation transfer** — probe trained on sensor-config A works zero-shot on config B
     (FAE invariant; ViT representation changes with grid). Cheap, runnable now.
  2. **Sparse-observation forecasting / inverse / data assimilation** — recover state/params/future
     from sparse, real, time-varying obs.
  3. **Heterogeneous foundation pretraining** — one FAE across mixed resolutions/geometries that ViTs
     can't co-train on. Biggest claim, biggest data lift.

---

## 8. Difficulties / risks (honest list)

1. **The "why" (§6) is unresolved.** Without it, the strong technical results lack a reason to exist.
2. **Our benchmark predicts a known simulation parameter** → doesn't demonstrate the "why."
3. **Observation-invariance is the architecture, not FAE's objective** (Senseiver already has it) — so
   "FAE's twoview objective" is hard to credit; the win is "coordinate encoders > grid encoders," for
   which a Senseiver might suffice.
4. **FAE doesn't plug into REPA/RAE as a peer** — set vs grid; generation comparisons are confounded.
5. **The clean-sim regime favors competitors** (supervised operators); FAE's case needs real/sparse/
   heterogeneous data, which is a data/engineering lift and harder to benchmark cleanly.
6. **Competition**: PDE foundation models (Poseidon, MPP) and FunDiff/GeoFunFlow already occupy
   adjacent ground.

**Bottom line:** we have a clean *representation-quality* win and a clear *necessity* axis (sparse),
but no resolved *motivation* for why PDEs need learned representations. Cracking §6 — ideally a
sparse/few-shot/real-data task where the answer is genuinely unavailable — is the blocker.
