# ChatGPT image prompts — REPA vs Reverse-REPA, explained in 5 images

A **sequence** of 5 single-idea images that build up the concept. Paste them **into the same ChatGPT
conversation, one at a time, in order** — that way it keeps the visual style consistent across the set.
If a label renders garbled, just say "regenerate, fix the text." Keep labels short.

## Shared style (state once; ChatGPT will carry it through the series)
> Flat **vector infographic**, scientific/textbook (Distill.pub) aesthetic. White background, crisp
> sans-serif labels, thin clean outlines, soft rounded corners, lots of whitespace, landscape.
> **Fixed palette for the whole series:** BLUE = grid / ViT / "discretized space"; ORANGE = set / FAE /
> "function space"; GREEN = valid alignment; RED = invalid / broken; GREY = the frozen FAE "lens."
> **Two token motifs, used consistently:** a *grid token* is a BLUE rounded chip with a 📍 location-pin
> and a position number; a *set token* is an ORANGE rounded chip with a ❓ question-mark and a slot
> number. Tokens are drawn as chips/vectors, NOT as literal squares.

---

## IMAGE 1 — "Two ways to represent a field" (the root difference)

> Title: **"Same field, two representations."** White background, flat vector style (use the shared
> palette). In the CENTER-LEFT draw a single continuous wavy 2D field surface labeled **`u(x,y)` — a
> continuous function**. From it, two branches go right:
> **Top branch (BLUE) — ViT / "grid space":** an arrow "chop into patches" to a 4×4 blue checkerboard,
> then an arrow "embed" to a neat ROW of ~6 blue token-chips, each with a 📍 pin and a position number
> (1,2,3…). Branch label: **"GRID tokens — tagged by POSITION (patch i = place i)."**
> **Bottom branch (ORANGE) — FAE / "function space":** an arrow "ask 128 fixed questions" to a loose
> floating CLOUD of ~6 orange token-chips, each with a ❓ and a slot number, drawn scattered/unordered
> (no grid). Branch label: **"SET tokens — tagged by ROLE (slot i = a fixed question, NO position)."**
> Bottom caption (bold): **"ViT discretizes the function onto a grid. FAE keeps it functional — a set
> of role-slots you can decode at any coordinate."**

## IMAGE 2 — "How REPA works, and why ViT fits"

> Title: **"REPA: match by POSITION."** Same style/palette. LEFT: a stack labeled **"DiT (image
> generator)"** with, at a middle layer, a horizontal ROW of BLUE grid token-chips (📍 1…6) labeled
> **"DiT patches."** RIGHT: a box labeled **"frozen ViT encoder (DINOv2 / MAE)"** fed a small clean
> image, outputting an identical ROW of BLUE grid token-chips (📍 1…6) labeled **"encoder patches."**
> Between them, **straight parallel GREEN arrows** connecting chip-1↔chip-1, 2↔2, … 6↔6, with a big
> green ✓. Speech-bubble from the arrows: **"each DiT patch: 'become what the encoder sees HERE.'"**
> Caption: **"Both sides are position-tagged grids → patch i ↔ patch i. Dense, local, works."**

## IMAGE 3 — "Why REPA can't use FAE"

> Title: **"REPA + FAE: a place can't point to a role."** Same style. LEFT: the BLUE **"DiT patches"**
> grid row (📍 1…6). RIGHT: an ORANGE **"FAE latent — 128 role-slots"** floating cloud (❓ 1…6, no
> position). Draw arrows from grid chips to set chips but make them **crossing, tangled RED** with a big
> **red ✗**. A small inset at bottom labeled **"the trap"**: an arrow "decode FAE at patch centers →"
> leading to a tiny pixel image with a thermometer reading **"cosine 0.95 — basically the pixels."**
> Caption: **"FAE slots have a ROLE, not a place — nothing for a position-arrow to land on. Forcing it
> through the decoder collapses to pixel-matching (trivial)."**

## IMAGE 4 — "Reverse-REPA: match by ROLE"

> Title: **"Reverse-REPA: bring the DiT into FAE's set."** Same style. Show TWO parallel rows feeding a
> shared GREY funnel/lens labeled **"frozen FAE 'lens' — the 128 fixed queries"**:
> ROW A: a small clean field `u(x,y)` → into the GREY lens → out as an ORANGE set cloud labeled
> **"FAE-set (target)"** (❓ 1…6).
> ROW B: the BLUE **"DiT patches"** grid → into the SAME GREY lens → out as an ORANGE set cloud labeled
> **"DiT-set"** (❓ 1…6).
> Then **straight GREEN arrows** matching DiT-set slot-1↔FAE-set slot-1 … with a big green ✓. Small note
> by the lens: **"same frozen queries on both sides → slot 5 = the same question for both."**
> Caption: **"Don't drag FAE onto a grid — drag the DiT into FAE's function-space set. Match by ROLE."**

## IMAGE 5 — "REPA is a special case of Reverse-REPA"

> Title: **"One idea, two settings."** Same style. A horizontal axis/spectrum. On the RIGHT, labeled
> **"latent IS an ordered grid (ViT)"**: show the GREY lens shrunk to a dotted "= identity (no pooling)"
> box, so grid → grid unchanged, then GREEN position-arrows → small tag **"= standard REPA."** On the
> LEFT, labeled **"latent is a SET (FAE)"**: show the full GREY lens pooling grid → set, then GREEN
> role-arrows → tag **"= Reverse-REPA (the only option)."** Bottom banner (bold): **"Reverse-REPA is the
> general recipe; REPA is the special case where the pooling is identity. Grid encoder → use REPA. Set
> encoder → no grid exists, so you must pool → Reverse-REPA."**

---

## The narrative the 5 images tell (read this even if the text in the images is imperfect)

1. **Image 1 — the root cause.** A field is a *function* `u(x,y)`. ViT *discretizes* it onto a grid →
   tokens tagged by **position** (📍 patch i = place i). FAE keeps it *functional* → tokens tagged by
   **role** (❓ slot i = a fixed question), with **no position**. Everything downstream follows from
   this one difference.
2. **Image 2 — REPA's hidden requirement.** REPA aligns DiT-patch-`i` to encoder-token-`i` — a
   **position match**. It works for ViT because both sides are position-tagged grids, giving dense,
   per-location supervision.
3. **Image 3 — why FAE breaks it.** FAE's slots have a role, not a place, so a position-arrow has
   nothing to land on. The only "fix" (decode FAE at patch centers) lands one linear layer from pixels →
   cosine ≈ 0.95 → trivial, no semantic guidance.
4. **Image 4 — the fix.** Stop demanding a grid. Run *both* the clean field and the DiT's features
   through the **same frozen FAE lens** (the 128 fixed queries) → both become **role-tagged sets** →
   match slot-`i` ↔ slot-`i`. Same queries ⇒ slot 5 means the same role on both sides ⇒ the alignment
   is well-defined with no grid.
5. **Image 5 — the relationship.** Reverse-REPA is the **general** recipe (pool → set → match by role);
   standard REPA is the **special case** where the pooling is the identity (grid stays a grid → match by
   position). For a grid encoder you'd use plain REPA; for a set encoder there's no grid, so pooling is
   forced, and Reverse-REPA is the only option.

**One-sentence takeaway:** *ViT lives in grid space and is matched by position (REPA); FAE lives in
function space and is matched by role (Reverse-REPA); the frozen FAE encoder is the shared lens that
turns the generator's grid into role-slots so the two can be compared.*
