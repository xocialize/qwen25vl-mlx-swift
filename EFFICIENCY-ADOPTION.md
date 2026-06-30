# Efficiency Adoption Brief — `qwen25vl-mlx-swift` (Qwen2.5-VL-3B, `imageAnalysis`)

> **For a session-specific agent.** Adopt the engine 1.14 efficiency contract (engine 0.15.0). Load the
> `mlx-swift-integration` skill; read references/package-efficiency.md (four levers + **"Measurement
> findings"** — the LLM prefill-scratch lesson applies here) + references/memory-harness.md. Audited
> 2026-06-30.

## Why this one is interesting (it combines two levers)
A VLM = **vision tower + LM**. So it has *both*: (1) **per-stage eviction** — the vision tower encodes the
image once, then is idle through LM generation (evict it, the encoder pattern); and (2) the **autoregressive
prefill-scratch transient** from the Qwen-LLM finding — *amplified*, because **image tokens inflate the
prefill** (a high-res image → thousands of vision tokens → large prefill scratch on top of the text).

## Package at a glance
- Wrapper `MLXQwen25VL` (`Qwen25VLPackage`) over core `Qwen25VL` (consumes `MLXVLM`/mlx-swift-lm).
  Components: **vision tower** (`Qwen25VLVision`) + **LM**. Capability `imageAnalysis`. Single size (3B) × quant.
- **Footprints today (flat):** bf16 **9.6 GB** · int4 **4.5 GB**. No split.
- Config `Qwen25VLConfiguration: PackageConfiguration, ModelStorable` — confirm a `quant` field for `QuantConfigured`.

## Engine dependency status
- Pins `mlx-engine-swift` **`from: "0.3.0"`**. **P0 = `swift package update`** → 0.15.0; build + fix any
  drift (the `imageAnalysis` surface is old/stable; verify).

## Audit vs. the four levers

| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡 | from 0.3.0 → 0.15.0 | **P0** |
| 1. Split footprint | ❌ | flat 9.6/4.5 GB; transient (prefill scratch + KV-cache, image-token-inflated) unaccounted | **P1** |
| 2. Per-stage evict | 🟡 | evict the vision tower after image-encode, before LM gen (modest — vision ≪ LM — but real) | **P2** |
| 3. mmap/lazy | 🟡 verify | MLXVLM loader — confirm lazy/mmap (floor ≈ on-disk) | note |
| 4. BudgetAware | ➖ | quant config-chosen; defer | defer |

---

## P1 — Declare the split  (apply the Qwen-LLM lesson: MEASURE the transient)
- `QuantConfigured` (bf16/int4 — single size, so quant alone suffices; no FootprintConfigured needed here).
- `residentBytes` = weights floor (vision + LM, after P2 the LM-gen-phase resident; cheaply ≈ on-disk).
- `peakActivationBytes` = the **measured** transient at a documented envelope — and per the Qwen-LLM
  finding, **measure it, don't derive it**: the LM prefill scratch (not just the KV-cache) dominates, and
  for a VLM it's inflated by the image-token count. Declare at a documented **(image resolution × maxTokens)**
  envelope; note both as the basis (image res drives the vision-token count → prefill size).

## P2 — Evict the vision tower after encode
The vision tower is used once to produce image embeddings, then idle through LM generation. Stage it:
load vision → encode image → evict (`nil` + `Memory.clearCache()`) → LM prefill+decode. Modest (the 3B LM
dominates the vision tower) but real, and it lowers the LM-gen-phase resident floor. Watch the Swift 6
`#isolation` gotcha if the staged path goes async (recurred on LTX + Qwen-Image-Edit).

## Defer/verify — P3 (MLXVLM lazy/mmap, note only), P4 (BudgetAware: quant config-chosen).

## Measurement (moderate — VLM inference)
Smoke/CLI target via `xcodebuild`: load → measure weights floor → run an **image + prompt → N tokens**
generation at a documented image resolution to build the (image-inflated) prefill + KV-cache → measure
peak. `peakActivationBytes ≈ peak − floor`. Report the image-res/maxTokens envelope. Re-measure both quants
if feasible; flag the unmeasured one (width-scale won't apply cleanly across quants — measure each).

## Definition of done
- [ ] `swift package update` → 0.15.0; build green.
- [ ] `QuantConfigured`; P2 vision-tower evict; split declared per quant (`residentBytes` + measured `peakActivationBytes`) at a documented image-res × maxTokens envelope.
- [ ] Parity/smoke green; an image→answer run produces a valid response; record the split + envelope.
- [ ] BudgetAware deferred (note). Registry: qwen25vl row Eff ⬜→✅, Eng→0.15.0.

## Report back
Flat→split per quant, the vision-tower-evict effect, the **measured** prefill-scratch transient + the
image-res/maxTokens envelope, drift since 0.3.0, effort, commit SHAs. STAY IN SCOPE — four-lever adoption +
brief + registry row only; no testing-app/shell changes; stop-and-report if a bigger change seems needed.
