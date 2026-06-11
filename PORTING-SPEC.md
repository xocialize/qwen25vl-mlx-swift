# qwen25vl-mlx-swift — Porting Spec

**Goal:** Swift/MLX package serving **stock Qwen2.5-VL-3B-Instruct** for MLXEngine's
`imageAnalysis` + `videoAnalysis` capabilities (one checkpoint registers both, per the
capability contract). Replaces the deferred Lance x2t path.

**Method: SALVAGE, not port.** `~/Development/MLXEngine/lance-mlx-swift` contains a
fully-verified Qwen2.5-VL implementation (Lance = Qwen2.5-VL + MoT experts). Strip the
Lance-specifics, retarget the loader, switch the template. Total source: 1,658 lines.
Carries two fixes upstream `mlx-swift-lm` still lacks: the ViT window-attention mask
(was discarded + all-false) and the decoder mRoPE section split (list-REPEAT
`[16,24,24,16,24,24]`, not element-doubled `[32,48,48]`).

## Resource constraint (2026-06-11)

SCAIL-2 port is actively consuming GPU + memory on this machine. Until it frees up:
**no model loads, no GPU tests, no xcodebuild of the full workspace.** Allowed: source
extraction/editing, `swift build` of this package alone (CPU, small), weight downloads
(network/disk). Gate runs and parity validation wait.

## File-by-file salvage map (from lance-mlx-swift @ 61ca687)

| Source file (lines) | Action | Notes |
|---|---|---|
| `Adapted/Qwen25VLVision.swift` (489) | **KEEP as-is** | Window-mask fix verified: cosine 1.000000/stage vs PT on CPU stream. MIT attribution header stays. |
| `Adapted/QwenVLComponents.swift` (70) | **KEEP as-is** | PatchEmbed + VisionRotaryEmbedding. |
| `LancePILResize.swift` (144) | **KEEP**, rename `PILResize` | PIL-exact bicubic (±1 LSB byte-gated). For STOCK Qwen2.5-VL, HF smart-resize IS the trained preprocessing — the Lance bucket-crop does NOT apply here. |
| `LanceUnderstanding.swift` (479) | **ADAPT** → `Qwen25VLPipeline` | (a) Template → stock Qwen chat template with `<|image_pad|>` (NO video-pad substitution — that was Lance's training convention); (b) keep smart-resize/patchify (stock-correct); (c) keep `get_rope_index`-equivalent position ids (verified exact vs HF transformers 5.9); (d) keep dual-EOS stop {151645, 151643}; (e) **add logit slice to `len(tokenizer)`=151665 before argmax** (lm_head has 151936 rows, 271 untrained). |
| `LanceModel.swift` (312) | **ADAPT** → `Qwen25VLModel` | STRIP: all 10 `*_moe_gen` branches, `PositionGroup` routing, per-expert final norm, **QK-norm (`q_norm`/`k_norm`)** — stock Qwen2.5-VL attention has NO QK-norm (it was a Lance addition; do not carry it). KEEP: mRoPE with list-repeat sections, GQA (16 q / 2 kv heads), SwiGLU MLP (bias-free — E5 lesson), KV cache. |
| `LanceConfig.swift` (118) | **REWRITE** → `Qwen25VLConfig` | Key contract for the stock checkpoint (below). 3B is **tied-head** (`tie_word_embeddings: true`) — loader must support lm_head-from-embeddings; Lance was untied. Keep the two-way key verify (refuse partial loads). |
| `LanceLoader.swift` (46) | **ADAPT** | Same strict-load discipline, stock key map. |

## Weights & key layout

- Runtime: `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` (~2.2 GB). Reference for parity:
  `mlx-community/Qwen2.5-VL-3B-Instruct-bf16` (~7 GB). Both Apache-2.0.
- **Verify the actual key prefixes on download** (mlx-vlm convention is
  `language_model.model.*` + `vision_tower.*`; HF original is `model.*` + `visual.*`)
  — write the loader against what the files actually ship (pitfall #13: keys, not config).
- Processor constants from `Qwen/Qwen2.5-VL-3B-Instruct` `preprocessor_config.json`
  (smart-resize min/max pixels; do NOT hardcode from memory — read the file).
- Tokenizer via swift-transformers from the same repo (as lance-mlx-swift already does).

## What changes vs Lance (checklist of deltas — each was a verified Lance-side finding)

1. MoT/`_moe_gen`: REMOVE (stock has single tower).
2. QK-norm: REMOVE (Lance-only).
3. Head: TIED for 3B (Lance untied). 7B+ variants differ — make it config-driven.
4. Template: stock `apply_chat_template` form: `<|im_start|>user\n<|vision_start|><|image_pad|>…<|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n` with the standard system prompt; `<|image_pad|>` stays `<|image_pad|>` (Lance substituted video_pad).
5. Preprocessing: HF smart-resize (PIL-exact resampler we already have). Bucket-crop is Lance-only.
6. Attention: fully causal (the bidirectional vision span was Lance's training regime; stock Qwen2.5-VL prefill is causal everywhere).
7. Logit masking to `len(tokenizer)` before argmax.
8. mRoPE positions: unchanged (verified exact vs HF `get_rope_index`).

## Validation plan (DEFERRED until SCAIL-2 frees the GPU)

Per the parity doctrine (`mlx-porting` skill, parity-testing.md § exact-match ceiling):

1. Strict load: 0 missing / 0 unused keys, bf16 + 4bit.
2. Integer artifacts exact: prompt token ids vs HF `apply_chat_template`; position ids
   vs HF `get_rope_index` (script exists: lance-mlx `scripts/` pattern); pixel_values
   byte-equal vs HF processor on the 6 Lance oracle images.
3. Op parity on the **CPU stream** vs mlx-vlm (Python) ≤1e-5 fp32: ViT final features,
   first-step logits. (Do NOT chase GPU-stream deltas — M5 matmul noise ≈8e-4/op.)
4. **Semantic gate, not token gate:** the 6 Lance oracle images (smoke only — expect
   correct chart reads "29%"/"1.3 billion": stock Qwen2.5-VL is benchmark-strong exactly
   there) + a ChartQA/DocVQA subset (N≥50) scored vs published Qwen2.5-VL-3B numbers.
5. Latency/memory on M-series for the MLXEngine validation table; 4bit vs bf16 semantic
   delta on the N≥50 set.

## MLXEngine integration (after validation)

- Package registers `imageAnalysis` + `videoAnalysis` (two capabilities, one model) per
  the capability contract; `mode` tags for caption/vqa/ocr/grounding rather than new
  capability cases.
- Video path (phase 2): frame sampling (even count — temporal_patch_size=2 pairs frames),
  port of the Python pipeline's `generate_video` sampling logic.
- RosettaCast hook candidate: on-screen text OCR feeding DubEngine translation context.

## Non-goals

- No Lance compatibility. No MoT. No generation (t2i/t2v) — understanding only.
- No upstream mlx-swift-lm dependency for the vision tower (theirs carries the mask
  no-op + mRoPE transcription bugs; our Adapted/ copies are the fixed reference until
  the upstream report lands).
