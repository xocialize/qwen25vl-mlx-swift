# qwen25vl-mlx-swift

Swift/MLX package serving **Qwen2.5-VL-3B-Instruct** (Apache-2.0) for MLXEngine's
`imageAnalysis` / `videoAnalysis` capabilities — VQA, captioning, OCR/document reads,
grounding. Salvaged from the verified `lance-mlx-swift` Qwen2.5-VL components
(ViT window-mask fix + decoder mRoPE list-repeat fix, both absent upstream).

**Read `PORTING-SPEC.md` first.** Status: scaffolded; extraction pending
(GPU/memory currently allocated to the SCAIL-2 port — source work only until freed).
