# qwen25vl-mlx-swift

> **Licence: the Qwen2.5-VL-3B-Instruct weights are under the Qwen RESEARCH License Agreement —
> research / evaluation only, no commercial use** (§1(i), §2(a); commercial use needs a separate
> licence from Alibaba Cloud, §2(b)). Upstream's model card says `license_name: qwen-research`, and
> so do the `mlx-community/Qwen2.5-VL-3B-Instruct-{bf16,4bit}` conversions. The family is split by
> size: the 7B and 32B are Apache-2.0, the 72B is under the Qwen licence, and the 3B is research-only.
> The Swift port code in this repository is Apache-2.0 (`LICENSE`); that doesn't change the weights'
> terms.
>
> Since 0.5.0, `MLXQwen25VL` declares `weightLicense = LicenseRef-Qwen-Research`, which is never on a
> permissive allowlist. A `.permissiveOnly` + `.blocking` engine refuses the package; an `.advisory`
> engine registers it with a licence advisory. Don't route to it by default (AB-A-0142, AB-D-0085).
> Before 0.5.0 it was declared Apache-2.0 by mistake.
>
> The `Qwen25VL` core carries no weights. `qwen-image-edit-swift` uses it with Qwen-Image's
> Qwen2.5-VL-7B text encoder, which is Apache-2.0.

Swift/MLX package serving **Qwen2.5-VL-3B-Instruct** (Qwen Research licence) for MLXEngine's
`imageAnalysis` / `videoAnalysis` capabilities — VQA, captioning, OCR/document reads,
grounding. Salvaged from the verified `lance-mlx-swift` Qwen2.5-VL components
(ViT window-mask fix + decoder mRoPE list-repeat fix, both absent upstream in
mlx-swift-lm).

**Status (2026-06-11):** bf16 + 4bit both load strictly and answer all six Lance
Phase-0 oracle images semantically correctly — including the two chart-value reads
(case-02 "29", case-04 "$1.3 billion") that no Lance configuration ever passed
together. Per-answer 0.8–20 s on M-series.

```swift
import Qwen25VL
let pipe = try await Qwen25VLPipeline.load(directory: snapshotURL)  // mlx-community/Qwen2.5-VL-3B-Instruct-{bf16,4bit}
let answer = try pipe.generate(image: ciImage, prompt: "What does this chart show?")
```

GPU tests: `QVL_GPU_TESTS=1 swift test` (copy the `mlx-swift_Cmlx.bundle` from a
DerivedData build into `.build/debug/` first — SPM doesn't compile Metal shaders).
`QVL_WEIGHTS_DIR` overrides the checkpoint path (defaults to the bf16 snapshot).

Remaining validation (tracked): integer-artifact parity vs HF processor, CPU-stream
op parity vs mlx-vlm, ChartQA/DocVQA N≥50 semantic eval, latency/memory table,
video path. See `PORTING-SPEC.md`.

## Consuming it

This package is the **VL encoder backbone consumed by [`qwen-image-edit-swift`](https://github.com/xocialize/qwen-image-edit-swift)** — salvaged from the dropped Lance port, it now serves `imageAnalysis` / `videoAnalysis` standalone (via the `MLXQwen25VL` engine wrapper) **and** backs the Qwen-Image-Edit wrapper, so both consume one verified VL core by version.

Public + version-tagged on github.com/xocialize. Add by tagged URL:
`.package(url: "https://github.com/xocialize/qwen25vl-mlx-swift", from: "0.1.0")`, then import `Qwen25VL` for the raw pipeline, or `MLXQwen25VL` for the conformant `imageAnalysis` / `videoAnalysis` engine package.
