// Qwen25VLGate — on-box regression gate for the NAX split-K GEMM bug (ml-explore/mlx#3797,
// fixed in mlx-swift 0.32.3 via mlx#3810) at THIS package's own (K,N) dims, per the fleet verification
// norm: exposure is measured on-box, never inferred from config dims alone.
//
//   swift run Qwen25VLGate --matmul-probe-rand
//   swift run -c release Qwen25VLGate --patch-embed-probe   (AB-T-0466, see that section)
//
// Two sections per model tier (3B: K=11008,N=2048 · 7B: K=18944,N=3584):
//   raw     — plain bf16 matmul vs fp32 reference across the M dispatch boundary.
//             Informational: documents whether THIS box/mlx-swift build is exposed
//             (in-window rows go garbage on affected NAX builds).
//   mlp     — the real `QVLMLP` forward (bf16, single fused down_proj since the
//             row-chunk was removed) vs an fp32 reference of the same SwiGLU math.
//             This is the GATE: cos ≥ 0.999 at every M or exit 1.
//
// Weights-free: seeded host-side LCG randoms, no MLXRandom, no snapshot required.

import Foundation
import MLX
import MLXExactConv
import MLXNN
import MLXRandom
import Qwen25VL

func err(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.asType(.float32).reshaped(-1)
    let y = b.asType(.float32).reshaped(-1)
    let dot = (x * y).sum().item(Float.self)
    let nx = sqrt((x * x).sum()).item(Float.self)
    let ny = sqrt((y * y).sum()).item(Float.self)
    return dot / (nx * ny)
}

// Seeded host-side uniform(-1,1) randoms — deterministic, no MLXRandom dep.
struct LCG {
    var state: UInt64 = 0x9E37_79B9_7F4A_7C15
    mutating func randArray(_ count: Int) -> [Float] {
        var out = [Float]()
        out.reserveCapacity(count)
        for _ in 0 ..< count {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            out.append(Float(Int64(bitPattern: state >> 11)) / Float(Int64.max >> 11))
        }
        return out
    }
}
var lcg = LCG()


let gate = CommandLine.arguments.dropFirst().first ?? "--matmul-probe-rand"
guard gate == "--matmul-probe-rand" || gate == "--patch-embed-probe" else {
    err("usage: Qwen25VLGate --matmul-probe-rand | --patch-embed-probe")
    exit(2)
}

// AB-T-0466: `QVLPatchEmbed` is a matmul, not a conv. mlx 0.32.x's `pad_gpu` loses values past 2^31
// destination elements (ml-explore/mlx#4735), so the old conv route (3 channels padded to 16) garbled
// every patch row past some count. Weights-free (no 3B snapshot needed), real 3B shape: patch 14,
// temporal 2, 3 channels, 1280 out, bf16, no bias.
//   (1) matmul vs the old ExactConv route vs an fp32 reference, up to and past 262,144 patches;
//   (2) a multi-image batch over 262,144 patches: the last image's embeddings must match it alone.
// Only the patch embed is exercised at that size: `QVLVisionModel` builds a dense (S, S) attention
// mask (~73 GB at 270K patches), and the pipeline encodes one pixel-capped image per call.
if gate == "--patch-embed-probe" {
    let (o, kt, p, c) = (1280, 2, 14, 3)
    let rowLen = c * kt * p * p
    let cut = 262_144
    let embed = QVLPatchEmbed(patchSize: p, temporalPatchSize: kt, inChannels: c, outputDimensions: o)
    let w = (MLXRandom.normal([o, kt, p, p, c], key: MLXRandom.key(3)) * 0.02).asType(.bfloat16)
    try! embed.update(parameters: ModuleParameters.unflattened([("proj.weight", w)]), verify: .none)
    var ok = true

    // The conv's input padded to 16 channels is 2·14·14·16 = 6,272 elements per patch, so 2^31 is crossed
    // at patch 342,393 for this shape (Qwen3-VL's 2·16·16·16 crosses at 262,145).
    for n in [4096, cut + 1, 342_392, 342_393, 350_000] {
        let x = MLXRandom.uniform(low: -1, high: 1, [n, rowLen], key: MLXRandom.key(0)).asType(.bfloat16)
        let patches = x.reshaped(n, c, kt, p, p).movedAxis(source: 1, destination: 4)
        let new = embed(x).asType(.float32)
        let ref = matmul(patches.reshaped(n, -1).asType(.float32), w.reshaped(o, -1).asType(.float32).T)
        let old = ExactConv.conv3d(patches, weight: w, stride: (kt, p, p)).reshaped(n, o).asType(.float32)
        eval(new, ref, old)
        let rowNewOld = abs(new - old).max(axis: 1)
        let rowNewRef = abs(new - ref).max(axis: 1)
        let rowOldRef = abs(old - ref).max(axis: 1)
        func span(_ r: MLXArray, _ lo: Int, _ hi: Int) -> String {
            hi > lo ? String(format: "%.5g", r[lo ..< hi].max().item(Float.self)) : "-"
        }
        let h = min(n, cut)
        let refMax = abs(ref).max().item(Float.self)
        let newErr = rowNewRef.max().item(Float.self)
        // First row where the old route is off by more than 100× the first 4096 rows' worst.
        let bad = rowOldRef .> (rowOldRef[0 ..< 4096].max() * 100 + 1e-3)
        let firstBad = bad.any().item(Bool.self) ? argMax(bad.asType(.int32)).item(Int.self) : -1
        let pass = newErr <= refMax / 64  // bf16 output rounding is ≤ 2^-8 relative
        ok = ok && pass
        err("(1) N=\(n)  |new-old| rows<2^18 \(span(rowNewOld, 0, h)) rows≥2^18 \(span(rowNewOld, h, n))"
            + "  |new-ref| \(span(rowNewRef, 0, h))/\(span(rowNewRef, h, n))"
            + "  |old-ref| \(span(rowOldRef, 0, h))/\(span(rowOldRef, h, n))"
            + "  max|ref| \(String(format: "%.4g", refMax))  old first bad row \(firstBad) \(pass ? "✅" : "❌")")
        GPU.clearCache()
    }

    // (2) K images of 1008×1008 (72×72 patches, t = 1), past both boundaries; the last one differs.
    let per = 72 * 72
    let k = 350_000 / per + 1  // 68 images, 352,512 patches
    let rest = MLXRandom.uniform(low: -1, high: 1, [(k - 1) * per, rowLen], key: MLXRandom.key(1))
    let last = MLXRandom.uniform(low: -1, high: 1, [per, rowLen], key: MLXRandom.key(2))
    let all = concatenated([rest, last], axis: 0).asType(.bfloat16)
    let lastOnly = last.asType(.bfloat16)
    let alone = embed(lastOnly)
    let tail = embed(all)[((k - 1) * per)...]
    let oldTail = ExactConv.conv3d(
        all.reshaped(-1, c, kt, p, p).movedAxis(source: 1, destination: 4), weight: w, stride: (kt, p, p)
    ).reshaped(-1, o)[((k - 1) * per)...]
    eval(alone, tail, oldTail)
    let cos = cosine(tail, alone)
    let pass = cos >= 0.99999
    ok = ok && pass
    err("(2) \(k) images × \(per) patches = \(k * per): last image vs alone, matmul cos \(cos)"
        + " max|Δ| \(abs(tail.asType(.float32) - alone.asType(.float32)).max().item(Float.self))"
        + "  (old ExactConv cos \(cosine(oldTail, alone))) \(pass ? "✅" : "❌")")
    err(ok ? "[gate] PASS — patch embed exact past 262,144 patches" : "[gate] FAIL")
    exit(ok ? 0 : 1)
}

// (tier, hidden N, ffn K) for the checkpoints this package serves. Window opens at
// M ≥ 2048²/N (and M ≤ K/3): 3B ≥ 2048 rows, 7B ≥ 1171 rows.
let tiers: [(label: String, n: Int, k: Int)] = [
    ("3B", 2048, 11008),
    ("7B", 3584, 18944),
]
var pass = true

for tier in tiers {
    let (n, k) = (tier.n, tier.k)
    let boundary = Int((4_194_304 + n - 1) / n)  // ceil(2048² / N)
    let ms = [896, boundary - 64, boundary, boundary + 64, 2304, min(4096, k / 3)]
        .filter { $0 > 0 }.sorted()

    err("[\(tier.label)] raw bf16 [M,\(k)]×[\(k),\(n)] vs fp32 (window opens M=\(boundary)):")
    let b = MLXArray(lcg.randArray(n * k), [n, k]).asType(.bfloat16)
    for m in ms {
        let a = MLXArray(lcg.randArray(m * k), [m, k]).asType(.bfloat16)
        let y = matmul(a, b.T)
        let yRef = matmul(a.asType(.float32), b.asType(.float32).T)
        eval(y, yRef)
        let mab = abs(y.asType(.float32) - yRef).max().item(Float.self)
        err("  M=\(m) cos \(cosine(y, yRef)) max_abs_vs_fp32 \(mab)")
    }

    // The real QVLMLP (single fused down_proj) vs fp32 SwiGLU reference — the gate.
    err("[\(tier.label)] QVLMLP(dim \(n), ffn \(k)) bf16 vs fp32 reference:")
    let mlp = QVLMLP(dimensions: n, hiddenDimensions: k)
    let gw = MLXArray(lcg.randArray(k * n), [k, n]) * 0.02
    let uw = MLXArray(lcg.randArray(k * n), [k, n]) * 0.02
    let dw = MLXArray(lcg.randArray(n * k), [n, k]) * 0.02
    let params = ModuleParameters.unflattened([
        ("gate_proj.weight", gw.asType(.bfloat16)),
        ("up_proj.weight", uw.asType(.bfloat16)),
        ("down_proj.weight", dw.asType(.bfloat16)),
    ])
    try! mlp.update(parameters: params, verify: .none)
    for m in ms {
        let x = MLXArray(lcg.randArray(m * n), [1, m, n])
        let y = mlp(x.asType(.bfloat16))
        let hRef = silu(matmul(x, gw.T)) * matmul(x, uw.T)
        let yRef = matmul(hRef, dw.T)
        eval(y, yRef)
        let cos = cosine(y, yRef)
        let mab = abs(y.asType(.float32) - yRef).max().item(Float.self)
        let ok = cos >= 0.999
        pass = pass && ok
        err("  M=\(m) cos \(cos) max_abs_vs_fp32 \(mab) \(ok ? "✅" : "❌")")
    }
}

err(pass ? "[gate] PASS — down_proj clean at all probed M" : "[gate] FAIL")
exit(pass ? 0 : 1)
