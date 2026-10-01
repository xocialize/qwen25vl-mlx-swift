// Qwen25VLGate — on-box regression gate for the NAX split-K GEMM bug (ml-explore/mlx#3797,
// fixed in mlx-swift 0.32.3 via mlx#3810) at THIS package's own (K,N) dims, per the fleet verification
// norm: exposure is measured on-box, never inferred from config dims alone.
//
//   swift run Qwen25VLGate --matmul-probe-rand
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
import MLXNN
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
guard gate == "--matmul-probe-rand" else {
    err("usage: Qwen25VLGate --matmul-probe-rand")
    exit(2)
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
