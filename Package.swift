// swift-tools-version: 6.2
// qwen25vl-mlx-swift — Swift/MLX package serving stock Qwen2.5-VL-3B-Instruct for
// MLXEngine's imageAnalysis/videoAnalysis. Salvaged from lance-mlx-swift's verified
// Qwen2.5-VL components (ViT window-mask fix + decoder mRoPE list-repeat fix, both
// absent upstream). Consumes mlx-community/Qwen2.5-VL-3B-Instruct-{bf16,4bit} as
// published. See PORTING-SPEC.md.

import PackageDescription

let package = Package(
    name: "Qwen25VL",
    platforms: [
        // v26 to match the MLXEngine contract (MLXToolKit) the wrapper target links.
        .macOS(.v26)
    ],
    products: [
        .library(name: "Qwen25VL", targets: ["Qwen25VL"]),
        // The MLXEngine wrapper: a conformant `ModelPackage` over the core pipeline.
        .library(name: "MLXQwen25VL", targets: ["MLXQwen25VL"]),
    ],
    dependencies: [
        // 0.32.3 carries the NAX split-K GEMM fix (mlx#3810); QVLMLP's row-chunk workaround is gone,
        // so earlier versions would corrupt half-precision down_proj on long sequences on M5.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.32.3"),
        // Public Qwen2.5-VL utilities (configuration types, THW). The fileprivate
        // vision blocks are copy-adapted into Sources/Qwen25VL/Adapted/ with
        // attribution (MIT) — carrying our window-mask fix. See NOTICE.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // Native downloader for `load()` auto-materialization (WeightMaterializer).
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
        // MLXEngine contract (MLXToolKit) for the wrapper target; the core `Qwen25VL`
        // target stays engine-agnostic. ≥0.27.0: run-lifecycle V4 — the CAN
        // cancellation-conformance gate (MLXServeConformance CAN-1..3).
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.27.0"),
    ],
    targets: [
        .target(
            name: "Qwen25VL",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/Qwen25VL"
        ),
        .target(
            name: "MLXQwen25VL",
            dependencies: [
                "Qwen25VL",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/MLXQwen25VL"
        ),
        // On-box NAX split-K GEMM probe at THIS package's (K,N) dims — the verification
        // norm for the mlx#3797 row-chunk workaround (weights-free, seeded, minutes).
        .executableTarget(
            name: "Qwen25VLGate",
            dependencies: [
                "Qwen25VL",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/Qwen25VLGate"
        ),
        .testTarget(
            name: "Qwen25VLTests",
            dependencies: ["Qwen25VL"],
            path: "Tests/Qwen25VLTests"
        ),
        // Offline MAT gate (MAT-1..5) + WeightSourcing declaration tests for the wrapper.
        .testTarget(
            name: "MLXQwen25VLTests",
            dependencies: [
                "MLXQwen25VL",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXQwen25VLTests"
        ),
    ]
)
