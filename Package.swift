// swift-tools-version: 6.0
// qwen25vl-mlx-swift — Swift/MLX package serving stock Qwen2.5-VL-3B-Instruct for
// MLXEngine's imageAnalysis/videoAnalysis. Salvaged from lance-mlx-swift's verified
// Qwen2.5-VL components (ViT window-mask fix + decoder mRoPE list-repeat fix, both
// absent upstream). Consumes mlx-community/Qwen2.5-VL-3B-Instruct-{bf16,4bit} as
// published. See PORTING-SPEC.md.

import PackageDescription

let package = Package(
    name: "Qwen25VL",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "Qwen25VL", targets: ["Qwen25VL"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        // Public Qwen2.5-VL utilities (configuration types, THW). The fileprivate
        // vision blocks are copy-adapted into Sources/Qwen25VL/Adapted/ with
        // attribution (MIT) — carrying our window-mask fix. See NOTICE.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
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
        .testTarget(
            name: "Qwen25VLTests",
            dependencies: ["Qwen25VL"],
            path: "Tests/Qwen25VLTests"
        ),
    ]
)
