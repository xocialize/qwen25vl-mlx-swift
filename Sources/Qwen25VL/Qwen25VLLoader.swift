import Foundation
import MLX
import MLXNN

/// Loads mlx-community/Qwen2.5-VL-3B-Instruct-{bf16,4bit} checkpoints as published.
///
/// Verified layout (safetensors headers, 2026-06-11):
///   - `language_model.model.*` — text backbone (flat under that prefix)
///   - `vision_tower.{patch_embed,blocks,merger}.*` — ViT
///   - NO `lm_head.*` (3B is tied) · bf16 sharded ×2 · 4bit adds `.scales`/`.biases`
///
/// Same strict-load discipline as the Lance loader: module keys must all be filled,
/// leftover file keys are reported — a partial load emits garbage with no other symptom.
public enum Qwen25VLLoader {
    public struct LoadResult {
        public let model: Qwen25VLModel
        public let config: Qwen25VLCheckpointConfig
        /// Checkpoint LLM keys the module didn't consume (should be empty).
        public let unusedKeys: [String]
        /// ViT weights (prefix-stripped, unsanitized) for the vision tower.
        public let visionWeights: [String: MLXArray]
    }

    static let llmPrefix = "language_model.model."
    static let llmHeadPrefix = "language_model."   // lm_head lives here when untied
    static let visionPrefix = "vision_tower."

    /// Merge all *.safetensors in the snapshot directory (handles sharding).
    static func loadAllArrays(directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        guard !files.isEmpty else {
            throw Qwen25VLError.loading("no .safetensors files in \(directory.path)")
        }
        var merged: [String: MLXArray] = [:]
        for f in files {
            let arrays = try MLX.loadArrays(url: f)
            merged.merge(arrays) { a, _ in a }
        }
        return merged
    }

    public static func loadModel(directory: URL) throws -> LoadResult {
        let config = try Qwen25VLCheckpointConfig.load(from: directory)
        let model = Qwen25VLModel(config.text)

        let all = try loadAllArrays(directory: directory)

        // Split by prefix.
        var llm: [String: MLXArray] = [:]
        var vision: [String: MLXArray] = [:]
        for (k, v) in all {
            if k.hasPrefix(llmPrefix) {
                llm[String(k.dropFirst(llmPrefix.count))] = v
            } else if k.hasPrefix(visionPrefix) {
                vision[String(k.dropFirst(visionPrefix.count))] = v
            } else if k.hasPrefix(llmHeadPrefix) {
                // e.g. language_model.lm_head.weight on untied variants
                llm[String(k.dropFirst(llmHeadPrefix.count))] = v
            } else {
                throw Qwen25VLError.loading("unexpected key prefix: \(k)")
            }
        }

        // Quantized checkpoints: swap Linears for QuantizedLinears wherever the file
        // carries scales (mlx-swift-lm's loadWeights pattern).
        if let q = config.quantization {
            quantize(model: model, groupSize: q.groupSize, bits: q.bits) { path, _ in
                llm["\(path).scales"] != nil
            }
        }

        let moduleKeys = Set(model.parameters().flattened().map(\.0))
        let fileKeys = Set(llm.keys)

        let missing = moduleKeys.subtracting(fileKeys).sorted()
        guard missing.isEmpty else {
            throw Qwen25VLError.loading(
                "checkpoint missing \(missing.count) module keys, e.g. "
                + missing.prefix(5).joined(separator: ", "))
        }
        let unused = fileKeys.subtracting(moduleKeys).sorted()

        let consumed = llm.filter { moduleKeys.contains($0.key) }
        model.update(parameters: ModuleParameters.unflattened(consumed))
        eval(model)

        return LoadResult(
            model: model, config: config, unusedKeys: unused, visionWeights: vision)
    }
}
