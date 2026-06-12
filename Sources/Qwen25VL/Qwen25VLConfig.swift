import Foundation

/// Text-backbone configuration, decoded from the published checkpoint's `config.json`
/// as-is (mlx-community/Qwen2.5-VL-3B-Instruct-* repos are consumed unchanged).
public struct Qwen25VLTextConfig: Codable, Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var rmsNormEps: Float
    public var vocabSize: Int
    public var maxPositionEmbeddings: Int
    public var ropeTheta: Float
    /// 3B ships tied (`tie_word_embeddings: true`, no `lm_head.*` keys);
    /// larger variants may untie — config-driven.
    public var tieWordEmbeddings: Bool

    /// head_dim = hiddenSize / numAttentionHeads (128 for 3B).
    public var headDim: Int { hiddenSize / numAttentionHeads }
    /// mRoPE section split [temporal, height, width] over headDim/2.
    /// Matches the checkpoint's rope_scaling.mrope_section.
    public var mropeSection: [Int] { [16, 24, 24] }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? numAttentionHeads
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? 128_000
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? true
    }
}

/// MLX quantization parameters, present in quantized checkpoints' `config.json`.
public struct Qwen25VLQuantization: Codable, Sendable {
    public var groupSize: Int
    public var bits: Int

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}

/// Top-level checkpoint config: text config at the root, plus optional quantization.
public struct Qwen25VLCheckpointConfig {
    public let text: Qwen25VLTextConfig
    public let quantization: Qwen25VLQuantization?
    /// Raw vision_config dict (decoded by the pipeline into MLXVLM's VisionConfiguration).
    public let visionConfigJSON: [String: Any]

    public static func load(from directory: URL) throws -> Qwen25VLCheckpointConfig {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let text = try JSONDecoder().decode(Qwen25VLTextConfig.self, from: data)

        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw Qwen25VLError.config("config.json is not a JSON object") }

        var quantization: Qwen25VLQuantization?
        if let q = root["quantization"] as? [String: Any] {
            let qData = try JSONSerialization.data(withJSONObject: q)
            quantization = try JSONDecoder().decode(Qwen25VLQuantization.self, from: qData)
        }
        guard let vision = root["vision_config"] as? [String: Any] else {
            throw Qwen25VLError.config("config.json has no vision_config")
        }
        return Qwen25VLCheckpointConfig(
            text: text, quantization: quantization, visionConfigJSON: vision)
    }
}

/// Smart-resize bounds from the checkpoint's `preprocessor_config.json`
/// (self-contained in the mlx-community repos — no side-fetch needed).
public struct Qwen25VLProcessorConfig: Codable, Sendable {
    public var minPixels: Int
    public var maxPixels: Int

    enum CodingKeys: String, CodingKey {
        case minPixels = "min_pixels"
        case maxPixels = "max_pixels"
    }

    public static func load(from directory: URL) throws -> Qwen25VLProcessorConfig {
        let data = try Data(
            contentsOf: directory.appendingPathComponent("preprocessor_config.json"))
        return try JSONDecoder().decode(Qwen25VLProcessorConfig.self, from: data)
    }
}

/// Token-id constants (Qwen2.5-VL vocabulary).
public enum Qwen25VLTokens {
    /// `<|im_end|>` — primary EOS.
    public static let imEnd = 151645
    /// `<|endoftext|>` — secondary stop; honor both.
    public static let endOfText = 151643
}

public enum Qwen25VLError: Error {
    case config(String)
    case imageProcessing(String)
    case missingToken(String)
    case loading(String)
}
