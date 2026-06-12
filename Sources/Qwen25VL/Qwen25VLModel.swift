import Foundation
import MLX
import MLXFast
import MLXNN

// Stock Qwen2.5-VL text backbone — salvaged from lance-mlx-swift's parity-verified
// LanceModel with the Lance-specifics removed: no MoT/`_moe_gen` expert towers, no
// QK-norm (a Lance addition; stock Qwen2.5-VL attention has none), tied LM head for
// the 3B checkpoint. The mRoPE carries the list-REPEAT section-split fix (the
// element-doubled split upstream mlx-swift-lm transcribed from NumPy `list * 2`
// scrambles vision-token rotations in the decoder).

/// Minimal growing KV cache (concat + offset). One per layer.
public final class QVLKVCache {
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    public var offset: Int { keys?.dim(2) ?? 0 }

    public init() {}

    public func update(keys k: MLXArray, values v: MLXArray) -> (MLXArray, MLXArray) {
        if let keys, let values {
            self.keys = concatenated([keys, k], axis: 2)
            self.values = concatenated([values, v], axis: 2)
        } else {
            self.keys = k
            self.values = v
        }
        return (self.keys!, self.values!)
    }
}

/// 3-axis mRoPE: positionIds (3, B, T) over [temporal, height, width] with section split
/// [16, 24, 24] of headDim/2.
public enum MRoPE {
    public static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let d = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, d...], x[.ellipsis, ..<d]], axis: -1)
    }

    /// cos/sin of shape (B, 1, T, headDim), section-interleaved across the t/h/w planes.
    public static func cosSin(
        positionIds: MLXArray, headDim: Int, theta: Float, mropeSection: [Int]
    ) -> (MLXArray, MLXArray) {
        let invFreq = pow(
            MLXArray(theta), -MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) })
                / Float(headDim)
        )  // (half,)
        // (3, B, T, 1) * (half,) → (3, B, T, half)
        let freqs = positionIds.asType(.float32).expandedDimensions(axis: -1) * invFreq
        let emb = concatenated([freqs, freqs], axis: -1)  // (3, B, T, headDim)
        var cosT = cos(emb)
        var sinT = sin(emb)
        // HF/mlx-vlm split cos/sin by the section list REPEATED — [16,24,24,16,24,24]
        // picking planes t,h,w,t,h,w — so rotate-half pairs (j, j+64) stay within one
        // axis. NOT element-doubled [32,48,48] (the upstream mlx-swift-lm transcription
        // bug: NumPy `mrope_section * 2` is list repetition, not multiplication).
        let repeated = mropeSection + mropeSection
        var indices: [Int] = []
        var acc = 0
        for s in repeated.dropLast() { acc += s; indices.append(acc) }
        cosT = concatenated(
            split(cosT, indices: indices, axis: -1).enumerated().map { i, m in m[i % 3] },
            axis: -1
        )[0..., .newAxis, 0..., 0...]
        sinT = concatenated(
            split(sinT, indices: indices, axis: -1).enumerated().map { i, m in m[i % 3] },
            axis: -1
        )[0..., .newAxis, 0..., 0...]
        return (cosT, sinT)
    }

    /// Apply to q/k of shape (B, H, T, headDim).
    public static func apply(q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray)
        -> (MLXArray, MLXArray)
    {
        let qOut = (q * cos) + (rotateHalf(q) * sin)
        let kOut = (k * cos) + (rotateHalf(k) * sin)
        return (qOut, kOut)
    }
}

/// SwiGLU MLP, bias-free on all three projections (stock Qwen2.5).
public final class QVLMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

/// Stock Qwen2.5-VL GQA attention: q/k/v carry biases, output projection does not.
/// No QK-norm (that was a Lance addition).
public final class QVLAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let ropeTheta: Float
    let mropeSection: [Int]

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    public init(_ config: Qwen25VLTextConfig) {
        let dim = config.hiddenSize
        self.heads = config.numAttentionHeads
        self.kvHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)
        self.ropeTheta = config.ropeTheta
        self.mropeSection = config.mropeSection

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, positionIds: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: QVLKVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        var values = wv(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        let (cosT, sinT) = MRoPE.cosSin(
            positionIds: positionIds, headDim: headDim, theta: ropeTheta,
            mropeSection: mropeSection)
        (queries, keys) = MRoPE.apply(q: queries, k: keys, cos: cosT, sin: sinT)

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return wo(output)
    }
}

/// Standard pre-norm decoder layer.
public final class QVLDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: QVLAttention
    @ModuleInfo(key: "mlp") var mlp: QVLMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm

    public init(_ config: Qwen25VLTextConfig) {
        self._attention.wrappedValue = QVLAttention(config)
        self._mlp.wrappedValue = QVLMLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        self._inputNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, positionIds: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: QVLKVCache?
    ) -> MLXArray {
        let h = x + attention(inputNorm(x), positionIds: positionIds, mask: mask, cache: cache)
        return h + mlp(postNorm(h))
    }
}

/// The Qwen2.5-VL text backbone: embeddings → N layers → final norm → (tied) LM head.
public final class Qwen25VLModel: Module {
    public let config: Qwen25VLTextConfig

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [QVLDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    /// Present only when the checkpoint is untied (7B+); 3B ships no lm_head keys.
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Qwen25VLTextConfig) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            QVLDecoderLayer(config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._lmHead.wrappedValue = config.tieWordEmbeddings
            ? nil
            : Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    /// Forward over pre-built input embeddings (ViT features already merged).
    public func callAsFunction(
        inputEmbeddings: MLXArray, positionIds: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, caches: [QVLKVCache]?
    ) -> MLXArray {
        var h = inputEmbeddings
        for (i, layer) in layers.enumerated() {
            h = layer(h, positionIds: positionIds, mask: mask, cache: caches?[i])
        }
        return norm(h)
    }

    /// Logits via the tied embedding (3B) or the untied head (larger variants).
    public func logits(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        }
        return embedTokens.asLinear(hidden)
    }
}
