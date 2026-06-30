import CoreImage
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXVLM
import Tokenizers

/// Image preprocessing for the Qwen2.5-VL ViT — HF-processor-exact smart-resize +
/// patchify. For STOCK Qwen2.5-VL the HF smart-resize IS the trained preprocessing
/// (the aspect-ratio bucket crop seen in bytedance/Lance was Lance-specific).
/// Resize is PIL-exact bicubic on uint8 (±1 LSB byte-gated vs Pillow), in the HF
/// order: decode → resize(BICUBIC) on bytes → rescale 1/255 → normalize.
public enum Qwen25VLImageProcessing {
    public static let imageMean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    public static let imageStd: [Float] = [0.26862954, 0.26130258, 0.27577711]
    public static let patchSize = 14
    public static let mergeSize = 2
    public static let temporalPatchSize = 2

    /// Qwen2.5-VL smart resize: round to factor multiples, scale into the pixel budget.
    public static func targetSize(
        height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int
    ) throws -> (Int, Int) {
        guard height >= factor, width >= factor else {
            throw Qwen25VLError.imageProcessing(
                "image \(width)×\(height) smaller than patch factor \(factor)")
        }
        guard max(height, width) / min(height, width) <= 200 else {
            throw Qwen25VLError.imageProcessing("aspect ratio over 200:1")
        }
        var hBar = max(factor, Int(round(Float(height) / Float(factor))) * factor)
        var wBar = max(factor, Int(round(Float(width) / Float(factor))) * factor)
        if hBar * wBar > maxPixels {
            let beta = sqrt(Float(height * width) / Float(maxPixels))
            hBar = Int(floor(Float(height) / beta / Float(factor))) * factor
            wBar = Int(floor(Float(width) / beta / Float(factor))) * factor
        } else if hBar * wBar < minPixels {
            let beta = sqrt(Float(minPixels) / Float(height * width))
            hBar = Int(ceil(Float(height) * beta / Float(factor))) * factor
            wBar = Int(ceil(Float(width) * beta / Float(factor))) * factor
        }
        hBar = (hBar / factor) * factor
        wBar = (wBar / factor) * factor
        guard hBar > 0, wBar > 0 else {
            throw Qwen25VLError.imageProcessing("invalid target \(wBar)×\(hBar)")
        }
        return (hBar, wBar)
    }

    /// Patchify (C,H,W)-stacked frames into the ViT's flattened layout.
    public static func patchify(images: [MLXArray]) throws -> (MLXArray, THW) {
        guard let first = images.first else {
            throw Qwen25VLError.imageProcessing("no frames")
        }
        let resizedHeight = first.dim(-2)
        let resizedWidth = first.dim(-1)
        var patches = concatenated(images)

        let mod = patches.dim(0) % temporalPatchSize
        if mod != 0 {
            let lastPatch = patches[-1, .ellipsis]
            let repeated = tiled(lastPatch, repetitions: [temporalPatchSize - mod, 1, 1, 1])
            patches = concatenated([patches, repeated])
        }
        let channel = patches.dim(1)
        let gridT = patches.dim(0) / temporalPatchSize
        let gridH = resizedHeight / patchSize
        let gridW = resizedWidth / patchSize

        patches = patches.reshaped(
            gridT, temporalPatchSize, channel,
            gridH / mergeSize, mergeSize, patchSize,
            gridW / mergeSize, mergeSize, patchSize)
        patches = patches.transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
        let flattened = patches.reshaped(
            gridT * gridH * gridW,
            channel * temporalPatchSize * patchSize * patchSize)
        return (flattened, THW(gridT, gridH, gridW))
    }

    /// CIImage → patchified ViT input at the smart-resized resolution.
    public static func preprocess(
        image: CIImage, minPixels: Int, maxPixels: Int
    ) throws -> (MLXArray, THW) {
        let h = Int(image.extent.height)
        let w = Int(image.extent.width)
        let (targetH, targetW) = try targetSize(
            height: h, width: w, factor: patchSize * mergeSize,
            minPixels: minPixels, maxPixels: maxPixels)

        // Decode to interleaved RGB8 in sRGB (what PIL's convert("RGB") sees).
        let context = CIContext(options: [
            .workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!
        ])
        guard let cgImage = context.createCGImage(
            image, from: image.extent, format: .RGBA8,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        else { throw Qwen25VLError.imageProcessing("CIImage render failed") }
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        guard let cgContext = CGContext(
            data: &rgba, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw Qwen25VLError.imageProcessing("CGContext creation failed") }
        cgContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            rgb[i * 3] = rgba[i * 4]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }

        let resized = PILResize.resize(
            rgb: rgb, width: w, height: h, outWidth: targetW, outHeight: targetH)
        var chw = [Float](repeating: 0, count: 3 * targetH * targetW)
        let plane = targetH * targetW
        for y in 0..<targetH {
            for x in 0..<targetW {
                let p = (y * targetW + x) * 3
                let o = y * targetW + x
                chw[o] = (Float(resized[p]) / 255 - imageMean[0]) / imageStd[0]
                chw[plane + o] = (Float(resized[p + 1]) / 255 - imageMean[1]) / imageStd[1]
                chw[2 * plane + o] = (Float(resized[p + 2]) / 255 - imageMean[2]) / imageStd[2]
            }
        }
        let array = MLXArray(chw, [1, 3, targetH, targetW])
        return try patchify(images: [array])
    }
}

/// VQA / captioning over images with stock Qwen2.5-VL-3B-Instruct.
/// Stock conventions throughout: `<|image_pad|>` placeholders (no video-pad
/// substitution), fully causal prefill, standard chat template, greedy decode,
/// dual stop ids {151645, 151643}.
public final class Qwen25VLPipeline {
    public let model: Qwen25VLModel
    /// The resident vision tower. Held only between `load()` and the post-encode eviction in
    /// `generate()` — per-stage load→use→evict (the encoder pattern): the ViT runs once to produce
    /// image embeddings, then is idle through the LM decode loop, so it is dropped before decode and
    /// lazily rebuilt by `ensureVision()` on the next call. `nil` while evicted.
    public private(set) var vision: QVLVision.VisionModel?
    /// Rebuilds the vision tower from the snapshot (config + weights) after an eviction. Captured at
    /// `load()` time so the staged path needs no snapshot re-read. The closure realizes the module.
    private let visionBuilder: () throws -> QVLVision.VisionModel
    public let tokenizer: any Tokenizers.Tokenizer
    public let spatialMergeSize: Int
    public let minPixels: Int
    public let maxPixels: Int

    let imagePadId: Int
    let visionStartId: Int

    public static let defaultSystemPrompt = "You are a helpful assistant."

    public init(
        model: Qwen25VLModel, vision: QVLVision.VisionModel,
        visionBuilder: @escaping () throws -> QVLVision.VisionModel,
        tokenizer: any Tokenizers.Tokenizer, spatialMergeSize: Int,
        minPixels: Int, maxPixels: Int
    ) throws {
        self.model = model
        self.vision = vision
        self.visionBuilder = visionBuilder
        self.tokenizer = tokenizer
        self.spatialMergeSize = spatialMergeSize
        self.minPixels = minPixels
        self.maxPixels = maxPixels
        guard let imagePad = tokenizer.convertTokenToId("<|image_pad|>"),
              let visionStart = tokenizer.convertTokenToId("<|vision_start|>")
        else { throw Qwen25VLError.missingToken("vision special tokens") }
        self.imagePadId = imagePad
        self.visionStartId = visionStart
    }

    /// Bring the vision tower resident if it was evicted after a prior encode. Idempotent.
    private func ensureVision() throws -> QVLVision.VisionModel {
        if let vision { return vision }
        let rebuilt = try visionBuilder()
        vision = rebuilt
        return rebuilt
    }

    /// Drop the vision tower after image-encode, before LM decode (per-stage eviction). The ViT
    /// weights + its activation scratch are reclaimed so they don't sit resident through the
    /// autoregressive loop (where the LM's prefill/KV-cache transient is the live cost). `nil` + a
    /// GPU cache clear; the next `generate()` rebuilds via `ensureVision()`.
    private func evictVision() {
        vision = nil
        Memory.clearCache()
    }

    /// Load everything from a published mlx-community snapshot (self-contained:
    /// weights + config + preprocessor config + tokenizer files).
    public static func load(
        directory: URL, tokenizerSource: String = "Qwen/Qwen2.5-VL-3B-Instruct"
    ) async throws -> Qwen25VLPipeline {
        let loaded = try Qwen25VLLoader.loadModel(directory: directory)
        let processor = try Qwen25VLProcessorConfig.load(from: directory)

        // Vision config from config.json's vision_config (HF `in_chans` → `in_channels`).
        var visionDict = loaded.config.visionConfigJSON
        if let inChans = visionDict.removeValue(forKey: "in_chans") {
            visionDict["in_channels"] = inChans
        }
        visionDict["model_type"] = visionDict["model_type"] ?? "qwen2_5_vl"
        let visionData = try JSONSerialization.data(withJSONObject: visionDict)
        let visionConfig = try JSONDecoder().decode(
            Qwen25VLConfiguration.VisionConfiguration.self, from: visionData)

        // Vision-tower builder: constructs + loads + realizes the ViT from the (already-resident,
        // mmap-backed) checkpoint weights. Used for the initial residency AND to rebuild the tower
        // after the per-stage post-encode eviction in `generate()`. The captured `visionWeights`
        // are lazy MLXArrays over the mmap'd safetensors, so re-running the closure re-materializes
        // the ViT params on demand rather than holding a second eager copy.
        let quantization = loaded.config.quantization
        let visionWeights = loaded.visionWeights
        let buildVision: () throws -> QVLVision.VisionModel = {
            let vision = QVLVision.VisionModel(visionConfig)
            var vitWeights = visionWeights
            if let q = quantization {
                // Quantized repos may also quantize the ViT linears — mirror the file.
                quantize(model: vision, groupSize: q.groupSize, bits: q.bits) { path, _ in
                    vitWeights["\(path).scales"] != nil
                }
            }
            vitWeights = vision.sanitize(weights: vitWeights)
            try vision.update(
                parameters: ModuleParameters.unflattened(vitWeights), verify: [.noUnusedKeys])
            eval(vision)
            return vision
        }
        let vision = try buildVision()

        let tokenizer = try await AutoTokenizer.from(pretrained: tokenizerSource)
        return try Qwen25VLPipeline(
            model: loaded.model, vision: vision, visionBuilder: buildVision,
            tokenizer: tokenizer,
            spatialMergeSize: visionConfig.spatialMergeSize,
            minPixels: processor.minPixels, maxPixels: processor.maxPixels)
    }

    /// VQA over one image. Returns the decoded answer text.
    public func generate(
        image: CIImage, prompt: String,
        systemPrompt: String = Qwen25VLPipeline.defaultSystemPrompt,
        maxNewTokens: Int = 256
    ) throws -> String {
        // 1. Stock Qwen2.5-VL chat template.
        let text = "<|im_start|>system\n\(systemPrompt)<|im_end|>\n"
            + "<|im_start|>user\n"
            + "<|vision_start|><|image_pad|><|vision_end|>\(prompt)<|im_end|>\n"
            + "<|im_start|>assistant\n"

        // 2. Preprocess + ViT (per-stage: load → encode → evict). The tower runs once here, then is
        //    dropped before the LM decode loop so its weights+scratch don't sit resident through the
        //    autoregressive transient.
        let (patches, frame) = try Qwen25VLImageProcessing.preprocess(
            image: image, minPixels: minPixels, maxPixels: maxPixels)
        let vision = try ensureVision()
        let visionDtype = vision.patchEmbed.proj.weight.dtype
        let imageFeatures = vision(patches.asType(visionDtype), frames: [frame])  // (N, D)
        // Realize the encode before dropping the tower, so eviction reclaims the ViT — not a graph
        // still pending on its weights.
        eval(imageFeatures)
        evictVision()

        // 3. Tokenize, expand the single image_pad to the per-patch count.
        var ids = tokenizer.encode(text: text, addSpecialTokens: false)
        let padCount = frame.product / (spatialMergeSize * spatialMergeSize)
        if let idx = ids.firstIndex(of: imagePadId) {
            ids.replaceSubrange(idx...idx, with: Array(repeating: imagePadId, count: padCount))
        } else {
            throw Qwen25VLError.missingToken("<|image_pad|> not present after tokenization")
        }

        // 4. Embed + merge ViT features at the pad positions (slice + concat — never
        //    MLXArray subscript-setters; they are silent no-ops for this pattern).
        let inputIds = MLXArray(ids.map { Int32($0) }).expandedDimensions(axis: 0)  // (1, T)
        let textEmbeds = model.embedTokens(inputIds)
        let padPositions = ids.enumerated().filter { $0.element == imagePadId }.map(\.offset)
        guard padPositions.count == imageFeatures.dim(0) else {
            throw Qwen25VLError.imageProcessing(
                "pad count \(padPositions.count) != ViT tokens \(imageFeatures.dim(0))")
        }
        let embeds = try Self.mergeImageFeatures(
            textEmbeds: textEmbeds, imageFeatures: imageFeatures.asType(textEmbeds.dtype),
            padPositions: padPositions)

        // 5. 3D mRoPE position ids (verified exact vs HF get_rope_index).
        let (positionIds, nextPosition) = Self.positionIds(
            ids: ids, frame: frame, mergeSize: spatialMergeSize, imagePadId: imagePadId)

        // 6. Greedy decode with KV cache; both stop ids honored.
        let caches = (0..<model.config.numHiddenLayers).map { _ in QVLKVCache() }
        var hidden = model(
            inputEmbeddings: embeds, positionIds: positionIds, mask: .causal, caches: caches)
        var output: [Int] = []
        var position = nextPosition
        for _ in 0..<maxNewTokens {
            let logits = model.logits(hidden[0..., -1, 0...])
            let next = argMax(logits, axis: -1).item(Int.self)
            if next == Qwen25VLTokens.imEnd || next == Qwen25VLTokens.endOfText { break }
            output.append(next)

            let nextEmbed = model.embedTokens(
                MLXArray([Int32(next)]).expandedDimensions(axis: 0))
            let stepPos = MLXArray([Int32(position)]).reshaped(1, 1, 1)
            let stepIds = broadcast(stepPos, to: [3, 1, 1])
            hidden = model(
                inputEmbeddings: nextEmbed, positionIds: stepIds, mask: .none, caches: caches)
            position += 1
        }
        return tokenizer.decode(tokens: output)
    }

    /// Slot ViT features into the text embeddings at the (contiguous) pad positions.
    /// Slice + concatenate — MLXArray subscript-set scatter is a silent no-op here.
    static func mergeImageFeatures(
        textEmbeds: MLXArray, imageFeatures: MLXArray, padPositions: [Int]
    ) throws -> MLXArray {
        guard let start = padPositions.first, let last = padPositions.last,
              padPositions.count == last - start + 1
        else {
            throw Qwen25VLError.imageProcessing(
                "pad positions not a single contiguous block (\(padPositions.count) positions)")
        }
        let features = imageFeatures.ndim == 2
            ? imageFeatures[.newAxis, 0..., 0...]   // (1, N, D)
            : imageFeatures
        var parts: [MLXArray] = []
        if start > 0 { parts.append(textEmbeds[0..., ..<start, 0...]) }
        parts.append(features)
        let end = last + 1
        if end < textEmbeds.dim(1) { parts.append(textEmbeds[0..., end..., 0...]) }
        return concatenated(parts, axis: 1)
    }

    /// 3D position ids for a single-image prompt: text positions advance on all axes;
    /// vision tokens get the (t, h, w) grid anchored at the block start; text resumes
    /// after the largest grid coordinate. Returns (3, 1, T) + next position.
    static func positionIds(
        ids: [Int], frame: THW, mergeSize: Int, imagePadId: Int
    ) -> (MLXArray, Int) {
        let llmGridH = frame.h / mergeSize
        let llmGridW = frame.w / mergeSize

        var t = [Int32](); var h = [Int32](); var w = [Int32]()
        t.reserveCapacity(ids.count); h.reserveCapacity(ids.count); w.reserveCapacity(ids.count)

        var textPos: Int32 = 0
        var i = 0
        while i < ids.count {
            if ids[i] == imagePadId {
                let anchor = textPos
                for ti in 0..<frame.t {
                    for hi in 0..<llmGridH {
                        for wi in 0..<llmGridW {
                            t.append(anchor + Int32(ti))
                            h.append(anchor + Int32(hi))
                            w.append(anchor + Int32(wi))
                        }
                    }
                }
                i += frame.t * llmGridH * llmGridW
                textPos = anchor + Int32(max(frame.t, max(llmGridH, llmGridW)))
            } else {
                t.append(textPos); h.append(textPos); w.append(textPos)
                textPos += 1
                i += 1
            }
        }
        let stacked = stacked(
            [MLXArray(t), MLXArray(h), MLXArray(w)], axis: 0
        ).reshaped(3, 1, ids.count)
        return (stacked, Int(textPos))
    }
}
