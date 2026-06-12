import CoreImage
import Foundation
import MLX
import MLXNN
import Tokenizers
import XCTest

@testable import Qwen25VL

/// Integer-artifact + CPU-stream op parity vs the HF/mlx-vlm reference
/// (tools/dump_hf_reference.py). Gated behind QVL_PARITY_TESTS=1.
///
/// Doctrine (mlx-porting skill): integer artifacts must be EXACT; float ops are
/// compared fp32-vs-fp32 on the CPU stream (GPU fp32 matmul carries ~8e-4 rel
/// accumulation noise on M-series and must not be used for op parity).
final class ParityTests: XCTestCase {
    static let parityDir = URL(fileURLWithPath:
        "/Volumes/DEV_VOL1/VideoResearch/qwen25vl-mlx-models/parity")
    static let question = "What percentage of respondents want better border security?"

    func testCase02Parity() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QVL_PARITY_TESTS"] == "1",
            "set QVL_PARITY_TESTS=1 (and dump the reference) to run")

        Device.setDefault(device: Device.cpu)

        let ref = try MLX.loadArrays(
            url: Self.parityDir.appendingPathComponent("case02.safetensors"))
        guard let refIds = ref["input_ids"], let refPos = ref["position_ids"],
              let refPixels = ref["pixel_values"], let refGrid = ref["image_grid_thw"],
              let refFeats = ref["vit_features_fp32_cpu"]
        else { return XCTFail("reference artifacts missing") }

        // --- 1. Pixels: our PIL-exact preprocess vs the HF processor ----------
        let imageURL = OracleSmokeTests.imagesDir.appendingPathComponent(
            "image-understanding-case-02.png")
        let image = CIImage(contentsOf: imageURL)!
        let (patches, frame) = try Qwen25VLImageProcessing.preprocess(
            image: image, minPixels: 3136, maxPixels: 12_845_056)
        XCTAssertEqual(patches.shape, refPixels.shape, "pixel layout differs")
        XCTAssertEqual(
            [frame.t, frame.h, frame.w],
            refGrid[0].asArray(Int32.self).map(Int.init), "grid differs")
        let pixDiff = abs(patches - refPixels)
        let pixMax = pixDiff.max().item(Float.self)
        // ±1 uint8 LSB in normalized space = 1/(255·min(std)) ≈ 0.01504 — the known
        // tolerance of PIL-exact resize + CIImage sRGB decode vs PIL's PNG decode.
        let oneLSB: Float = 1.0 / (255.0 * 0.26130258) + 1e-4
        let fracOff = (pixDiff .> 1e-6).asType(.float32).mean().item(Float.self)
        print("pixel max|diff| = \(pixMax) (1 LSB = \(oneLSB)), frac>0: \(fracOff)")
        XCTAssertLessThanOrEqual(pixMax, oneLSB, "pixels differ by more than ±1 LSB")
        XCTAssertLessThan(fracOff, 0.05, "too many pixels off — not a resampler LSB effect")

        // --- 2. Input ids: template + pad expansion EXACT ----------------------
        let tokenizer = try await AutoTokenizer.from(
            pretrained: "Qwen/Qwen2.5-VL-3B-Instruct")
        let text = "<|im_start|>system\n\(Qwen25VLPipeline.defaultSystemPrompt)<|im_end|>\n"
            + "<|im_start|>user\n"
            + "<|vision_start|><|image_pad|><|vision_end|>\(Self.question)<|im_end|>\n"
            + "<|im_start|>assistant\n"
        var ids = tokenizer.encode(text: text, addSpecialTokens: false)
        let imagePadId = tokenizer.convertTokenToId("<|image_pad|>")!
        let padCount = frame.product / 4
        let idx = ids.firstIndex(of: imagePadId)!
        ids.replaceSubrange(idx...idx, with: Array(repeating: imagePadId, count: padCount))
        let refIdsArr = refIds.asArray(Int32.self).map(Int.init)
        XCTAssertEqual(ids, refIdsArr, "input_ids differ from HF apply_chat_template")

        // --- 3. Position ids: EXACT vs HF get_rope_index -----------------------
        let (positions, _) = Qwen25VLPipeline.positionIds(
            ids: ids, frame: frame, mergeSize: 2, imagePadId: imagePadId)
        let posEqual = (positions.asType(.int32) .== refPos.asType(.int32))
            .all().item(Bool.self)
        XCTAssertTrue(posEqual, "position_ids differ from HF get_rope_index")

        // --- 4. ViT features: fp32 CPU vs mlx-vlm fp32 CPU ---------------------
        let weightsDir = OracleSmokeTests.weightsDir
        let configData = try Data(
            contentsOf: weightsDir.appendingPathComponent("config.json"))
        let root = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
        var visionDict = root["vision_config"] as! [String: Any]
        if let inChans = visionDict.removeValue(forKey: "in_chans") {
            visionDict["in_channels"] = inChans
        }
        visionDict["model_type"] = visionDict["model_type"] ?? "qwen2_5_vl"
        let visionConfig = try JSONDecoder().decode(
            MLXVLMVisionConfigurationAlias.self,
            from: JSONSerialization.data(withJSONObject: visionDict))
        let vision = QVLVision.VisionModel(visionConfig)
        var vit = try Qwen25VLLoader.loadAllArrays(directory: weightsDir)
            .reduce(into: [String: MLXArray]()) { acc, kv in
                if kv.key.hasPrefix("vision_tower.") {
                    acc[String(kv.key.dropFirst("vision_tower.".count))] = kv.value
                }
            }
        vit = vision.sanitize(weights: vit)
        vit = vit.mapValues { $0.asType(.float32) }
        try vision.update(
            parameters: ModuleParameters.unflattened(vit), verify: [.noUnusedKeys])
        eval(vision)

        // OP parity: identical inputs (reference pixels) → identical features.
        let featsOp = vision(refPixels.asType(.float32), frames: [frame])
        XCTAssertEqual(featsOp.shape, refFeats.shape)
        func cosine(_ x: MLXArray, _ y: MLXArray) -> Float {
            let a = x.asType(.float32).flattened()
            let b = y.asType(.float32).flattened()
            return ((a * b).sum() / (sqrt(a.square().sum()) * sqrt(b.square().sum())))
                .item(Float.self)
        }
        let opCos = cosine(featsOp, refFeats)
        print("vit OP parity (ref pixels, fp32, cpu): cosine=\(opCos)")
        XCTAssertGreaterThan(opCos, 0.99999, "ViT op parity failed on the CPU stream")

        // Informational: end-to-end features from OUR pixels (carries the ±1 LSB).
        let featsOwn = vision(patches.asType(.float32), frames: [frame])
        print("vit e2e (own pixels): cosine=\(cosine(featsOwn, refFeats))")
    }
}

/// MLXVLM's VisionConfiguration, decoded from the checkpoint vision_config.
typealias MLXVLMVisionConfigurationAlias = MLXVLM.Qwen25VLConfiguration.VisionConfiguration

import MLXVLM
