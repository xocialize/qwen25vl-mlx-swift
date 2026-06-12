import XCTest
@testable import Qwen25VL

/// CPU-pure checks (no MLXArray allocation — SPM test runs lack the metallib).
final class SmartResizeTests: XCTestCase {
    /// HF smart_resize agreement on the values exercised by the oracle images:
    /// 800×557 at the stock budget (min 3136 / max 12845056) stays near-native,
    /// rounded to 28-multiples — 812×560 (grid 40×58), matching AutoProcessor.
    func testTargetSizeNativeImage() throws {
        let (h, w) = try Qwen25VLImageProcessing.targetSize(
            height: 557, width: 800, factor: 28,
            minPixels: 3136, maxPixels: 12_845_056)
        XCTAssertEqual(h, 560)
        XCTAssertEqual(w, 812)
    }

    /// Oversized image scales DOWN into the budget on 28-multiples.
    func testTargetSizeDownscale() throws {
        let (h, w) = try Qwen25VLImageProcessing.targetSize(
            height: 4000, width: 6000, factor: 28,
            minPixels: 3136, maxPixels: 12_845_056)
        XCTAssertLessThanOrEqual(h * w, 12_845_056)
        XCTAssertEqual(h % 28, 0)
        XCTAssertEqual(w % 28, 0)
    }

    /// Tiny image scales UP to the minimum budget.
    func testTargetSizeUpscale() throws {
        let (h, w) = try Qwen25VLImageProcessing.targetSize(
            height: 40, width: 40, factor: 28,
            minPixels: 3136, maxPixels: 12_845_056)
        XCTAssertGreaterThanOrEqual(h * w, 3136)
        XCTAssertEqual(h % 28, 0)
        XCTAssertEqual(w % 28, 0)
    }

    func testConfigDecode() throws {
        let json = """
        {"hidden_size": 2048, "num_hidden_layers": 36, "intermediate_size": 11008,
         "num_attention_heads": 16, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
         "vocab_size": 151936, "rope_theta": 1000000.0, "tie_word_embeddings": true}
        """
        let config = try JSONDecoder().decode(
            Qwen25VLTextConfig.self, from: Data(json.utf8))
        XCTAssertEqual(config.headDim, 128)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertEqual(config.mropeSection, [16, 24, 24])
    }
}
