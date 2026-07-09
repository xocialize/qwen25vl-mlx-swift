// TokenizerLocalLoadTests.swift — validates the snapshot-local tokenizer path
// (`AutoTokenizer.from(modelFolder:)`, the materialization retrofit's no-network load):
// the special tokens the pipeline keys on must resolve to the stock Qwen2.5-VL ids.
//
// Gated on QVL_TOKENIZER_DIR — a directory holding the published snapshot's tokenizer files
// (tokenizer.json + tokenizer_config.json + sidecars). Weights are NOT required, so this runs
// against a few MB where the full oracle needs the 7 GB snapshot:
//   QVL_TOKENIZER_DIR=/path/to/tokenizer-files swift test --filter TokenizerLocalLoadTests

import Foundation
import Tokenizers
import XCTest
@testable import Qwen25VL

final class TokenizerLocalLoadTests: XCTestCase {

    func testLocalFolderTokenizerCarriesVisionSpecialTokens() async throws {
        guard let dir = ProcessInfo.processInfo.environment["QVL_TOKENIZER_DIR"] else {
            throw XCTSkip("QVL_TOKENIZER_DIR not set")
        }
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: URL(fileURLWithPath: dir))

        // The ids `Qwen25VLPipeline` resolves at init / stops on during decode.
        XCTAssertEqual(tokenizer.convertTokenToId("<|image_pad|>"), 151655)
        XCTAssertEqual(tokenizer.convertTokenToId("<|vision_start|>"), 151652)
        XCTAssertEqual(tokenizer.convertTokenToId("<|im_end|>"), Qwen25VLTokens.imEnd)
        XCTAssertEqual(tokenizer.convertTokenToId("<|endoftext|>"), Qwen25VLTokens.endOfText)

        // The chat-template prefix must keep special tokens atomic (addSpecialTokens: false,
        // as the pipeline encodes).
        let ids = tokenizer.encode(
            text: "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>",
            addSpecialTokens: false)
        XCTAssertTrue(ids.contains(151655), "image_pad not atomic: \(ids)")
        XCTAssertTrue(ids.contains(151652), "vision_start not atomic: \(ids)")
    }
}
