import CoreImage
import XCTest
@testable import Qwen25VL

/// GPU integration smoke — gated behind QVL_GPU_TESTS=1 (needs the Cmlx metallib
/// bundle copied into .build/debug/; see the metallib workaround in CLAUDE.md).
/// Loads the bf16 snapshot and answers one oracle chart question.
final class OracleSmokeTests: XCTestCase {
    static let weightsDir = URL(fileURLWithPath:
        "/Volumes/DEV_VOL1/VideoResearch/qwen25vl-mlx-models/Qwen2.5-VL-3B-Instruct-bf16")
    static let imagesDir = URL(fileURLWithPath:
        "/Volumes/DEV_ARCHIVE/lance-mlx/tests/fixtures/images")

    func testLoadAndChartRead() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QVL_GPU_TESTS"] == "1",
            "set QVL_GPU_TESTS=1 (and copy the Cmlx bundle) to run")

        let pipe = try await Qwen25VLPipeline.load(directory: Self.weightsDir)

        let imageURL = Self.imagesDir.appendingPathComponent(
            "image-understanding-case-02.png")
        let image = CIImage(contentsOf: imageURL)!
        let answer = try pipe.generate(
            image: image,
            prompt: "What percentage of respondents want better border security?")
        print("case-02 answer: \(answer)")
        XCTAssertFalse(answer.isEmpty)
        // Semantic expectation (stock Qwen2.5-VL is benchmark-strong on charts):
        XCTAssertTrue(answer.contains("29"), "expected a 29% read, got: \(answer)")
    }
}
