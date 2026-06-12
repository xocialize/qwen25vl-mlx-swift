import CoreImage
import Foundation
import MLX
import XCTest

@testable import Qwen25VL

/// ChartQA subset runner (gated: QVL_CHARTQA=1). Writes answers.json next to the
/// manifest for Python relaxed-accuracy scoring. Uses the lmms-eval Qwen-VL prompt
/// convention ("single word or phrase") so answers are scoreable.
final class ChartQATests: XCTestCase {
    static let subsetDir = URL(fileURLWithPath:
        "/Volumes/DEV_VOL1/VideoResearch/qwen25vl-mlx-models/chartqa-subset")

    struct Case: Codable {
        let image: String
        let question: String
        let answer: String
    }

    func testChartQASubset() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QVL_CHARTQA"] == "1", "QVL_CHARTQA=1")

        let manifest = try JSONDecoder().decode(
            [Case].self,
            from: Data(contentsOf: Self.subsetDir.appendingPathComponent("manifest.json")))

        let t0 = Date()
        let pipe = try await Qwen25VLPipeline.load(directory: OracleSmokeTests.weightsDir)
        let loadTime = -t0.timeIntervalSinceNow

        var results: [[String: String]] = []
        var totalTime = 0.0
        for (i, c) in manifest.enumerated() {
            let url = Self.subsetDir.appendingPathComponent("images/\(c.image)")
            guard let image = CIImage(contentsOf: url) else {
                XCTFail("unreadable image \(c.image)"); continue
            }
            let prompt = c.question
                + "\nAnswer the question using a single word or phrase."
            let t = Date()
            let answer = try pipe.generate(image: image, prompt: prompt, maxNewTokens: 64)
            let dt = -t.timeIntervalSinceNow
            totalTime += dt
            results.append(["image": c.image, "question": c.question,
                            "gt": c.answer, "pred": answer])
            print("[\(i + 1)/\(manifest.count)] (\(String(format: "%.1f", dt))s) "
                + "gt=\(c.answer) pred=\(answer.prefix(60))")
        }

        let peakGB = Double(GPU.peakMemory) / 1_073_741_824
        print(String(format: "load %.1fs · mean %.2fs/case · peak GPU %.2f GB",
                     loadTime, totalTime / Double(manifest.count), peakGB))

        let out = try JSONSerialization.data(
            withJSONObject: results, options: [.prettyPrinted])
        try out.write(to: Self.subsetDir.appendingPathComponent("answers.json"))
    }
}
