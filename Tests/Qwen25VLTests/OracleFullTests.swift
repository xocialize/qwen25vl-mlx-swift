import CoreImage
import XCTest
@testable import Qwen25VL

/// All six Lance Phase-0 oracle images, semantic report (GPU-gated).
final class OracleFullTests: XCTestCase {
    func testAllSixOracleCases() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QVL_GPU_TESTS"] == "1", "QVL_GPU_TESTS=1")

        let pipe = try await Qwen25VLPipeline.load(directory: OracleSmokeTests.weightsDir)
        let cases: [(String, String)] = [
            ("01", "Is the largest segment greater than sum of all the other segments?"),
            ("02", "What percentage of respondents want better border security?"),
            ("03", "What is the license plate number of the car?"),
            ("04", "According to the data from the proprietary market research, how much amount was spent on the promotional meetings and events during 1998?"),
            ("05", "What is the appearance of the Colosseum in Rome, Italy?"),
            ("06", "How does a total solar eclipse look like from Earth?"),
        ]
        for (n, q) in cases {
            let url = OracleSmokeTests.imagesDir.appendingPathComponent(
                "image-understanding-case-\(n).png")
            let image = CIImage(contentsOf: url)!
            let t0 = Date()
            let answer = try pipe.generate(image: image, prompt: q)
            print("=== case \(n) (\(String(format: "%.1f", -t0.timeIntervalSinceNow))s)")
            print("    Q: \(q)")
            print("    A: \(answer)")
        }
    }
}
