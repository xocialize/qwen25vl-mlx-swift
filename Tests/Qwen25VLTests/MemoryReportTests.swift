import CoreImage
import MLX
import XCTest
@testable import Qwen25VL

/// Efficiency-adoption memory harness (engine 1.14 split footprint). Measures, per quant, the
/// resident weights floor and the **transient** activation peak of a real image→text generation,
/// so `Qwen25VLPackage.manifest` can declare `QuantFootprint(residentBytes:peakActivationBytes:)`
/// honestly instead of the old flat number.
///
/// Per the Qwen-LLM prefill-scratch lesson, the transient is MEASURED, not derived: for a VLM the
/// LM prefill scratch dominates and is inflated by the image-token count, so the peak is recorded
/// at a documented **(image resolution × maxTokens)** envelope.
///
/// Gated behind QVL_MEM_TESTS=1 (needs the Cmlx metallib bundle in .build/debug/; see CLAUDE.md).
/// Set QVL_WEIGHTS_DIR to the snapshot folder (defaults to the bf16 snapshot on DEV_VOL1).
final class MemoryReportTests: XCTestCase {
    static func weightsDir() -> URL {
        URL(fileURLWithPath:
            ProcessInfo.processInfo.environment["QVL_WEIGHTS_DIR"]
                ?? "/Volumes/DEV_VOL1/VideoResearch/qwen25vl-mlx-models/Qwen2.5-VL-3B-Instruct-bf16")
    }
    static let imageURL = URL(fileURLWithPath:
        ProcessInfo.processInfo.environment["QVL_IMAGE"]
            ?? "/Volumes/DEV_ARCHIVE/lance-mlx/tests/fixtures/images/image-understanding-case-02.png")

    func testMeasureSplitFootprint() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QVL_MEM_TESTS"] == "1",
            "set QVL_MEM_TESTS=1 (and copy the Cmlx bundle) to run the memory harness")

        let dir = Self.weightsDir()
        let pipe = try await Qwen25VLPipeline.load(directory: dir)

        // --- Resident floor: realize weights with one warmup, drop activations, read active. ---
        guard let image = CIImage(contentsOf: Self.imageURL) else {
            XCTFail("could not load \(Self.imageURL.path)"); return
        }
        let warmup = try pipe.generate(image: image, prompt: "Describe this image.", maxNewTokens: 8)
        XCTAssertFalse(warmup.isEmpty)
        Memory.clearCache()
        let floor = Memory.activeMemory

        // --- Transient peak: rebase peak to the floor, run the documented envelope, read peak. ---
        Memory.clearCache()
        Memory.peakMemory = 0   // rebase to current active (≈ weights)
        let maxTokens = 256     // realistic answer window (the declared envelope)
        let answer = try pipe.generate(
            image: image, prompt: "Describe this image in detail.", maxNewTokens: maxTokens)
        XCTAssertFalse(answer.isEmpty)
        let peak = Memory.peakMemory

        let transient = max(0, peak - floor)
        let gb = { (b: Int) in String(format: "%.2f GB", Double(b) / 1e9) }
        let imgW = Int(image.extent.width), imgH = Int(image.extent.height)
        print("""

        ===== Qwen2.5-VL memory report =====
        snapshot:        \(dir.lastPathComponent)
        envelope:        image \(imgW)x\(imgH), maxTokens=\(maxTokens)
        resident floor:  \(gb(floor))   (\(floor) B)  <- residentBytes
        worst peak:      \(gb(peak))   (\(peak) B)
        transient (peak-floor): \(gb(transient))   (\(transient) B)  <- peakActivationBytes
        ====================================

        """)
    }
}
