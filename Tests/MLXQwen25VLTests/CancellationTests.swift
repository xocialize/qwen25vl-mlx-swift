// CancellationTests.swift — Qwen2.5-VL through the engine's CAN gate (offline, no MLX kernels):
// CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before notLoaded
// validation or weights). CAN-3 is the document of record for the checkpoint cadence: the
// pipeline's greedy decode loop checks `try Task.checkCancellation()` once per generated token,
// with a stage checkpoint at the post-encode/pre-decode seam (after the vision-tower eviction);
// the vision encode itself is one monolithic ViT forward — no loop to checkpoint.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXQwen25VL

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = Qwen25VLPackage(configuration: Qwen25VLConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: ImageAnalysisRequest(
                image: Image(format: .png, data: Data([0])),
                prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // imageAnalysis is not a long-run capability, but the bf16 tier's measured peak
        // activation (3.0 GB — LM prefill/KV scratch, image-token-inflated) crosses the 2 GB
        // threshold, so the sub-second exemption is not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: Qwen25VLPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: Qwen25VLPackage.manifest,
            posture: .cadence([
                // The greedy LM decode loop checks `try Task.checkCancellation()` once per
                // generated token (Qwen25VLPipeline.generate, the maxNewTokens loop), plus a
                // stage checkpoint after vision-encode/eviction, before decode. `generate`
                // runs synchronously on the run's task, so the flag is visible and the
                // CancellationError rethrows unchanged through the throwing signature.
                .init(phase: .generate, unit: .token),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
