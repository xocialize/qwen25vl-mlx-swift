// MaterializationTests.swift — Qwen2.5-VL through the engine's MAT gate (offline, no network):
// the WeightSourcing declaration, fresh-machine honesty, explicit-path satisfaction, and the
// store-layout probe/resolution. One self-contained snapshot per quant tier (bf16 / int4), so
// the gate runs per selectable tier — the declaration follows the quant via `defaultRepo(for:)`.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXQwen25VL

final class MaterializationTests: XCTestCase {

    /// Temp dir holding the probe files (+ one weights shard) that make an explicit-dir
    /// configuration read as satisfied.
    private func satisfiedDir() throws -> (dir: URL, cleanup: () -> Void) {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "qwen25vl-mat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for file in Qwen25VLConfiguration.probeFiles + ["model-00001-of-00002.safetensors"] {
            FileManager.default.createFile(
                atPath: dir.appending(path: file).path, contents: Data([0]))
        }
        return (dir, { try? FileManager.default.removeItem(at: dir) })
    }

    // MARK: - Engine MAT gate (per selectable quant tier)

    func testMATGatePerQuantTier() throws {
        for quant in [Quant.bf16, Quant.int4] {
            let (dir, cleanup) = try satisfiedDir()
            let report = MaterializationConformance.check(
                freshConfiguration: Qwen25VLConfiguration(quant: quant),
                satisfiedConfiguration: Qwen25VLConfiguration(quant: quant, snapshotDirectory: dir))
            XCTAssertTrue(report.passed, "\(quant): \(report.summary)")
            cleanup()
        }
    }

    // MARK: - Source declaration shape

    func testDeclarationFollowsQuant() {
        let bf16 = Qwen25VLConfiguration()
        XCTAssertEqual(bf16.weightSources.map(\.role), ["main"])
        XCTAssertEqual(bf16.weightSources[0].repo, "mlx-community/Qwen2.5-VL-3B-Instruct-bf16")
        XCTAssertNil(bf16.weightSources[0].matching)   // whole self-contained snapshot

        let int4 = Qwen25VLConfiguration(quant: .int4)
        XCTAssertEqual(int4.weightSources[0].repo, "mlx-community/Qwen2.5-VL-3B-Instruct-4bit")

        // An explicit repo always wins over the quant-derived default.
        let pinned = Qwen25VLConfiguration(repo: "org/custom", quant: .int4)
        XCTAssertEqual(pinned.weightSources[0].repo, "org/custom")
    }

    // MARK: - Store-layout probe + resolution

    func testStoreLayoutSatisfiesAndResolves() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "qwen25vl-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = Qwen25VLConfiguration()
        // Empty store: the source is missing.
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 1)
        // A bare directory (materializer creates it before the download completes) is NOT enough.
        let dir = root.appending(path: cfg.repo)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 1)
        // The full probe set + a weights shard satisfies.
        for file in Qwen25VLConfiguration.probeFiles + ["model-00001-of-00002.safetensors"] {
            FileManager.default.createFile(
                atPath: dir.appending(path: file).path, contents: Data([0]))
        }
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
        // Resolution lands on the store layout; an explicit dir always wins.
        XCTAssertEqual(cfg.resolved(storeRoot: root).snapshotDirectory?.path, dir.path)
        let explicit = Qwen25VLConfiguration(snapshotDirectory: URL(fileURLWithPath: "/x"))
            .resolved(storeRoot: root)
        XCTAssertEqual(explicit.snapshotDirectory?.path, "/x")
    }

    func testPrewarmPathsUseResolvedStoreLayout() {
        let root = URL(fileURLWithPath: "/tmp/some-store")
        let cfg = Qwen25VLConfiguration(modelsRootDirectory: root)
        let expected = root.appending(path: "mlx-community/Qwen2.5-VL-3B-Instruct-bf16")
        XCTAssertEqual(
            cfg.prewarmPaths.map(\.path),
            [expected.appending(path: "model.safetensors.index.json").path, expected.path])
    }

    func testCodableRoundTrip() throws {
        let cfg = Qwen25VLConfiguration(
            quant: .int4, snapshotDirectory: URL(fileURLWithPath: "/x"))
        let decoded = try JSONDecoder().decode(Qwen25VLConfiguration.self,
                                               from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.repo, cfg.repo)
        XCTAssertEqual(decoded.quant, .int4)
        XCTAssertNil(decoded.snapshotDirectory)   // environment-specific, never encoded
    }
}
