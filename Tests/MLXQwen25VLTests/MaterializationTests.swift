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

    /// Write the full probe set + one weights shard into `dir`.
    private func populate(_ dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for file in Qwen25VLConfiguration.probeFiles + ["model-00001-of-00002.safetensors"] {
            FileManager.default.createFile(
                atPath: dir.appending(path: file).path, contents: Data([0]))
        }
    }

    private func tempStoreRoot() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "qwen25vl-store-\(UUID().uuidString)")
    }

    func testFlatStoreLayoutSatisfiesAndResolves() throws {
        // The engine-executed FLAT layout (contract 1.24): files directly under
        // `<root>/models--<org>--<name>/` — where MLXServeEngine's materializer (and this
        // package's defensive WeightMaterializer) land them.
        let root = tempStoreRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = Qwen25VLConfiguration()
        // Empty store: the source is missing.
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 1)
        // A bare directory (materializer creates it before the download completes) is NOT enough.
        let dir = root.appending(path: "models--mlx-community--Qwen2.5-VL-3B-Instruct-bf16")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 1)
        // config.json alone satisfies the engine's default probe but NOT this package's.
        FileManager.default.createFile(
            atPath: dir.appending(path: "config.json").path, contents: Data([0]))
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 1)
        // The full probe set + a weights shard satisfies.
        try populate(dir)
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
        // No hub snapshot exists → resolution lands on the flat repo dir; an explicit dir wins.
        XCTAssertEqual(cfg.resolvedModelDirectory(storeRoot: root)?.path, dir.path)
        XCTAssertEqual(cfg.resolved(storeRoot: root).snapshotDirectory?.path, dir.path)
        let explicit = Qwen25VLConfiguration(snapshotDirectory: URL(fileURLWithPath: "/x"))
        XCTAssertEqual(explicit.resolvedModelDirectory(storeRoot: root)?.path, "/x")
        // The sibling quant tier is a DIFFERENT repo dir — still missing.
        XCTAssertEqual(
            Qwen25VLConfiguration(quant: .int4).missingWeightSources(storeRoot: root).count, 1)
    }

    func testHubSnapshotLayoutSatisfiesAndResolvesSnapshotFirst() throws {
        // The hub-client layout (MS-1): `models--<org>--<name>/snapshots/<commit>/…` behind
        // `refs/main`. The probe accepts it, and resolution prefers the snapshot dir.
        let root = tempStoreRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let repoDir = root.appending(path: "models--mlx-community--Qwen2.5-VL-3B-Instruct-bf16")
        let snapshot = repoDir.appending(path: "snapshots/abc123")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: repoDir.appending(path: "refs"), withIntermediateDirectories: true)
        try Data("abc123".utf8).write(to: repoDir.appending(path: "refs/main"))
        let cfg = Qwen25VLConfiguration()
        // A half-landed snapshot (config.json only) is not a materialized model.
        FileManager.default.createFile(
            atPath: snapshot.appending(path: "config.json").path, contents: Data([0]))
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 1)
        try populate(snapshot)
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
        XCTAssertEqual(cfg.resolvedModelDirectory(storeRoot: root)?.path, snapshot.path)
    }

    func testLegacyNestedLayoutReadsAsMissing() throws {
        // The pre-MS-1 `<root>/<org>/<name>` form survives only as a MARKER-read tolerance —
        // weights there were never where the hub client lands them, so it must not satisfy.
        let root = tempStoreRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try populate(root.appending(path: "mlx-community/Qwen2.5-VL-3B-Instruct-bf16"))
        XCTAssertEqual(Qwen25VLConfiguration().missingWeightSources(storeRoot: root).count, 1)
    }

    func testPrewarmPathsUseResolvedStoreLayout() {
        let root = URL(fileURLWithPath: "/tmp/some-store")
        let cfg = Qwen25VLConfiguration(modelsRootDirectory: root)
        // Nothing materialized at this root → no snapshot to prefer → the flat repo dir
        // (MS-1 `models--<org>--<name>`).
        let expected = root.appending(path: "models--mlx-community--Qwen2.5-VL-3B-Instruct-bf16")
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
