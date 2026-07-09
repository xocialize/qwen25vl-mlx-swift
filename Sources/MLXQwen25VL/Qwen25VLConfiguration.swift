import Foundation
import MLXToolKit

/// Init-time configuration for `Qwen25VLPackage` (C9): which published snapshot, its quant, and
/// where it lives on disk. Per-request prompt/image ride the `ImageAnalysisRequest`, not here.
///
/// Loads from a **self-contained snapshot directory** (`mlx-community/Qwen2.5-VL-3B-Instruct-{bf16,4bit}`
/// as published — weights + config + preprocessor config + tokenizer files). `snapshotDirectory`
/// is the explicit-directory escape hatch (dev mode — never touches the network); when it is nil,
/// `load()` auto-materializes the declared `weightSources` into the engine-stamped
/// `modelsRootDirectory` (ModelStore layout, `<root>/<org>/<name>`) and loads from there.
public struct Qwen25VLConfiguration: PackageConfiguration, ModelStorable {
    /// Snapshot repo id — also the provenance repo. Defaults per quant tier via
    /// `defaultRepo(for:)`; pass explicitly to pin a different published snapshot.
    public var repo: String
    public var revision: String?
    public var quant: Quant
    /// Explicit local snapshot folder (dev escape hatch — never touches the network).
    /// Environment-specific, so excluded from `Codable`.
    public var snapshotDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Set by the engine from its
    /// `ModelStore`. Also environment-specific → excluded from `Codable`.
    public var modelsRootDirectory: URL?

    /// The published mlx-community snapshot for a quant tier (bf16 and 4bit are both published).
    public static func defaultRepo(for quant: Quant) -> String {
        switch quant {
        case .int4: return "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
        default: return "mlx-community/Qwen2.5-VL-3B-Instruct-bf16"
        }
    }

    public init(
        repo: String? = nil,
        revision: String? = nil,
        quant: Quant = .bf16,
        snapshotDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo ?? Self.defaultRepo(for: quant)
        self.revision = revision
        self.quant = quant
        self.snapshotDirectory = snapshotDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant
    }
}

/// Opt into per-quant footprint charging (ISSUES W1): the config already stores `quant`, so the
/// memory governor charges the matching declared `QuantFootprint` (bf16 vs int4) instead of the
/// largest-that-fits guess. Single-size model, so quant alone disambiguates — no `FootprintConfigured`.
extension Qwen25VLConfiguration: QuantConfigured {}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension Qwen25VLConfiguration: WeightSourcing {
    /// Files probed to decide a snapshot directory is materialized. Both published quant tiers
    /// carry all of them; `tokenizer.json` is load-bearing — the pipeline reads the tokenizer
    /// from the snapshot (no stock-repo network fetch).
    static let probeFiles = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ]

    /// ONE self-contained snapshot per quant tier — the whole repo, no globs (the quant axis is
    /// realized by which repo `defaultRepo(for:)` selects, not by file exclusion within one repo).
    public var weightSources: [WeightSource] {
        [WeightSource(role: "main", repo: repo, revision: revision)]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        // Explicit local directory first (dev escape hatch).
        if let dir = snapshotDirectory, Self.snapshotPresent(at: dir) { return [] }
        // Then the ModelStore layout (`<root>/<org>/<name>`).
        if let dir = ModelStore(root: storeRoot).directory(for: repo),
           Self.snapshotPresent(at: dir) {
            return []
        }
        return weightSources
    }

    /// All probe files present + at least one weights shard (the materializer creates the
    /// directory before the download completes, so bare-directory existence is not enough).
    static func snapshotPresent(at dir: URL) -> Bool {
        let fm = FileManager.default
        guard probeFiles.allSatisfy({ fm.fileExists(atPath: dir.appending(path: $0).path) })
        else { return false }
        let contents = (try? fm.contentsOfDirectory(atPath: dir.path)) ?? []
        return contents.contains { $0.hasSuffix(".safetensors") }
    }

    /// The configuration with a nil `snapshotDirectory` resolved to the store layout — what
    /// `load()` uses AFTER materialization. An explicit directory always wins.
    public func resolved(storeRoot: URL?) -> Qwen25VLConfiguration {
        var cfg = self
        if cfg.snapshotDirectory == nil {
            cfg.snapshotDirectory = ModelStore(root: storeRoot).directory(for: repo)
        }
        return cfg
    }
}

// MARK: - Cold-start prewarm

extension Qwen25VLConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        // Store-resolved snapshot; the prewarmer skips missing paths (first launch). The
        // directory is scanned recursively for the weight shards; the index file rides along as
        // a completeness probe so `needsDownload` doesn't read a half-materialized directory as
        // present.
        guard let dir = resolved(storeRoot: modelsRootDirectory).snapshotDirectory else { return [] }
        return [dir.appending(path: "model.safetensors.index.json"), dir]
    }
}
