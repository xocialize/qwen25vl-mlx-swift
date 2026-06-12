import Foundation
import MLXToolKit

/// Init-time configuration for `Qwen25VLPackage` (C9): which published snapshot, its quant, and
/// where it lives on disk. Per-request prompt/image ride the `ImageAnalysisRequest`, not here.
///
/// V1 loads from a **local self-contained snapshot directory** (`mlx-community/Qwen2.5-VL-3B-Instruct-{bf16,4bit}`
/// downloaded as published — weights + config + preprocessor config). `snapshotDirectory` is the
/// resolved folder; the engine sets `modelsRootDirectory` from its `ModelStore` and a future revision
/// will auto-download the repo there (see `Qwen25VLPackage.load`).
public struct Qwen25VLConfiguration: PackageConfiguration, ModelStorable {
    /// Provenance repo id (also the HF source for the stock tokenizer the pipeline fetches).
    public var repo: String
    public var revision: String?
    public var quant: Quant
    /// Resolved local snapshot folder (weights + config + preprocessor config). Required in V1.
    /// Environment-specific, so excluded from `Codable`.
    public var snapshotDirectory: URL?
    /// Engine-chosen models root (where a future revision auto-materializes the snapshot). Also
    /// environment-specific → excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public init(
        repo: String = "mlx-community/Qwen2.5-VL-3B-Instruct-bf16",
        revision: String? = nil,
        quant: Quant = .bf16,
        snapshotDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.snapshotDirectory = snapshotDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant
    }
}
