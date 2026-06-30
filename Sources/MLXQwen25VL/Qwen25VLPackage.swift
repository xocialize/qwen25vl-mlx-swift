import CoreImage
import Foundation
import MLX
import MLXToolKit
import Qwen25VL

/// MLXEngine package: stock Qwen2.5-VL-3B-Instruct exposing the canonical `imageAnalysis` surface
/// (VLM "look at this image and answer"). Returns canonical structured text.
///
/// Engine-owned lifecycle (C13): the engine constructs from a `Qwen25VLConfiguration`, pages the
/// snapshot in with `load()`, drives `run(_:)`, and reclaims with `unload()`. Lifecycle is isolated
/// to `InferenceActor`, so "runs only in the serialization domain, no private queue" is
/// compiler-enforced. The non-`Sendable` `Qwen25VLPipeline` is held as actor-isolated state and never
/// crosses the isolation boundary.
///
/// V1 backs **imageAnalysis** only; `videoAnalysis` (frame-sampling path) is a future additive surface
/// against the same loaded model.
@InferenceActor
public final class Qwen25VLPackage: ModelPackage {
    public typealias Configuration = Qwen25VLConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Qwen2.5-VL weights are Apache-2.0; this port code is Apache-2.0 too.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "mlx-community/Qwen2.5-VL-3B-Instruct-bf16",
                revision: "main",
                tier: 1
            ),
            requirements: RequirementsManifest(
                // 3B VLM, split footprint (engine 1.14). MEASURED via the gated MemoryReportTests
                // harness at the envelope: image 800x557 (~0.45 MP → ~580 vision tokens after the
                // 28-px smart-resize factor) × maxTokens 256, after the per-stage vision-tower
                // eviction (the ViT is dropped before the LM decode loop).
                //   bf16: resident floor 7.51 GB, worst peak 9.98 GB → transient 2.47 GB.
                //   int4: resident floor 3.07 GB, worst peak 4.04 GB → transient 0.97 GB.
                // residentBytes = measured floor + a little overhead; peakActivationBytes = the
                // measured transient + ~20% headroom (it's the LM prefill/KV-cache scratch, image-
                // token-inflated — measured, not derived; it does NOT scale cleanly across quants,
                // so each quant is measured separately). The transient scales with the (image-res ×
                // maxTokens) envelope, like a resolution envelope — re-measure if it changes.
                footprints: [
                    QuantFootprint(
                        quant: .bf16, residentBytes: 7_700_000_000,
                        peakActivationBytes: 3_000_000_000),
                    QuantFootprint(
                        quant: .int4, residentBytes: 3_200_000_000,
                        peakActivationBytes: 1_200_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 15, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [
                SpecialtyWeight(.general, strength: 0.7),
            ],
            surfaces: [
                ImageAnalysisContract.descriptor(
                    name: "qwen2.5-vl-image",
                    summary: "Qwen2.5-VL-3B image understanding / VQA (MLX).",
                    modes: []
                )
            ]
        )
    }

    private let configuration: Configuration
    /// The resident pipeline, paged in by `load()`. `nil` until loaded. Non-`Sendable`, held as
    /// `InferenceActor`-isolated state.
    private var pipeline: Qwen25VLPipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    /// Page the snapshot in. Idempotent when already resident.
    ///
    /// V1 loads from the resolved local snapshot directory. The engine-driven HF auto-download into
    /// `modelsRootDirectory` is the next additive step; until then a missing `snapshotDirectory` is a
    /// configuration error (the published `mlx-community/Qwen2.5-VL-3B-Instruct-*` snapshot is
    /// self-contained: weights + config + preprocessor config).
    public func load() async throws {
        guard pipeline == nil else { return }
        guard let directory = configuration.snapshotDirectory else {
            throw PackageError.configurationMismatch(
                expected: "a local Qwen2.5-VL snapshot directory (snapshotDirectory)",
                got: "nil — HF auto-download into modelsRootDirectory is not yet wired"
            )
        }
        pipeline = try await Qwen25VLPipeline.load(directory: directory)
    }

    public func unload() async {
        pipeline = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    /// Run one `imageAnalysis` call. Decodes the canonical request, builds a `CIImage` from the
    /// artifact bytes, generates, and returns canonical text. Honors cancellation at the call
    /// boundary (the greedy decode loop is a single synchronous call — per-token cancellation is a
    /// future core enhancement, mirrored across the VLM packages).
    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .imageAnalysis,
              let analysis = request as? ImageAnalysisRequest
        else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        guard let image = CIImage(data: analysis.image.data) else {
            throw Qwen25VLPackageError.imageDecodeFailed
        }
        let text = try pipeline.generate(image: image, prompt: analysis.prompt)
        return ImageAnalysisResponse(text: text)
    }
}

extension Qwen25VLPackage {
    /// The author one-liner the engine registers: manifest + license-gated factory.
    public nonisolated static var registration: PackageRegistration {
        .of(Qwen25VLPackage.self)
    }
}

/// Errors raised at the package boundary, distinct from the engine's `PackageError`.
public enum Qwen25VLPackageError: Error, Sendable, Equatable {
    /// The `Image` artifact bytes did not decode to a `CIImage`.
    case imageDecodeFailed
}
