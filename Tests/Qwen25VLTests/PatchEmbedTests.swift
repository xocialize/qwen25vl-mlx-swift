import MLX
import MLXExactConv
import MLXNN
import MLXRandom
import XCTest
@testable import Qwen25VL

/// The vision patch embed is a 3-channel conv3d with kernel == stride. A plain conv3d at this shape
/// takes mlx's channel-pad helpers — `pad_and_slice_conv_3D_gpu`'s use-after-free in 0.32.2 (AB-T-0214)
/// and `pad_gpu`'s loss past 2^31 elements, which garbled requests over 262,144 patches (AB-T-0466) —
/// so `QVLPatchEmbed` computes it as a matmul. Its output must still match a plain conv3d on the CPU. (The
/// >262,144-patch check runs on the GPU in a CLI probe, not here: SPM-test kernels are unreliable.)
final class PatchEmbedTests: XCTestCase {
    func testPatchEmbedMatchesCPUConv3d() {
        let embed = QVLPatchEmbed(patchSize: 14, temporalPatchSize: 2, inChannels: 3, outputDimensions: 64)
        // The vision tower's input: flattened (C, T, P, P) patches, one row per patch.
        let x = MLXRandom.uniform(low: -1, high: 1, [37, 3 * 2 * 14 * 14], key: MLXRandom.key(7))
        let y = Device.withDefaultDevice(Device(.gpu)) { embed(x) }
        eval(y)

        let patches = x.reshaped(-1, 3, 2, 14, 14).movedAxis(source: 1, destination: 4)
        XCTAssertTrue(
            ExactConv.takesPadHelper3D(input: patches, weight: embed.proj.weight),
            "a plain Conv3d at this shape would take mlx's pad helper")
        var reference = MLXArray(0)
        withMLXCPU {
            reference = conv3d(patches, embed.proj.weight, stride: [2, 14, 14]).reshaped(-1, 64)
            eval(reference)
        }
        XCTAssertEqual(y.shape, [37, 64])
        // fp32 GEMM on NAX (M5) GPUs is TF32-class by MLX's default (R-NUM-1: 7.7e-4 vs fp64 at 2048³);
        // the old conv route took the exact implicit-GEMM path. bf16 production is unaffected.
        XCTAssertLessThan(abs(y - reference).max().item(Float.self), 2e-3)
    }
}
