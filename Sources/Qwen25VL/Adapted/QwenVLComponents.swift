// Adapted from mlx-swift-lm (https://github.com/ml-explore/mlx-swift-lm, MIT License)
// MLXVLM/Models/QwenVL.swift — VisionRotaryEmbedding + PatchEmbed, whose initializers
// are internal upstream. See NOTICE for attribution.

import Foundation
import MLX
import MLXNN

public class QVLVisionRotaryEmbedding {
    let dimensions: Int
    let theta: Float
    let inverseFreq: MLXArray

    public init(dimensions: Int, theta: Float) {
        self.dimensions = dimensions
        self.theta = theta
        let p = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / dimensions
        self.inverseFreq = 1.0 / pow(theta, p)
    }

    public func callAsFunction(sequenceLength: Int) -> MLXArray {
        let seq = MLXArray(0 ..< sequenceLength).asType(inverseFreq.dtype)
        let freqs = outer(seq, inverseFreq)
        return freqs
    }
}

public class QVLPatchEmbed: Module, UnaryLayer {
    @ModuleInfo var proj: Conv3d

    let patchSize: Int
    let temporalPatchSize: Int
    let inChannels: Int
    let outputDimensions: Int

    public convenience init(
        patchSize: Int, temporalPatchSize: Int, inChannels: Int, hiddenSize: Int
    ) {
        self.init(
            patchSize: patchSize, temporalPatchSize: temporalPatchSize,
            inChannels: inChannels, outputDimensions: hiddenSize)
    }

    public init(patchSize: Int, temporalPatchSize: Int, inChannels: Int, outputDimensions: Int) {
        self.patchSize = patchSize
        self.temporalPatchSize = temporalPatchSize
        self.inChannels = inChannels
        self.outputDimensions = outputDimensions

        let kernelSize = IntOrTriple([temporalPatchSize, patchSize, patchSize])
        self._proj.wrappedValue = Conv3d(
            inputChannels: inChannels,
            outputChannels: outputDimensions,
            kernelSize: kernelSize,
            stride: kernelSize,
            bias: false
        )
        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var hiddenStates = hiddenStates.reshaped(
            -1, inChannels, temporalPatchSize, patchSize, patchSize
        ).movedAxis(source: 1, destination: 4)

        // A matmul, not a conv (AB-T-0466). Kernel == stride, so each output row is one dot product of
        // a (T, P, P, C) patch with the (O, kT, kH, kW, C) weight — the same channels-last layout. The
        // conv route (ExactConv, and plain conv3d before it) padded the 3 channels to 16, and mlx's
        // `pad_gpu` loses values past 2^31 destination elements (ml-explore/mlx#4735; the fix,
        // ml-explore/mlx#4744, is unreleased in 0.32.x), so a request over 262,144 patches got garbage
        // rows for its newest images. The matmul pads nothing, so it also stays clear of mlx 0.32.2's
        // `pad_and_slice_conv_3D_gpu` use-after-free (AB-T-0214). `proj` keeps the weight and its key.
        let n = hiddenStates.dim(0)
        hiddenStates = matmul(
            hiddenStates.reshaped(n, -1), proj.weight.reshaped(outputDimensions, -1).T)
        if let bias = proj.bias { hiddenStates = hiddenStates + bias }
        return hiddenStates
    }
}
