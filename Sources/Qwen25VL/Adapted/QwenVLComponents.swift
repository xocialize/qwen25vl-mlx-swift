// Adapted from mlx-swift-lm (https://github.com/ml-explore/mlx-swift-lm, MIT License)
// MLXVLM/Models/QwenVL.swift — VisionRotaryEmbedding + PatchEmbed, whose initializers
// are internal upstream. See NOTICE for attribution.

import Foundation
import MLX
import MLXExactConv
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

        // Through ExactConv, not `proj(_:)`: a 3-channel conv3d is what mlx 0.32.2's
        // `pad_and_slice_conv_3D_gpu` serves, and that helper frees its zero fill value while the GPU
        // still reads it — under a capped buffer cache (the Image Engine's 2 GiB) the process dies
        // with "Invalid Resource" (AB-T-0214). ExactConv pads the channels to 16 in the graph and
        // runs the same implicit GEMM; `proj` keeps the weight and its key.
        hiddenStates = ExactConv.conv3d(
            hiddenStates, weight: proj.weight, bias: proj.bias,
            stride: (temporalPatchSize, patchSize, patchSize))
        hiddenStates = hiddenStates.reshaped(-1, outputDimensions)
        return hiddenStates
    }
}
