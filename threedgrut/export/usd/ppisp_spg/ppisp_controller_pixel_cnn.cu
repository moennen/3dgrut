// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 2/3 of the PPISP controller pipeline: pixel-wise CNN encoder.
//
// Mirrors the per-ds-pixel slice of the PyTorch:
//
//     conv1 (3 -> 16, 1x1)
//     MaxPool 3x3 stride 3 (folded into the inner loop)
//     ReLU
//     conv2 (16 -> 32, 1x1)
//     ReLU
//     conv3 (32 -> 64, 1x1)
//
// One thread per ds-pixel, ~256 threads/block (16x16). At 256x256 input
// the grid is 29 x 29 blocks ~ 850 blocks ~ saturates a modern GPU.
// All conv weights stay in L1/L2 after the first warp loads them.
//
// Inputs/outputs (lua names):
//   inputs:Resized       -> cudaTextureObject_t (target-res RGBA float4)
//   inputs:weights       -> const float* (full controller weights buffer)
//   outputs:PixelFeatures -> float* device buffer, layout
//                            [dy * dsW + dx][c], 64 channels, length dsH*dsW*64.
//
// dsH = inH / 3, dsW = inW / 3 (the 3x MaxPool stride). The lua sets the
// output buffer shape accordingly.

// Layout constants (must match ppisp_controller.cu / ppisp_controller_writer.py)
#define INPUT_DOWNSAMPLING  3
#define CNN_FEATURE_DIM     64

// Weight offsets (from ppisp_controller_writer.flatten_controller_weights)
#define OFF_CONV1_W   0
#define OFF_CONV1_B   (OFF_CONV1_W + 16 * 3)            // 48
#define OFF_CONV2_W   (OFF_CONV1_B + 16)                // 64
#define OFF_CONV2_B   (OFF_CONV2_W + 32 * 16)           // 576
#define OFF_CONV3_W   (OFF_CONV2_B + 32)                // 608
#define OFF_CONV3_B   (OFF_CONV3_W + 64 * 32)           // 2656

extern "C" __global__ void pixelCnnProcess(
    int inW,
    int inH,
    int dsW,
    int dsH,
    const float* __restrict__ weights,
    cudaTextureObject_t inputTex,
    float* __restrict__ pixelFeatures)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dsW || dy >= dsH)
        return;

    int x0 = dx * INPUT_DOWNSAMPLING;
    int y0 = dy * INPUT_DOWNSAMPLING;
    int x1 = min(x0 + INPUT_DOWNSAMPLING, inW);
    int y1 = min(y0 + INPUT_DOWNSAMPLING, inH);

    // conv1 -> max-pool over 3x3.
    float pooled[16];
    #pragma unroll
    for (int c = 0; c < 16; ++c)
        pooled[c] = -3.402823e+38f;

    for (int yy = y0; yy < y1; ++yy)
    {
        for (int xx = x0; xx < x1; ++xx)
        {
            float4 s = tex2D<float4>(inputTex, xx, yy);
            float t[16];
            #pragma unroll
            for (int o = 0; o < 16; ++o)
            {
                float v = weights[OFF_CONV1_B + o];
                v += s.x * weights[OFF_CONV1_W + o * 3 + 0];
                v += s.y * weights[OFF_CONV1_W + o * 3 + 1];
                v += s.z * weights[OFF_CONV1_W + o * 3 + 2];
                t[o] = v;
            }
            #pragma unroll
            for (int c = 0; c < 16; ++c)
                pooled[c] = fmaxf(pooled[c], t[c]);
        }
    }

    // ReLU after max-pool (matches PyTorch order).
    #pragma unroll
    for (int c = 0; c < 16; ++c)
        pooled[c] = fmaxf(0.0f, pooled[c]);

    // conv2 (16 -> 32) + ReLU.
    float feat32[32];
    #pragma unroll
    for (int o = 0; o < 32; ++o)
    {
        float v = weights[OFF_CONV2_B + o];
        #pragma unroll
        for (int i = 0; i < 16; ++i)
            v += pooled[i] * weights[OFF_CONV2_W + o * 16 + i];
        feat32[o] = fmaxf(0.0f, v);
    }

    // conv3 (32 -> 64). No ReLU (matches PyTorch -- AdaptiveAvgPool follows directly).
    float* outPtr = pixelFeatures + (dy * dsW + dx) * CNN_FEATURE_DIM;
    #pragma unroll
    for (int o = 0; o < CNN_FEATURE_DIM; ++o)
    {
        float v = weights[OFF_CONV3_B + o];
        #pragma unroll
        for (int i = 0; i < 32; ++i)
            v += feat32[i] * weights[OFF_CONV3_W + o * 32 + i];
        outPtr[o] = v;
    }
}
