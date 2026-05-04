// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 3/3 of the PPISP controller pipeline: AdaptiveAvgPool + MLP trunk
// + heads.
//
// Reads the [dsH * dsW * 64] pixel-feature buffer produced by the
// pixel-CNN stage, runs:
//
//   AdaptiveAvgPool2d((5, 5))       -> 25 cells x 64 channels = 1600 floats
//   Flatten (channel-major)         -> 1600 floats
//   concat priorExposure            -> 1601 floats
//   trunk0 (1601 -> 128) + ReLU
//   trunk1 (128  -> 128) + ReLU
//   trunk2 (128  -> 128) + ReLU
//   exposure_head (128 -> 1)
//   color_head    (128 -> 8)
//
// One block, 128 threads. Reduction across cell pixels is per-thread
// (each thread owns 12-13 of the 1600 (cell, channel) outputs); no
// cross-thread atomics needed because cells are non-overlapping.
//
// Inputs/outputs (lua names):
//   inputs:PixelFeatures   -> const float* (from pixel CNN stage)
//   inputs:priorExposure   -> float
//   inputs:weights         -> const float* (full controller weights buffer)
//   outputs:ControllerParams -> 1x9 float surface

#define CNN_FEATURE_DIM        64
#define POOL_GRID_H            5
#define POOL_GRID_W            5
#define POOL_CELL_COUNT        (POOL_GRID_H * POOL_GRID_W)        // 25
#define POOL_FEATURE_LEN       (POOL_CELL_COUNT * CNN_FEATURE_DIM) // 1600
#define MLP_INPUT_DIM          (POOL_FEATURE_LEN + 1)              // 1601
#define MLP_HIDDEN_DIM         128
#define COLOR_PARAMS_PER_FRAME 8

// Weight offsets (continuation of pixel_cnn.cu's). MUST match
// ppisp_controller_writer.flatten_controller_weights.
#define OFF_TRUNK0_W   2720                                        // OFF_CONV3_B + 64
#define OFF_TRUNK0_B   (OFF_TRUNK0_W + MLP_HIDDEN_DIM * MLP_INPUT_DIM)
#define OFF_TRUNK1_W   (OFF_TRUNK0_B + MLP_HIDDEN_DIM)
#define OFF_TRUNK1_B   (OFF_TRUNK1_W + MLP_HIDDEN_DIM * MLP_HIDDEN_DIM)
#define OFF_TRUNK2_W   (OFF_TRUNK1_B + MLP_HIDDEN_DIM)
#define OFF_TRUNK2_B   (OFF_TRUNK2_W + MLP_HIDDEN_DIM * MLP_HIDDEN_DIM)
#define OFF_EXP_W      (OFF_TRUNK2_B + MLP_HIDDEN_DIM)
#define OFF_EXP_B      (OFF_EXP_W + MLP_HIDDEN_DIM)
#define OFF_COL_W      (OFF_EXP_B + 1)
#define OFF_COL_B      (OFF_COL_W + COLOR_PARAMS_PER_FRAME * MLP_HIDDEN_DIM)

extern "C" __global__ void poolMlpProcess(
    int dsW,
    int dsH,
    float priorExposure,
    const float* __restrict__ weights,
    const float* __restrict__ pixelFeatures,
    cudaSurfaceObject_t outputSurf)
{
    __shared__ float gsPooled[POOL_FEATURE_LEN];   // 1600 floats =  6.4 KB
    __shared__ float gsHiddenA[MLP_HIDDEN_DIM];    //  128 floats =  0.5 KB
    __shared__ float gsHiddenB[MLP_HIDDEN_DIM];    //  128 floats =  0.5 KB

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Stage B: AdaptiveAvgPool2d -> Flatten (channel-major).
    //
    // gsPooled is laid out channel-major (matches PyTorch nn.Flatten on
    // [C, H, W]): gsPooled[c * 25 + cell].
    //
    // Memory access: each warp owns a cell, the warp's 32 lanes read
    // adjacent channels of the same pixel (lane reads channel `lane`
    // and `lane + 32`). 32 contiguous floats per warp instruction = 1
    // L2 cache line, perfectly coalesced. This is the hot path -- the
    // pixel-feature buffer is the only large blob the kernel touches,
    // and the channel stride (64) made the naive layout misuse cache.
    const int wid    = tid >> 5;     // warp id within block
    const int lane   = tid & 31;     // lane within warp
    const int nwarps = nthreads >> 5;

    for (int cell = wid; cell < POOL_CELL_COUNT; cell += nwarps)
    {
        int gy = cell / POOL_GRID_W;
        int gx = cell - gy * POOL_GRID_W;

        int hStart = (gy * dsH) / POOL_GRID_H;
        int hEnd   = ((gy + 1) * dsH + POOL_GRID_H - 1) / POOL_GRID_H;
        hEnd = min(hEnd, dsH);
        int wStart = (gx * dsW) / POOL_GRID_W;
        int wEnd   = ((gx + 1) * dsW + POOL_GRID_W - 1) / POOL_GRID_W;
        wEnd = min(wEnd, dsW);

        float sum0 = 0.0f, sum1 = 0.0f;
        int count = 0;
        for (int dy = hStart; dy < hEnd; ++dy)
        {
            for (int dx = wStart; dx < wEnd; ++dx)
            {
                int base = (dy * dsW + dx) * CNN_FEATURE_DIM;
                sum0 += pixelFeatures[base + lane];          // channel = lane
                sum1 += pixelFeatures[base + lane + 32];     // channel = lane + 32
                ++count;
            }
        }
        float invCount = (count > 0) ? 1.0f / (float)count : 0.0f;
        gsPooled[(lane     ) * POOL_CELL_COUNT + cell] = sum0 * invCount;
        gsPooled[(lane + 32) * POOL_CELL_COUNT + cell] = sum1 * invCount;
    }
    __syncthreads();

    // Stage C: trunk0 (1601 -> 128). Distribute outputs across threads.
    for (int o = tid; o < MLP_HIDDEN_DIM; o += nthreads)
    {
        float v = weights[OFF_TRUNK0_B + o];
        const float* row = weights + OFF_TRUNK0_W + o * MLP_INPUT_DIM;
        for (int i = 0; i < POOL_FEATURE_LEN; ++i)
            v += gsPooled[i] * row[i];
        v += priorExposure * row[POOL_FEATURE_LEN];
        gsHiddenA[o] = fmaxf(0.0f, v);
    }
    __syncthreads();

    // trunk1 (128 -> 128).
    for (int o = tid; o < MLP_HIDDEN_DIM; o += nthreads)
    {
        float v = weights[OFF_TRUNK1_B + o];
        const float* row = weights + OFF_TRUNK1_W + o * MLP_HIDDEN_DIM;
        for (int i = 0; i < MLP_HIDDEN_DIM; ++i)
            v += gsHiddenA[i] * row[i];
        gsHiddenB[o] = fmaxf(0.0f, v);
    }
    __syncthreads();

    // trunk2 (128 -> 128).
    for (int o = tid; o < MLP_HIDDEN_DIM; o += nthreads)
    {
        float v = weights[OFF_TRUNK2_B + o];
        const float* row = weights + OFF_TRUNK2_W + o * MLP_HIDDEN_DIM;
        for (int i = 0; i < MLP_HIDDEN_DIM; ++i)
            v += gsHiddenB[i] * row[i];
        gsHiddenA[o] = fmaxf(0.0f, v);
    }
    __syncthreads();

    // heads. Single thread does exposure_head; first 8 threads do color_head.
    if (tid == 0)
    {
        float v = weights[OFF_EXP_B];
        const float* row = weights + OFF_EXP_W;
        for (int i = 0; i < MLP_HIDDEN_DIM; ++i)
            v += gsHiddenA[i] * row[i];
        surf2Dwrite<float>(v, outputSurf, 0 * (int)sizeof(float), 0);
    }
    if (tid < COLOR_PARAMS_PER_FRAME)
    {
        int o = tid;
        float v = weights[OFF_COL_B + o];
        const float* row = weights + OFF_COL_W + o * MLP_HIDDEN_DIM;
        for (int i = 0; i < MLP_HIDDEN_DIM; ++i)
            v += gsHiddenA[i] * row[i];
        surf2Dwrite<float>(v, outputSurf, (1 + o) * (int)sizeof(float), 0);
    }
}
