// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

// PPISP Controller SPG CUDA kernel.
//
// CUDA port of ppisp_controller.slang. The slang variant is unsupported
// today because SPG's slang plugin cannot bind a USD float[] attribute
// to a StructuredBuffer<float> -- weights silently never reach the
// shader. The CUDA plugin already handles tensor-shaped USD params via
// its tensor-upload path (cudaXxx 'tensors:' input bindings), so the
// 241,961 trained weights flow through cleanly here.
//
// Architecture mirrors ppisp._PPISPController (default config):
//
//   Conv1x1(3->16, +bias)
//   MaxPool 3x3 stride 3
//   ReLU
//   Conv1x1(16->32, +bias)
//   ReLU
//   Conv1x1(32->64, +bias)
//   AdaptiveAvgPool2d((5,5))
//   Flatten -> 1600
//   concat prior_exposure -> 1601
//   MLP: 1601 -> 128 -> 128 -> 128, ReLU after each hidden layer
//   exposure_head: 128 -> 1
//   color_head:    128 -> 8
//
// Output surface (1x9 float, single-channel):
//   pixel (0,0): exposureOffset
//   pixel (1,0)..(8,0): color latents
//        [colorBlue.x, colorBlue.y,
//         colorRed.x,  colorRed.y,
//         colorGreen.x, colorGreen.y,
//         colorNeutral.x, colorNeutral.y]
//
// Single 32-thread block. Threads cooperate via shared memory through
// the pooled CNN features (1600 floats), then through the MLP hidden
// vectors (128 floats x 2 ping-pong).

// ---------------------------------------------------------------------------
// Architecture sizes (must match _PPISPController defaults).
// ---------------------------------------------------------------------------
#define CNN_FEATURE_DIM        64
#define POOL_GRID_H            5
#define POOL_GRID_W            5
#define POOL_CELL_COUNT        (POOL_GRID_H * POOL_GRID_W)        // 25
#define POOL_FEATURE_LEN       (POOL_CELL_COUNT * CNN_FEATURE_DIM) // 1600
#define MLP_INPUT_DIM          (POOL_FEATURE_LEN + 1)              // 1601
#define MLP_HIDDEN_DIM         128
#define COLOR_PARAMS_PER_FRAME 8
#define INPUT_DOWNSAMPLING     3
#define THREAD_GROUP_SIZE      32

// ---------------------------------------------------------------------------
// Weight buffer offsets (the Python writer flattens weights in this order
// into a single float buffer that gets bound as the `weights` USD attribute).
// MUST stay in lockstep with ppisp_controller.slang and
// flatten_controller_weights() in ppisp_controller_writer.py.
// ---------------------------------------------------------------------------
#define OFF_CONV1_W    0                                     // 16 * 3       = 48
#define OFF_CONV1_B    (OFF_CONV1_W + 16 * 3)                // + 16          = 64
#define OFF_CONV2_W    (OFF_CONV1_B + 16)                    // + 32 * 16     = 576
#define OFF_CONV2_B    (OFF_CONV2_W + 32 * 16)               // + 32          = 608
#define OFF_CONV3_W    (OFF_CONV2_B + 32)                    // + 64 * 32     = 2656
#define OFF_CONV3_B    (OFF_CONV3_W + 64 * 32)               // + 64          = 2720
#define OFF_TRUNK0_W   (OFF_CONV3_B + 64)                    // + 128 * 1601  = 207648
#define OFF_TRUNK0_B   (OFF_TRUNK0_W + MLP_HIDDEN_DIM * MLP_INPUT_DIM)
#define OFF_TRUNK1_W   (OFF_TRUNK0_B + MLP_HIDDEN_DIM)
#define OFF_TRUNK1_B   (OFF_TRUNK1_W + MLP_HIDDEN_DIM * MLP_HIDDEN_DIM)
#define OFF_TRUNK2_W   (OFF_TRUNK1_B + MLP_HIDDEN_DIM)
#define OFF_TRUNK2_B   (OFF_TRUNK2_W + MLP_HIDDEN_DIM * MLP_HIDDEN_DIM)
#define OFF_EXP_W      (OFF_TRUNK2_B + MLP_HIDDEN_DIM)
#define OFF_EXP_B      (OFF_EXP_W + MLP_HIDDEN_DIM)
#define OFF_COL_W      (OFF_EXP_B + 1)
#define OFF_COL_B      (OFF_COL_W + COLOR_PARAMS_PER_FRAME * MLP_HIDDEN_DIM)

// ---------------------------------------------------------------------------
// Per-pixel CNN building blocks
// ---------------------------------------------------------------------------

__device__ __forceinline__ void conv1Forward(
    const float* __restrict__ weights,
    float r, float g, float b,
    float feat[16])
{
    #pragma unroll
    for (int o = 0; o < 16; ++o)
    {
        float v = weights[OFF_CONV1_B + o];
        v += r * weights[OFF_CONV1_W + o * 3 + 0];
        v += g * weights[OFF_CONV1_W + o * 3 + 1];
        v += b * weights[OFF_CONV1_W + o * 3 + 2];
        feat[o] = v;
    }
}

__device__ __forceinline__ void conv2Forward(
    const float* __restrict__ weights,
    const float fin[16],
    float fout[32])
{
    #pragma unroll
    for (int o = 0; o < 32; ++o)
    {
        float v = weights[OFF_CONV2_B + o];
        #pragma unroll
        for (int i = 0; i < 16; ++i)
            v += fin[i] * weights[OFF_CONV2_W + o * 16 + i];
        fout[o] = v;
    }
}

__device__ __forceinline__ void conv3Forward(
    const float* __restrict__ weights,
    const float fin[32],
    float fout[CNN_FEATURE_DIM])
{
    #pragma unroll
    for (int o = 0; o < CNN_FEATURE_DIM; ++o)
    {
        float v = weights[OFF_CONV3_B + o];
        #pragma unroll
        for (int i = 0; i < 32; ++i)
            v += fin[i] * weights[OFF_CONV3_W + o * 32 + i];
        fout[o] = v;
    }
}

__device__ __forceinline__ void cnnForwardAtDownsampledPixel(
    const float* __restrict__ weights,
    cudaTextureObject_t inputTex,
    int inW, int inH,
    int dx, int dy,
    float feat64[CNN_FEATURE_DIM])
{
    int x0 = dx * INPUT_DOWNSAMPLING;
    int y0 = dy * INPUT_DOWNSAMPLING;
    int x1 = min(x0 + INPUT_DOWNSAMPLING, inW);
    int y1 = min(y0 + INPUT_DOWNSAMPLING, inH);

    float pooled[16];
    #pragma unroll
    for (int c = 0; c < 16; ++c)
        pooled[c] = -3.402823e+38f;

    for (int yy = y0; yy < y1; ++yy)
    {
        for (int xx = x0; xx < x1; ++xx)
        {
            float4 sample = tex2D<float4>(inputTex, xx, yy);
            float conv1Out[16];
            conv1Forward(weights, sample.x, sample.y, sample.z, conv1Out);
            #pragma unroll
            for (int c = 0; c < 16; ++c)
                pooled[c] = fmaxf(pooled[c], conv1Out[c]);
        }
    }

    #pragma unroll
    for (int c = 0; c < 16; ++c)
        pooled[c] = fmaxf(0.0f, pooled[c]);

    float feat32[32];
    conv2Forward(weights, pooled, feat32);
    #pragma unroll
    for (int c = 0; c < 32; ++c)
        feat32[c] = fmaxf(0.0f, feat32[c]);

    conv3Forward(weights, feat32, feat64);
}

__device__ __forceinline__ void adaptiveCellAverage(
    const float* __restrict__ weights,
    cudaTextureObject_t inputTex,
    int inW, int inH,
    int dsW, int dsH,
    int gx, int gy,
    float cellFeat[CNN_FEATURE_DIM])
{
    int hStart = (gy * dsH) / POOL_GRID_H;
    int hEnd   = ((gy + 1) * dsH + POOL_GRID_H - 1) / POOL_GRID_H;
    int wStart = (gx * dsW) / POOL_GRID_W;
    int wEnd   = ((gx + 1) * dsW + POOL_GRID_W - 1) / POOL_GRID_W;
    hEnd = min(hEnd, dsH);
    wEnd = min(wEnd, dsW);

    #pragma unroll
    for (int c = 0; c < CNN_FEATURE_DIM; ++c)
        cellFeat[c] = 0.0f;

    int count = 0;
    for (int dy = hStart; dy < hEnd; ++dy)
    {
        for (int dx = wStart; dx < wEnd; ++dx)
        {
            float feat64[CNN_FEATURE_DIM];
            cnnForwardAtDownsampledPixel(weights, inputTex, inW, inH, dx, dy, feat64);
            #pragma unroll
            for (int c = 0; c < CNN_FEATURE_DIM; ++c)
                cellFeat[c] += feat64[c];
            count += 1;
        }
    }

    float invCount = (count > 0) ? (1.0f / (float)count) : 0.0f;
    #pragma unroll
    for (int c = 0; c < CNN_FEATURE_DIM; ++c)
        cellFeat[c] *= invCount;
}

// ---------------------------------------------------------------------------
// Compute kernel: 1 block, THREAD_GROUP_SIZE threads.
// ---------------------------------------------------------------------------
extern "C" __global__ void controllerProcess(
    int inW,
    int inH,
    float priorExposure,
    const float* __restrict__ weights,
    cudaTextureObject_t inputTex,
    cudaSurfaceObject_t outputSurf)
{
    __shared__ float gsPooled[POOL_FEATURE_LEN];   // 1600 floats
    __shared__ float gsHiddenA[MLP_HIDDEN_DIM];    //  128 floats
    __shared__ float gsHiddenB[MLP_HIDDEN_DIM];    //  128 floats

    const int tid = threadIdx.x;

    int dsW = max(1, inW / INPUT_DOWNSAMPLING);
    int dsH = max(1, inH / INPUT_DOWNSAMPLING);

    // Phase 1: pool cells. 25 cells distributed across THREAD_GROUP_SIZE=32
    // threads -- only the first 25 threads do work in this phase.
    //
    // Layout note: PyTorch's nn.Flatten on the [N, C, H, W] CNN output
    // produces a *channel-major* flat layout -- feat[c * H*W + h*W + w].
    // The trunk0 weight matrix was trained against that layout, so
    // gsPooled MUST be stored channel-major as well, i.e.
    //     gsPooled[c * POOL_CELL_COUNT + cell].
    // (cell-major would silently permute every controller output.)
    if (tid < POOL_CELL_COUNT)
    {
        int cell = tid;
        int gy = cell / POOL_GRID_W;
        int gx = cell % POOL_GRID_W;

        float cellFeat[CNN_FEATURE_DIM];
        adaptiveCellAverage(weights, inputTex, inW, inH, dsW, dsH, gx, gy, cellFeat);

        #pragma unroll
        for (int c = 0; c < CNN_FEATURE_DIM; ++c)
            gsPooled[c * POOL_CELL_COUNT + cell] = cellFeat[c];
    }
    __syncthreads();

    // Phase 2: trunk0 (1601 -> 128). Output rows distributed across threads.
    for (int o = tid; o < MLP_HIDDEN_DIM; o += THREAD_GROUP_SIZE)
    {
        float v = weights[OFF_TRUNK0_B + o];
        for (int i = 0; i < POOL_FEATURE_LEN; ++i)
            v += gsPooled[i] * weights[OFF_TRUNK0_W + o * MLP_INPUT_DIM + i];
        v += priorExposure
             * weights[OFF_TRUNK0_W + o * MLP_INPUT_DIM + POOL_FEATURE_LEN];
        gsHiddenA[o] = fmaxf(0.0f, v);
    }
    __syncthreads();

    // Phase 3: trunk1 (128 -> 128). gsHiddenA -> gsHiddenB.
    for (int o = tid; o < MLP_HIDDEN_DIM; o += THREAD_GROUP_SIZE)
    {
        float v = weights[OFF_TRUNK1_B + o];
        for (int i = 0; i < MLP_HIDDEN_DIM; ++i)
            v += gsHiddenA[i] * weights[OFF_TRUNK1_W + o * MLP_HIDDEN_DIM + i];
        gsHiddenB[o] = fmaxf(0.0f, v);
    }
    __syncthreads();

    // Phase 4: trunk2 (128 -> 128). gsHiddenB -> gsHiddenA.
    for (int o = tid; o < MLP_HIDDEN_DIM; o += THREAD_GROUP_SIZE)
    {
        float v = weights[OFF_TRUNK2_B + o];
        for (int i = 0; i < MLP_HIDDEN_DIM; ++i)
            v += gsHiddenB[i] * weights[OFF_TRUNK2_W + o * MLP_HIDDEN_DIM + i];
        gsHiddenA[o] = fmaxf(0.0f, v);
    }
    __syncthreads();

    // Phase 5: heads. exposure_head + color_head share gsHiddenA as input.
    if (tid == 0)
    {
        float v = weights[OFF_EXP_B];
        for (int i = 0; i < MLP_HIDDEN_DIM; ++i)
            v += gsHiddenA[i] * weights[OFF_EXP_W + i];
        surf2Dwrite<float>(v, outputSurf, 0 * sizeof(float), 0);
    }
    if (tid < COLOR_PARAMS_PER_FRAME)
    {
        int o = tid;
        float v = weights[OFF_COL_B + o];
        for (int i = 0; i < MLP_HIDDEN_DIM; ++i)
            v += gsHiddenA[i] * weights[OFF_COL_W + o * MLP_HIDDEN_DIM + i];
        surf2Dwrite<float>(v, outputSurf, (1 + o) * sizeof(float), 0);
    }
}
