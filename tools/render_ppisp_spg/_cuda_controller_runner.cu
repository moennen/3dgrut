// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Standalone harness for ppisp_controller.cu used by validate_controller_cuda.py.
//
// Mirrors the SPG dispatch path so we exercise the exact same kernel code:
//   - HDR input  -> cudaArray + cudaTextureObject_t (float4, integer addressing)
//   - Weights    -> raw device pointer (cudaMalloc + cudaMemcpy)
//   - Output     -> 1x9 float cudaArray + cudaSurfaceObject_t
//
// Inputs and outputs are exchanged via binary files so a Python driver can
// produce/consume them without needing cupy/pycuda. CLI:
//
//   _cuda_controller_runner <width> <height> <priorExposure> \
//                           <hdr_rgba32f.bin> <weights_f32.bin> <out_f32_9.bin>
//
// hdr_rgba32f.bin: width*height*4 float32 values in row-major order.
// weights_f32.bin: TOTAL_WEIGHTS float32 values.
// out_f32_9.bin:    9 float32 values written by the runner.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

// Pull in the kernel source. _cuda_controller_runner.cu and ppisp_controller.cu
// live in different directories; the build command provides
// `--include-path=.../ppisp_spg/` so this resolves.
#include "ppisp_controller.cu"

#define CHECK(call)                                                                       \
    do                                                                                    \
    {                                                                                     \
        cudaError_t _e = (call);                                                          \
        if (_e != cudaSuccess)                                                            \
        {                                                                                 \
            std::fprintf(stderr, "CUDA error %s at %s:%d -- %s\n",                        \
                         cudaGetErrorName(_e), __FILE__, __LINE__, cudaGetErrorString(_e)); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

static std::vector<unsigned char> readFile(const char* path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    auto size = static_cast<size_t>(in.tellg());
    in.seekg(0);
    std::vector<unsigned char> buf(size);
    in.read(reinterpret_cast<char*>(buf.data()), size);
    return buf;
}

static void writeFile(const char* path, const void* data, size_t size)
{
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char*>(data), size);
}

int main(int argc, char** argv)
{
    if (argc != 7)
    {
        std::fprintf(stderr,
                     "usage: %s <width> <height> <priorExposure> "
                     "<hdr_rgba32f.bin> <weights_f32.bin> <out_f32_9.bin>\n",
                     argv[0]);
        return 1;
    }
    int   width   = std::atoi(argv[1]);
    int   height  = std::atoi(argv[2]);
    float prior   = std::strtof(argv[3], nullptr);
    auto  hdrBuf  = readFile(argv[4]);
    auto  wBuf    = readFile(argv[5]);

    size_t expectedHdrBytes = size_t(width) * height * 4 * sizeof(float);
    if (hdrBuf.size() != expectedHdrBytes)
    {
        std::fprintf(stderr,
                     "hdr size mismatch: got %zu bytes, expected %zu (%dx%d float4)\n",
                     hdrBuf.size(), expectedHdrBytes, width, height);
        return 1;
    }

    // ---- Input texture: float4, unnormalized integer addressing ----
    cudaChannelFormatDesc inDesc = cudaCreateChannelDesc(32, 32, 32, 32,
                                                         cudaChannelFormatKindFloat);
    cudaArray_t inArr = nullptr;
    CHECK(cudaMallocArray(&inArr, &inDesc, width, height));
    // copy 2D row-major. spitch = width*16, dpitch handled by cuda runtime.
    CHECK(cudaMemcpy2DToArray(inArr, 0, 0,
                              hdrBuf.data(), size_t(width) * 16,
                              size_t(width) * 16, height,
                              cudaMemcpyHostToDevice));

    cudaResourceDesc inResDesc{};
    inResDesc.resType         = cudaResourceTypeArray;
    inResDesc.res.array.array = inArr;

    cudaTextureDesc inTexDesc{};
    inTexDesc.addressMode[0]   = cudaAddressModeClamp;
    inTexDesc.addressMode[1]   = cudaAddressModeClamp;
    inTexDesc.filterMode       = cudaFilterModePoint;
    inTexDesc.readMode         = cudaReadModeElementType;
    inTexDesc.normalizedCoords = 0;

    cudaTextureObject_t inTex = 0;
    CHECK(cudaCreateTextureObject(&inTex, &inResDesc, &inTexDesc, nullptr));

    // ---- Output surface: 1x9 single-channel float ----
    cudaChannelFormatDesc outDesc = cudaCreateChannelDesc(32, 0, 0, 0,
                                                          cudaChannelFormatKindFloat);
    cudaArray_t outArr = nullptr;
    CHECK(cudaMallocArray(&outArr, &outDesc, 9, 1, cudaArraySurfaceLoadStore));

    cudaResourceDesc outResDesc{};
    outResDesc.resType         = cudaResourceTypeArray;
    outResDesc.res.array.array = outArr;

    cudaSurfaceObject_t outSurf = 0;
    CHECK(cudaCreateSurfaceObject(&outSurf, &outResDesc));

    // ---- Weights buffer ----
    float* dWeights = nullptr;
    CHECK(cudaMalloc(&dWeights, wBuf.size()));
    CHECK(cudaMemcpy(dWeights, wBuf.data(), wBuf.size(), cudaMemcpyHostToDevice));

    // ---- Launch ----
    controllerProcess<<<dim3(1, 1, 1), dim3(THREAD_GROUP_SIZE, 1, 1)>>>(
        width, height, prior, dWeights, inTex, outSurf);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // ---- Read back 9 floats ----
    float out[9];
    CHECK(cudaMemcpy2DFromArray(out, 9 * sizeof(float),
                                outArr, 0, 0,
                                9 * sizeof(float), 1,
                                cudaMemcpyDeviceToHost));
    writeFile(argv[6], out, sizeof(out));

    CHECK(cudaDestroySurfaceObject(outSurf));
    CHECK(cudaDestroyTextureObject(inTex));
    CHECK(cudaFreeArray(outArr));
    CHECK(cudaFreeArray(inArr));
    CHECK(cudaFree(dWeights));
    return 0;
}
