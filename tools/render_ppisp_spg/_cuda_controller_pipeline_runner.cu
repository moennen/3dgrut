// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Standalone harness that exercises the 2-node CUDA controller pipeline
// (pixel CNN -> pool/MLP) the way SPG would chain it, with cudaEvent
// timings per stage. The pixel CNN runs at the full input resolution so
// numerical results match torch's CNN forward pass exactly.
//
// CLI:
//
//   _cuda_controller_pipeline_runner <inW> <inH> <priorExposure> \
//                                    <hdr_rgba32f.bin> <weights_f32.bin> \
//                                    <out_f32_9.bin> [<warmup>] [<iters>]
//
// hdr_rgba32f.bin: inW*inH*4 float32 values, row-major.
// weights_f32.bin: TOTAL_WEIGHTS float32 values.
// out_f32_9.bin:    9 float32 values (the controller output).
//
// On stdout: one line per timing report:
//   pixel_cnn : <ms> ms (avg over N)
//   pool_mlp  : <ms> ms (avg over N)
//   total     : <ms> ms (avg over N)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "ppisp_controller_pixel_cnn.cu"
#include "ppisp_controller_pool_mlp.cu"

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
    if (argc < 7 || argc > 9)
    {
        std::fprintf(stderr,
                     "usage: %s <inW> <inH> <priorExposure> "
                     "<hdr_rgba32f.bin> <weights_f32.bin> <out_f32_9.bin> "
                     "[<warmup>] [<iters>]\n",
                     argv[0]);
        return 1;
    }
    int   inW    = std::atoi(argv[1]);
    int   inH    = std::atoi(argv[2]);
    float prior  = std::strtof(argv[3], nullptr);
    auto  hdrBuf = readFile(argv[4]);
    auto  wBuf   = readFile(argv[5]);

    int warmup = (argc >= 8) ? std::atoi(argv[7]) : 1;
    int iters  = (argc >= 9) ? std::atoi(argv[8]) : 5;
    if (warmup < 0) warmup = 0;
    if (iters  < 1) iters  = 1;

    size_t expectedHdrBytes = size_t(inW) * inH * 4 * sizeof(float);
    if (hdrBuf.size() != expectedHdrBytes)
    {
        std::fprintf(stderr,
                     "hdr size mismatch: got %zu bytes, expected %zu (%dx%d float4)\n",
                     hdrBuf.size(), expectedHdrBytes, inW, inH);
        return 1;
    }

    int dsW = std::max(1, inW / 3);
    int dsH = std::max(1, inH / 3);

    // ---------------------------------------------------------------- input texture
    cudaChannelFormatDesc inDesc = cudaCreateChannelDesc(32, 32, 32, 32,
                                                         cudaChannelFormatKindFloat);
    cudaArray_t inArr = nullptr;
    CHECK(cudaMallocArray(&inArr, &inDesc, inW, inH));
    CHECK(cudaMemcpy2DToArray(inArr, 0, 0,
                              hdrBuf.data(), size_t(inW) * 16,
                              size_t(inW) * 16, inH,
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

    // ---------------------------------------------------------------- pixel features
    float* dPixelFeatures = nullptr;
    size_t pixelFeaturesBytes = size_t(dsH) * dsW * 64 * sizeof(float);
    CHECK(cudaMalloc(&dPixelFeatures, pixelFeaturesBytes));

    // ---------------------------------------------------------------- output surface
    cudaChannelFormatDesc outChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0,
                                                                 cudaChannelFormatKindFloat);
    cudaArray_t outArr = nullptr;
    CHECK(cudaMallocArray(&outArr, &outChannelDesc, 9, 1, cudaArraySurfaceLoadStore));
    cudaResourceDesc outResDesc{};
    outResDesc.resType         = cudaResourceTypeArray;
    outResDesc.res.array.array = outArr;
    cudaSurfaceObject_t outSurf = 0;
    CHECK(cudaCreateSurfaceObject(&outSurf, &outResDesc));

    // ---------------------------------------------------------------- weights buffer
    float* dWeights = nullptr;
    CHECK(cudaMalloc(&dWeights, wBuf.size()));
    CHECK(cudaMemcpy(dWeights, wBuf.data(), wBuf.size(), cudaMemcpyHostToDevice));

    // ---------------------------------------------------------------- launch sizes
    dim3 cnnBlock(16, 16, 1);
    dim3 cnnGrid((dsW + 15) / 16, (dsH + 15) / 16, 1);
    dim3 mlpBlock(128, 1, 1);
    dim3 mlpGrid(1, 1, 1);

    // ---------------------------------------------------------------- run
    cudaEvent_t evStart, evCnn, evMlp;
    CHECK(cudaEventCreate(&evStart));
    CHECK(cudaEventCreate(&evCnn));
    CHECK(cudaEventCreate(&evMlp));

    auto launchOne = [&](){
        pixelCnnProcess<<<cnnGrid, cnnBlock>>>(
            inW, inH, dsW, dsH, dWeights, inTex, dPixelFeatures);
        CHECK(cudaGetLastError());
        cudaEventRecord(evCnn);
        poolMlpProcess<<<mlpGrid, mlpBlock>>>(
            dsW, dsH, prior, dWeights, dPixelFeatures, outSurf);
        CHECK(cudaGetLastError());
        cudaEventRecord(evMlp);
    };

    for (int i = 0; i < warmup; ++i)
    {
        cudaEventRecord(evStart);
        launchOne();
        CHECK(cudaEventSynchronize(evMlp));
    }

    float msCnn = 0.f, msMlp = 0.f, msTotal = 0.f;
    for (int i = 0; i < iters; ++i)
    {
        cudaEventRecord(evStart);
        launchOne();
        CHECK(cudaEventSynchronize(evMlp));
        float c = 0.f, m = 0.f, t = 0.f;
        cudaEventElapsedTime(&c, evStart, evCnn);
        cudaEventElapsedTime(&m, evCnn,   evMlp);
        cudaEventElapsedTime(&t, evStart, evMlp);
        msCnn   += c;
        msMlp   += m;
        msTotal += t;
    }

    std::fprintf(stdout, "pixel_cnn : %.3f ms (avg over %d)\n", msCnn   / iters, iters);
    std::fprintf(stdout, "pool_mlp  : %.3f ms (avg over %d)\n", msMlp   / iters, iters);
    std::fprintf(stdout, "total     : %.3f ms (avg over %d)\n", msTotal / iters, iters);

    // ---------------------------------------------------------------- read back
    float out[9];
    CHECK(cudaMemcpy2DFromArray(out, 9 * sizeof(float),
                                outArr, 0, 0,
                                9 * sizeof(float), 1,
                                cudaMemcpyDeviceToHost));
    writeFile(argv[6], out, sizeof(out));

    cudaEventDestroy(evStart);
    cudaEventDestroy(evCnn);
    cudaEventDestroy(evMlp);
    cudaDestroySurfaceObject(outSurf);
    cudaFreeArray(outArr);
    cudaFree(dPixelFeatures);
    cudaDestroyTextureObject(inTex);
    cudaFreeArray(inArr);
    cudaFree(dWeights);
    return 0;
}
