// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// PPISP (Physically Plausible ISP) -- controller-aware variant, CUDA port.
//
// Per-pixel ISP applied after gaussian rendering:
//   1. Exposure offset (read from controller texture)
//   2. Per-channel vignetting (cubic poly in r^2)
//   3. Color correction via 3x3 homography derived from 4 ZCA latents
//      (also read from controller texture)
//   4. Per-channel CRF tone mapping (toe / shoulder / gamma / center)
//
// One thread per output pixel. Output is rgba8_unorm via surf2Dwrite<uchar4>.
//
// This is a 1:1 port of ppisp_usd_spg_dyn.slang -- maintained as a CUDA
// alternative because Kit's current slang plugin sometimes drops resource
// names from reflection (we observed empty-name errors at bindings 1/2/3
// for the slang variant). The CUDA path goes through SPG's CUDA plugin,
// which doesn't share that reflection limitation.

#include <cuda_fp16.h>
#include <math_constants.h>

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

__device__ __forceinline__ float saturatef(float x)
{
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__device__ __forceinline__ float boundedSoftplus(float raw, float minVal)
{
    return minVal + log1pf(expf(raw));
}

__device__ __forceinline__ float sigmoidF(float raw)
{
    return 1.0f / (1.0f + expf(-raw));
}

__device__ __forceinline__ float applyVignetting(
    float value, float dx, float dy, float a1, float a2, float a3)
{
    float r2 = dx * dx + dy * dy;
    float falloff = 1.0f;
    float r2Pow = r2;
    falloff += a1 * r2Pow;
    r2Pow *= r2;
    falloff += a2 * r2Pow;
    r2Pow *= r2;
    falloff += a3 * r2Pow;
    return value * fminf(fmaxf(falloff, 0.0f), 1.0f);
}

__device__ __forceinline__ float applyCRF(
    float x, float toeRaw, float shoulderRaw, float gammaRaw, float centerRaw)
{
    x = saturatef(x);
    float toe      = boundedSoftplus(toeRaw, 0.3f);
    float shoulder = boundedSoftplus(shoulderRaw, 0.3f);
    float gamma    = boundedSoftplus(gammaRaw, 0.1f);
    float center   = sigmoidF(centerRaw);

    float lerpVal = (shoulder - toe) * center + toe;
    float a = (shoulder * center) / lerpVal;
    float b = 1.0f - a;
    float y;
    if (x <= center)
        y = a * powf(x / center, toe);
    else
        y = 1.0f - b * powf((1.0f - x) / (1.0f - center), shoulder);
    return powf(fmaxf(0.0f, y), gamma);
}

// 2x2 ZCA matrices (must match the slang variant's hard-coded values).
__device__ __forceinline__ void mulZcaBlue(float& ox, float& oy, float ix, float iy)
{
    ox = 0.0480542f * ix + (-0.0043631f) * iy;
    oy = (-0.0043631f) * ix + 0.0481283f * iy;
}
__device__ __forceinline__ void mulZcaRed(float& ox, float& oy, float ix, float iy)
{
    ox = 0.0580570f * ix + (-0.0179872f) * iy;
    oy = (-0.0179872f) * ix + 0.0431061f * iy;
}
__device__ __forceinline__ void mulZcaGreen(float& ox, float& oy, float ix, float iy)
{
    ox = 0.0433336f * ix + (-0.0180537f) * iy;
    oy = (-0.0180537f) * ix + 0.0580500f * iy;
}
__device__ __forceinline__ void mulZcaNeutral(float& ox, float& oy, float ix, float iy)
{
    ox = 0.0128369f * ix + (-0.0034654f) * iy;
    oy = (-0.0034654f) * ix + 0.0128158f * iy;
}

// computeHomography: 1:1 port of the slang version. Layout note: slang
// constructors fill matrices row-major: float3x3(r0r1r2_first_row, ...).
// We store as 9 floats H[r*3+c].
__device__ __forceinline__ void computeHomography(
    float bLatX, float bLatY,
    float rLatX, float rLatY,
    float gLatX, float gLatY,
    float nLatX, float nLatY,
    float H[9])
{
    float bdx, bdy, rdx, rdy, gdx, gdy, ndx, ndy;
    mulZcaBlue   (bdx, bdy, bLatX, bLatY);
    mulZcaRed    (rdx, rdy, rLatX, rLatY);
    mulZcaGreen  (gdx, gdy, gLatX, gLatY);
    mulZcaNeutral(ndx, ndy, nLatX, nLatY);

    float tBx = 0.0f + bdx,        tBy = 0.0f + bdy,        tBz = 1.0f;
    float tRx = 1.0f + rdx,        tRy = 0.0f + rdy,        tRz = 1.0f;
    float tGx = 0.0f + gdx,        tGy = 1.0f + gdy,        tGz = 1.0f;
    float tGrx = 1.0f / 3.0f + ndx, tGry = 1.0f / 3.0f + ndy, tGrz = 1.0f;

    // T row-major: row 0 = [tBx, tRx, tGx], row 1 = [tBy, tRy, tGy], row 2 = [tBz, tRz, tGz]
    float Trm[9] = {
        tBx, tRx, tGx,
        tBy, tRy, tGy,
        tBz, tRz, tGz,
    };

    // skew matrix from tGray (slang: float3x3(0,-tGz,tGy, tGz,0,-tGx, -tGy,tGx,0))
    float Srm[9] = {
        0.0f,   -tGrz,  tGry,
        tGrz,    0.0f, -tGrx,
       -tGry,    tGrx,  0.0f,
    };

    // M = skew @ T  (row-major)
    float Mrm[9];
    #pragma unroll
    for (int r = 0; r < 3; ++r)
    {
        #pragma unroll
        for (int c = 0; c < 3; ++c)
        {
            float s = 0.0f;
            #pragma unroll
            for (int k = 0; k < 3; ++k)
                s += Srm[r * 3 + k] * Trm[k * 3 + c];
            Mrm[r * 3 + c] = s;
        }
    }

    float r0x = Mrm[0], r0y = Mrm[1], r0z = Mrm[2];
    float r1x = Mrm[3], r1y = Mrm[4], r1z = Mrm[5];
    float r2x = Mrm[6], r2y = Mrm[7], r2z = Mrm[8];

    // lam = cross(r0, r1) -- with cascading fallbacks for degenerate rows.
    float lamx, lamy, lamz;
    lamx = r0y * r1z - r0z * r1y;
    lamy = r0z * r1x - r0x * r1z;
    lamz = r0x * r1y - r0y * r1x;
    if (lamx * lamx + lamy * lamy + lamz * lamz < 1.0e-20f)
    {
        lamx = r0y * r2z - r0z * r2y;
        lamy = r0z * r2x - r0x * r2z;
        lamz = r0x * r2y - r0y * r2x;
        if (lamx * lamx + lamy * lamy + lamz * lamz < 1.0e-20f)
        {
            lamx = r1y * r2z - r1z * r2y;
            lamy = r1z * r2x - r1x * r2z;
            lamz = r1x * r2y - r1y * r2x;
        }
    }

    // Sinv = float3x3(-1,-1,1, 1,0,0, 0,1,0)
    float Sinv[9] = {
        -1.0f, -1.0f, 1.0f,
         1.0f,  0.0f, 0.0f,
         0.0f,  1.0f, 0.0f,
    };

    // D = diag(lam.xyz)
    float D[9] = {
        lamx, 0.0f, 0.0f,
        0.0f, lamy, 0.0f,
        0.0f, 0.0f, lamz,
    };

    // H = T @ D @ Sinv  (row-major)
    float TD[9];
    #pragma unroll
    for (int r = 0; r < 3; ++r)
    {
        #pragma unroll
        for (int c = 0; c < 3; ++c)
        {
            float s = 0.0f;
            #pragma unroll
            for (int k = 0; k < 3; ++k)
                s += Trm[r * 3 + k] * D[k * 3 + c];
            TD[r * 3 + c] = s;
        }
    }
    #pragma unroll
    for (int r = 0; r < 3; ++r)
    {
        #pragma unroll
        for (int c = 0; c < 3; ++c)
        {
            float s = 0.0f;
            #pragma unroll
            for (int k = 0; k < 3; ++k)
                s += TD[r * 3 + k] * Sinv[k * 3 + c];
            H[r * 3 + c] = s;
        }
    }

    float sN = H[8];
    if (fabsf(sN) > 1.0e-20f)
    {
        float invS = 1.0f / sN;
        #pragma unroll
        for (int i = 0; i < 9; ++i)
            H[i] *= invS;
    }
}

__device__ __forceinline__ void applyColorCorrection(
    float& r, float& g, float& b, const float H[9])
{
    float intensity = r + g + b;
    float ix = r;
    float iy = g;
    float iz = intensity;

    // mul(H, rgi)  (H row-major, vector multiplied as column)
    float ox = H[0] * ix + H[1] * iy + H[2] * iz;
    float oy = H[3] * ix + H[4] * iy + H[5] * iz;
    float oz = H[6] * ix + H[7] * iy + H[8] * iz;

    float scale = intensity / (oz + 1.0e-5f);
    ox *= scale;
    oy *= scale;
    oz *= scale;

    r = ox;
    g = oy;
    b = oz - ox - oy;
}

// --------------------------------------------------------------------------
// Kernel
// --------------------------------------------------------------------------

extern "C" __global__ void ppispProcessDyn(
    int width,
    int height,
    // Vignetting (per-channel: center.xy, alpha1, alpha2, alpha3)
    const float* __restrict__ vignettingCenterR,
    float vignettingAlpha1R,
    float vignettingAlpha2R,
    float vignettingAlpha3R,
    const float* __restrict__ vignettingCenterG,
    float vignettingAlpha1G,
    float vignettingAlpha2G,
    float vignettingAlpha3G,
    const float* __restrict__ vignettingCenterB,
    float vignettingAlpha1B,
    float vignettingAlpha2B,
    float vignettingAlpha3B,
    // CRF (per-channel: toe, shoulder, gamma, center)
    float crfToeR, float crfShoulderR, float crfGammaR, float crfCenterR,
    float crfToeG, float crfShoulderG, float crfGammaG, float crfCenterG,
    float crfToeB, float crfShoulderB, float crfGammaB, float crfCenterB,
    cudaTextureObject_t inputTex,        // float4 HDR
    cudaTextureObject_t controllerTex,   // 1x9 float, controller params
    cudaSurfaceObject_t outputSurf)      // uchar4 LDR
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    float4 pixel = tex2D<float4>(inputTex, x, y);
    float r = pixel.x, g = pixel.y, b = pixel.z;

    // Centred normalised coordinates -- match slang's `(tid + 0.5 - dim/2) / max(w, h)`.
    float maxRes = (width > height) ? (float)width : (float)height;
    float ux = ((float)x + 0.5f - 0.5f * (float)width)  / maxRes;
    float uy = ((float)y + 0.5f - 0.5f * (float)height) / maxRes;

    // Controller output (1x9 float texture). Reads at integer x, y=0.
    float exposureOffset      = tex2D<float>(controllerTex, 0, 0);
    float colorLatentBlueX    = tex2D<float>(controllerTex, 1, 0);
    float colorLatentBlueY    = tex2D<float>(controllerTex, 2, 0);
    float colorLatentRedX     = tex2D<float>(controllerTex, 3, 0);
    float colorLatentRedY     = tex2D<float>(controllerTex, 4, 0);
    float colorLatentGreenX   = tex2D<float>(controllerTex, 5, 0);
    float colorLatentGreenY   = tex2D<float>(controllerTex, 6, 0);
    float colorLatentNeutralX = tex2D<float>(controllerTex, 7, 0);
    float colorLatentNeutralY = tex2D<float>(controllerTex, 8, 0);

    // 1. Exposure.
    float expScale = exp2f(exposureOffset);
    r *= expScale;
    g *= expScale;
    b *= expScale;

    // 2. Vignetting.
    float dxR = ux - vignettingCenterR[0], dyR = uy - vignettingCenterR[1];
    float dxG = ux - vignettingCenterG[0], dyG = uy - vignettingCenterG[1];
    float dxB = ux - vignettingCenterB[0], dyB = uy - vignettingCenterB[1];
    r = applyVignetting(r, dxR, dyR, vignettingAlpha1R, vignettingAlpha2R, vignettingAlpha3R);
    g = applyVignetting(g, dxG, dyG, vignettingAlpha1G, vignettingAlpha2G, vignettingAlpha3G);
    b = applyVignetting(b, dxB, dyB, vignettingAlpha1B, vignettingAlpha2B, vignettingAlpha3B);

    // 3. Color correction.
    float H[9];
    computeHomography(
        colorLatentBlueX,    colorLatentBlueY,
        colorLatentRedX,     colorLatentRedY,
        colorLatentGreenX,   colorLatentGreenY,
        colorLatentNeutralX, colorLatentNeutralY,
        H);
    applyColorCorrection(r, g, b, H);

    // 4. CRF.
    r = applyCRF(r, crfToeR, crfShoulderR, crfGammaR, crfCenterR);
    g = applyCRF(g, crfToeG, crfShoulderG, crfGammaG, crfCenterG);
    b = applyCRF(b, crfToeB, crfShoulderB, crfGammaB, crfCenterB);

    // 5. Quantise to uchar4 (matches slang variant's slang.uchar4 output).
    uchar4 out;
    out.x = (unsigned char)(saturatef(r) * 255.0f + 0.5f);
    out.y = (unsigned char)(saturatef(g) * 255.0f + 0.5f);
    out.z = (unsigned char)(saturatef(b) * 255.0f + 0.5f);
    out.w = 255;
    surf2Dwrite<uchar4>(out, outputSurf, x * (int)sizeof(uchar4), y);
}
