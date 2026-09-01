// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <3dgut/kernels/cuda/common/rayPayload.cuh>

template <int FeatN>
struct RayPayloadBackward : public RayPayload<FeatN>, public TOptionalNormalGradient {
    float transmittanceBackward;
    float transmittanceGradient;
    float hitTBackward;
    float hitTGradient;
#if GAUSSIAN_ENABLE_HIT_DISTANCE_SQ
    float hitTSqBackward;
    float hitTSqGradient;
#endif
#if GAUSSIAN_ENABLE_FEATURE_SQ
    tcnn::vec<FeatN> featuresSqBackward;
    tcnn::vec<FeatN> featuresSqGradient;
#endif
    tcnn::vec<FeatN> featuresBackward;
    tcnn::vec<FeatN> featuresGradient;

    // Compile-time nullptr when the moment is off, so the Slang backward reads it the same way
    // it reads an absent normal and call sites need no preprocessor branch of their own.
    __device__ __inline__ float* hitTSqBackwardPtr() {
#if GAUSSIAN_ENABLE_HIT_DISTANCE_SQ
        return &hitTSqBackward;
#else
        return nullptr;
#endif
    }
    __device__ __inline__ float* hitTSqGradientPtr() {
#if GAUSSIAN_ENABLE_HIT_DISTANCE_SQ
        return &hitTSqGradient;
#else
        return nullptr;
#endif
    }
};

// The backward payload lives in registers on the hot path, so the normal gradient must cost
// nothing once normals are compiled out. OptionalNormalGradient<false> is empty and is a
// distinct type from the OptionalNormal base RayPayload already carries, so the empty base
// optimization applies; this pins that rather than trusting it.
static_assert(GAUSSIAN_PARTICLE_ENABLE_NORMAL ||
                  sizeof(RayPayloadBackward<RAY_FEATURE_DIM>) ==
                      sizeof(RayPayload<RAY_FEATURE_DIM>) +
                          (GAUSSIAN_ENABLE_HIT_DISTANCE_SQ ? 6 : 4) * sizeof(float) +
                          (GAUSSIAN_ENABLE_FEATURE_SQ ? 4 : 2) * sizeof(tcnn::vec<RAY_FEATURE_DIM>),
              "compiling normals out must not grow the backward ray payload");

template <typename RayPayloadT>
__device__ __inline__ RayPayloadT initializeBackwardRay(const threedgut::RenderParameters& params,
                                                        const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                                        const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                                                        const float* __restrict__ worldHitDistancePtr,
                                                        const float* __restrict__ worldHitDistanceGradientPtr,
                                                        const TFeatureDensityElem* __restrict__ featuresDensityPtr,
                                                        const float* __restrict__ featuresDensityGradientPtr,
                                                        const tcnn::mat4x3& sensorToWorldTransform,
                                                        const tcnn::vec3* __restrict__ worldHitNormalPtr         = nullptr,
                                                        const tcnn::vec3* __restrict__ worldHitNormalGradientPtr = nullptr,
                                                        const float* __restrict__ worldHitDistanceSqPtr          = nullptr,
                                                        const float* __restrict__ worldHitDistanceSqGradientPtr  = nullptr,
                                                        const float* __restrict__ worldFeatureSqPtr              = nullptr,
                                                        const float* __restrict__ worldFeatureSqGradientPtr      = nullptr) {

    // NB : no backpropagation through the forward ray initialization / finalization
    RayPayloadT ray = initializeRay<RayPayloadT>(params,
                                                 sensorRayOriginPtr,
                                                 sensorRayDirectionPtr,
                                                 sensorToWorldTransform);

    if (ray.isAlive()) {
        constexpr uint32_t stride = RayPayloadT::FeatDim + 1;
        const uint32_t base       = ray.idx * stride;
        // Forward features: fp16 when FEATURE_OUTPUT_HALF=1, fp32 otherwise.
        // Gradient buffer: always fp32 — keeps backward numerically stable regardless of forward dtype.
#if FEATURE_OUTPUT_HALF
#pragma unroll
        for (int i = 0; i < RayPayloadT::FeatDim; ++i) {
            ray.featuresBackward[i] = __half2float(featuresDensityPtr[base + i]);
            ray.featuresGradient[i] = featuresDensityGradientPtr[base + i];
        }
        ray.transmittanceBackward = 1.f - __half2float(featuresDensityPtr[base + RayPayloadT::FeatDim]);
        ray.transmittanceGradient = -1.f * featuresDensityGradientPtr[base + RayPayloadT::FeatDim];
#else
#pragma unroll
        for (int i = 0; i < RayPayloadT::FeatDim; ++i) {
            ray.featuresBackward[i] = featuresDensityPtr[base + i];
            ray.featuresGradient[i] = featuresDensityGradientPtr[base + i];
        }
        ray.transmittanceBackward = 1.f - featuresDensityPtr[base + RayPayloadT::FeatDim];
        ray.transmittanceGradient = -1.f * featuresDensityGradientPtr[base + RayPayloadT::FeatDim];
#endif
        ray.hitTBackward = worldHitDistancePtr[ray.idx];
        ray.hitTGradient = worldHitDistanceGradientPtr[ray.idx];

#if GAUSSIAN_ENABLE_HIT_DISTANCE_SQ
        // Replayed the same way as the depth: seed with the forward total and carry the
        // upstream gradient. A caller that renders the moment but takes no gradient from it
        // passes null, which must leave a zero gradient rather than reading uninitialized
        // memory -- `hitTSq` itself is reset by `initializeRay`.
        ray.hitTSqBackward = (worldHitDistanceSqPtr != nullptr) ? worldHitDistanceSqPtr[ray.idx] : 0.0f;
        ray.hitTSqGradient = (worldHitDistanceSqGradientPtr != nullptr) ? worldHitDistanceSqGradientPtr[ray.idx] : 0.0f;
#endif

#if GAUSSIAN_ENABLE_FEATURE_SQ
#pragma unroll
        for (int i = 0; i < RayPayloadT::FeatDim; ++i) {
            const uint32_t featureIdx = ray.idx * RayPayloadT::FeatDim + i;
            ray.featuresSqBackward[i] = (worldFeatureSqPtr != nullptr) ? worldFeatureSqPtr[featureIdx] : 0.0f;
            ray.featuresSqGradient[i] =
                (worldFeatureSqGradientPtr != nullptr) ? worldFeatureSqGradientPtr[featureIdx] : 0.0f;
        }
#endif

#if GAUSSIAN_PARTICLE_ENABLE_NORMAL
        // The normal has no separate `*Backward` slot: the compositing replay unwinds the
        // accumulator in place, exactly as the k-buffer backward does for features. So seed
        // the inherited accumulator with the forward result rather than the zero left by
        // initializeRay, and carry the upstream gradient alongside it.
        if ((worldHitNormalPtr != nullptr) && (worldHitNormalGradientPtr != nullptr)) {
            ray.normalVec         = worldHitNormalPtr[ray.idx];
            ray.normalGradientVec = worldHitNormalGradientPtr[ray.idx];
        } else {
            ray.normalGradientVec = tcnn::vec3(0.0f);
        }
#endif
    }

    return ray;
}
