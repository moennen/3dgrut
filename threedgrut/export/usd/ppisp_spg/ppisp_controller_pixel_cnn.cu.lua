-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Stage 1/2 of the PPISP controller pipeline.
--
-- Reads the full-resolution HDR colour buffer and the trained controller
-- weights; writes the [dsH * dsW * 64] pixel-feature buffer that the
-- pool/MLP stage consumes. dsH = inH/3, dsW = inW/3.
--
-- One thread per ds-pixel, ~256 threads/block (16x16). At megapixel
-- inputs the grid saturates the GPU.

local INPUT_DOWNSAMPLING = 3
local CNN_FEATURE_DIM    = 64

function pixelCnnProcess(inputs, outputs, params)
    local in_rgba = inputs["HdrColor"]
    assert(in_rgba and #in_rgba.shape == 2,
           "HdrColor input must be a 2D image")

    local inH = in_rgba.shape[1]
    local inW = in_rgba.shape[2]
    local dsH = math.max(1, math.floor(inH / INPUT_DOWNSAMPLING))
    local dsW = math.max(1, math.floor(inW / INPUT_DOWNSAMPLING))

    -- 1D buffer holding [dy * dsW + dx][c]. The next node sees a flat
    -- float buffer through cuda.array().
    outputs["PixelFeatures"] = cuda.empty({ dsH * dsW * CNN_FEATURE_DIM }, cuda.float)

    return cuda.kernel({
        args = {
            cuda.int(inW),
            cuda.int(inH),
            cuda.int(dsW),
            cuda.int(dsH),
            cuda.array(params["weights"], cuda.float),
            cuda.TextureObject(in_rgba),
            cuda.array(outputs["PixelFeatures"]),
        },
        block = { 16, 16, 1 },
        grid  = { math.ceil(dsW / 16), math.ceil(dsH / 16), 1 },
    })
end
