-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Stage 2/2 of the PPISP controller pipeline.
--
-- Reads the pixel-feature buffer from stage 1 plus the controller
-- weights, runs the AdaptiveAvgPool + 3-layer MLP trunk + heads, and
-- writes the 1x9 ControllerParams float surface.

local INPUT_DOWNSAMPLING = 3
local CNN_FEATURE_DIM    = 64

function poolMlpProcess(inputs, outputs, params)
    -- We need (dsW, dsH) of the pixel-feature buffer from stage 1.
    -- Recompute them from the HdrColor input wired through to this node.
    local in_rgba = inputs["HdrColor"]
    assert(in_rgba and #in_rgba.shape == 2,
           "HdrColor input must be a 2D image (wired from RenderProduct AOV)")
    local inH = in_rgba.shape[1]
    local inW = in_rgba.shape[2]
    local dsH = math.max(1, math.floor(inH / INPUT_DOWNSAMPLING))
    local dsW = math.max(1, math.floor(inW / INPUT_DOWNSAMPLING))

    local features = inputs["PixelFeatures"]
    assert(features, "poolMlpProcess needs the PixelFeatures buffer input")

    -- 1x9 single-channel float surface (matches the existing controller
    -- output, so ppisp_usd_spg_dyn.slang reads it unchanged).
    outputs["ControllerParams"] = cuda.image(9, 1, cuda.float)

    return cuda.kernel({
        args = {
            cuda.int(dsW),
            cuda.int(dsH),
            cuda.float(params["priorExposure"] or 0.0),
            cuda.array(params["weights"], cuda.float),
            cuda.array(features),
            cuda.SurfaceObject(outputs["ControllerParams"]),
        },
        block = { 128, 1, 1 },
        grid  = { 1, 1, 1 },
    })
end
