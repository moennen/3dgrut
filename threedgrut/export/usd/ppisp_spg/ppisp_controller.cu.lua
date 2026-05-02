-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- PPISP Controller SPG Launcher (CUDA backend).
--
-- Single shared launcher for every camera. Per-camera differences are
-- carried by the `weights` USD attribute, so this file does not need
-- to be regenerated.
--
-- Why CUDA: the slang variant cannot bind the 241,961-float weight
-- buffer because SPG's slang plugin has no path from a USD float[]
-- attribute to a StructuredBuffer<float> (see ppisp_controller.cu's
-- header for the diagnosis). cuda.array() routes the tensor through
-- the SPG tensor-upload pipeline; the kernel receives a device pointer.

function controllerProcess(inputs, outputs, params)
    local in_rgba = inputs["HdrColor"]
    assert(in_rgba and #in_rgba.shape == 2,
           "HdrColor input must be a 2D image")

    -- shape is { height, width } (tensor convention).
    local height = in_rgba.shape[1]
    local width  = in_rgba.shape[2]

    -- 1x9 single-channel float surface. Pixel (x, 0) holds:
    --   x=0           -> exposureOffset
    --   x=1..8        -> colour latents (Blue.xy, Red.xy, Green.xy, Neutral.xy)
    -- ppisp_usd_spg_dyn.slang reads this surface back as a Texture2D<float>.
    outputs["ControllerParams"] = cuda.image(9, 1, cuda.float)

    return cuda.kernel({
        args = {
            cuda.int(width),
            cuda.int(height),
            cuda.float(params["priorExposure"] or 0.0),
            -- USD float[] -> device pointer via SPG tensor upload.
            cuda.array(params["weights"], cuda.float),
            cuda.TextureObject(in_rgba),
            cuda.SurfaceObject(outputs["ControllerParams"]),
        },
        -- One block of 32 threads. The kernel is intentionally sequential
        -- across phases (CNN cells, trunk MLPs, heads); the only
        -- parallelism we exploit is across the 25 pool cells in phase 1
        -- and across MLP rows in the trunk phases.
        block = { 32, 1, 1 },
        grid  = { 1, 1, 1 },
    })
end
