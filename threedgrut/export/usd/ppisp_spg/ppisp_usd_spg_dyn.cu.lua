-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- PPISP SPG Launcher (controller-aware variant) -- CUDA backend.
--
-- Reads exposureOffset and 8 colour latents from the controller's
-- 1x9 float texture; reads vignette/CRF scalars from per-camera USD
-- attributes. HdrColor comes from the RenderProduct's primary AOV.

function ppispProcessDyn(inputs, outputs, params)
    local in_rgba = inputs["HdrColor"]
    assert(in_rgba and #in_rgba.shape == 2,
           "HdrColor input must be a 2D image")

    local controller = inputs["ControllerParams"]
    assert(controller, "ppispProcessDyn needs a ControllerParams input texture")

    local height = in_rgba.shape[1]
    local width  = in_rgba.shape[2]

    -- uchar4 surface (matches the slang variant's slang.uchar4 output).
    outputs["PPISPColor"] = cuda.image(width, height, cuda.uchar4)

    local function vec2(name)
        local p = params[name]
        return p and cuda.float2(p) or cuda.float2(0.0, 0.0)
    end

    return cuda.kernel({
        args = {
            cuda.int(width),
            cuda.int(height),
            -- Vignetting (R, G, B): center (float2) + 3 alpha scalars.
            vec2("vignettingCenterR"),
            cuda.float(params["vignettingAlpha1R"] or 0.0),
            cuda.float(params["vignettingAlpha2R"] or 0.0),
            cuda.float(params["vignettingAlpha3R"] or 0.0),
            vec2("vignettingCenterG"),
            cuda.float(params["vignettingAlpha1G"] or 0.0),
            cuda.float(params["vignettingAlpha2G"] or 0.0),
            cuda.float(params["vignettingAlpha3G"] or 0.0),
            vec2("vignettingCenterB"),
            cuda.float(params["vignettingAlpha1B"] or 0.0),
            cuda.float(params["vignettingAlpha2B"] or 0.0),
            cuda.float(params["vignettingAlpha3B"] or 0.0),
            -- CRF (R, G, B): toe / shoulder / gamma / center.
            cuda.float(params["crfToeR"]      or 0.013659),
            cuda.float(params["crfShoulderR"] or 0.013659),
            cuda.float(params["crfGammaR"]    or 0.378165),
            cuda.float(params["crfCenterR"]   or 0.0),
            cuda.float(params["crfToeG"]      or 0.013659),
            cuda.float(params["crfShoulderG"] or 0.013659),
            cuda.float(params["crfGammaG"]    or 0.378165),
            cuda.float(params["crfCenterG"]   or 0.0),
            cuda.float(params["crfToeB"]      or 0.013659),
            cuda.float(params["crfShoulderB"] or 0.013659),
            cuda.float(params["crfGammaB"]    or 0.378165),
            cuda.float(params["crfCenterB"]   or 0.0),
            -- Resources.
            cuda.TextureObject(in_rgba),
            cuda.TextureObject(controller),
            cuda.SurfaceObject(outputs["PPISPColor"]),
        },
        block = { 16, 16, 1 },
        grid  = { math.ceil(width / 16), math.ceil(height / 16), 1 },
    })
end
