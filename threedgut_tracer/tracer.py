# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
from ncore.data import FThetaCameraModelParameters, ShutterType
from omegaconf import OmegaConf

from threedgrut.datasets.protocols import Batch

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
#

_3dgut_plugin = None
_3dgut_plugin_variant = None


def _plugin_variant(conf) -> tuple:
    """The config fields that are compiled into the binary rather than read at runtime.

    Must list every field `setup_3dgut` turns into a `-D`, since two configs agreeing here
    are interchangeable and any disagreement means the cached binary is the wrong one.
    """
    return (
        conf.render.particle_kernel_degree,
        bool(conf.render.enable_normals),
        conf.render.primitive_type,
        bool(getattr(conf.render, "enable_depth_variance", False)),
        conf.render.splat.k_buffer_size,
        bool(conf.render.splat.fine_grained_load_balancing),
    )


def load_3dgut_plugin(conf):
    """Compile-and-load the 3DGUT extension, or hand back the one already loaded.

    A process holds exactly one `lib3dgut_cc`: the module is imported by name, so a second
    config cannot load a second binary and would silently run against the first one. That
    is invisible at the call site and produces plausible-but-wrong output -- with
    `enable_normals` off, for instance, the tracer returns a *constant* placeholder normal,
    so a test of normal direction keeps passing on whatever the first config compiled. Fail
    loudly instead, and isolate configs that genuinely differ into separate processes.
    """
    global _3dgut_plugin, _3dgut_plugin_variant
    variant = _plugin_variant(conf)
    if _3dgut_plugin is not None:
        if variant != _3dgut_plugin_variant:
            raise RuntimeError(
                "the 3DGUT extension is already loaded for a different build variant, and a "
                "process can only hold one:\n"
                f"  loaded:    {_3dgut_plugin_variant}\n"
                f"  requested: {variant}\n"
                "(fields: particle_kernel_degree, enable_normals, primitive_type, "
                "enable_depth_variance, k_buffer_size, fine_grained_load_balancing)\n"
                "Reusing the loaded binary would silently render with the wrong kernel. Run "
                "the two configurations in separate processes."
            )
        return

    import torch

    try:
        from . import lib3dgut_cc as tdgut  # type: ignore
    except ImportError:
        from .setup_3dgut import setup_3dgut

        tdgut = setup_3dgut(conf)
    _3dgut_plugin = tdgut
    _3dgut_plugin_variant = variant


# ----------------------------------------------------------------------------
#


@dataclass
class SensorPose3D:
    T_world_sensors: list  # represents two tquat [t,q] poses
    timestamps_us: list


class SensorPose3DModel:
    def __init__(self, R, T, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        super(SensorPose3DModel, self).__init__()
        self.R = R
        self.T = T
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        self.world_view_transform = (
            torch.tensor(SensorPose3DModel.__getWorld2View2(R, T, trans, scale)).transpose(0, 1).cpu()
        )
        # self.camera_center = self.world_view_transform.inverse()[3, :3].cpu()

    @staticmethod
    def __getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)

    @staticmethod
    def __so3_matrix_to_quat(R: torch.Tensor | np.ndarray, unbatch: bool = True) -> torch.Tensor:
        """
        Converts a singe / batch of SO3 rotation matrices (3x3) to unit quaternion representation.

        Args:
            R: single / batch of SO3 rotation matrices [bs, 3, 3] or [3,3]
            unbatch: if the single example should be unbatched (first dimension removed) or not

        Returns:
            single / batch of unit quaternions (XYZW convention)  [bs, 4] or [4]
        """

        # Convert numpy array to torch tensor
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R)

        R = R.reshape((-1, 3, 3))  # batch dimensions unconditionally
        num_rotations, D1, D2 = R.shape
        assert (D1, D2) == (3, 3), "so3_matrix_to_quat: Input has to be a Bx3x3 tensor."

        decision_matrix = torch.empty((num_rotations, 4), dtype=R.dtype, device=R.device)
        quat = torch.empty((num_rotations, 4), dtype=R.dtype, device=R.device)

        decision_matrix[:, :3] = R.diagonal(dim1=1, dim2=2)
        decision_matrix[:, -1] = decision_matrix[:, :3].sum(dim=1)
        choices = decision_matrix.argmax(dim=1)

        ind = torch.nonzero(choices != 3, as_tuple=True)[0]
        i = choices[ind]
        j = (i + 1) % 3
        k = (j + 1) % 3

        quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * R[ind, i, i]
        quat[ind, j] = R[ind, j, i] + R[ind, i, j]
        quat[ind, k] = R[ind, k, i] + R[ind, i, k]
        quat[ind, 3] = R[ind, k, j] - R[ind, j, k]

        ind = torch.nonzero(choices == 3, as_tuple=True)[0]
        quat[ind, 0] = R[ind, 2, 1] - R[ind, 1, 2]
        quat[ind, 1] = R[ind, 0, 2] - R[ind, 2, 0]
        quat[ind, 2] = R[ind, 1, 0] - R[ind, 0, 1]
        quat[ind, 3] = 1 + decision_matrix[ind, -1]

        quat = quat / torch.norm(quat, dim=1)[:, None]

        if unbatch:  # unbatch dimensions conditionally
            quat = quat.squeeze()

        return quat  # (N,4) or (4,)

    def get_sensor_pose(self):
        # construct rolling-shutter parameters for perfect pinhole (global shutter / static pose / no timestamps available)
        T_world_sensor_t = self.world_view_transform[
            3, :3
        ]  # translation is last *row* in current "transposed" conventions
        T_world_sensor_R = self.world_view_transform[:3, :3].transpose(
            0, 1
        )  # rotation matrix ("transposed" in current conventions)
        T_world_sensor_quat = SensorPose3DModel.__so3_matrix_to_quat(T_world_sensor_R)
        T_world_sensor_tquat = torch.hstack([T_world_sensor_t.cpu(), T_world_sensor_quat.cpu()])
        return SensorPose3D(
            T_world_sensors=[T_world_sensor_tquat, T_world_sensor_tquat],  # same pose for both timestamps
            timestamps_us=[0, 1],  # arbitrary timestamps
        )


# ----------------------------------------------------------------------------
#


class Tracer:
    class _Autograd(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            tracer_wrapper,
            frame_id,
            n_active_features,
            ray_ori,
            ray_dir,
            mog_pos,
            mog_rot,
            mog_scl,
            mog_dns,
            mog_sph,
            sensor_params,
            sensor_poses,
            dist_sq_differentiable,
        ):
            particle_density = torch.concat(
                [mog_pos, mog_dns, mog_rot, mog_scl, torch.zeros_like(mog_dns)], dim=1
            ).contiguous()
            particle_features = mog_sph.contiguous()

            ray_time = (
                torch.ones(
                    (ray_ori.shape[0], ray_ori.shape[1], ray_ori.shape[2], 1), device=ray_ori.device, dtype=torch.long
                )
                * sensor_poses.timestamps_us[0]
            )

            (
                ray_features_density,
                ray_hit_distance,
                ray_hit_count,
                mog_visibility,
                ray_hit_normal,
                ray_hit_distance_sq,
            ) = tracer_wrapper.trace(
                frame_id,
                n_active_features,
                particle_density,
                particle_features,
                ray_ori.contiguous(),
                ray_dir.contiguous(),
                ray_time.contiguous(),
                sensor_params,
                sensor_poses.timestamps_us[0],
                sensor_poses.timestamps_us[1],
                sensor_poses.T_world_sensors[0],
                sensor_poses.T_world_sensors[1],
            )

            ctx.save_for_backward(
                ray_ori,
                ray_dir,
                ray_time,
                ray_features_density,
                ray_hit_distance,
                particle_density,
                particle_features,
                # Needed by the backward: the compositing replay unwinds the accumulated
                # normal in place, so it has to start from the forward result.
                ray_hit_normal,
                # Likewise for the second moment.
                ray_hit_distance_sq,
            )

            # The second moment only has a backward on the hand-written CUDA compositing
            # path. Where it does not, mark it non-differentiable so autograd raises on any
            # loss built from it rather than silently returning a zero gradient.
            if not dist_sq_differentiable:
                ctx.mark_non_differentiable(ray_hit_distance_sq)

            ctx.frame_id = frame_id
            ctx.n_active_features = n_active_features
            ctx.sensor_params = sensor_params
            ctx.sensor_poses = sensor_poses
            ctx.tracer_wrapper = tracer_wrapper

            return (
                ray_features_density.float(),  # always fp32 to caller; fp16 saved in ctx for trace_bwd
                ray_hit_distance,
                ray_hit_count,
                mog_visibility,
                # Empty unless normals are compiled in, in which case it is differentiable
                # w.r.t. position / rotation / scale / density.
                ray_hit_normal,
                # Empty unless render.enable_depth_variance is set; differentiable only on
                # the configurations listed in Tracer._dist_sq_differentiable.
                ray_hit_distance_sq,
            )

        @staticmethod
        def backward(
            ctx,
            ray_features_density_grd,  # always fp32 (gradient buffer is never fp16)
            ray_hit_distance_grd,
            ray_hit_count_grd_UNUSED,
            mog_visibility_grd_UNUSED,
            ray_hit_normal_grd,
            ray_hit_distance_sq_grd,  # None unless ctx.dist_sq_differentiable
        ):
            (
                ray_ori,
                ray_dir,
                ray_time,
                ray_features_density,
                ray_hit_distance,
                particle_density,
                particle_features,
                ray_hit_normal,
                ray_hit_distance_sq,
            ) = ctx.saved_variables

            # Autograd passes None when nothing downstream consumed the normals, and the
            # kernel reads the buffer unconditionally once normals are compiled in.
            if ray_hit_normal_grd is None:
                ray_hit_normal_grd = torch.zeros_like(ray_hit_normal)
            # Same for the second moment: None both when it is non-differentiable and when
            # nothing downstream used it. The kernel reads whatever it is handed, so an
            # explicit zero is what keeps it from adding a stale gradient.
            if ray_hit_distance_sq_grd is None:
                ray_hit_distance_sq_grd = torch.zeros_like(ray_hit_distance_sq)

            frame_id = ctx.frame_id
            n_active_features = ctx.n_active_features
            sensor_params = ctx.sensor_params
            sensor_poses = ctx.sensor_poses

            particle_density_grd, particle_features_grd = ctx.tracer_wrapper.trace_bwd(
                frame_id,
                n_active_features,
                particle_density,
                particle_features,
                ray_ori,
                ray_dir,
                ray_time,
                sensor_params,
                sensor_poses.timestamps_us[0],
                sensor_poses.timestamps_us[1],
                sensor_poses.T_world_sensors[0],
                sensor_poses.T_world_sensors[1],
                ray_features_density,
                ray_features_density_grd,
                ray_hit_distance,
                ray_hit_distance_grd,
                ray_hit_normal,
                ray_hit_normal_grd.contiguous(),
                ray_hit_distance_sq,
                ray_hit_distance_sq_grd.contiguous(),
            )

            mog_pos_grd, mog_dns_grd, mog_rot_grd, mog_scl_grd, _ = torch.split(
                particle_density_grd, [3, 1, 4, 3, 1], dim=1
            )
            mog_sph_grd = particle_features_grd

            return (
                None,  # tracer_wrapper
                None,  # frame_id
                None,  # n_active_features
                None,  # ray_ori
                None,  # ray_dir
                mog_pos_grd.contiguous(),
                mog_rot_grd.contiguous(),
                mog_scl_grd.contiguous(),
                mog_dns_grd.contiguous(),
                mog_sph_grd.contiguous(),
                None,  # sensor_params
                None,  # sensor_poses
                None,  # dist_sq_differentiable
            )

    def __init__(self, conf):
        self.device = "cuda"
        self.conf = conf

        torch.zeros(1, device=self.device)  # Create a dummy tensor to force cuda context init
        load_3dgut_plugin(conf)

        self.tracer_wrapper = _3dgut_plugin.SplatRaster(OmegaConf.to_container(conf))

    def _dist_sq_differentiable(self) -> bool:
        """Whether a gradient can flow from the depth second moment.

        All three backward compositing paths now unwind the moment -- the hand-written CUDA
        `processHitBwd` and both Slang autodiff entry points -- so this reduces to whether the
        moment was rendered at all. It still has to be checked: with the feature off the buffer
        is *empty* rather than absent, and a loss built on an empty tensor is silently zero.
        Marking it non-differentiable turns that into a raise.
        """
        return bool(getattr(self.conf.render, "enable_depth_variance", False))

    @property
    def timings(self):
        return self.tracer_wrapper.collect_times()

    def build_acc(self, gaussians, rebuild=True):
        pass  # no-op for 3DGUT

    def render(self, gaussians, gpu_batch: Batch, train=False, frame_id=0):
        rays_o = gpu_batch.rays_ori
        rays_d = gpu_batch.rays_dir

        sensor, poses = Tracer.__create_camera_parameters(gpu_batch)

        num_gaussians = gaussians.num_gaussians
        with torch.cuda.nvtx.range(f"model.forward({num_gaussians} gaussians)"):
            (
                pred_features_alpha,
                pred_dist,
                hits_count,
                mog_visibility,
                pred_normals,
                pred_dist_sq,
            ) = Tracer._Autograd.apply(
                self.tracer_wrapper,
                frame_id,
                gaussians.n_active_features,
                rays_o.contiguous(),
                rays_d.contiguous(),
                gaussians.positions.contiguous(),
                gaussians.get_rotation().contiguous(),
                gaussians.get_scale().contiguous(),
                gaussians.get_density().contiguous(),
                gaussians.get_features().contiguous(),
                sensor,
                poses,
                self._dist_sq_differentiable(),
            )

            # pred_features_alpha is [..., RAY_FEATURE_DIM + 1]: features (fp32) + density
            ray_feature_dim = gaussians.ray_feature_dim
            pred_features = pred_features_alpha[..., :ray_feature_dim].unsqueeze(0).contiguous()
            pred_opacity = pred_features_alpha[..., ray_feature_dim:].unsqueeze(0).contiguous()
            pred_dist = pred_dist.unsqueeze(0).contiguous()
            if pred_dist_sq.numel() > 0:
                pred_dist_sq = pred_dist_sq.unsqueeze(0).contiguous()
            hits_count = hits_count.unsqueeze(0).contiguous()
            pred_normal_accum = pred_normals.unsqueeze(0).contiguous()
            pred_normals = self.__resolve_normals(pred_normals, pred_features)

            timings = self.tracer_wrapper.collect_times()

        return {
            "pred_features": pred_features,
            "pred_opacity": pred_opacity,
            "pred_dist": pred_dist,
            "pred_dist_sq": pred_dist_sq,
            "pred_normal_accum": pred_normal_accum,
            "pred_normals": pred_normals,
            "hits_count": hits_count,
            "frame_time_ms": timings["forward_render"] if "forward_render" in timings else 0.0,
            "mog_visibility": mog_visibility,
        }

    def __resolve_normals(self, ray_hit_normal: torch.Tensor, pred_features: torch.Tensor) -> torch.Tensor:
        """Unit-length world-space normals, or the legacy constant when normals are off.

        The kernel accumulates alpha-premultiplied normals, so the magnitude carries the
        coverage of the ray and only the direction is meaningful. Rays that hit nothing
        accumulate exactly zero and stay zero rather than being normalized into an
        arbitrary direction, so consumers can tell "no surface" from "a surface facing
        somewhere". `enable_normals=false` builds return an empty tensor, in which case
        the previous placeholder is preserved for backwards compatibility.
        """
        if not self.conf.render.enable_normals or ray_hit_normal.numel() == 0:
            return torch.nn.functional.normalize(torch.ones_like(pred_features), dim=3)
        normals = ray_hit_normal.unsqueeze(0).contiguous()
        magnitude = normals.norm(dim=3, keepdim=True)
        return torch.where(magnitude > 1e-6, normals / magnitude.clamp_min(1e-6), torch.zeros_like(normals))

    @staticmethod
    def __fov2focal(fov_radians: float, pixels: int):
        return pixels / (2 * math.tan(fov_radians / 2))

    @staticmethod
    def __focal2fov(focal: float, pixels: int):
        return 2 * math.atan(pixels / (2 * focal))

    @staticmethod
    def __create_sensor_pose_from_R_T(R_start, T_start, R_end, T_end):
        """
        Create SensorPose3D from start and end R, T matrices.
        Supports both global shutter (R_start == R_end) and rolling shutter.
        """
        # Convert START pose to quaternion
        T_world_sensor_t_start = torch.from_numpy(T_start).float()
        T_world_sensor_R_start = torch.from_numpy(R_start.transpose()).float()
        T_world_sensor_quat_start = SensorPose3DModel._SensorPose3DModel__so3_matrix_to_quat(T_world_sensor_R_start)
        T_world_sensor_tquat_start = torch.hstack([T_world_sensor_t_start.cpu(), T_world_sensor_quat_start.cpu()])

        # Convert END pose to quaternion
        T_world_sensor_t_end = torch.from_numpy(T_end).float()
        T_world_sensor_R_end = torch.from_numpy(R_end.transpose()).float()
        T_world_sensor_quat_end = SensorPose3DModel._SensorPose3DModel__so3_matrix_to_quat(T_world_sensor_R_end)
        T_world_sensor_tquat_end = torch.hstack([T_world_sensor_t_end.cpu(), T_world_sensor_quat_end.cpu()])

        return SensorPose3D(
            T_world_sensors=[T_world_sensor_tquat_start, T_world_sensor_tquat_end],
            timestamps_us=[0, 1],  # Relative timestamps (start=0, end=1)
        )

    @staticmethod
    def __create_camera_parameters(gpu_batch):
        SHUTTER_TYPE_MAP = {
            ShutterType.ROLLING_TOP_TO_BOTTOM.name: _3dgut_plugin.ShutterType.ROLLING_TOP_TO_BOTTOM,
            ShutterType.ROLLING_LEFT_TO_RIGHT.name: _3dgut_plugin.ShutterType.ROLLING_LEFT_TO_RIGHT,
            ShutterType.ROLLING_BOTTOM_TO_TOP.name: _3dgut_plugin.ShutterType.ROLLING_BOTTOM_TO_TOP,
            ShutterType.ROLLING_RIGHT_TO_LEFT.name: _3dgut_plugin.ShutterType.ROLLING_RIGHT_TO_LEFT,
            ShutterType.GLOBAL.name: _3dgut_plugin.ShutterType.GLOBAL,
        }

        # Check if rays are already in world space (rolling shutter with per-pixel poses)
        rays_in_world_space = gpu_batch.rays_in_world_space if hasattr(gpu_batch, "rays_in_world_space") else False

        if rays_in_world_space:
            # Rays are already in world space with correct per-pixel poses baked in
            # Use identity transform (no-op) so rays stay in world space
            R_start = R_end = np.eye(3, dtype=np.float32)
            T_start = T_end = np.zeros(3, dtype=np.float32)
        else:
            # Process the camera extrinsics (START and END poses for rolling shutter)
            pose_start = gpu_batch.T_to_world.squeeze()
            assert pose_start.ndim == 2

            # Extract END pose for rolling shutter (if available)
            if gpu_batch.T_to_world_end is not None:
                pose_end = gpu_batch.T_to_world_end.squeeze()
                assert pose_end.ndim == 2
            else:
                # Global shutter: END pose = START pose
                pose_end = pose_start

            # Helper function to convert pose to world-to-camera parameters
            def pose_to_world_to_camera(pose):
                C2W = np.concatenate((pose[:3, :4].cpu().detach().numpy(), np.zeros((1, 4))))
                C2W[3, 3] = 1.0
                W2C = np.linalg.inv(C2W)
                R = np.transpose(W2C[:3, :3])
                T = W2C[:3, 3]
                return R, T

            R_start, T_start = pose_to_world_to_camera(pose_start)
            R_end, T_end = pose_to_world_to_camera(pose_end)

        # Process the camera intrinsics
        if (K := gpu_batch.intrinsics) is not None:
            focalx, focaly, cx, cy = K[0], K[1], K[2], K[3]
            orig_w = int(2 * cx)
            orig_h = int(2 * cy)
            FovX = Tracer.__focal2fov(focalx, orig_w)
            FovY = Tracer.__focal2fov(focaly, orig_h)
            # Compute the camera model parameters
            camera_model_parameters = _3dgut_plugin.fromOpenCVPinholeCameraModelParameters(
                resolution=np.array([orig_w, orig_h], dtype=np.uint32),
                shutter_type=_3dgut_plugin.ShutterType.GLOBAL,
                principal_point=np.array([orig_w, orig_h], dtype=np.float32) / 2,
                focal_length=np.array(
                    [orig_w / (2.0 * math.tan(FovX * 0.5)), orig_h / (2.0 * math.tan(FovY * 0.5))], dtype=np.float32
                ),
                radial_coeffs=np.zeros((6,), dtype=np.float32),
                tangential_coeffs=np.zeros((2,), dtype=np.float32),
                thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
            )
            return camera_model_parameters, Tracer.__create_sensor_pose_from_R_T(R_start, T_start, R_end, T_end)

        elif (K := gpu_batch.intrinsics_OpenCVPinholeCameraModelParameters) is not None:
            camera_model_parameters = _3dgut_plugin.fromOpenCVPinholeCameraModelParameters(
                resolution=K["resolution"],
                shutter_type=SHUTTER_TYPE_MAP[K["shutter_type"]],
                principal_point=K["principal_point"],
                focal_length=K["focal_length"],
                radial_coeffs=K["radial_coeffs"],
                tangential_coeffs=K["tangential_coeffs"],
                thin_prism_coeffs=K.get("thin_prism_coeffs", np.zeros((4,), dtype=np.float32)),
            )
            return camera_model_parameters, Tracer.__create_sensor_pose_from_R_T(R_start, T_start, R_end, T_end)

        elif (K := gpu_batch.intrinsics_OpenCVFisheyeCameraModelParameters) is not None:
            camera_model_parameters = _3dgut_plugin.fromOpenCVFisheyeCameraModelParameters(
                resolution=K["resolution"],
                shutter_type=SHUTTER_TYPE_MAP[K["shutter_type"]],
                principal_point=K["principal_point"],
                focal_length=K["focal_length"],
                radial_coeffs=K["radial_coeffs"],
                max_angle=K["max_angle"],
            )
            return camera_model_parameters, Tracer.__create_sensor_pose_from_R_T(R_start, T_start, R_end, T_end)

        elif (K := gpu_batch.intrinsics_FThetaCameraModelParameters) is not None:
            POLYNOMIAL_TYPE_MAP = {
                FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE.name: _3dgut_plugin.PolynomialType.PIXELDIST_TO_ANGLE,
                FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST.name: _3dgut_plugin.PolynomialType.ANGLE_TO_PIXELDIST,
            }
            camera_model_parameters = _3dgut_plugin.fromFThetaCameraModelParameters(
                resolution=K["resolution"],
                shutter_type=SHUTTER_TYPE_MAP[K["shutter_type"]],
                principal_point=K["principal_point"],
                reference_poly=POLYNOMIAL_TYPE_MAP[K["reference_poly"]],
                pixeldist_to_angle_poly=K["pixeldist_to_angle_poly"],
                angle_to_pixeldist_poly=K["angle_to_pixeldist_poly"],
                max_angle=K["max_angle"],
                linear_cde=K["linear_cde"],
            )
            return camera_model_parameters, Tracer.__create_sensor_pose_from_R_T(R_start, T_start, R_end, T_end)

        raise ValueError(
            f"Camera intrinsics unavailable or unsupported, input keys are [{', '.join(gpu_batch.keys())}]"
        )
