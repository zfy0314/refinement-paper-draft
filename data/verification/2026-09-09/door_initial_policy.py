import torch
from torch import nn
import torch.nn.functional as F

class OpenDoorPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Handle geometry and orientation
        self.w_handle_r = nn.Parameter(torch.tensor(0.12))   # +corr: door normal -> outward handle offset (m)
        self.w_handle_z = nn.Parameter(torch.tensor(0.0))    # small vertical offset (m)
        self.w_normal_sign = nn.Parameter(torch.tensor(1.0)) # flips door normal if needed

        # Motion gains
        self.k_approach = nn.Parameter(torch.tensor(1.0))    # +corr: approach magnitude
        self.k_open = nn.Parameter(torch.tensor(0.5))        # +corr: tangential open magnitude
        self.k_z = nn.Parameter(torch.tensor(1.0))           # +corr: keep height at handle

        # Proximity / grasp gating
        self.close_slope = nn.Parameter(torch.tensor(4.0))   # +corr: distance -> proximity
        self.w_near_coef = nn.Parameter(torch.tensor(0.7))   # +corr: proximity -> grasp
        self.w_hold_coef = nn.Parameter(torch.tensor(0.3))   # +corr: closedness -> grasp

        # Gripper command shaping (1=open, 0=close)
        self.w_close_from_near = nn.Parameter(torch.tensor(1.0))  # +corr: near -> close
        self.w_close_from_hold = nn.Parameter(torch.tensor(1.0))  # +corr: holding -> keep closed

        # Action covariance (diagonal L)
        # Softplus(diag_raw) > 0; init small std for position, slightly larger for gripper
        self.cov_diag_raw = nn.Parameter(torch.tensor([-5.0, -5.0, -5.0, -3.0]))

    @staticmethod
    def _quat_to_rotmat(q):
        # q: (B,4) as (x, y, z, w). Handle NaNs and zero-norm gracefully.
        q = torch.nan_to_num(q, nan=0.0)
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        n = torch.sqrt(x * x + y * y + z * z + w * w)
        inv_n = torch.where(n > 0, 1.0 / n, torch.zeros_like(n))
        x = x * inv_n
        y = y * inv_n
        z = z * inv_n
        w = w * inv_n

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        r11 = 1 - 2 * (yy + zz)
        r12 = 2 * (xy - wz)
        r13 = 2 * (xz + wy)

        r21 = 2 * (xy + wz)
        r22 = 1 - 2 * (xx + zz)
        r23 = 2 * (yz - wx)

        r31 = 2 * (xz - wy)
        r32 = 2 * (yz + wx)
        r33 = 1 - 2 * (xx + yy)

        R = torch.stack(
            [
                torch.stack([r11, r12, r13], dim=-1),
                torch.stack([r21, r22, r23], dim=-1),
                torch.stack([r31, r32, r33], dim=-1),
            ],
            dim=-2,
        )
        return R

    def forward(self, x):
        # x: (B, 39)
        x = torch.nan_to_num(x, nan=0.0)
        B = x.shape[0]

        # Parse inputs
        g_pos = x[:, 0:3]
        g_open = x[:, 3]                        # assume 1=open, 0=close
        d_pos = x[:, 4:7]
        d_quat = x[:, 7:11]
        target_pos = x[:, 36:39]

        # Door frame orientation
        R = self._quat_to_rotmat(d_quat)        # (B,3,3)
        door_x_axis = R[:, :, 0]                # (B,3) local x-axis in world
        door_normal_xy = door_x_axis[:, 0:2]    # (B,2)
        norm_n = torch.sqrt(torch.clamp(door_normal_xy[:, 0] * door_normal_xy[:, 0] +
                                        door_normal_xy[:, 1] * door_normal_xy[:, 1], min=0.0)).unsqueeze(-1)
        inv_norm_n = torch.where(norm_n > 0, 1.0 / norm_n, torch.zeros_like(norm_n))
        door_normal_xy_unit = door_normal_xy * inv_norm_n

        # Allow flipping sign of normal if needed
        signed_n = door_normal_xy_unit * self.w_normal_sign
        norm_sn = torch.sqrt(torch.clamp(signed_n[:, 0] * signed_n[:, 0] +
                                         signed_n[:, 1] * signed_n[:, 1], min=0.0)).unsqueeze(-1)
        inv_norm_sn = torch.where(norm_sn > 0, 1.0 / norm_sn, torch.zeros_like(norm_sn))
        door_normal_xy_signed = signed_n * inv_norm_sn

        # Tangent = 90° rotation in XY
        t_xy = torch.stack([-door_normal_xy_signed[:, 1], door_normal_xy_signed[:, 0]], dim=-1)
        norm_t = torch.sqrt(torch.clamp(t_xy[:, 0] * t_xy[:, 0] + t_xy[:, 1] * t_xy[:, 1], min=0.0)).unsqueeze(-1)
        inv_norm_t = torch.where(norm_t > 0, 1.0 / norm_t, torch.zeros_like(norm_t))
        tangent_xy_unit = t_xy * inv_norm_t

        # Approximate handle position
        handle_xy = d_pos[:, 0:2] + self.w_handle_r * door_normal_xy_signed
        handle_z = d_pos[:, 2] + self.w_handle_z
        handle_pos = torch.cat([handle_xy, handle_z.unsqueeze(-1)], dim=-1)

        # Approach
        g_to_handle = handle_pos - g_pos
        dist = torch.sqrt(torch.clamp(torch.sum(g_to_handle * g_to_handle, dim=-1), min=0.0))
        proximity = torch.clamp(1 - F.softplus(self.close_slope) * dist, 0.0, 1.0)

        # Holding estimate
        hold_level = torch.clamp(1 - g_open, 0.0, 1.0)

        # Grasp probability
        p_grasp_raw = F.softplus(self.w_near_coef) * proximity + F.softplus(self.w_hold_coef) * hold_level
        p_grasp = torch.clamp(p_grasp_raw, 0.0, 1.0)

        # Opening drive along tangent
        door_to_target_xy = target_pos[:, 0:2] - d_pos[:, 0:2]
        open_drive = torch.sum(tangent_xy_unit * door_to_target_xy, dim=-1, keepdim=True)
        open_vector_xy = tangent_xy_unit * open_drive

        # Motions
        approach_delta = F.softplus(self.k_approach) * g_to_handle
        open_xy = F.softplus(self.k_open) * open_vector_xy
        z_err = handle_pos[:, 2] - g_pos[:, 2]
        open_z = F.softplus(self.k_z) * z_err.unsqueeze(-1)
        open_delta = torch.cat([open_xy, open_z], dim=-1)

        mix_grasp = p_grasp.unsqueeze(-1)
        pos_delta_mean = (1 - mix_grasp) * approach_delta + mix_grasp * open_delta

        # Gripper mean (1=open, 0=close)
        close_drive = F.softplus(self.w_close_from_near) * proximity + F.softplus(self.w_close_from_hold) * hold_level
        grip_mean = torch.clamp(1 - close_drive, 0.0, 1.0)

        action_mean = torch.cat([pos_delta_mean, grip_mean.unsqueeze(-1)], dim=-1)

        # Diagonal Cholesky
        diag = F.softplus(self.cov_diag_raw)            # (4,)
        diag_b = diag.unsqueeze(0).expand(B, -1)        # (B,4)
        action_cov_tril = torch.diag_embed(diag_b)      # (B,4,4)

        # Info dict
        info = {}

        # Weights (as (1,))
        info["weight_handle_radius"] = self.w_handle_r.reshape(1,)
        info["weight_handle_height"] = self.w_handle_z.reshape(1,)
        info["weight_normal_sign"] = self.w_normal_sign.reshape(1,)
        info["weight_k_approach"] = self.k_approach.reshape(1,)
        info["weight_k_open"] = self.k_open.reshape(1,)
        info["weight_k_z"] = self.k_z.reshape(1,)
        info["weight_close_slope"] = self.close_slope.reshape(1,)
        info["weight_near_coef"] = self.w_near_coef.reshape(1,)
        info["weight_hold_coef"] = self.w_hold_coef.reshape(1,)
        info["weight_close_from_near"] = self.w_close_from_near.reshape(1,)
        info["weight_close_from_hold"] = self.w_close_from_hold.reshape(1,)
        info["weight_cov_diag_raw_dx"] = self.cov_diag_raw[0].reshape(1,)
        info["weight_cov_diag_raw_dy"] = self.cov_diag_raw[1].reshape(1,)
        info["weight_cov_diag_raw_dz"] = self.cov_diag_raw[2].reshape(1,)
        info["weight_cov_diag_raw_grip"] = self.cov_diag_raw[3].reshape(1,)

        # Latents (as (B,))
        info["latent_door_normal_x"] = door_normal_xy_signed[:, 0]
        info["latent_door_normal_y"] = door_normal_xy_signed[:, 1]
        info["latent_tangent_x"] = tangent_xy_unit[:, 0]
        info["latent_tangent_y"] = tangent_xy_unit[:, 1]
        info["latent_handle_pos_x"] = handle_pos[:, 0]
        info["latent_handle_pos_y"] = handle_pos[:, 1]
        info["latent_handle_pos_z"] = handle_pos[:, 2]
        info["latent_gripper_to_handle_dx"] = g_to_handle[:, 0]
        info["latent_gripper_to_handle_dy"] = g_to_handle[:, 1]
        info["latent_gripper_to_handle_dz"] = g_to_handle[:, 2]
        info["latent_approach_distance"] = dist
        info["latent_proximity"] = proximity
        info["latent_p_grasp"] = p_grasp
        info["latent_open_drive"] = open_drive.squeeze(-1)
        info["latent_open_vector_x"] = open_vector_xy[:, 0]
        info["latent_open_vector_y"] = open_vector_xy[:, 1]
        info["latent_action_mean_dx"] = action_mean[:, 0]
        info["latent_action_mean_dy"] = action_mean[:, 1]
        info["latent_action_mean_dz"] = action_mean[:, 2]
        info["latent_action_mean_grip"] = action_mean[:, 3]

        return action_mean, action_cov_tril, info
