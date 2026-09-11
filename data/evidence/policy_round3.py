import torch
from torch import nn

class RacecarPolicy(nn.Module):
    def __init__(self, lookahead: int = 6):
        super().__init__()
        self.lookahead = lookahead

        # --- Steering weights (directional: sign matters) ---
        self.weight_steer_closest_x = nn.Parameter(torch.tensor([0.10], dtype=torch.float32), requires_grad=True)
        self.weight_steer_lookahead_theta = nn.Parameter(torch.tensor([-0.20], dtype=torch.float32), requires_grad=True)
        self.weight_steer_heading_indicator = nn.Parameter(torch.tensor([-0.50], dtype=torch.float32), requires_grad=True)

        # omega damping: store raw params but use abs() in forward to make damping magnitude positive
        self.weight_steer_omega_raw = nn.Parameter(torch.tensor([0.05], dtype=torch.float32), requires_grad=True)
        self.weight_steer_omega_speed_scale_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32), requires_grad=True)

        # steering gain (positive)
        self.weight_steering_gain_raw = nn.Parameter(torch.tensor([1.00], dtype=torch.float32), requires_grad=True)
        self.weight_steer_speed_scale_raw = nn.Parameter(torch.tensor([0.50], dtype=torch.float32), requires_grad=True)

        # --- Desired speed weights ---
        # base speed (positive)
        self.weight_speed_base = nn.Parameter(torch.tensor([0.68], dtype=torch.float32), requires_grad=True)
        # the following are semantically "speed-reducing" -> constrain to be negative
        self.weight_speed_curv_raw = nn.Parameter(torch.tensor([0.18], dtype=torch.float32), requires_grad=True)
        self.weight_speed_lat_raw = nn.Parameter(torch.tensor([0.54], dtype=torch.float32), requires_grad=True)
        self.weight_speed_slip_raw = nn.Parameter(torch.tensor([0.20], dtype=torch.float32), requires_grad=True)
        self.weight_speed_omega_raw = nn.Parameter(torch.tensor([0.13], dtype=torch.float32), requires_grad=True)
        self.weight_speed_lookahead_theta_abs_raw = nn.Parameter(torch.tensor([0.10], dtype=torch.float32), requires_grad=True)
        self.weight_speed_lookahead_theta_abs_speed_raw = nn.Parameter(torch.tensor([0.12], dtype=torch.float32), requires_grad=True)
        self.weight_speed_steer_interaction_raw = nn.Parameter(torch.tensor([0.20], dtype=torch.float32), requires_grad=True)
        # additional sharper-curvature latent (max abs theta)
        self.weight_speed_lookahead_theta_abs_max_raw = nn.Parameter(torch.tensor([0.08], dtype=torch.float32), requires_grad=True)

        # small learned slip normalization (positive)
        self.slip_scale_raw = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)

        # --- actuator gains (positive) ---
        self.weight_accel_gain_raw = nn.Parameter(torch.tensor([1.00], dtype=torch.float32), requires_grad=True)
        self.weight_brake_gain_raw = nn.Parameter(torch.tensor([2.50], dtype=torch.float32), requires_grad=True)

        # --- uncertainty (raw, enforced positive in forward) ---
        self.weight_var_steer_raw = nn.Parameter(torch.tensor([0.12], dtype=torch.float32), requires_grad=True)
        self.weight_var_steer_sharp_raw = nn.Parameter(torch.tensor([0.04], dtype=torch.float32), requires_grad=True)
        self.weight_var_steer_speed_raw = nn.Parameter(torch.tensor([0.04], dtype=torch.float32), requires_grad=True)

        self.weight_var_accel_raw = nn.Parameter(torch.tensor([0.06], dtype=torch.float32), requires_grad=True)
        self.weight_var_accel_sharp_raw = nn.Parameter(torch.tensor([0.06], dtype=torch.float32), requires_grad=True)

        self.weight_var_brake_raw = nn.Parameter(torch.tensor([0.03], dtype=torch.float32), requires_grad=True)
        self.weight_var_brake_sharp_raw = nn.Parameter(torch.tensor([0.04], dtype=torch.float32), requires_grad=True)

        # Small constant bounds for std (kept as constants on module)
        self._std_min = 1e-3
        self._std_max = 1.0

    def forward(self, tiles: torch.Tensor, indicators: torch.Tensor):
        """
        tiles: (B, L, 4) -> (x, y, theta, border)
        indicators: (B, 7) -> [speed, heading_abs, omega, wheel_FL, wheel_FR, wheel_RL, wheel_RR]
        Returns:
          action_mean: (B,3) steer, accel, brake
          action_cov_tril: (B,3,3) lower-triangular L (diagonal here)
          info_dict: dict with keys weight_* (shape (1,)) and latent_* (shape (B,))
        """
        # handle NaNs gracefully
        tiles_clean = torch.nan_to_num(tiles, nan=0.0, posinf=0.0, neginf=0.0)
        indicators_clean = torch.nan_to_num(indicators, nan=0.0, posinf=0.0, neginf=0.0)

        B = tiles_clean.shape[0]

        # --- input-derived scalars (latents) ---
        closest_x = tiles_clean[:, 0, 0]                 # (B,)
        closest_theta = tiles_clean[:, 0, 2]             # (B,)

        lookahead_tiles = tiles_clean[:, : self.lookahead, :]  # (B, lookahead, 4)
        lookahead_theta = lookahead_tiles[:, :, 2]             # (B, lookahead)
        lookahead_theta_mean = lookahead_theta.mean(dim=1)     # (B,)
        lookahead_theta_abs_mean = lookahead_theta.abs().mean(dim=1)  # (B,)
        # new: max absolute lookahead theta (captures sharpest upcoming turn)
        lookahead_theta_abs_max = lookahead_theta.abs().max(dim=1)[0]  # (B,)

        # distance-weighted lookahead abs mean: nearer tiles count more (1/(1+distance))
        lookahead_y = lookahead_tiles[:, :, 1]  # (B, lookahead)
        inv_dist_weights = 1.0 / (1.0 + torch.abs(lookahead_y))  # (B, lookahead)
        weighted_lookahead_abs = (lookahead_theta.abs() * inv_dist_weights).sum(dim=1) / (
            inv_dist_weights.sum(dim=1) + self._std_min
        )  # (B,)

        lookahead_border = lookahead_tiles[:, :, 3]            # (B, lookahead)
        sharp_vals = torch.max(lookahead_border, dim=1)[0]     # (B,)

        wheel_abs_mean = indicators_clean[:, 3:7].mean(dim=1)  # (B,)
        speed = indicators_clean[:, 0]                         # (B,)
        heading_indicator = indicators_clean[:, 1]             # (B,)
        omega = indicators_clean[:, 2]                         # (B,)

        # Relative wheel slip (positive if wheels rotate faster than vehicle speed)
        wheel_slip_rel = wheel_abs_mean - speed               # (B,)
        # NEW: only consider positive slip (wheels spinning faster than vehicle)
        wheel_slip_pos = torch.relu(wheel_slip_rel)           # (B,)
        # compress slip values to avoid requiring huge slip weights
        slip_scale = torch.abs(self.slip_scale_raw) + 1e-6
        wheel_slip_pos_norm = wheel_slip_pos / (1.0 + slip_scale * wheel_slip_pos)  # (B,)

        # --- steering (sparse linear combination) ---
        centering = self.weight_steer_closest_x * closest_x
        theta_lookahead_contrib = self.weight_steer_lookahead_theta * lookahead_theta_mean
        heading_contrib = self.weight_steer_heading_indicator * heading_indicator

        # Damping uses magnitude of angular velocity and enforces negative sign (damping)
        omega_damping = -torch.abs(self.weight_steer_omega_raw) * torch.abs(omega) * (
            1.0 + torch.abs(self.weight_steer_omega_speed_scale_raw) * speed
        )

        steering_raw = centering + theta_lookahead_contrib + heading_contrib + omega_damping  # (B,)

        # steering gain: ensure positive
        steering_gain_effective = torch.abs(self.weight_steering_gain_raw) / (
            1.0 + torch.abs(self.weight_steer_speed_scale_raw) * speed
        )  # (B,)
        steer = steering_raw * steering_gain_effective
        steer_clamped = torch.clamp(steer, min=-1.0, max=1.0)
        steer_abs = torch.abs(steer_clamped)

        # --- desired speed and throttle/brake ---
        # effective speed-reducing weights: constrained negative so they cannot flip sign
        weight_speed_curv = -torch.abs(self.weight_speed_curv_raw)
        weight_speed_lat = -torch.abs(self.weight_speed_lat_raw)
        weight_speed_slip = -torch.abs(self.weight_speed_slip_raw)
        weight_speed_omega = -torch.abs(self.weight_speed_omega_raw)
        weight_speed_lookahead_theta_abs = -torch.abs(self.weight_speed_lookahead_theta_abs_raw)
        weight_speed_lookahead_theta_abs_speed = -torch.abs(self.weight_speed_lookahead_theta_abs_speed_raw)
        weight_speed_steer_interaction = -torch.abs(self.weight_speed_steer_interaction_raw)
        weight_speed_lookahead_theta_abs_max = -torch.abs(self.weight_speed_lookahead_theta_abs_max_raw)

        desired_preclamp = (
            self.weight_speed_base
            + weight_speed_curv * sharp_vals
            + weight_speed_lat * torch.abs(closest_x)
            + weight_speed_slip * wheel_slip_pos_norm
            + weight_speed_omega * torch.abs(omega)
            + weight_speed_lookahead_theta_abs * lookahead_theta_abs_mean
            + weight_speed_lookahead_theta_abs_speed * lookahead_theta_abs_mean * speed
            + weight_speed_steer_interaction * steer_abs * speed
            # additional strong-corner penalty
            + weight_speed_lookahead_theta_abs_max * lookahead_theta_abs_max
            # distance-weighted lookahead is available to the model (smaller magnitude contribution)
            + weight_speed_lookahead_theta_abs * weighted_lookahead_abs * 0.5
        )  # (B,)

        # enforce desired speed within [0,1]
        desired_speed = torch.clamp(desired_preclamp, min=0.0, max=1.0)
        speed_error = desired_speed - speed  # (B,)

        accel_raw = torch.relu(speed_error * torch.abs(self.weight_accel_gain_raw))
        accel_clamped = torch.clamp(accel_raw, min=0.0, max=1.0)

        brake_raw = torch.relu(-speed_error * torch.abs(self.weight_brake_gain_raw))
        brake_clamped = torch.clamp(brake_raw, min=0.0, max=1.0)

        action_mean = torch.stack([steer_clamped, accel_clamped, brake_clamped], dim=1)  # (B,3)

        # --- action covariance (lower-triangular L), diagonal stds from positive params ---
        diag_std_steer = torch.abs(self.weight_var_steer_raw) + torch.abs(self.weight_var_steer_sharp_raw) * sharp_vals + torch.abs(self.weight_var_steer_speed_raw) * speed  # (B,)
        diag_std_accel = torch.abs(self.weight_var_accel_raw) + torch.abs(self.weight_var_accel_sharp_raw) * sharp_vals  # (B,)
        diag_std_brake = torch.abs(self.weight_var_brake_raw) + torch.abs(self.weight_var_brake_sharp_raw) * sharp_vals  # (B,)

        diag_std = torch.stack([diag_std_steer, diag_std_accel, diag_std_brake], dim=1)  # (B,3)
        # Clamp stds to avoid degenerate covariances
        diag_std = torch.clamp(diag_std, min=self._std_min, max=self._std_max)

        # produce batch of diagonal lower-triangular matrices
        action_cov_tril = torch.diag_embed(diag_std).contiguous()  # (B,3,3)

        # --- info dict: effective weights (shape (1,)) and latents (shape (B,)) ---
        info_dict = {
            # steering weights (provide effective values used in forward)
            "weight_steer_closest_x": self.weight_steer_closest_x.view(1),
            "weight_steer_lookahead_theta": self.weight_steer_lookahead_theta.view(1),
            "weight_steer_heading_indicator": self.weight_steer_heading_indicator.view(1),
            # report the damping magnitude as positive value (the forward uses it as magnitude)
            "weight_steer_omega": torch.abs(self.weight_steer_omega_raw).view(1),
            "weight_steer_omega_speed_scale": torch.abs(self.weight_steer_omega_speed_scale_raw).view(1),
            "weight_steering_gain": torch.abs(self.weight_steering_gain_raw).view(1),
            "weight_steer_speed_scale": torch.abs(self.weight_steer_speed_scale_raw).view(1),

            # speed weights: report effective signed values used (negative where intended)
            "weight_speed_base": self.weight_speed_base.view(1),
            "weight_speed_curv": (-torch.abs(self.weight_speed_curv_raw)).view(1),
            "weight_speed_lat": (-torch.abs(self.weight_speed_lat_raw)).view(1),
            "weight_speed_slip": (-torch.abs(self.weight_speed_slip_raw)).view(1),
            "weight_speed_omega": (-torch.abs(self.weight_speed_omega_raw)).view(1),
            "weight_speed_lookahead_theta_abs": (-torch.abs(self.weight_speed_lookahead_theta_abs_raw)).view(1),
            "weight_speed_lookahead_theta_abs_speed": (-torch.abs(self.weight_speed_lookahead_theta_abs_speed_raw)).view(1),
            "weight_speed_steer_interaction": (-torch.abs(self.weight_speed_steer_interaction_raw)).view(1),
            "weight_speed_lookahead_theta_abs_max": (-torch.abs(self.weight_speed_lookahead_theta_abs_max_raw)).view(1),

            # actuator gains (report positive effective gains)
            "weight_accel_gain": torch.abs(self.weight_accel_gain_raw).view(1),
            "weight_brake_gain": torch.abs(self.weight_brake_gain_raw).view(1),

            # variance params (positive)
            "weight_var_steer": torch.abs(self.weight_var_steer_raw).view(1),
            "weight_var_steer_sharp": torch.abs(self.weight_var_steer_sharp_raw).view(1),
            "weight_var_steer_speed": torch.abs(self.weight_var_steer_speed_raw).view(1),
            "weight_var_accel": torch.abs(self.weight_var_accel_raw).view(1),
            "weight_var_accel_sharp": torch.abs(self.weight_var_accel_sharp_raw).view(1),
            "weight_var_brake": torch.abs(self.weight_var_brake_raw).view(1),
            "weight_var_brake_sharp": torch.abs(self.weight_var_brake_sharp_raw).view(1),

            # slip scale (report positive)
            "weight_slip_scale": torch.abs(self.slip_scale_raw).view(1),

            # latents (shape (B,))
            "latent_closest_x": closest_x,
            "latent_closest_theta": closest_theta,
            "latent_lookahead_theta_mean": lookahead_theta_mean,
            "latent_lookahead_theta_abs_mean": lookahead_theta_abs_mean,
            "latent_lookahead_theta_abs_max": lookahead_theta_abs_max,
            "latent_weighted_lookahead_theta_abs_mean": weighted_lookahead_abs,
            "latent_sharp_ahead": sharp_vals,
            "latent_wheel_abs_mean": wheel_abs_mean,
            "latent_wheel_slip_rel": wheel_slip_rel,
            "latent_wheel_slip_pos": wheel_slip_pos,
            "latent_wheel_slip_pos_norm": wheel_slip_pos_norm,
            "latent_speed": speed,
            "latent_heading_indicator": heading_indicator,
            "latent_omega": omega,

            "latent_steering_raw": steering_raw,
            "latent_steering_gain_effective": steering_gain_effective,
            "latent_steer_clamped": steer_clamped,
            "latent_steer_abs": steer_abs,

            "latent_desired_speed_preclamp": desired_preclamp,
            "latent_desired_speed": desired_speed,
            "latent_speed_error": speed_error,
            "latent_accel_raw": accel_raw,
            "latent_brake_raw": brake_raw,
        }

        return action_mean, action_cov_tril, info_dict