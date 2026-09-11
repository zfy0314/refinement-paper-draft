import torch
from torch import nn

class RacecarPolicy(nn.Module):
    def __init__(self, lookahead: int = 6):
        super().__init__()
        self.lookahead = lookahead

        # --- Steering weights (initialized based on observed scales/signs) ---
        self.weight_steer_closest_x = nn.Parameter(torch.tensor([0.10], dtype=torch.float32), requires_grad=True)   # positive: center correction
        self.weight_steer_lookahead_theta = nn.Parameter(torch.tensor([-0.20], dtype=torch.float32), requires_grad=True)  # negative: lookahead theta positive -> steer left
        self.weight_steer_heading_indicator = nn.Parameter(torch.tensor([-0.50], dtype=torch.float32), requires_grad=True) # negative: heading positive -> steer left
        # magnitude parameter for omega damping; we will always apply it as negative * abs(omega)
        self.weight_steer_omega = nn.Parameter(torch.tensor([0.05], dtype=torch.float32), requires_grad=True)
        self.weight_steering_gain = nn.Parameter(torch.tensor([1.00], dtype=torch.float32), requires_grad=True)  # base steering scale
        # speed-dependent steering scale: effective_gain = steering_gain / (1 + abs(scale) * speed)
        self.weight_steer_speed_scale = nn.Parameter(torch.tensor([0.50], dtype=torch.float32), requires_grad=True)

        # --- Desired speed weights ---
        self.weight_speed_base = nn.Parameter(torch.tensor([0.68], dtype=torch.float32), requires_grad=True)   # base desired speed in [0,1]
        self.weight_speed_curv = nn.Parameter(torch.tensor([-0.18], dtype=torch.float32), requires_grad=True)    # negative when sharp corner ahead
        self.weight_speed_lat = nn.Parameter(torch.tensor([-0.54], dtype=torch.float32), requires_grad=True)     # negative with lateral offset magnitude
        # slip: now applied to wheel_slip_rel = mean_wheels - speed; initialize negative to reduce speed on slip
        self.weight_speed_slip = nn.Parameter(torch.tensor([-0.20], dtype=torch.float32), requires_grad=True)
        # use magnitude of angular velocity to reduce speed
        self.weight_speed_omega = nn.Parameter(torch.tensor([-0.13], dtype=torch.float32), requires_grad=True)
        # reduce speed for large lookahead turning magnitude
        self.weight_speed_lookahead_theta_abs = nn.Parameter(torch.tensor([-0.10], dtype=torch.float32), requires_grad=True)

        # --- actuator gains (initialize reasonably) ---
        self.weight_accel_gain = nn.Parameter(torch.tensor([1.00], dtype=torch.float32), requires_grad=True)
        self.weight_brake_gain = nn.Parameter(torch.tensor([2.50], dtype=torch.float32), requires_grad=True)

        # --- uncertainty (base + state-dependent multipliers) ---
        # For backward compatibility we expose weight_var_steer/accel/brake (base)
        self.weight_var_steer = nn.Parameter(torch.tensor([0.12], dtype=torch.float32), requires_grad=True)
        self.weight_var_steer_sharp = nn.Parameter(torch.tensor([0.04], dtype=torch.float32), requires_grad=True)
        self.weight_var_steer_speed = nn.Parameter(torch.tensor([0.04], dtype=torch.float32), requires_grad=True)

        self.weight_var_accel = nn.Parameter(torch.tensor([0.06], dtype=torch.float32), requires_grad=True)
        self.weight_var_accel_sharp = nn.Parameter(torch.tensor([0.06], dtype=torch.float32), requires_grad=True)

        self.weight_var_brake = nn.Parameter(torch.tensor([0.03], dtype=torch.float32), requires_grad=True)
        self.weight_var_brake_sharp = nn.Parameter(torch.tensor([0.04], dtype=torch.float32), requires_grad=True)

    def forward(self, tiles: torch.Tensor, indicators: torch.Tensor):
        """
        tiles: (B, L, 4) -> (x, y, theta, border)
        indicators: (B, 7) -> [speed, heading_abs, omega, wheel_FL, wheel_FR, wheel_RL, wheel_RR]
        Returns:
          action_mean: (B,3) steer, accel, brake
          action_cov_tril: (B,3,3) lower-triangular L such that Sigma = L L^T (diagonal here)
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
        lookahead_border = lookahead_tiles[:, :, 3]            # (B, lookahead)
        sharp_vals, _ = torch.max(lookahead_border, dim=1)     # (B,)

        wheel_abs_mean = indicators_clean[:, 3:7].mean(dim=1)  # (B,)
        speed = indicators_clean[:, 0]                         # (B,)
        heading_indicator = indicators_clean[:, 1]             # (B,)
        omega = indicators_clean[:, 2]                         # (B,)

        # Relative wheel slip (positive if wheels rotate faster than vehicle speed)
        wheel_slip_rel = wheel_abs_mean - speed               # (B,)

        # --- steering (sparse linear combination) ---
        centering = self.weight_steer_closest_x * closest_x
        theta_lookahead_contrib = self.weight_steer_lookahead_theta * lookahead_theta_mean
        heading_contrib = self.weight_steer_heading_indicator * heading_indicator
        # Damping uses magnitude of angular velocity and enforces negative sign
        omega_damping = -torch.abs(self.weight_steer_omega) * torch.abs(omega)

        steering_raw = centering + theta_lookahead_contrib + heading_contrib + omega_damping  # (B,)

        # speed-dependent steering gain (reduces steering authority with speed)
        steering_gain_effective = self.weight_steering_gain / (1.0 + torch.abs(self.weight_steer_speed_scale) * speed)  # (B,)
        steer = steering_raw * steering_gain_effective
        steer_clamped = torch.clamp(steer, min=-1.0, max=1.0)

        # --- desired speed and throttle/brake ---
        desired_preclamp = (
            self.weight_speed_base
            + self.weight_speed_curv * sharp_vals
            + self.weight_speed_lat * torch.abs(closest_x)
            + self.weight_speed_slip * wheel_slip_rel
            + self.weight_speed_omega * torch.abs(omega)
            + self.weight_speed_lookahead_theta_abs * lookahead_theta_abs_mean
        )  # (B,)

        # enforce desired speed within [0,1] to avoid runaway errors
        desired_speed = torch.clamp(desired_preclamp, min=0.0, max=1.0)
        speed_error = desired_speed - speed  # (B,)

        accel_raw = torch.relu(speed_error * self.weight_accel_gain)
        accel_clamped = torch.clamp(accel_raw, min=0.0, max=1.0)

        brake_raw = torch.relu(-speed_error * self.weight_brake_gain)
        brake_clamped = torch.clamp(brake_raw, min=0.0, max=1.0)

        action_mean = torch.stack([steer_clamped, accel_clamped, brake_clamped], dim=1)  # (B,3)

        # --- action covariance (lower-triangular L), diagonal stds from params but state-dependent ---
        diag_std_steer = torch.abs(self.weight_var_steer) + torch.abs(self.weight_var_steer_sharp) * sharp_vals + torch.abs(self.weight_var_steer_speed) * speed  # (B,)
        diag_std_accel = torch.abs(self.weight_var_accel) + torch.abs(self.weight_var_accel_sharp) * sharp_vals  # (B,)
        diag_std_brake = torch.abs(self.weight_var_brake) + torch.abs(self.weight_var_brake_sharp) * sharp_vals  # (B,)

        diag_std = torch.stack([diag_std_steer, diag_std_accel, diag_std_brake], dim=1)  # (B,3)
        # produce batch of diagonal lower-triangular matrices
        action_cov_tril = torch.diag_embed(diag_std).contiguous()  # (B,3,3)

        # --- info dict: weights (shape (1,)) and latents (shape (B,)) ---
        info_dict = {
            # weights (expose both original-style names and new params)
            "weight_steer_closest_x": self.weight_steer_closest_x.view(1),
            "weight_steer_lookahead_theta": self.weight_steer_lookahead_theta.view(1),
            "weight_steer_heading_indicator": self.weight_steer_heading_indicator.view(1),
            "weight_steer_omega": self.weight_steer_omega.view(1),
            "weight_steering_gain": self.weight_steering_gain.view(1),
            "weight_steer_speed_scale": self.weight_steer_speed_scale.view(1),

            "weight_speed_base": self.weight_speed_base.view(1),
            "weight_speed_curv": self.weight_speed_curv.view(1),
            "weight_speed_lat": self.weight_speed_lat.view(1),
            "weight_speed_slip": self.weight_speed_slip.view(1),
            "weight_speed_omega": self.weight_speed_omega.view(1),
            "weight_speed_lookahead_theta_abs": self.weight_speed_lookahead_theta_abs.view(1),

            "weight_accel_gain": self.weight_accel_gain.view(1),
            "weight_brake_gain": self.weight_brake_gain.view(1),

            # variance params (expose base names for compatibility)
            "weight_var_steer": self.weight_var_steer.view(1),
            "weight_var_steer_sharp": self.weight_var_steer_sharp.view(1),
            "weight_var_steer_speed": self.weight_var_steer_speed.view(1),
            "weight_var_accel": self.weight_var_accel.view(1),
            "weight_var_accel_sharp": self.weight_var_accel_sharp.view(1),
            "weight_var_brake": self.weight_var_brake.view(1),
            "weight_var_brake_sharp": self.weight_var_brake_sharp.view(1),

            # latents
            "latent_closest_x": closest_x,
            "latent_closest_theta": closest_theta,
            "latent_lookahead_theta_mean": lookahead_theta_mean,
            "latent_lookahead_theta_abs_mean": lookahead_theta_abs_mean,
            "latent_sharp_ahead": sharp_vals,
            "latent_wheel_abs_mean": wheel_abs_mean,
            "latent_wheel_slip_rel": wheel_slip_rel,
            "latent_speed": speed,
            "latent_heading_indicator": heading_indicator,
            "latent_omega": omega,

            "latent_steering_raw": steering_raw,
            "latent_steering_gain_effective": steering_gain_effective,

            "latent_desired_speed_preclamp": desired_preclamp,
            "latent_desired_speed": desired_speed,
            "latent_speed_error": speed_error,
            "latent_accel_raw": accel_raw,
            "latent_brake_raw": brake_raw,
        }

        return action_mean, action_cov_tril, info_dict