import torch
from torch import nn

class RacecarPolicy(nn.Module):
    def __init__(self, lookahead: int = 6):
        super().__init__()
        # how many tiles ahead to summarize (not a magic number in forward)
        self.lookahead = lookahead

        # Steering weights (signs reflect correlation)
        self.weight_steer_closest_x = nn.Parameter(torch.tensor([2.0], dtype=torch.float32), requires_grad=True)   # positive: if left-of-center -> steer right
        self.weight_steer_lookahead_theta = nn.Parameter(torch.tensor([-0.3], dtype=torch.float32), requires_grad=True)  # negative: lookahead theta positive -> steer left
        self.weight_steer_heading_indicator = nn.Parameter(torch.tensor([-0.2], dtype=torch.float32), requires_grad=True) # negative: heading indicator positive -> steer left
        self.weight_steer_omega = nn.Parameter(torch.tensor([-0.1], dtype=torch.float32), requires_grad=True)  # negative damping on angular velocity
        self.weight_steering_gain = nn.Parameter(torch.tensor([0.8], dtype=torch.float32), requires_grad=True)  # scale to (-1,1)

        # Desired speed (bias + negative corrections)
        self.weight_speed_base = nn.Parameter(torch.tensor([0.75], dtype=torch.float32), requires_grad=True)   # base desired speed in [0,1]
        self.weight_speed_curv = nn.Parameter(torch.tensor([-0.5], dtype=torch.float32), requires_grad=True)    # negative when sharp corner ahead
        self.weight_speed_lat = nn.Parameter(torch.tensor([-0.6], dtype=torch.float32), requires_grad=True)     # negative with lateral offset magnitude
        self.weight_speed_slip = nn.Parameter(torch.tensor([-0.3], dtype=torch.float32), requires_grad=True)    # negative with wheel slip
        self.weight_speed_omega = nn.Parameter(torch.tensor([-0.2], dtype=torch.float32), requires_grad=True)   # negative with angular velocity

        # Throttle / brake gains
        self.weight_accel_gain = nn.Parameter(torch.tensor([2.0], dtype=torch.float32), requires_grad=True)
        self.weight_brake_gain = nn.Parameter(torch.tensor([3.0], dtype=torch.float32), requires_grad=True)

        # Uncertainty (we treat these as std devs; ensure positive via abs in forward)
        self.weight_var_steer = nn.Parameter(torch.tensor([0.10], dtype=torch.float32), requires_grad=True)
        self.weight_var_accel = nn.Parameter(torch.tensor([0.05], dtype=torch.float32), requires_grad=True)
        self.weight_var_brake = nn.Parameter(torch.tensor([0.05], dtype=torch.float32), requires_grad=True)

    def forward(self, tiles: torch.Tensor, indicators: torch.Tensor):
        """
        tiles: (B, L=12, 4) -> (x, y, theta, border)
        indicators: (B, 7) -> [speed, heading_abs, omega, wheel_FL, wheel_FR, wheel_RL, wheel_RR]
        Returns:
          action_mean: (B,3) steer, accel, brake
          action_cov_tril: (B,3,3) lower-triangular L such that Sigma = L L^T
          info_dict: dict with keys weight_* (shape (1,)) and latent_* (shape (B,))
        """
        # handle NaNs gracefully
        tiles_clean = torch.nan_to_num(tiles, nan=0.0, posinf=0.0, neginf=0.0)
        indicators_clean = torch.nan_to_num(indicators, nan=0.0, posinf=0.0, neginf=0.0)

        B = tiles_clean.shape[0]

        # --- input-derived scalars (latents) ---
        closest_x = tiles_clean[:, 0, 0]                 # (B,)
        closest_theta = tiles_clean[:, 0, 2]             # (B,)
        # whether any sharp corner in lookahead window
        lookahead_slice = tiles_clean[:, : self.lookahead, 3]  # (B, lookahead)
        sharp_vals, _ = torch.max(lookahead_slice, dim=1)      # (B,)
        lookahead_theta_mean = tiles_clean[:, : self.lookahead, 2].mean(dim=1)  # (B,)

        wheel_slip = indicators_clean[:, 3:7].mean(dim=1)     # (B,)
        speed = indicators_clean[:, 0]                        # (B,)
        heading_indicator = indicators_clean[:, 1]            # (B,)
        omega = indicators_clean[:, 2]                        # (B,)

        # --- steering (sparse linear combination) ---
        centering = self.weight_steer_closest_x * closest_x
        theta_lookahead_contrib = self.weight_steer_lookahead_theta * lookahead_theta_mean
        heading_contrib = self.weight_steer_heading_indicator * heading_indicator
        omega_contrib = self.weight_steer_omega * omega

        steering_raw = centering + theta_lookahead_contrib + heading_contrib + omega_contrib  # (B,)
        steer = steering_raw * self.weight_steering_gain
        steer_clamped = torch.clamp(steer, min=-1.0, max=1.0)

        # --- desired speed and throttle/brake ---
        desired_speed = (
            self.weight_speed_base
            + self.weight_speed_curv * sharp_vals
            + self.weight_speed_lat * torch.abs(closest_x)
            + self.weight_speed_slip * wheel_slip
            + self.weight_speed_omega * omega
        )  # (B,)
        speed_error = desired_speed - speed  # (B,)

        accel_raw = torch.relu(speed_error * self.weight_accel_gain)
        accel_clamped = torch.clamp(accel_raw, min=0.0, max=1.0)

        brake_raw = torch.relu(-speed_error * self.weight_brake_gain)
        brake_clamped = torch.clamp(brake_raw, min=0.0, max=1.0)

        action_mean = torch.stack([steer_clamped, accel_clamped, brake_clamped], dim=1)  # (B,3)

        # --- action covariance (lower-triangular L), diagonal stds from params ---
        diag_std = torch.cat(
            [
                torch.abs(self.weight_var_steer),
                torch.abs(self.weight_var_accel),
                torch.abs(self.weight_var_brake),
            ],
            dim=0,
        ).to(action_mean.dtype)  # (3,)
        L_single = torch.diag(diag_std)  # (3,3)
        action_cov_tril = L_single.unsqueeze(0).expand(B, 3, 3).contiguous()  # (B,3,3)

        # --- info dict: weights (shape (1,)) and latents (shape (B,)) ---
        info_dict = {
            # weights (biases included under weight_*)
            "weight_steer_closest_x": self.weight_steer_closest_x.view(1),
            "weight_steer_lookahead_theta": self.weight_steer_lookahead_theta.view(1),
            "weight_steer_heading_indicator": self.weight_steer_heading_indicator.view(1),
            "weight_steer_omega": self.weight_steer_omega.view(1),
            "weight_steering_gain": self.weight_steering_gain.view(1),
            "weight_speed_base": self.weight_speed_base.view(1),
            "weight_speed_curv": self.weight_speed_curv.view(1),
            "weight_speed_lat": self.weight_speed_lat.view(1),
            "weight_speed_slip": self.weight_speed_slip.view(1),
            "weight_speed_omega": self.weight_speed_omega.view(1),
            "weight_accel_gain": self.weight_accel_gain.view(1),
            "weight_brake_gain": self.weight_brake_gain.view(1),
            "weight_var_steer": self.weight_var_steer.view(1),
            "weight_var_accel": self.weight_var_accel.view(1),
            "weight_var_brake": self.weight_var_brake.view(1),
            # latents
            "latent_closest_x": closest_x,
            "latent_closest_theta": closest_theta,
            "latent_lookahead_theta_mean": lookahead_theta_mean,
            "latent_sharp_ahead": sharp_vals,
            "latent_wheel_slip": wheel_slip,
            "latent_speed": speed,
            "latent_heading_indicator": heading_indicator,
            "latent_omega": omega,
            "latent_steering_raw": steering_raw,
            "latent_desired_speed": desired_speed,
            "latent_speed_error": speed_error,
            "latent_accel_raw": accel_raw,
            "latent_brake_raw": brake_raw,
        }

        return action_mean, action_cov_tril, info_dict
