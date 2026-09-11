import torch
from torch import nn

class SpeedControl(nn.Module):
  def __init__(self):
    # Trainable weights
    self.target_speed = nn.Parameter(
        torch.tensor(0.6))
    self.gain = nn.Parameter(
        torch.tensor(1.0))
  def forward(self, speed):
    speed_error = self.target_speed - speed
    throttle = self.gain * speed_error.relu()
    return throttle.clamp(0, 1), {
        "speed_error": speed_error}
