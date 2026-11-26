"""Uncertainty-aware gating at the camera/LiDAR BEV fusion point (Phase 4).

GatedConvFuser keeps ConvFuser's conv/BN/ReLU (same parameter names, so the
released checkpoint loads into it) and adds a small gating network that
re-weights the camera and LiDAR BEV features before they are concatenated:

    w_cam, w_lidar = 2 * sigmoid(Gate([F_cam, F_lidar]))
    F_fused = ConvFuser([w_cam * F_cam, w_lidar * F_lidar])

The gate's last layer is zero-initialised, so every weight starts at exactly
1.0 and the untrained gated model reproduces the original fused model; training
moves the weights from there (range 0-2).
"""
from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

from .conv import ConvFuser

__all__ = ["GatedConvFuser"]

GATE_TYPES = ("modality", "channel", "spatial")


@FUSERS.register_module()
class GatedConvFuser(ConvFuser):
    """ConvFuser with a learned per-sample camera/LiDAR gate.

    Args:
        in_channels: [camera, lidar] channels, e.g. [80, 256].
        out_channels: Output channels of the fusion conv.
        gate_type:
            "modality" (default): one weight per modality per sample (SE-style:
                global average pool -> MLP -> 2 weights). Easiest to interpret.
            "channel": one weight per input channel per sample (SE over all
                80 + 256 channels).
            "spatial": one weight per modality per BEV cell (two 3x3 convs on
                the concatenated features), so the gate can trust the camera
                near the vehicle and not far away.
        hidden_channels: Width of the gate's hidden layer.
        camera_drop_prob: Training only. Probability, per sample, of zeroing
            the camera features before gating, so the fuser and decoder learn
            to work from LiDAR alone (graceful fallback).
        camera_scale: As in ConvFuser; 1.0 leaves the camera unchanged.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        gate_type: str = "modality",
        hidden_channels: int = 64,
        camera_drop_prob: float = 0.0,
        camera_scale: float = 1.0,
    ) -> None:
        super().__init__(in_channels, out_channels, camera_scale=camera_scale)
        if gate_type not in GATE_TYPES:
            raise ValueError(f"gate_type must be one of {GATE_TYPES}, got {gate_type!r}")
        if len(in_channels) != 2:
            raise ValueError("GatedConvFuser expects [camera, lidar] inputs")
        self.gate_type = gate_type
        self.camera_drop_prob = float(camera_drop_prob)
        total = sum(in_channels)
        if gate_type == "spatial":
            self.gate = nn.Sequential(
                nn.Conv2d(total, hidden_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(True),
                nn.Conv2d(hidden_channels, 2, 3, padding=1),
            )
        else:
            out = 2 if gate_type == "modality" else total
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(total, hidden_channels),
                nn.ReLU(True),
                nn.Linear(hidden_channels, out),
            )
        last = self.gate[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        # Mean camera / LiDAR weight of the last forward pass, for logging.
        self.last_weights = None

    def _weights(self, cam: torch.Tensor, lidar: torch.Tensor):
        logits = self.gate(torch.cat([cam, lidar], dim=1))
        w = 2.0 * torch.sigmoid(logits.float()).to(cam.dtype)
        c_cam = cam.shape[1]
        if self.gate_type == "modality":  # (B, 2)
            w_cam, w_lidar = w[:, 0:1, None, None], w[:, 1:2, None, None]
        elif self.gate_type == "channel":  # (B, C_cam + C_lidar)
            w_cam, w_lidar = w[:, :c_cam, None, None], w[:, c_cam:, None, None]
        else:  # spatial: (B, 2, H, W)
            w_cam, w_lidar = w[:, 0:1], w[:, 1:2]
        return w_cam, w_lidar

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        cam, lidar = inputs
        assert cam.shape[1] == self.in_channels[0], (
            f"expected camera features first; got {cam.shape[1]} channels, "
            f"in_channels {self.in_channels}"
        )
        if self.camera_scale != 1.0:
            cam = cam * self.camera_scale
        if self.training and self.camera_drop_prob > 0:
            keep = (
                torch.rand(cam.shape[0], 1, 1, 1, device=cam.device)
                >= self.camera_drop_prob
            ).to(cam.dtype)
            cam = cam * keep
        w_cam, w_lidar = self._weights(cam, lidar)
        self.last_weights = (
            float(w_cam.detach().float().mean()),
            float(w_lidar.detach().float().mean()),
        )
        x = torch.cat([cam * w_cam, lidar * w_lidar], dim=1)
        # Apply ConvFuser's conv/BN/ReLU explicitly: nn.Sequential.forward would
        # also run self.gate, which is a registered child module.
        return self[2](self[1](self[0](x)))
