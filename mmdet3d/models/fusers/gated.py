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
import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from mmdet3d.models.builder import FUSERS

from .conv import ConvFuser

__all__ = ["GatedConvFuser"]

GATE_TYPES = ("modality", "channel", "spatial")

# Camera reliability for the supervised gate: r = exp(-a * severity) when the
# camera is degraded by that weather, 1 when clean, 0 when dropped. The rates
# follow the Phase 3 camera-only losses at severity 1.0 (fog -4.6, snow -2.6,
# rain -2.1 mAP points), scaled so fog at severity 1 gives r = 0.37 and fog at
# severity 3 gives r = 0.05.
DEFAULT_CAMERA_RELIABILITY = dict(fog=1.0, snow=0.6, rain=0.45)


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
        reliability_loss_weight: Training only. If > 0, adds
            weight * MSE(mean camera gate weight, camera reliability target)
            as "loss/gate/reliability". The target comes from the weather
            condition WeatherDegradation recorded for each sample (needs
            "weather_cond" in Collect3D's meta_lis_keys) and is 0 for samples
            whose camera was dropped. 0 (default) = unsupervised gate.
        camera_reliability_coeffs: Per-weather decay rates a in
            r = exp(-a * severity); default DEFAULT_CAMERA_RELIABILITY.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        gate_type: str = "modality",
        hidden_channels: int = 64,
        camera_drop_prob: float = 0.0,
        camera_scale: float = 1.0,
        reliability_loss_weight: float = 0.0,
        camera_reliability_coeffs: Optional[Dict[str, float]] = None,
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
        self.reliability_loss_weight = float(reliability_loss_weight)
        self.camera_reliability_coeffs = dict(
            camera_reliability_coeffs or DEFAULT_CAMERA_RELIABILITY
        )
        # Mean camera / LiDAR weight of the last forward pass, for logging.
        self.last_weights = None
        # Set by BEVFusion before each forward / read by it afterwards.
        self.current_metas = None
        self.aux_losses = {}

    def set_metas(self, metas) -> None:
        """Per-sample metadata for the next forward (used for supervision)."""
        self.current_metas = metas

    def camera_targets(self, metas, dropped: torch.Tensor) -> torch.Tensor:
        """Camera reliability in [0, 1] per sample."""
        r = []
        for meta in metas:
            cond = (meta or {}).get("weather_cond") or {}
            weather, severity = cond.get("weather"), float(cond.get("severity", 0.0))
            if (
                cond.get("modality") in ("both", "camera")
                and weather in self.camera_reliability_coeffs
            ):
                r.append(math.exp(-self.camera_reliability_coeffs[weather] * severity))
            else:
                r.append(1.0)
        target = torch.tensor(r, device=dropped.device, dtype=torch.float32)
        return torch.where(dropped, torch.zeros_like(target), target)

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
        dropped = torch.zeros(cam.shape[0], dtype=torch.bool, device=cam.device)
        if self.training and self.camera_drop_prob > 0:
            dropped = (
                torch.rand(cam.shape[0], device=cam.device) < self.camera_drop_prob
            )
            cam = cam * (~dropped).to(cam.dtype).view(-1, 1, 1, 1)
        w_cam, w_lidar = self._weights(cam, lidar)
        self.aux_losses = {}
        metas, self.current_metas = self.current_metas, None
        if self.training and self.reliability_loss_weight > 0:
            if metas is None or len(metas) != cam.shape[0]:
                raise RuntimeError(
                    "Supervised gate needs per-sample metas: BEVFusion must call "
                    "fuser.set_metas(metas) before the fuser"
                )
            target = self.camera_targets(metas, dropped)
            pred = w_cam.float().mean(dim=(1, 2, 3))
            self.aux_losses["reliability"] = (
                self.reliability_loss_weight * F.mse_loss(pred, target)
            )
        self.last_weights = (
            float(w_cam.detach().float().mean()),
            float(w_lidar.detach().float().mean()),
        )
        x = torch.cat([cam * w_cam, lidar * w_lidar], dim=1)
        # Apply ConvFuser's conv/BN/ReLU explicitly: nn.Sequential.forward would
        # also run self.gate, which is a registered child module.
        return self[2](self[1](self[0](x)))
