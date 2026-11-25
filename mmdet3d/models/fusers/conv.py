from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvFuser"]


@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    """Concatenate BEV feature maps along channels, then 3x3 conv + BN + ReLU.

    Args:
        in_channels: Channels of each input, in input order. For camera+lidar
            BEVFusion the inputs arrive as [camera (80), lidar (256)].
        out_channels: Output channels.
        camera_scale: Multiplier on the first (camera) input before
            concatenation. 1.0 (default) is the original model; 0.0 removes the
            camera entirely, emulating a hard gate (w_cam = 0) without any
            retraining. Has no parameters, so checkpoints load unchanged.
    """

    def __init__(
        self, in_channels: int, out_channels: int, camera_scale: float = 1.0
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.camera_scale = float(camera_scale)
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        if self.camera_scale != 1.0:
            assert inputs[0].shape[1] == self.in_channels[0], (
                "camera_scale expects the camera features first; got "
                f"{[x.shape[1] for x in inputs]} channels for in_channels "
                f"{self.in_channels}"
            )
            inputs = [inputs[0] * self.camera_scale, *inputs[1:]]
        return super().forward(torch.cat(inputs, dim=1))
