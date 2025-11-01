# ----------------------------------------------------------
# Grad-CAM utilities for ResNet-like models
# ----------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


@dataclass
class GradCamResult:
    heatmap: np.ndarray  # HxW float32 in [0,1]
    overlay_bgr: np.ndarray  # HxWx3 uint8 BGR (OpenCV)


class GradCAM:
    """Simple Grad-CAM for CNN models.

    Usage:
        target_layer = model.layer4[-1]  # for ResNet-18
        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(input_tensor, class_idx)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            # grad_output is a tuple with length 1 for most layers
            self.gradients = grad_output[0].detach()

        self.fh = target_layer.register_forward_hook(forward_hook)
        self.bh = target_layer.register_full_backward_hook(backward_hook)  # full for safety

    def remove_hooks(self):
        self.fh.remove()
        self.bh.remove()

    @torch.no_grad()
    def _normalize_cam(self, cam: torch.Tensor) -> np.ndarray:
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy().astype(np.float32)

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single input image tensor (1xCxHxW).
        Returns HxW float32 array in [0,1].
        """
        assert input_tensor.ndim == 4 and input_tensor.size(0) == 1, "input_tensor must be 1xCxHxW"

        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        score.backward()

        assert self.gradients is not None and self.activations is not None, "Hooks didn't capture tensors"

        grads = self.gradients  # [B, C, H', W']
        acts = self.activations  # [B, C, H', W']

        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * acts).sum(dim=1, keepdim=False)  # [B, H', W']
        cam = torch.relu(cam)[0]

        # Upsample to input spatial size
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]

        return self._normalize_cam(cam)


def overlay_heatmap_on_image(
    rgb_image: np.ndarray,  # HxWx3 RGB uint8
    heatmap: np.ndarray,  # HxW float32 in [0,1]
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Create overlay of heatmap on image. Returns BGR uint8 (OpenCV style)."""
    h, w = rgb_image.shape[:2]
    hm = (heatmap * 255.0).clip(0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, colormap)  # BGR
    img_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 1.0, hm_color, alpha, 0)
    return overlay
