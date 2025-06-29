
from __future__ import annotations
import random
from typing import Tuple, Literal

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image

BackgroundT = Literal["none", "padded"]
ViewModeT   = Literal["random", "uni-direction", "bi-direction"]
UniViewT    = Literal["random", "right", "left", "top", "bottom"]
BiViewT     = Literal["random",
                      "right-bottom", "right-top",
                      "left-bottom",  "left-top"]


class MobiusMPDTransform:
    """Möbius Perspective Distortion augmentation with view control."""

    def __init__(
        self,
        p: float = 1.0,
        *,
        min: float = 0.1,
        max: float = 0.3,
        background: BackgroundT = "none",
        view_mode: ViewModeT = "random",
        view: str = "random",
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0,1]")
        if min <= 0 or max <= 0 or max < min:
            raise ValueError("Require 0 < min <= max")
        if background not in ("none", "padded"):
            raise ValueError("background must be 'none' or 'padded'")

        self.p = p
        self.min_mag = float(min)
        self.max_mag = float(max)
        self.background = background
        self.view_mode = view_mode
        self.view = view

        # fixed coefficients a=d=1, b=0
        self.a_real = 1.0
        self.a_imag = 0.0
        self.b_real = 0.0
        self.b_imag = 0.0
        self.d_real = 1.0
        self.d_imag = 0.0

    # ------------------------------------------------------------ #
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        c_real, c_imag = self._choose_c()

        w, h = img.size
        device = torch.device("cpu")
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, dtype=torch.float32, device=device),
            torch.linspace(-1, 1, w, dtype=torch.float32, device=device),
            indexing="ij",
        )
        z_real, z_imag = x, y

        num_real = self.a_real * z_real - self.a_imag * z_imag + self.b_real
        num_imag = self.a_real * z_imag + self.a_imag * z_real + self.b_imag

        den_real = c_real * z_real - c_imag * z_imag + self.d_real
        den_imag = c_real * z_imag + c_imag * z_real + self.d_imag
        denom = den_real.pow(2) + den_imag.pow(2)

        xt = (num_real * den_real + num_imag * den_imag) / denom
        yt = (num_imag * den_real - num_real * den_imag) / denom

        if self.background == "padded":
            xt = torch.clamp(xt, -1, 1)
            yt = torch.clamp(yt, -1, 1)

        grid = torch.stack((xt, yt), dim=-1)[None, ...]
        img_t = TF.pil_to_tensor(img).float().unsqueeze(0) / 255.0
        out = F.grid_sample(img_t, grid, mode="bilinear", align_corners=True)
        return TF.to_pil_image(out[0].clamp(0, 1))

    # ------------------------------------------------------------ #
    def _choose_c(self) -> Tuple[float, float]:
        mag_r = random.uniform(self.min_mag, self.max_mag)
        mag_i = random.uniform(self.min_mag, self.max_mag)

        if self.view_mode == "random":
            return self._legacy_random(mag_r, mag_i)

        if self.view_mode == "uni-direction":
            view = self.view if self.view != "random" else random.choice(
                ["right", "left", "bottom", "top"])
            mapping = {
                "right":  ( +mag_r, 0.0 ),
                "left":   ( -mag_r, 0.0 ),
                "bottom": ( 0.0, +mag_i ),
                "top":    ( 0.0, -mag_i ),
            }
            if view not in mapping:
                raise ValueError(f"invalid uni-direction view {view}")
            return mapping[view]

        if self.view_mode == "bi-direction":
            view = self.view if self.view != "random" else random.choice(
                ["right-bottom", "right-top", "left-bottom", "left-top"])
            mapping = {
                "right-bottom": ( +mag_r, +mag_i ),
                "right-top":    ( +mag_r, -mag_i ),
                "left-bottom":  ( -mag_r, +mag_i ),
                "left-top":     ( -mag_r, -mag_i ),
            }
            if view not in mapping:
                raise ValueError(f"invalid bi-direction view {view}")
            return mapping[view]

        raise ValueError(f"invalid view_mode {self.view_mode}")

    def _legacy_random(self, mag_r: float, mag_i: float) -> Tuple[float, float]:
        if random.random() < 0.5:
            return random.choice([-mag_r, +mag_r]), 0.0
        return 0.0, random.choice([-mag_i, +mag_i])

    # ------------------------------------------------------------ #
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(p={self.p}, min={self.min_mag}, "
                f"max={self.max_mag}, background='{self.background}', "
                f"view_mode='{self.view_mode}', view='{self.view}')")
