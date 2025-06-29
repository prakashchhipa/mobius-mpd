
import random
from typing import Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image


class MobiusMPDTransform:
    """Möbius Perspective‑Distortion augmentation.

    Parameters
    ----------
    p : float, default=1.0
        Probability of applying the transform.
    min : float, default=0.1
        Minimum absolute value for the coefficient |c|.
    max : float, default=0.3
        Maximum absolute value for the coefficient |c|.
    interpolate_bg : bool, default=False
        Clamp the sampling grid to [-1,1] (avoids black borders).

    Notes
    -----
    This follows the implementation described in:

        Chhipa, Prakash C., et al.
        "Möbius transform for mitigating perspective distortions in representation learning."
        *European Conference on Computer Vision* (ECCV) 2024.

    The Möbius map is

        f(z) = (a z + b) / (c z + d)

    where we fix a = d = 1, b = 0 and sample exactly one non‑zero component
    of c per call (real or imaginary) within [min, max].

    """

    def __init__(
        self,
        p: float = 1.0,
        min: float = 0.1,
        max: float = 0.3,
        interpolate_bg: bool = False,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        if min <= 0 or max <= 0 or max < min:
            raise ValueError("Require 0 < min ≤ max")

        self.p = p
        self.min_mag = float(min)
        self.max_mag = float(max)
        self.interpolate_bg = interpolate_bg

        # Möbius parameters (a=1, b=0, d=1 are fixed)
        self.a_real, self.a_imag = 1.0, 0.0
        self.b_real, self.b_imag = 0.0, 0.0
        self.d_real, self.d_imag = 1.0, 0.0

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        width, height = img.size
        device = torch.device("cpu")

        c_real, c_imag = self._sample_c()

        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device, dtype=torch.float32),
            torch.linspace(-1, 1, width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        z_real, z_imag = x, y

        num_real = self.a_real * z_real - self.a_imag * z_imag + self.b_real
        num_imag = self.a_real * z_imag + self.a_imag * z_real + self.b_imag

        den_real = c_real * z_real - c_imag * z_imag + self.d_real
        den_imag = c_real * z_imag + c_imag * z_real + self.d_imag

        denom = den_real.pow(2) + den_imag.pow(2)
        zt_real = (num_real * den_real + num_imag * den_imag) / denom
        zt_imag = (num_imag * den_real - num_real * den_imag) / denom

        xt = zt_real
        yt = zt_imag

        if self.interpolate_bg:
            xt = torch.clamp(xt, -1, 1)
            yt = torch.clamp(yt, -1, 1)

        grid = torch.stack((xt, yt), dim=-1)[None, ...]  # (1, H, W, 2)

        img_t = TF.pil_to_tensor(img).float().unsqueeze(0) / 255.0
        out = F.grid_sample(img_t, grid, mode="bilinear", align_corners=True)
        return TF.to_pil_image(out[0].clamp(0, 1))

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _sample_c(self) -> Tuple[float, float]:
        magnitude = random.uniform(self.min_mag, self.max_mag)
        sign = random.choice([-1.0, 1.0])
        if random.random() < 0.5:
            return magnitude * sign, 0.0
        else:
            return 0.0, magnitude * sign

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(p={self.p}, min={self.min_mag}, "
            f"max={self.max_mag}, interpolate_bg={self.interpolate_bg})"
        )
