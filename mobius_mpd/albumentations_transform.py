
"""Albumentations binding for Möbius Perspective‑Distortion."""

import numpy as np
from PIL import Image
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

from .torch_transform import MobiusMPDTransform


class A_MobiusMPD(ImageOnlyTransform):
    """Albumentations wrapper around :class:`mobius_mpd.MobiusMPDTransform`.

    Parameters
    ----------
    p : float
        Probability of applying the transform.
    min : float
        Minimum |c|.
    max : float
        Maximum |c|.
    interpolate_bg : bool

    The random coefficient **c** is sampled uniformly in [min, max] each call.

    Citation
    --------
    Chhipa, P. C., *et al.* "Möbius transform for mitigating perspective distortions
    in representation learning." ECCV 2024.
    """

    def __init__(
        self,
        p: float = 1.0,
        min: float = 0.1,
        max: float = 0.3,
        interpolate_bg: bool = False,
        always_apply: bool = False,
        **kwargs
    ):
        super().__init__(always_apply=always_apply, p=p, **kwargs)
        # we set p=1 for internal transform; albumentations handles probability
        self._mpd = MobiusMPDTransform(
            p=1.0,
            min=min,
            max=max,
            interpolate_bg=interpolate_bg,
        )

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return np.array(self._mpd(Image.fromarray(img)))

    def get_transform_init_args_names(self):
        return ("min", "max", "interpolate_bg")
