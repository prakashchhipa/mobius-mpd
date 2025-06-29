
from __future__ import annotations
from typing import Tuple
import numpy as np
from PIL import Image
from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A

from .torch_transform import MobiusMPDTransform, BackgroundT, ViewModeT


class A_MobiusMPDTransform(ImageOnlyTransform):
    """Albumentations wrapper around MobiusMPDTransform."""

    def __init__(
        self,
        p: float = 1.0,
        *,
        min: float = 0.1,
        max: float = 0.3,
        background: BackgroundT = "none",
        view_mode: ViewModeT = "random",
        view: str = "random",
        always_apply: bool = False,
        **kwargs
    ):
        super().__init__(always_apply=always_apply, p=p, **kwargs)
        self._mpd = MobiusMPDTransform(
            p=1.0,
            min=min,
            max=max,
            background=background,
            view_mode=view_mode,
            view=view,
        )

    def apply(self, img: np.ndarray, **params) -> np.ndarray:  # type: ignore[override]
        return np.array(self._mpd(Image.fromarray(img)))

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("min", "max", "background", "view_mode", "view")
