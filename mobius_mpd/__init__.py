
"""mobius_mpd

A friction‑free Python package that exposes the Möbius Perspective‑Distortion (MPD)
data‑augmentation transform for PyTorch **and** Albumentations.

Paper to cite
-------------
Chhipa, Prakash Chandra, et al. *"Möbius transform for mitigating perspective distortions in representation learning."* European Conference on Computer Vision (ECCV) 2024.

If you use this package in academic work, **please cite the paper above**.
"""

from .torch_transform import MobiusMPDTransform
from .albumentations_transform import A_MobiusMPD

__all__ = ["MobiusMPDTransform", "A_MobiusMPDTransform"]

__version__ = "0.1.3"
