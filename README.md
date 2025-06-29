
# mobius‑mpd

> **Möbius Perspective Distortion (MPD)** augmentation for PyTorch & Albumentations  
> **Paper**: Chhipa, Prakash Chandra, *et al.* “Möbius transform for mitigating
> perspective distortions in representation learning.” **ECCV 2024**.

![demo](https://raw.githubusercontent.com/yourusername/mobius-mpd/main/docs/demo_grid.gif)

## Why MPD?

Perspective distortions (steep viewpoint changes) are a major failure mode for
CNNs and ViTs. The ECCV 2024 paper shows that augmenting with a simple single‑parameter
Möbius transform

```
f(z) = (a z + b) / (c z + d),   with  a = d = 1, b = 0
```

yields **+10 pp** RobustBench ImageNet‑PD accuracy, improves crowd‑counting,
object detection and person Re‑ID – *without harming clean accuracy*.

## Installation

```bash
pip install mobius-mpd
```

or in editable mode:

```bash
git clone https://github.com/yourusername/mobius-mpd
cd mobius-mpd
pip install -e .
```

## Usage

### PyTorch / torchvision

```python
from torchvision import transforms
from mobius_mpd import MobiusMPDTransform

train_aug = transforms.Compose([
    MobiusMPDTransform(
        p=0.5,      # apply 50% of the time
        min=0.1,    # minimum |c|
        max=0.3,    # maximum |c|
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
```

### Albumentations

```python
import albumentations as A
from mobius_mpd import A_MobiusMPD

aug = A.Compose([
    A_MobiusMPD(p=0.7, min=0.05, max=0.25),
    A.RandomBrightnessContrast(p=0.3),
])
```

**Parameters**

| name | default | description |
|------|---------|-------------|
| `p`  | `1.0`   | probability of applying the transform |
| `min`| `0.1`   | lower bound for the sampled coefficient \(|c|\) |
| `max`| `0.3`   | upper bound for the sampled coefficient \(|c|\) |
| `interpolate_bg` | `False` | clamp coordinates to avoid black borders |

## BibTeX

```bibtex
@inproceedings{chhipa2024mobius,
  title     = {Möbius transform for mitigating perspective distortions in representation learning},
  author    = {Chhipa, Prakash Chandra and ...},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024},
  publisher = {Springer Nature Switzerland}
}
```

If this library helps your research, **please cite the paper above** 🙏.

## License

MIT
