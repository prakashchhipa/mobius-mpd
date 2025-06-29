
# mobius‑mpd

> **Möbius Perspective Distortion (MPD)** augmentation for PyTorch & Albumentations  

> Chhipa, Prakash Chandra, et al. "Möbius transform for mitigating perspective distortions in representation learning." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

# Möbius-MPD — Perspective-Distortion Augmentation

<div align="center">

<img src="https://raw.githubusercontent.com/prakashchhipa/mobius-mpd/main/assets/two_cats_mpd_transition.gif"   width="45%"/>
<img src="https://raw.githubusercontent.com/prakashchhipa/mobius-mpd/main/assets/burj_mpd_transition.gif"        width="45%"/><br/>
<img src="https://raw.githubusercontent.com/prakashchhipa/mobius-mpd/main/assets/parking_mpd_transition.gif"     width="45%"/>
<img src="https://raw.githubusercontent.com/prakashchhipa/mobius-mpd/main/assets/milan_cathedral_mpd_transition.gif" width="45%"/>

</div>

---

### 1 What is perspective distortion?  
> A camera viewed from an oblique pose **changes the apparent shape, size, orientation and angles of objects in the image plane**.:contentReference[oaicite:0]{index=0}

---

### 2 Why perspective distortion is troublesome for computer-vision models?  
> **Camera parameters are hard to estimate**, so PD can’t be synthesised easily for training.:contentReference[oaicite:1]{index=1}  
> Existing augmentation methods are affine and lienar in nature are not able to model perspective distortion.:contentReference[oaicite:2]{index=2}  
> Lack of perspective distortion data leaves models brittle in the wild for real-world applications—crowd counting, fisheye recognition, person re-ID and object detection all degrade when PD is present.:contentReference[oaicite:3]{index=3}

---

### 3 What does Möbius-MPD offer?  
> Möbius-MPD **mathmetically models perspective distortion and translate it directly in pixel space** with a conformal Möbius mapping  

$$
\Phi(z)=\frac{a z + b}{c z + d}, \qquad c!=0
$$

> and the real and imaginery compoents of complex parameter **c** controls the perspectively distorted view generations.  

> **Orientation & intensity control** – the signs and magnitudes of \(\operatorname{Re}(c)\) and \(\operatorname{Im}(c)\) yield left / right / top / bottom or corner views, scaled continuously.:contentReference[oaicite:4]{index=4}  
> **No camera parameters or real PD images required** – the transform alone synthesises realistic PD.:contentReference[oaicite:5]{index=5}  
> **Padding variant** – optionally fills black corners with edge pixels.:contentReference[oaicite:6]{index=6}  
> **Proven gains** – +10 pp on ImageNet-PD and improvements across crowd counting, fisheye recognition, person re-ID and COCO object detection.:contentReference[oaicite:7]{index=7}

---

## Installation

```bash
pip install mobius-mpd
```

or in editable mode:

```bash
git clone https://github.com/prakashchhipa/mobius-mpd
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

| name          | default     | description                                                                                         |
|---------------|-------------|-----------------------------------------------------------------------------------------------------|
| `p`           | `1.0`       | probability of applying the transform                                                               |
| `min`         | `0.1`       | lower bound for the sampled coefficient \(|c|\)                                                     |
| `max`         | `0.3`       | upper bound for the sampled coefficient \(|c|\)                                                     |
| `background`  | `"none"`    | `"none"` → black corners · `"padded"` → edge-pixel padding                                          |
| `view_mode`   | `"random"`  | `"random"`, `"uni-direction"`, or `"bi-direction"`                                                  |
| `view`        | `"random"`  | orientation; for **uni**: `left / right / top / bottom` · for **bi**: `left-top / left-bottom / right-top / right-bottom` |


**Parent project page:** <https://prakashchhipa.github.io/projects/mpd/>

## BibTeX

```bibtex
@inproceedings{chhipa2024mobius,
  title     = {Möbius transform for mitigating perspective distortions in representation learning},
  author    = {Chhipa, Prakash Chandra et al.},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024},
  publisher = {Springer Nature Switzerland}
}
```

If this library helps your research, **please cite the paper above** 🙏.

## License

MIT
