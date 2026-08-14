from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


FROZEN_VIEW_NAMES = (
    "clean", "crop", "color_jitter", "blur", "occlusion", "resize"
)


def _resampling(name: str):
    enum = getattr(Image, "Resampling", Image)
    return getattr(enum, name)


def fixed_view(image: Image.Image, view: str) -> Image.Image:
    """Apply one preregistered, deterministic CIFAR multi-view perturbation."""
    if view not in FROZEN_VIEW_NAMES:
        raise ValueError(f"unknown fixed view: {view}")
    image = image.convert("RGB")
    width, height = image.size
    if view == "clean":
        return image.copy()
    if view == "crop":
        crop_width = max(1, int(round(width * 28 / 32)))
        crop_height = max(1, int(round(height * 28 / 32)))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), _resampling("BICUBIC"))
    if view == "color_jitter":
        result = ImageEnhance.Brightness(image).enhance(0.8)
        result = ImageEnhance.Contrast(result).enhance(1.2)
        return ImageEnhance.Color(result).enhance(0.8)
    if view == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0))
    if view == "occlusion":
        array = np.asarray(image, dtype=np.uint8).copy()
        occ_width = max(1, int(round(width * 8 / 32)))
        occ_height = max(1, int(round(height * 8 / 32)))
        left = (width - occ_width) // 2
        top = (height - occ_height) // 2
        array[top:top + occ_height, left:left + occ_width] = 128
        return Image.fromarray(array, mode="RGB")
    if view == "resize":
        small_width = max(1, int(round(width * 20 / 32)))
        small_height = max(1, int(round(height * 20 / 32)))
        small = image.resize((small_width, small_height), _resampling("BILINEAR"))
        return small.resize((width, height), _resampling("BICUBIC"))
    raise AssertionError("unreachable")


def materialize_fixed_views(
    images: Sequence[Image.Image],
    transform: Callable,
) -> dict[str, list]:
    """Materialize all views in fixed order with a caller-provided test transform."""
    return {
        view: [transform(fixed_view(image, view)) for image in images]
        for view in FROZEN_VIEW_NAMES
    }
