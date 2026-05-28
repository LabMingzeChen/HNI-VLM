"""
hni_vlm.utils.image_utils
-------------------------
Image loading and encoding helpers.

Most VLM APIs accept either:
    (a) a direct image URL, or
    (b) a base64-encoded data URL (more reliable, avoids hot-link issues).

We always try (a) first and fall back to (b). This logic is extracted
from the user's `run_qwen_api_50.py` / `run_qwen_intensity_only.py`.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Union

import requests
from PIL import Image


# Type alias for any input the user might pass us
ImageInput = Union[str, Path, Image.Image, bytes]


DEFAULT_MAX_SIDE = 768   # downscale long side to this many px before sending
DEFAULT_TIMEOUT  = 20    # seconds, for URL downloads


# ----------------------------------------------------------------------
def _resize_long_side(img: Image.Image, max_side: int) -> Image.Image:
    """Resize so the longest side is at most `max_side` pixels."""
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    return img.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.BICUBIC,
    )


def _pil_to_data_url(img: Image.Image, quality: int = 90) -> str:
    """Convert a PIL image to a data: URL (base64 JPEG)."""
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def load_image(source: ImageInput) -> Image.Image:
    """
    Load an image from any reasonable input and return a PIL.Image.

    Parameters
    ----------
    source : str | Path | PIL.Image.Image | bytes
        - str starting with http/https : downloaded over the network.
        - str / Path to a local file   : opened from disk.
        - PIL.Image.Image              : returned as-is.
        - bytes                        : interpreted as raw image bytes.

    Returns
    -------
    PIL.Image.Image (RGB)
    """
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")

    if isinstance(source, (str, Path)):
        s = str(source)
        if s.startswith(("http://", "https://")):
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(s, timeout=DEFAULT_TIMEOUT,
                             allow_redirects=True, headers=headers)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        return Image.open(s).convert("RGB")

    raise TypeError(f"Unsupported image source type: {type(source)}")


def to_data_url(
    source: ImageInput,
    max_side: int = DEFAULT_MAX_SIDE,
    quality: int = 90,
) -> str:
    """
    Convert any supported image source into a base64 data URL,
    downscaled to `max_side` pixels on its longest edge.

    This is the safest way to ship an image to a VLM API: it avoids
    hot-link blocks, missing Content-Length errors, and huge uploads.
    """
    img = load_image(source)
    img = _resize_long_side(img, max_side)
    return _pil_to_data_url(img, quality=quality)
