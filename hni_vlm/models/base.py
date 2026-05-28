"""
hni_vlm.models.base
-------------------
Abstract base class for VLM backends.

Adding a new model (e.g. Claude, LLaVA, InternVL) only requires
subclassing `BaseVLM` and implementing `_call(image_url, prompt)`.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

from ..utils.image_utils import to_data_url, ImageInput


class BaseVLM(ABC):
    """
    Minimal interface every VLM backend must implement.

    Subclasses only need to define `_call(image_url, prompt) -> str`.
    The retry / fallback logic (try direct URL first, then base64) lives
    here so every backend gets it for free.
    """

    def __init__(
        self,
        model_name: str,
        timeout: int = 20,
        retries: int = 3,
        sleep_between: float = 0.2,
        max_image_side: int = 768,
    ):
        self.model_name     = model_name
        self.timeout        = timeout
        self.retries        = retries
        self.sleep_between  = sleep_between
        self.max_image_side = max_image_side

    # ------------------------------------------------------------------
    @abstractmethod
    def _call(self, image_url: str, prompt: str) -> str:
        """
        Send (image_url, prompt) to the underlying VLM and return the raw
        text response. Implementations should NOT do any retry / fallback.
        """
        ...

    # ------------------------------------------------------------------
    def predict(
        self,
        image: ImageInput,
        prompt: str,
    ) -> str:
        """
        Robust single-image prediction.

        Strategy
        --------
        1. If `image` is already an http(s) URL, try sending the URL
           directly (fastest).
        2. On any failure, fall back to base64 data URL (most reliable).
        3. Retry up to `self.retries` times.
        """
        last_err: Optional[str] = None

        # Decide if we can try direct URL mode
        direct_url = isinstance(image, str) and image.startswith(("http://", "https://"))

        for attempt in range(1, self.retries + 1):
            try:
                if direct_url:
                    return self._call(image, prompt)
                # Local file / PIL / bytes → must encode
                data_url = to_data_url(image, max_side=self.max_image_side)
                return self._call(data_url, prompt)
            except Exception as e:                          # noqa: BLE001
                last_err = str(e)

                # If direct URL failed once, force base64 from now on
                if direct_url:
                    try:
                        data_url = to_data_url(image, max_side=self.max_image_side)
                        return self._call(data_url, prompt)
                    except Exception as e2:                  # noqa: BLE001
                        last_err = str(e2)

                if attempt < self.retries:
                    time.sleep(0.8 * attempt)

        raise RuntimeError(
            f"{self.model_name} prediction failed after {self.retries} retries. "
            f"Last error: {last_err}"
        )
