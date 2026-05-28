"""
hni_vlm.classifier
------------------
Main user-facing API: `HNIClassifier`.

Designed to mirror the ergonomics of `ultralytics.YOLO`:

    >>> from hni_vlm import HNIClassifier
    >>> model = HNIClassifier(backend="qwen")
    >>> result = model.predict("park.jpg")
    >>> print(result.primary_category)   # "setting"

Batch processing, CSV input, and resumable runs are supported via
`predict_batch()`.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional, Union

from .models     import BaseVLM, QwenVLM
from .prompts    import PROMPT_FULL_HNI
from .schemas    import HNIResult, LABEL_SETS
from .utils      import ImageInput


# ----------------------------------------------------------------------
# Helpers — label normalization (gracefully handle case / whitespace /
# common synonyms produced by VLMs).
# ----------------------------------------------------------------------
def _norm(s: object) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip().strip('"').strip("'")
    s_low = s.lower()
    if s_low in {"na", "n/a", "none", "null", ""}:
        return "NA"
    return s_low


def _extract_json(text: str) -> dict:
    """
    Robust JSON extraction. Some VLMs wrap responses in code fences or
    include preamble. We try plain `json.loads` first, then fall back to
    regex-extracting the last `{...}` block.
    """
    t = (text or "").strip()
    t = t.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    candidates = re.findall(r"\{[\s\S]*?\}", t)
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except Exception:
            continue
    raise ValueError(f"No valid JSON found. Output head: {t[:200]}")


def _parse_full_response(raw: str, image_path: Optional[str] = None) -> HNIResult:
    """Parse the JSON returned by `PROMPT_FULL_HNI` into an `HNIResult`."""
    obj = _extract_json(raw)
    out = HNIResult(
        primary_category        = _norm(obj.get("primary_category")),
        social_context          = _norm(obj.get("social_context")),
        activity_intensity      = _norm(obj.get("activity_intensity")),
        photographer_engagement = _norm(obj.get("photographer_engagement")),
        image_path              = image_path,
        raw_response            = raw,
    )

    # Validate against the allowed label sets; mark error if anything off.
    for field_name, allowed in LABEL_SETS.items():
        v = getattr(out, field_name)
        if v not in allowed:
            out.error = f"Invalid {field_name}: {v!r}"
            break
    return out


# ======================================================================
# Main user-facing class
# ======================================================================
class HNIClassifier:
    """
    Classify Human-Nature Interaction in images using a VLM backend.

    Parameters
    ----------
    backend : str | BaseVLM, default "qwen"
        Which backend to use. Currently supported: "qwen".
        Power users can pass a custom `BaseVLM` instance.
    api_key : str, optional
        Forwarded to the backend. If None, the backend reads from env.
    **backend_kwargs
        Additional keyword arguments passed to the backend constructor
        (e.g., `model_name`, `temperature`, `retries`).

    Example
    -------
    >>> from hni_vlm import HNIClassifier
    >>> model = HNIClassifier(backend="qwen")           # reads DASHSCOPE_API_KEY
    >>> r = model.predict("https://example.com/park.jpg")
    >>> r.primary_category, r.activity_intensity
    ('setting', 'moderate')
    """

    def __init__(
        self,
        backend: Union[str, BaseVLM] = "qwen",
        api_key: Optional[str] = None,
        **backend_kwargs,
    ):
        if isinstance(backend, BaseVLM):
            self.backend = backend
            return

        backend = backend.lower()
        if backend == "qwen":
            self.backend = QwenVLM(api_key=api_key, **backend_kwargs)
        else:
            raise ValueError(
                f"Unknown backend: {backend!r}. "
                "Supported: 'qwen'. "
                "For GPT-4 / Gemini, contribute a subclass of BaseVLM."
            )

    # ------------------------------------------------------------------
    def predict(self, image: ImageInput) -> HNIResult:
        """
        Classify a single image and return an `HNIResult`.

        Parameters
        ----------
        image : str | Path | PIL.Image | bytes
            - URL                  : sent directly (with base64 fallback).
            - Local path           : loaded and base64-encoded.
            - PIL.Image / bytes    : base64-encoded.

        Returns
        -------
        HNIResult
        """
        image_path = str(image) if isinstance(image, (str, Path)) else None
        try:
            raw = self.backend.predict(image, PROMPT_FULL_HNI)
            return _parse_full_response(raw, image_path=image_path)
        except Exception as exc:                              # noqa: BLE001
            return HNIResult(image_path=image_path, error=str(exc))

    # ------------------------------------------------------------------
    def predict_batch(
        self,
        images: Iterable[ImageInput],
        sleep_between: float = 0.2,
        verbose: bool = True,
    ) -> List[HNIResult]:
        """
        Classify a list / generator of images.

        Parameters
        ----------
        images : iterable of image inputs
            See `predict()` for accepted types.
        sleep_between : float
            Seconds to sleep between calls (gentle rate-limiting).
        verbose : bool
            If True, show a tqdm progress bar (if tqdm is installed).

        Returns
        -------
        list[HNIResult]
        """
        images = list(images)

        iterator: Iterable
        if verbose:
            try:
                from tqdm import tqdm                         # noqa: WPS433
                iterator = tqdm(images, desc="HNI predict")
            except ImportError:
                iterator = images
        else:
            iterator = images

        results: List[HNIResult] = []
        for img in iterator:
            results.append(self.predict(img))
            if sleep_between:
                time.sleep(sleep_between)
        return results
