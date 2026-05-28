"""
hni_vlm.models.qwen_backend
---------------------------
Qwen3-Omni-Flash backend, accessed through Alibaba DashScope's
OpenAI-compatible endpoint.

This is the model selected as the backbone in:
    Chen, M. (2026). "Vision-language models for human-nature interaction
    in urban environments." PhD Dissertation, MIT.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import BaseVLM


class QwenVLM(BaseVLM):
    """
    Qwen3-Omni-Flash backend.

    Reads the API key from `api_key` argument or env var `DASHSCOPE_API_KEY`.

    Example
    -------
    >>> from hni_vlm.models import QwenVLM
    >>> backend = QwenVLM()
    >>> backend.predict("https://example.com/cat.jpg",
    ...                 "Describe this image.")
    """

    DEFAULT_MODEL = "qwen3-omni-flash"
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)

        # Lazy import so users without the openai package can still
        # `import hni_vlm` and use other backends.
        try:
            from openai import OpenAI                         # noqa: WPS433
        except ImportError as exc:
            raise ImportError(
                "QwenVLM needs the `openai` package.\n"
                "    pip install openai"
            ) from exc

        key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError(
                "Qwen backend requires a DashScope API key. "
                "Pass it as `QwenVLM(api_key=...)` or set the "
                "DASHSCOPE_API_KEY environment variable."
            )

        self._client = OpenAI(api_key=key, base_url=base_url)
        self._temperature = temperature

    # ------------------------------------------------------------------
    def _call(self, image_url: str, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text",      "text":  prompt},
                ],
            }],
            temperature=self._temperature,
        )
        return (resp.choices[0].message.content or "").strip()
