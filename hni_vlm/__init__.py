"""
HNI-VLM: A vision-language toolkit for Human-Nature Interaction classification.

Quick start
-----------
    >>> from hni_vlm import HNIClassifier
    >>> model = HNIClassifier(backend="qwen")   # needs DASHSCOPE_API_KEY
    >>> result = model.predict("park.jpg")
    >>> print(result.primary_category)   # "setting" | "subject" | "partner"

See https://github.com/<your-username>/HNI-VLM for full documentation.
"""
from .classifier import HNIClassifier
from .schemas    import HNIResult, LABEL_SETS
from .evaluate   import evaluate_predictions

__all__ = [
    "HNIClassifier",
    "HNIResult",
    "LABEL_SETS",
    "evaluate_predictions",
]

__version__ = "0.1.0"
__author__  = "Mingze Chen"
