"""
HNI-VLM: A vision-language toolkit for Human-Nature Interaction classification.

Quick start
-----------
    >>> from hni_vlm import HNIClassifier
    >>> model = HNIClassifier(backend="qwen")   # needs DASHSCOPE_API_KEY
    >>> result = model.predict("park.jpg")
    >>> print(result.primary_category)   # "setting" | "subject" | "partner"

See https://github.com/LabMingzeChen/HNI-VLM for full documentation.
"""
from .classifier import HNIClassifier
from .schemas    import HNIResult, LABEL_SETS

# evaluate_predictions is optional: it needs scikit-learn. Loading it should
# never crash an inference-only environment (e.g. a Gradio Space).
try:
    from .evaluate import evaluate_predictions
except ImportError:  # scikit-learn not installed
    evaluate_predictions = None

__all__ = [
    "HNIClassifier",
    "HNIResult",
    "LABEL_SETS",
    "evaluate_predictions",
]

__version__ = "0.1.0"
__author__  = "Mingze Chen"
