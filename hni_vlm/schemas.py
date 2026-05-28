"""
hni_vlm.schemas
---------------
Typed result containers for HNI predictions.

Using a dataclass instead of a raw dict gives users IDE auto-complete and
makes downstream code self-documenting.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Literal, Dict, Any


PrimaryCategory   = Literal["setting", "subject", "partner"]
SocialContext     = Literal["solo", "group", "NA"]
ActivityIntensity = Literal["sedentary", "moderate", "vigorous", "NA"]
Photographer      = Literal["documented", "implied"]


# Canonical label sets (used by evaluator and validators)
LABEL_SETS: Dict[str, set] = {
    "primary_category":        {"setting", "subject", "partner"},
    "social_context":          {"solo", "group", "NA"},
    "activity_intensity":      {"sedentary", "moderate", "vigorous", "NA"},
    "photographer_engagement": {"documented", "implied"},
}


@dataclass
class HNIResult:
    """
    Result of an HNI classification on a single image.

    Attributes
    ----------
    primary_category : str
        Functional role of nature: "setting" | "subject" | "partner".
    social_context : str
        "solo" | "group" | "NA".
    activity_intensity : str
        "sedentary" | "moderate" | "vigorous" | "NA".
    photographer_engagement : str
        "documented" | "implied".
    image_path : str, optional
        Path / URL of the source image (for traceability in batch runs).
    raw_response : str, optional
        The raw model response string (useful for debugging).
    error : str, optional
        Error message if prediction failed. None if everything worked.
    """
    primary_category:        Optional[PrimaryCategory]   = None
    social_context:          Optional[SocialContext]     = None
    activity_intensity:      Optional[ActivityIntensity] = None
    photographer_engagement: Optional[Photographer]      = None

    image_path:   Optional[str] = None
    raw_response: Optional[str] = field(default=None, repr=False)
    error:        Optional[str] = None

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a flat dictionary (handy for `pd.DataFrame([r.to_dict()])`)."""
        return asdict(self)

    def is_valid(self) -> bool:
        """True iff all four labels are filled in and within the allowed set."""
        if self.error:
            return False
        checks = {
            "primary_category":        self.primary_category,
            "social_context":          self.social_context,
            "activity_intensity":      self.activity_intensity,
            "photographer_engagement": self.photographer_engagement,
        }
        for field_name, value in checks.items():
            if value is None or value not in LABEL_SETS[field_name]:
                return False
        return True
