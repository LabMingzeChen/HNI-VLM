"""Smoke tests — make sure imports and basic plumbing work without an API key."""
import pytest


def test_import_top_level():
    import hni_vlm
    assert hasattr(hni_vlm, "HNIClassifier")
    assert hasattr(hni_vlm, "HNIResult")
    assert hasattr(hni_vlm, "evaluate_predictions")
    assert hni_vlm.__version__


def test_label_sets_complete():
    from hni_vlm import LABEL_SETS
    assert "setting"  in LABEL_SETS["primary_category"]
    assert "subject"  in LABEL_SETS["primary_category"]
    assert "partner"  in LABEL_SETS["primary_category"]
    assert "solo"     in LABEL_SETS["social_context"]
    assert "vigorous" in LABEL_SETS["activity_intensity"]
    assert "implied"  in LABEL_SETS["photographer_engagement"]


def test_result_to_dict_roundtrip():
    from hni_vlm import HNIResult
    r = HNIResult(
        primary_category="setting",
        social_context="group",
        activity_intensity="moderate",
        photographer_engagement="implied",
    )
    d = r.to_dict()
    assert d["primary_category"] == "setting"
    assert r.is_valid()


def test_prompt_templates_present():
    from hni_vlm.prompts import PROMPTS
    expected = {
        "people_count", "primary_category", "social_context",
        "activity_intensity", "photographer_engagement", "full",
    }
    assert expected.issubset(PROMPTS.keys())
    for v in PROMPTS.values():
        assert isinstance(v, str) and len(v) > 0


def test_invalid_label_marks_invalid():
    from hni_vlm import HNIResult
    r = HNIResult(
        primary_category="not_a_category",
        social_context="group",
        activity_intensity="moderate",
        photographer_engagement="implied",
    )
    assert not r.is_valid()
