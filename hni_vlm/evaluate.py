"""
hni_vlm.evaluate
----------------
Benchmarking utilities for HNI predictions vs. human annotations.

This is a refactor of the user's standalone `evaluate_multitask_labels.py`
into a reusable function:

    >>> from hni_vlm.evaluate import evaluate_predictions
    >>> summary = evaluate_predictions(
    ...     csv_path="qwen_result_500.csv",
    ...     output_csv="qwen_result_500_metrics.csv",
    ... )

Required CSV columns
--------------------
    primary_category_A,         primary_category_H
    social_context_A,           social_context_H
    Activity_Intensity_A,       Activity_Intensity_H
    photographer_engagement_A,  photographer_engagement_H

Where `_A` is the model (Assistant) prediction and `_H` is the Human label.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )
except ImportError as exc:                                  # pragma: no cover
    raise ImportError(
        "hni_vlm.evaluate needs scikit-learn.\n"
        "    pip install scikit-learn"
    ) from exc


# Task definitions: (task_name, pred_col, true_col, label_order)
TASKS: List[Tuple[str, str, str, List[str]]] = [
    ("primary_category",
        "primary_category_A", "primary_category_H",
        ["setting", "subject", "partner"]),
    ("social_context",
        "social_context_A", "social_context_H",
        ["NA", "solo", "group"]),
    ("activity_intensity",
        "Activity_Intensity_A", "Activity_Intensity_H",
        ["NA", "sedentary", "moderate", "vigorous"]),
    ("photographer_engagement",
        "photographer_engagement_A", "photographer_engagement_H",
        ["implied", "documented"]),
]


# ----------------------------------------------------------------------
def _normalize(x: object) -> Optional[str]:
    """Normalize a single label cell to reduce trivial mismatches."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    s_low = s.lower()
    if s_low in {"na", "n/a", "none", "null", ""}:
        return "NA"
    # Everything else: lowercase trimmed.
    return s_low


def _task_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    label_order: List[str],
) -> Dict[str, object]:
    """Compute accuracy + macro P/R/F1 + confusion matrix for one task."""
    mask = y_true.notna() & y_pred.notna()
    y_t, y_p = y_true[mask].tolist(), y_pred[mask].tolist()

    acc = accuracy_score(y_t, y_p)
    p, r, f1, _ = precision_recall_fscore_support(
        y_t, y_p,
        labels=label_order, average="macro", zero_division=0,
    )
    cm = confusion_matrix(y_t, y_p, labels=label_order)

    return dict(
        n_used=len(y_t),
        accuracy=acc,
        macro_precision=p,
        macro_recall=r,
        macro_f1=f1,
        confusion_matrix=cm,
        labels=label_order,
    )


# ======================================================================
def evaluate_predictions(
    csv_path: str | Path,
    output_csv: Optional[str | Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute per-task metrics + overall label-wise and joint accuracy.

    Parameters
    ----------
    csv_path : str | Path
        CSV with both model (`_A`) and human (`_H`) columns. See module docstring.
    output_csv : str | Path, optional
        If given, the summary DataFrame is written here.
    verbose : bool
        Print per-task metrics + confusion matrices to stdout.

    Returns
    -------
    pd.DataFrame  (one row per task + 2 overall summary rows)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
    if verbose:
        print(f"Loaded: {csv_path}  ({len(df)} rows)")
        print("-" * 70)

    # Normalize in-place into new `_norm` columns.
    for _, pred_col, true_col, _order in TASKS:
        for col in (pred_col, true_col):
            if col not in df.columns:
                raise KeyError(f"Missing required column: {col}")
        df[pred_col + "_norm"] = df[pred_col].map(_normalize)
        df[true_col + "_norm"] = df[true_col].map(_normalize)

    # Per-task
    rows = []
    correct_flags = []
    for name, pred_col, true_col, label_order in TASKS:
        m = _task_metrics(
            df[true_col + "_norm"],
            df[pred_col + "_norm"],
            label_order,
        )
        rows.append({
            "task":            name,
            "n_used":          m["n_used"],
            "accuracy":        m["accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall":    m["macro_recall"],
            "macro_f1":        m["macro_f1"],
        })
        correct_flags.append(
            (df[pred_col + "_norm"] == df[true_col + "_norm"])
            & df[pred_col + "_norm"].notna()
            & df[true_col + "_norm"].notna()
        )
        if verbose:
            print(f"[{name}]  n_used={m['n_used']}")
            print(f"  Accuracy        : {m['accuracy']:.4f}")
            print(f"  Macro-Precision : {m['macro_precision']:.4f}")
            print(f"  Macro-Recall    : {m['macro_recall']:.4f}")
            print(f"  Macro-F1        : {m['macro_f1']:.4f}")
            print(f"  Confusion matrix (rows=GT, cols=Pred), labels: {m['labels']}")
            print(m["confusion_matrix"])
            print("-" * 70)

    # Overall
    correct_matrix       = pd.concat(correct_flags, axis=1)   # [N, 4]
    label_wise_accuracy  = correct_matrix.mean(axis=1).mean()
    joint_accuracy       = correct_matrix.all(axis=1).mean()

    if verbose:
        print("OVERALL")
        print(f"  Label-wise accuracy (avg over 4 tasks)        : {label_wise_accuracy:.4f}")
        print(f"  Joint (exact-match) accuracy (all 4 correct)  : {joint_accuracy:.4f}")

    summary = pd.DataFrame(rows)
    summary.loc[len(summary)] = {
        "task": "OVERALL_label_wise_accuracy", "n_used": len(df),
        "accuracy": label_wise_accuracy,
        "macro_precision": None, "macro_recall": None, "macro_f1": None,
    }
    summary.loc[len(summary)] = {
        "task": "OVERALL_joint_accuracy", "n_used": len(df),
        "accuracy": joint_accuracy,
        "macro_precision": None, "macro_recall": None, "macro_f1": None,
    }

    if output_csv:
        summary.to_csv(output_csv, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"Summary saved to: {output_csv}")
    return summary
