# The HNI Semantic Framework

HNI-VLM is built around a **hierarchical semantic framework** for classifying Human–Nature Interaction in images. The framework was developed and validated as part of Mingze Chen's PhD dissertation at MIT.

The core insight: the same visible activity can represent very different human–nature relationships depending on whether nature acts as a **background condition**, an **object of attention**, or an **active medium of interaction**. Object-detection pipelines tell you *what is in a scene*. HNI-VLM tells you *the functional role nature plays*.

---

## Primary categories — the functional role of nature

| Category | Definition | Example activities |
|---|---|---|
| **Setting** | Nature is the environmental backdrop for activities that are NOT inherently oriented toward natural elements. | Walking on a park path, social gathering on a lawn, exercising in green space. |
| **Subject** | Natural elements are the primary focus of visual / cognitive attention. Engagement is **perceptual**, not physical. | Landscape photography, wildlife watching, scenic viewing. |
| **Partner** | People directly and physically engage with natural elements. Nature is an **active medium** of the interaction. | Gardening, touching plants, swimming in natural water, climbing rocks. |

The three categories are grounded in three distinct theoretical traditions:

- **Setting** ↔ Behavior-setting theory (environments structure activities without being focal).
- **Subject** ↔ Aesthetic / restorative perception (Ulrich, 1991).
- **Partner** ↔ Affordance theory (environments invite specific embodied actions).

---

## Secondary tags — contextual attributes

Three orthogonal contextual dimensions are predicted jointly with the primary category:

### 1. Social context

| Value | Definition |
|---|---|
| `solo`  | Exactly one real, physically-present person. |
| `group` | Two or more real people. |
| `NA`    | No people visible. |

### 2. Activity intensity

| Value | Definition |
|---|---|
| `sedentary` | Minimal bodily movement — sitting, lying, standing still, posing. |
| `moderate`  | Light to moderate movement — walking, casual exploration, fishing. |
| `vigorous`  | Sustained physical exertion — running, cycling, swimming, sports. |
| `NA`        | No people visible. |

(Classes follow WHO guidelines on physical activity.)

### 3. Photographer engagement

| Value | Definition |
|---|---|
| `documented` | A person in the image is visibly photographing / filming nature. |
| `implied`    | Photographer is not in frame, but composition shows intentional aesthetic attention to nature. |

---

## Why this matters for urban planning

The primary category translates directly into different planning priorities:

- A space dominated by **Setting** interactions may support **circulation and rest** but not perceptual or embodied engagement.
- **Subject**-dominated spaces point to **scenic, aesthetic, or restorative value**.
- Rare **Partner** interactions may indicate **limited affordances** for touch, play, stewardship, water contact, or ecological participation.

By scaling this distinction to thousands of images, HNI-VLM lets planners compare urban nature spaces by **interaction quality**, not only by accessibility, greenness, or visitation volume.

---

## Joint output schema

A single inference call returns a structured object:

```python
HNIResult(
    primary_category        = "subject",
    social_context          = "solo",
    activity_intensity      = "sedentary",
    photographer_engagement = "documented",
)
```

This compositional representation supports downstream tasks like spatial-temporal aggregation, comparative analysis across cities, and integration with GIS layers.

---

## Conservative tie-breaks (built into the prompts)

When two labels are visually plausible, the prompts choose the more conservative interpretation:

- `subject` vs `setting`   → **`setting`**
- `partner` vs `subject`   → **`subject`**
- `vigorous` vs `moderate` → **`moderate`**
- `moderate` vs `sedentary` → **`sedentary`**

These rules were tuned iteratively on the 495-image gold-standard dataset to align model behavior with conservative human judgment.

---

## Reference

Chen, M. (2026). *Vision-Language Models for Human–Nature Interaction in Urban Environments.* PhD Dissertation, Massachusetts Institute of Technology.
