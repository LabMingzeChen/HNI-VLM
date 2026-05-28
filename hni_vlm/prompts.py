"""
hni_vlm.prompts
---------------
Centralized prompt templates for Human-Nature Interaction (HNI) classification.

Each prompt corresponds to one task in the HNI hierarchical semantic framework:

Primary task:
    - PROMPT_PRIMARY_CATEGORY   :  Setting | Subject | Partner

Secondary tasks:
    - PROMPT_SOCIAL_CONTEXT     :  solo | group | NA
    - PROMPT_ACTIVITY_INTENSITY :  sedentary | moderate | vigorous | NA
    - PROMPT_PHOTOGRAPHER       :  documented | implied

Composite prompt:
    - PROMPT_FULL_HNI : returns all four labels as a JSON object in one call.

These prompts are the prompt-tuned versions reported in:
    Chen, M. (2026). "Vision-language models for human-nature interaction
    in urban environments." PhD Dissertation, MIT.
"""

# ----------------------------------------------------------------------
# Helper used to detect whether people are visible (Task 0)
# Lifted from the user's `run_qwen_api_50.py`.
# ----------------------------------------------------------------------
PROMPT_PEOPLE_COUNT = """You are given ONE image.

Your task is to determine how many REAL, PHYSICALLY PRESENT people are visible
in the image.

Choose EXACTLY ONE of the following three options:
- "0":  no people visible
- "1":  exactly one person visible
- "2+": two or more people visible

Counting rules:
- Count only real humans physically present in the scene.
- A person counts if any visible body part is human (head, face, torso,
  arms, or legs).
- Very small, distant, partially occluded, or low-light humans MUST be
  counted if they are recognizable as human.
- Do NOT count statues, mannequins, posters, photos, reflections,
  shadows, or people on screens.
- If you are not sure whether something is human, do NOT count it.

Output ONLY one of the following values, without any explanation:
0
1
2+
"""

# ----------------------------------------------------------------------
# Task 1 - Primary HNI category (Setting / Subject / Partner)
# ----------------------------------------------------------------------
PROMPT_PRIMARY_CATEGORY = r"""
You are given ONE image. Classify the FUNCTIONAL ROLE of nature in the scene.

Output EXACTLY one of:
- "setting" : nature is the environmental backdrop for activities that are
              NOT inherently oriented toward nature (e.g., jogging on a
              park path, social gathering on a lawn, sitting on a bench).
- "subject" : nature is the primary focus of visual / cognitive attention
              (e.g., landscape photography, wildlife watching, scenic
              viewing where people face nature).
- "partner" : people directly and physically engage with natural elements
              (e.g., gardening, touching plants, swimming in natural
              water, feeding animals, climbing rocks).

Decision rules:
- Use only what is visible in the image.
- If uncertain between "subject" and "setting", prefer "setting".
- If uncertain between "partner" and "subject", prefer "subject".
- Output ONE word only, lowercase, no punctuation, no explanation.
""".strip()


# ----------------------------------------------------------------------
# Task 2 - Social context
# ----------------------------------------------------------------------
PROMPT_SOCIAL_CONTEXT = r"""
You are given ONE image. Classify the SOCIAL CONTEXT of the people visible.

Output EXACTLY one of:
- "solo"  : exactly one person visible.
- "group" : two or more people visible.
- "NA"    : no people visible at all.

Counting rules:
- A person counts if any human body part is visible (head, face, torso,
  arms, legs).
- Do NOT count statues, mannequins, photos, screens, or reflections.
- Very small / distant / partially occluded humans STILL count.

Output ONE token only: solo, group, or NA.
""".strip()


# ----------------------------------------------------------------------
# Task 3 - Activity intensity (your improved V2 prompt)
# ----------------------------------------------------------------------
PROMPT_ACTIVITY_INTENSITY = r"""
You are given ONE image. Output the OVERALL activity intensity of the MAIN
group of people (majority).

You must do TWO internal steps:

Step 1) Roughly estimate how many visible people belong to each class:
    - sedentary_count : still (lying / sitting / standing still / posing)
    - moderate_count  : light movement (walking, fishing, small body motions)
    - vigorous_count  : vigorous / sports (more intense than walking)

Step 2) Decide the FINAL label by majority rule:
    - If one count is clearly the largest, choose that label.
    - If tied / unclear, choose the more conservative label:
        vigorous vs moderate  -> moderate
        moderate vs sedentary -> sedentary
        vigorous vs sedentary -> moderate (middle)

Definitions (STRICT):
A) sedentary  - lying, sitting, standing still, posing / looking at camera.
B) moderate   - walking, fishing (ALWAYS moderate), light arm/hand motions,
                casual small actions.
C) vigorous   - swimming, cycling (any kind), ball sports, skating, skiing,
                snowboarding, running, workouts. Anything more intense
                than walking.

Special cases:
- If NO people are visible, output "NA".

Output ONE word only:
sedentary | moderate | vigorous | NA
""".strip()


# ----------------------------------------------------------------------
# Task 4 - Photographer engagement
# ----------------------------------------------------------------------
PROMPT_PHOTOGRAPHER_ENGAGEMENT = r"""
You are given ONE image. Classify the PHOTOGRAPHER engagement type.

Output EXACTLY one of:
- "documented" : a person inside the image is visibly photographing /
                 filming a natural element (holding a camera / phone
                 toward nature).
- "implied"    : the photographer is not visible inside the frame, but the
                 composition (centered landscape, framed wildlife close-up,
                 deliberate aesthetic framing toward nature) shows the
                 image-maker is paying intentional attention to nature.

If neither applies clearly, prefer "implied".

Output ONE word only: documented or implied.
""".strip()


# ----------------------------------------------------------------------
# Composite prompt - all 4 tasks in one call (JSON output)
# Recommended for production / batch use.
# ----------------------------------------------------------------------
PROMPT_FULL_HNI = r"""
You are given ONE image. Analyze the human-nature interaction (HNI) it
depicts, then return a STRICT JSON object with FOUR keys.

Schema:
{
  "primary_category":        "setting" | "subject" | "partner",
  "social_context":          "solo"    | "group"   | "NA",
  "activity_intensity":      "sedentary" | "moderate" | "vigorous" | "NA",
  "photographer_engagement": "documented" | "implied"
}

Definitions:

A) primary_category  (the FUNCTIONAL ROLE of nature)
   - setting : nature is the backdrop for activities NOT oriented toward
               nature (jogging, gathering, sitting on a bench).
   - subject : nature is the focus of attention (landscape photo,
               wildlife watching, scenic viewing).
   - partner : people physically engage WITH nature (gardening, touching
               plants, swimming in natural water, climbing rocks).

B) social_context
   - solo  : exactly one real, physically-present person.
   - group : two or more real people.
   - NA    : no people visible at all.

C) activity_intensity
   - sedentary : sitting / lying / standing still / posing.
   - moderate  : walking, fishing, light movement.
   - vigorous  : swimming, cycling, running, sports.
   - NA        : no people visible.

D) photographer_engagement
   - documented : a person in the image is visibly photographing or
                  filming a natural element.
   - implied    : photographer not in frame, but composition shows
                  intentional aesthetic attention toward nature.

Conservative tie-breaks:
   - subject vs setting  -> setting
   - partner vs subject  -> subject
   - vigorous vs moderate -> moderate
   - moderate vs sedentary -> sedentary

Output STRICTLY the JSON object. NO markdown, NO code fences, NO extra
text before or after.
""".strip()


# Convenience lookup
PROMPTS = {
    "people_count":            PROMPT_PEOPLE_COUNT,
    "primary_category":        PROMPT_PRIMARY_CATEGORY,
    "social_context":          PROMPT_SOCIAL_CONTEXT,
    "activity_intensity":      PROMPT_ACTIVITY_INTENSITY,
    "photographer_engagement": PROMPT_PHOTOGRAPHER_ENGAGEMENT,
    "full":                    PROMPT_FULL_HNI,
}
