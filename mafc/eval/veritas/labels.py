from typing import Mapping
from mafc.common.label import BaseLabel


class Veritas3Label(BaseLabel):
    INTACT = "intact"
    COMPROMISED = "compromised"
    UNKNOWN = "unknown"


class Veritas7Label(BaseLabel):
    INTACT_CERTAIN = "intact (certain)"
    INTACT_RATHER_CERTAIN = "intact (rather certain)"
    INTACT_RATHER_UNCERTAIN = "intact (rather uncertain)"
    UNKNOWN = "unknown"
    COMPROMISED_RATHER_UNCERTAIN = "compromised (rather uncertain)"
    COMPROMISED_RATHER_CERTAIN = "compromised (rather certain)"
    COMPROMISED_CERTAIN = "compromised (certain)"


# Benchmark-provided mapping from human-visible class strings to enum values
CLASS_MAPPING_3: Mapping[str, BaseLabel] = {
    "Intact": Veritas3Label.INTACT,
    "Compromised": Veritas3Label.COMPROMISED,
    "Unknown": Veritas3Label.UNKNOWN,
}

CLASS_DEFINITIONS_3: Mapping[BaseLabel, str] = {
    Veritas3Label.INTACT: "The claim has intact integrity (score >= 0.33). The claim is factually accurate, "
    "and any media is authentic and properly contextualized.",
    Veritas3Label.COMPROMISED: "The claim has compromised integrity (score <= -0.33). The claim is factually inaccurate, "
    "misleading, or contains manipulated/out-of-context media.",
    Veritas3Label.UNKNOWN: "The integrity of the claim is unknown or uncertain (-0.33 < score < 0.33). "
    "There is insufficient evidence to determine whether the claim is intact or compromised.",
}

EXTRA_JUDGE_RULES_3 = """* Holistic Integrity Assessment: The integrity verdict should reflect:
    - High integrity (Intact): Claim is factually accurate AND any media is authentic and properly contextualized
    - Low integrity (Compromised): Claim is factually inaccurate OR media is manipulated/out-of-context
    - Uncertain integrity (Unknown): Insufficient evidence to make a determination
    * Media Impact: Even if text is accurate, misused media can compromise integrity.
    * Scoring Thresholds:
      - Intact: integrity >= 0.33
      - Unknown: -0.33 < integrity < 0.33
      - Compromised: integrity <= -0.33
"""

THRESHOLDS_3 = {
    "intact": 0.33,  # score >= 0.33
    "compromised": -0.33,  # score <= -0.33
}


CLASS_MAPPING_7: Mapping[str, BaseLabel] = {
    "Intact (certain)": Veritas7Label.INTACT_CERTAIN,
    "Intact (rather certain)": Veritas7Label.INTACT_RATHER_CERTAIN,
    "Intact (rather uncertain)": Veritas7Label.INTACT_RATHER_UNCERTAIN,
    "Unknown": Veritas7Label.UNKNOWN,
    "Compromised (rather uncertain)": Veritas7Label.COMPROMISED_RATHER_UNCERTAIN,
    "Compromised (rather certain)": Veritas7Label.COMPROMISED_RATHER_CERTAIN,
    "Compromised (certain)": Veritas7Label.COMPROMISED_CERTAIN,
}

CLASS_DEFINITIONS_7: Mapping[BaseLabel, str] = {
    Veritas7Label.INTACT_CERTAIN: "The claim is factually accurate with strong, unequivocal evidence. "
    "Any associated media is authentic and properly contextualized.",
    Veritas7Label.INTACT_RATHER_CERTAIN: "The claim appears factually accurate with strong but not fully definitive evidence. "
    "Media appears authentic and properly contextualized.",
    Veritas7Label.INTACT_RATHER_UNCERTAIN: "The claim weakly appears factually accurate based on limited evidence. "
    "There is some indication of integrity but not enough for confidence.",
    Veritas7Label.UNKNOWN: "There is insufficient evidence to determine the claim's accuracy or integrity.",
    Veritas7Label.COMPROMISED_RATHER_UNCERTAIN: "The claim weakly appears inaccurate or misleading based on limited evidence. "
    "There is some indication of compromised integrity but not enough for confidence.",
    Veritas7Label.COMPROMISED_RATHER_CERTAIN: "The claim appears inaccurate or misleading with strong but not fully definitive evidence. "
    "Media appears manipulated or used out of context.",
    Veritas7Label.COMPROMISED_CERTAIN: "The claim is factually inaccurate, misleading, or contains manipulated/miscontextualized "
    "media with strong, unequivocal evidence.",
}

EXTRA_JUDGE_RULES_7 = """* Holistic Integrity Assessment with Uncertainty: The integrity verdict should reflect
    both the direction (intact vs compromised) and your confidence level (certain, rather certain, rather uncertain).
    - Intact (certain): Claim is factually accurate AND any media is authentic with strong, unequivocal evidence
    - Intact (rather certain): Claim appears accurate with strong but not fully definitive evidence
    - Intact (rather uncertain): Claim weakly appears accurate based on limited evidence
    - Unknown: Insufficient evidence to determine integrity in either direction
    - Compromised (rather uncertain): Claim weakly appears inaccurate based on limited evidence
    - Compromised (rather certain): Claim appears inaccurate with strong but not fully definitive evidence
    - Compromised (certain): Claim is clearly inaccurate, misleading, or contains manipulated media
    * Media Impact: Even if text is accurate, misused media can compromise integrity.
    * Confidence Calibration: Choose the uncertainty level that best reflects the strength of available evidence.
      Only use "certain" when evidence is overwhelming and unambiguous.
"""

THRESHOLDS_7 = [
    (-5 / 6, Veritas7Label.COMPROMISED_CERTAIN),
    (-3 / 6, Veritas7Label.COMPROMISED_RATHER_CERTAIN),
    (-1 / 6, Veritas7Label.COMPROMISED_RATHER_UNCERTAIN),
    (1 / 6, Veritas7Label.UNKNOWN),
    (3 / 6, Veritas7Label.INTACT_RATHER_UNCERTAIN),
    (5 / 6, Veritas7Label.INTACT_RATHER_CERTAIN),
    (float("inf"), Veritas7Label.INTACT_CERTAIN),
]
