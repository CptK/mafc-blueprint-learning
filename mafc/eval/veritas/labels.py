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

# Numeric integrity-score value of each label (enum-keyed; used e.g. by the
# judge's n-sample aggregation and mirrored as strings in eval/veritas/metrics.py).
LABEL_NUMERIC_3: Mapping[BaseLabel, float] = {
    Veritas3Label.INTACT: 1.0,
    Veritas3Label.UNKNOWN: 0.0,
    Veritas3Label.COMPROMISED: -1.0,
}

LABEL_NUMERIC_7: Mapping[BaseLabel, float] = {
    Veritas7Label.INTACT_CERTAIN: 1.0,
    Veritas7Label.INTACT_RATHER_CERTAIN: 2 / 3,
    Veritas7Label.INTACT_RATHER_UNCERTAIN: 1 / 3,
    Veritas7Label.UNKNOWN: 0.0,
    Veritas7Label.COMPROMISED_RATHER_UNCERTAIN: -1 / 3,
    Veritas7Label.COMPROMISED_RATHER_CERTAIN: -2 / 3,
    Veritas7Label.COMPROMISED_CERTAIN: -1.0,
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
    * Confidence Calibration — apply these anchors strictly:
      Use "certain" only when BOTH conditions hold:
        (a) A direct primary source (official document/transcript, original upload, authoritative database entry)
            unambiguously confirms or refutes the claim with no meaningful alternative interpretation.
        (b) The evidence leaves no residual doubt — not merely a consistent picture, but definitive proof.
      Use "rather certain" when the direction is clear and evidence is strong, but at least one caveat applies:
        - No single primary source; conclusion rests on multiple corroborating secondary sources.
        - A primary source exists but requires a small inferential step to apply to the specific claim.
        - Minor alternative interpretations cannot be fully ruled out.
      Use "rather uncertain" when evidence is limited, indirect, or not from authoritative sources,
        or when the conclusion depends on significant inference beyond what the evidence directly states.
      Use "unknown" when evidence is absent, contradictory, or insufficient to determine direction.
      Default toward "rather certain" over "certain" when in doubt — reserve "certain" for cases
        where a fact-checker would stake their reputation on the finding without qualification.
    * Claim Date Semantics: The "Date:" line under the claim is the date the claim was POSTED or
      shared — it is NOT a date asserted by the claim and NOT the date of the depicted event.
      - Sharing older, authentic footage at a later date is NOT miscontextualization by itself.
        Only treat timing as a compromise if the claim TEXT asserts a specific event date/recency
        that the evidence contradicts.
      - Never call a claim inaccurate, "impossible", or manipulated because its posting date seems
        late or "in the future". Dates through mid-2026 are in the past relative to this benchmark;
        your training data may end earlier — a 2025/2026 date is never by itself evidence of
        fabrication or manipulation.
    * Same-Media Referent Rule: A fact-check, debunk, or reverse-image result counts against THIS
      claim's media only if the evidence indicates it refers to the SAME footage/image (matching
      frames or visuals, same source post, or an explicit match). Major events attract many
      different videos, some fake — a debunk of a DIFFERENT video of the same event (e.g., an
      AI-generated or miscaptioned clip circulating alongside) is NOT evidence against this media.
      "Similar videos were debunked" alone never justifies compromised.
    * Weak Visual-Similarity Matches: Generic reverse-image hits (stock-photo sites, unrelated
      business listings, educational or entertainment pages, viral reposts) show visual
      similarity, not identity. Such hits are weak evidence in EITHER direction: do not conclude
      the media is stock/generic/unrelated footage from them alone, and equally do not treat
      "search results link/associate the visuals with reporting" as authentication. Identity
      requires an exact match or a source explicitly tying THIS media to the other content.
    * Event ≠ Media Provenance: Evidence that the claimed EVENT happened does NOT establish that
      this media shows it — fabricated or recycled media routinely accompanies real events. For
      claims of the form "this video/image shows X", any intact label requires POSITIVE
      provenance for the media itself: the original upload, an exact reverse-image match to
      reporting of the claimed event, or a credible source explicitly authenticating this
      footage. Phrases like "visuals align with", "consistent with reports", or corroboration of
      the event alone are NOT provenance — in that case the ceiling is "unknown" or
      "intact (rather uncertain)"; never "intact (rather certain)" or "intact (certain)".
    * Exaggeration and Denied Statements: If the claim asserts a superlative or specific figure
      ("best in history", "exclusively", exact percentages) that the sources do not actually
      state, or reports a statement/event that the subject later denied, clarified, or that
      fact-checkers flagged as misleading framing, treat the claim as compromised — do not
      round it up to intact because its gist or underlying event is real.
    * Central Assertion Focus: Purely nominal imprecision in a secondary detail (an exact venue
      or event name, an honorific/title) may be reflected as lower confidence rather than an
      automatic flip to compromised — but ONLY when no evidence suggests the framing misleads.
      Any indication of misleading framing, missing context, or partial fabrication (e.g., some
      of the depicted media unrelated) keeps the claim compromised.
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
