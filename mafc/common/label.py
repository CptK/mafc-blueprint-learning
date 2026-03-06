from enum import Enum


class Label(Enum):
    SUPPORTED = "supported"
    NEI = "not enough information"
    REFUTED = "refuted"
    CONFLICTING = "conflicting evidence"
    CHERRY_PICKING = "cherry-picking"
    REFUSED_TO_ANSWER = "error: refused to answer"
    OUT_OF_CONTEXT = "out of context"
    MISCAPTIONED = "miscaptioned"
    INTACT = "intact"
    COMPROMISED = "compromised"
    UNKNOWN = "unknown"

    # 7-class integrity labels (with uncertainty levels)
    INTACT_CERTAIN = "intact (certain)"
    INTACT_RATHER_CERTAIN = "intact (rather certain)"
    INTACT_RATHER_UNCERTAIN = "intact (rather uncertain)"
    COMPROMISED_RATHER_UNCERTAIN = "compromised (rather uncertain)"
    COMPROMISED_RATHER_CERTAIN = "compromised (rather certain)"
    COMPROMISED_CERTAIN = "compromised (certain)"