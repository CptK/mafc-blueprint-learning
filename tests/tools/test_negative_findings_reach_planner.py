"""Every media detector must report back, including when it finds nothing.

A tool that returns no summary leaves the planner with no answer at all — it
cannot tell "already checked, nothing there" from "not checked yet", so it can
re-delegate the same task indefinitely. The reverse image search path already
treats its no-match case this way ("RIS found nothing" is a finding); these
tests hold the manipulation detectors to the same rule.

The other half of the rule: reporting a negative must not turn it into positive
evidence. `Results.is_useful()` stays False so relevance ranking still treats it
as a non-signal, and the text must say plainly that it implies nothing about
authenticity.
"""

import pytest

from mafc.tools.media.c2pa_checker import C2PAProvenanceResults
from mafc.tools.media.gend.tool import DeepfakeDetectionResults
from mafc.tools.media.oracle import OracleManipulationResults
from mafc.tools.media.sightengine.tool import SightengineDetectionResults

# One "found nothing" result per detector, with the phrase that must survive
# into the planner-visible text so the negative cannot be read as a positive.
EMPTY_RESULTS = [
    pytest.param(
        C2PAProvenanceResults(notes=["no C2PA manifest"]),
        "says nothing about authenticity",
        id="c2pa-no-manifest",
    ),
    pytest.param(
        SightengineDetectionResults(),
        "not evidence about authenticity",
        id="sightengine-no-scores",
    ),
    pytest.param(
        DeepfakeDetectionResults(p_fake=None, n_faces=0),
        "not evidence either way",
        id="gend-no-face",
    ),
    pytest.param(
        OracleManipulationResults(label="unknown", found=True),
        "not evidence that it is authentic",
        id="oracle-unassessed",
    ),
]


@pytest.mark.parametrize("result, disclaimer", EMPTY_RESULTS)
def test_empty_result_still_renders_a_report(result, disclaimer) -> None:
    rendered = str(result)
    assert rendered.strip(), "a detector that found nothing still has to say so"
    assert "No useful results" not in rendered, "placeholder text is not a report"
    assert len(rendered) > 40, f"too terse to be actionable: {rendered!r}"


@pytest.mark.parametrize("result, disclaimer", EMPTY_RESULTS)
def test_empty_result_is_not_positive_evidence(result, disclaimer) -> None:
    """It must be reported, and it must not read as a clean bill of health."""
    assert result.is_useful() is False
    assert disclaimer in str(result).lower() or disclaimer in str(
        result
    ), f"missing the disclaimer that makes this safe to report: {str(result)!r}"


@pytest.mark.parametrize("result, disclaimer", EMPTY_RESULTS)
def test_summarize_does_not_swallow_the_negative(result, disclaimer) -> None:
    """The tool-level guard: _summarize must not return None just because the
    result carries no positive signal. That is the gate that used to make C2PA
    and Sightengine invisible to the planner."""
    from mafc.tools.media.c2pa_checker import C2PAChecker
    from mafc.tools.media.sightengine.tool import SightengineChecker

    checkers = {
        C2PAProvenanceResults: C2PAChecker,
        SightengineDetectionResults: SightengineChecker,
    }
    checker_cls = checkers.get(type(result))
    if checker_cls is None:
        pytest.skip("covered by the tool's own test module")

    checker = checker_cls.__new__(checker_cls)  # no network/model setup needed
    assert checker._summarize(result) is not None, "negative finding was swallowed"
