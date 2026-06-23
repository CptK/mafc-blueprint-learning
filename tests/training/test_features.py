import numpy as np

from mafc.training.claims_io import ClaimRecord
from mafc.training.features import (
    FeatureExtractorConfig,
    _domain,
    dispersion_stats,
    extract_row,
)
from mafc.training.trace_io import EvidenceView, NormalisedTrace


def test_domain_parsing() -> None:
    assert _domain("https://www.aljazeera.com/video/x") == "aljazeera.com"
    assert _domain("http://factcheck.afp.com/doc") == "factcheck.afp.com"
    assert _domain(None) is None
    assert _domain("not a url") is None


def test_dispersion_stats_nan_for_singletons() -> None:
    stats = dispersion_stats(np.zeros((1, 4)))
    assert all(np.isnan(v) for v in stats.values())
    two = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    stats2 = dispersion_stats(two)
    # orthogonal vectors -> cosine distance 1.0
    assert abs(stats2["emb_disp_mean"] - 1.0) < 1e-6


def test_extract_row_target_and_leakage_free() -> None:
    trace = NormalisedTrace(
        claim_id="x1",
        trace_kind="fact_check",
        judge_label="compromised (certain)",
        judge_justification="The evidence appears to suggest manipulation.",
        evidence=[
            EvidenceView(source="https://a.com/p", takeaways_text="t1", is_useful=True),
            EvidenceView(source="https://b.com/p", takeaways_text=None, is_useful=False),
        ],
        evidence_count=2,
        n_iterations=2,
        max_iterations=2,
        hit_max_iterations=True,
        n_delegated_tasks=3,
        n_errors=1,
        retrieval_failures=1,
        evidence_growth=[0, 5],
    )
    claim = ClaimRecord(id="x1", text="some claim text", integrity_score=-0.9,
                        language="en", n_media=1, has_media=True)
    row = extract_row(trace, claim, FeatureExtractorConfig())
    assert row.target == 0.9  # abs(integrity_score)
    # judge direction is a feature; signed score / true label are NOT present
    assert row.features["judge_direction"] == "compromised"
    assert "true_label" not in row.features
    assert "gt_score" not in row.features
    assert row.features["evidence_count"] == 2.0
    assert row.features["useful_ratio"] == 0.5
    assert row.features["n_distinct_domains"] == 2.0
    assert row.features["evidence_growth_total"] == 5.0
    # blueprint-coupled / pruned features must NOT be emitted
    assert "blueprint_name" not in row.features
    assert "hit_max_iterations" not in row.features
    assert "hedge_count" not in row.features
    assert not any(k.startswith("cf_") for k in row.features)
