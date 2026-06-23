from mafc.training.claims_io import ClaimRecord
from mafc.training.sampler import SamplerConfig, build_pool, sample, stratum_counts


def _claim(cid: str, score: float) -> ClaimRecord:
    return ClaimRecord(id=cid, integrity_score=score)


def _synthetic() -> list[ClaimRecord]:
    records: list[ClaimRecord] = []
    # intact: mix of hard (|score|>=0.5) and easy
    for i in range(20):
        records.append(_claim(f"i_hard_{i}", 0.9))
    for i in range(20):
        records.append(_claim(f"i_easy_{i}", 0.4))
    # compromised
    for i in range(20):
        records.append(_claim(f"c_hard_{i}", -0.8))
    # unknown
    for i in range(20):
        records.append(_claim(f"u_{i}", 0.1))
    return records


def test_build_pool_assigns_strata_and_weights() -> None:
    cfg = SamplerConfig(hard_weight=3.0, easy_weight=1.0)
    pool = build_pool(_synthetic(), cfg)
    by_id = {s.id: s for s in pool}
    assert by_id["i_hard_0"].stratum == "intact/hard"
    assert by_id["i_hard_0"].weight == 3.0
    assert by_id["i_easy_0"].stratum == "intact/easy"
    assert by_id["i_easy_0"].weight == 1.0
    assert by_id["c_hard_0"].direction == "compromised"
    assert by_id["u_0"].direction == "unknown"


def test_sample_is_deterministic() -> None:
    cfg = SamplerConfig(target_n=24, seed=7)
    a = [s.id for s in sample(_synthetic(), cfg)]
    b = [s.id for s in sample(_synthetic(), cfg)]
    assert a == b


def test_sample_balances_directions() -> None:
    cfg = SamplerConfig(target_n=30, seed=1, balance_directions=True)
    selected = sample(_synthetic(), cfg)
    dirs = {}
    for s in selected:
        dirs[s.direction] = dirs.get(s.direction, 0) + 1
    # 3 directions, budget 30 -> 10 each.
    assert dirs == {"intact": 10, "compromised": 10, "unknown": 10}


def test_boundary_oversampling_prefers_hard_band() -> None:
    # With strong hard weighting, the intact stratum's selection should be
    # dominated by hard-band claims.
    cfg = SamplerConfig(target_n=30, seed=3, hard_weight=10.0, easy_weight=1.0)
    selected = sample(_synthetic(), cfg)
    intact = [s for s in selected if s.direction == "intact"]
    n_hard = sum(1 for s in intact if s.stratum == "intact/hard")
    assert n_hard > len(intact) / 2


def test_target_n_none_returns_all() -> None:
    cfg = SamplerConfig(target_n=None)
    selected = sample(_synthetic(), cfg)
    assert len(selected) == 80
    assert sum(stratum_counts(selected).values()) == 80
