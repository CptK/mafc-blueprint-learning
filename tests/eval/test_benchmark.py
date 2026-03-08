from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from mafc.common.label import BaseLabel
from mafc.eval.benchmark import Benchmark


class DummyLabel(BaseLabel):
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass
class DummySample:
    id: str
    label: BaseLabel
    justification: dict[str, Any] | None = None


class DummyBenchmark(Benchmark[DummySample]):
    name = "Dummy"
    shorthand = "dummy"
    class_mapping = {"Positive": DummyLabel.POSITIVE, "Negative": DummyLabel.NEGATIVE}
    class_definitions = {
        DummyLabel.POSITIVE: "Positive class",
        DummyLabel.NEGATIVE: "Negative class",
    }
    available_actions: list[type] = []

    def _load_data(self) -> list[DummySample]:
        return [
            DummySample(id="a", label=DummyLabel.POSITIVE, justification={"k": 1}),
            DummySample(id="b", label=DummyLabel.NEGATIVE, justification=None),
            DummySample(id="c", label=DummyLabel.POSITIVE, justification={"k": 2}),
        ]


def test_basic_collection_behavior() -> None:
    bench = DummyBenchmark(variant="test")

    assert len(bench) == 3
    assert [s.id for s in bench] == ["a", "b", "c"]
    assert bench.labels == [DummyLabel.POSITIVE, DummyLabel.NEGATIVE, DummyLabel.POSITIVE]
    assert bench.get_by_id("b").label == DummyLabel.NEGATIVE
    assert bench.get_class_name(DummyLabel.POSITIVE) == "Positive"


def test_shuffle_is_deterministic() -> None:
    bench_1 = DummyBenchmark(variant="test")
    bench_2 = DummyBenchmark(variant="test")

    bench_1.shuffle()
    bench_2.shuffle()

    assert [s.id for s in bench_1] == [s.id for s in bench_2]


def test_process_output_saves_prediction_and_stats(monkeypatch) -> None:
    bench = DummyBenchmark(variant="test")
    saved_prediction: dict[str, Any] = {}
    saved_stats: dict[str, Any] = {}

    def fake_save_next_prediction(**kwargs) -> None:
        saved_prediction.update(kwargs)

    def fake_save_next_instance_stats(stats, claim_id) -> None:
        saved_stats["stats"] = stats
        saved_stats["claim_id"] = claim_id

    monkeypatch.setattr("mafc.eval.benchmark.logger.save_next_prediction", fake_save_next_prediction)
    monkeypatch.setattr("mafc.eval.benchmark.logger.save_next_instance_stats", fake_save_next_instance_stats)
    monkeypatch.setattr("mafc.eval.benchmark.logger.log", lambda *args, **kwargs: None)

    class DummyClaim:
        id = "a"

        def describe(self) -> str:
            return "claim-a"

    doc = SimpleNamespace(claim=DummyClaim(), verdict=DummyLabel.POSITIVE, justification="because")
    meta = {"Statistics": {"tokens": 123}}

    bench.process_output((doc, meta))

    assert saved_prediction["sample_index"] == "a"
    assert saved_prediction["target"] == DummyLabel.POSITIVE
    assert saved_prediction["predicted"] == DummyLabel.POSITIVE
    assert saved_prediction["justification"] == "because"
    assert saved_prediction["gt_justification"] == {"k": 1}
    assert saved_stats == {"stats": {"tokens": 123}, "claim_id": "a"}
