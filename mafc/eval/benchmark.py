from abc import ABC, abstractmethod
from pathlib import Path
import random
from typing import Any, MutableSequence, Iterator, Mapping, Protocol, TypeVar, Generic

from config.globals import random_seed, data_root_dir
from mafc.common.label import BaseLabel
from mafc.common.logger import logger
from mafc.utils.console import bold, green, red


class BenchmarkSampleProtocol(Protocol):
    id: str
    label: BaseLabel
    justification: dict[str, Any] | None


TBenchmarkSample = TypeVar("TBenchmarkSample", bound=BenchmarkSampleProtocol)


class Benchmark(ABC, Generic[TBenchmarkSample]):
    """Abstract class for all benchmarks. Inherit from this class when you want to add
    a new benchmark."""

    name: str
    shorthand: str  # Used for naming files/directories

    data: MutableSequence[TBenchmarkSample]

    class_mapping: Mapping[str, BaseLabel]  # Maps benchmark-specific labels (strings) to the label enum used
    class_definitions: Mapping[BaseLabel, str]  # Explains (to the LLM) the meaning of each class/label

    file_path: Path

    available_actions: list[type]

    extra_prepare_rules: str | None = (
        None  # Additional, benchmark-specific instructions to guide LLM's initial reasoning
    )
    extra_plan_rules: str | None = (
        None  # Additional, benchmark-specific instructions to guide LLM's action planning
    )
    extra_judge_rules: str | None = (
        None  # Additional, benchmark-specific instructions to guide LLM's verdict prediction
    )

    def __init__(self, variant: str, file_path: Path | str | None = None):
        """Base class for benchmarks.

        Args:
            variant: The variant of the benchmark to use. Typically one of "train", "val", or "test".
            file_path: The path to the file (relative to the base data dir) that contains
                the data of the specified split.
        """
        self.variant = variant
        self.full_name = f"{self.name} ({variant})"

        if file_path:
            self.file_path = data_root_dir / file_path
            if not self.file_path.exists():
                raise ValueError(
                    f"Unable to locate {self.name} at '{self.file_path.as_posix()}'. "
                    f"See README.md for setup instructions."
                )

        self.data = self._load_data()

    @abstractmethod
    def _load_data(self) -> MutableSequence[TBenchmarkSample]:
        """Reads the data from the disk and turns them into ready-to-use instances."""
        pass

    @property
    def labels(self) -> list[BaseLabel]:
        """Returns the ground truth labels of this dataset as a list."""
        return [instance.label for instance in self]

    def get_classes(self) -> list[BaseLabel]:
        """Returns a list of distinct labels representing the classes occurring in this dataset."""
        return list(self.class_definitions.keys())

    def shuffle(self) -> None:
        """Reorders the samples randomly."""
        random.seed(random_seed)
        random.shuffle(self.data)

    def get_by_id(self, claim_id: str) -> TBenchmarkSample:
        """Returns the instance with the given ID (different from the instance's index)."""
        for instance in self:
            if instance.id == claim_id:
                return instance
        raise ValueError(f"Benchmark does not contain any instance with ID {claim_id}.")

    def get_class_name(self, label: BaseLabel) -> str:
        """Returns the original class name for the given standard Label."""
        for name, cls in self.class_mapping.items():
            if cls == label:
                return name
        raise ValueError(f"Unknown label {label}.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> TBenchmarkSample:
        return self.data[idx]

    def __iter__(self) -> Iterator[TBenchmarkSample]:
        return iter(self.data)

    def process_output(self, output: tuple[Any, Mapping[str, Any]]) -> None:
        """Handles the model's output and evaluates whether it is correct."""
        doc, meta = output
        claim = doc.claim
        instance = self.get_by_id(claim.id)
        prediction = doc.verdict
        self._save_prediction(doc, meta, claim, prediction, instance.label, instance.justification)

    def _save_prediction(
        self,
        doc: Any,
        meta: Mapping[str, Any],
        claim: Any,
        prediction: BaseLabel,
        target_label: BaseLabel | None = None,
        gt_justification: dict[str, Any] | None = None,
    ) -> None:
        claim_str = doc.claim.describe() if hasattr(doc.claim, "describe") else str(doc.claim)
        logger.save_next_prediction(
            sample_index=claim.id,
            claim=claim_str,
            target=target_label,
            justification=doc.justification,
            predicted=prediction,
            gt_justification=gt_justification,
        )
        logger.save_next_instance_stats(meta["Statistics"], claim.id)

        if target_label:
            prediction_is_correct = target_label == prediction
            if prediction_is_correct:
                logger.log(bold(green("✅ CORRECT\n")))
            else:
                logger.log(bold(red(f"❌ WRONG - Ground truth: {target_label.value}\n")))
