import csv
import json
import logging
import os.path
import sys
from typing import Any, Literal
from pathlib import Path

import pandas as pd
from filelock import FileLock

from mafc.common.label import BaseLabel
from mafc.utils.console import remove_string_formatters, bold, red, yellow, gray, green
from mafc.utils.utils import flatten_dict

# Suppress unwanted logging from other libraries
logging.getLogger("bs4").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("git").setLevel(logging.WARNING)
logging.getLogger("wandb").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("duckduckgo_search").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("timm.models._builder").setLevel(logging.ERROR)
logging.getLogger("timm.models._hub").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3.connection").setLevel(logging.ERROR)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("ezMM").propagate = False


LOG_LEVELS = {
    "critical": 50,
    "error": 40,
    "warning": 30,
    "info": 20,
    "log": 15,
    "debug": 10,
}


class Logger:

    log_filename = "log.txt"
    predictions_filename = "predictions.csv"
    instance_stats_filename = "instance_stats.csv"
    results_filename = "results.jsonl"
    summary_filename = "summary.json"

    def __init__(self):
        self.experiment_dir: Path | None = None
        self.print_log_level = "debug"

        logging.basicConfig(level=logging.DEBUG)

        self.logger = logging.getLogger("mafc")
        self.logger.propagate = False
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(LOG_LEVELS[self.print_log_level])
        self.logger.addHandler(stdout_handler)
        self.logger.setLevel(logging.DEBUG)

    def set_experiment_dir(self, path: str | Path) -> None:
        """Set the run directory for all output files. Creates the directory if needed."""
        self.experiment_dir = Path(path)
        self._update_file_handler()

    def set_log_level(self, level: Literal["critical", "error", "warning", "info", "log", "debug"]) -> None:
        """Set the console log verbosity level."""
        self.print_log_level = level
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(LOG_LEVELS[level])

    def _update_file_handler(self) -> None:
        assert self.experiment_dir is not None
        self._remove_file_handlers()
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.logger.addHandler(_make_file_handler(self.log_path))

    def _remove_file_handlers(self) -> None:
        from logging.handlers import RotatingFileHandler

        for handler in list(self.logger.handlers):
            if isinstance(handler, RotatingFileHandler):
                self.logger.removeHandler(handler)
                handler.close()

    # --- Standard logging ---

    def critical(self, *args) -> None:
        self.logger.critical(bold(red(compose_message(*args))))

    def error(self, *args) -> None:
        self.logger.error(red(compose_message(*args)))

    def warning(self, *args) -> None:
        self.logger.warning(yellow(compose_message(*args)))

    def info(self, *args) -> None:
        self.logger.info(green(compose_message(*args)))

    def log(self, *args, level: int = 15) -> None:
        self.logger.log(level, compose_message(*args))

    def debug(self, *args) -> None:
        self.logger.debug(gray(compose_message(*args)))

    # --- Benchmark evaluation results ---

    def save_benchmark_result(self, result: dict[str, Any]) -> None:
        """Append one sample result to results.jsonl. Concurrent-safe via FileLock."""
        assert self.experiment_dir is not None, "Call set_experiment_dir first"
        serialized = json.dumps(result, ensure_ascii=False) + "\n"
        lock = FileLock(str(self.results_path) + ".lock")
        with lock:
            with open(self.results_path, "a", encoding="utf-8") as f:
                f.write(serialized)

    def load_completed_ids(self) -> set[str]:
        """Read back completed claim IDs from results.jsonl (for resume)."""
        if not self.results_path.exists():
            return set()
        completed: set[str] = set()
        with open(self.results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        completed.add(json.loads(line)["claim_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        return completed

    def write_benchmark_summary(self, summary: dict[str, Any]) -> None:
        """Write summary.json to the experiment directory."""
        assert self.experiment_dir is not None, "Call set_experiment_dir first"
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # --- Legacy benchmark.py evaluation helpers ---

    def save_next_prediction(
        self,
        sample_index: int | str,
        claim: str,
        target: BaseLabel | None,
        predicted: BaseLabel,
        justification: str | dict[str, Any] | list[Any] | None,
        gt_justification: str | dict[str, Any] | list[Any] | None,
    ) -> None:
        assert self.experiment_dir is not None, "Call set_experiment_dir first"

        if not os.path.exists(self.predictions_path):
            self._init_predictions_csv()

        is_correct = target == predicted if target is not None else None
        justification_payload = (
            json.dumps(justification, ensure_ascii=False)
            if isinstance(justification, (dict, list))
            else justification
        )
        gt_justification_payload = (
            json.dumps(gt_justification, ensure_ascii=False)
            if isinstance(gt_justification, (dict, list))
            else gt_justification
        )
        with open(self.predictions_path, "a") as f:
            csv.writer(f).writerow(
                (
                    sample_index,
                    claim,
                    target.name if target is not None else None,
                    predicted.name,
                    justification_payload,
                    is_correct,
                    gt_justification_payload,
                )
            )

    def save_next_instance_stats(self, stats: dict, claim_id: int | str) -> None:
        assert self.experiment_dir is not None, "Call set_experiment_dir first"

        instance_stats = flatten_dict(stats)
        instance_stats["ID"] = claim_id

        df = pd.DataFrame([instance_stats]).set_index("ID")
        file_exists = os.path.exists(self.instance_stats_path)
        df.to_csv(self.instance_stats_path, mode="a", header=not file_exists)

    def _init_predictions_csv(self) -> None:
        with open(self.predictions_path, "w") as f:
            csv.writer(f).writerow(
                (
                    "sample_index",
                    "claim",
                    "target",
                    "predicted",
                    "justification",
                    "correct",
                    "gt_justification",
                )
            )

    # --- Properties ---

    @property
    def log_path(self) -> Path:
        assert self.experiment_dir is not None
        return self.experiment_dir / self.log_filename

    @property
    def predictions_path(self) -> Path:
        assert self.experiment_dir is not None
        return self.experiment_dir / self.predictions_filename

    @property
    def instance_stats_path(self) -> Path:
        assert self.experiment_dir is not None
        return self.experiment_dir / self.instance_stats_filename

    @property
    def results_path(self) -> Path:
        assert self.experiment_dir is not None
        return self.experiment_dir / self.results_filename

    @property
    def summary_path(self) -> Path:
        assert self.experiment_dir is not None
        return self.experiment_dir / self.summary_filename


class RemoveStringFormattingFormatter(logging.Formatter):
    def format(self, record):
        return remove_string_formatters(record.getMessage())


def _make_file_handler(path: Path) -> logging.FileHandler:
    from logging.handlers import RotatingFileHandler

    file_handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(RemoveStringFormattingFormatter())
    return file_handler


def compose_message(*args) -> str:
    msg = " ".join([str(a) for a in args])
    return msg.encode("utf-8", "ignore").decode("utf-8")


logger = Logger()
