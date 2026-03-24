#!/usr/bin/env python3
"""Run a full benchmark evaluation from a YAML config file."""

from __future__ import annotations

import argparse
import faulthandler
import resource
import shutil
from datetime import datetime
from pathlib import Path

from mafc.common.logger import logger
from mafc.eval.run_config import BenchmarkRunConfig
from mafc.eval.runner import run_benchmark

faulthandler.enable()

# Raise the open-file-descriptor limit to avoid EMFILE errors when many
# parallel model/retrieval connections are open simultaneously.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _hard), _hard))

DEFAULT_OUT_DIR = "out"


def _make_run_dir(config: BenchmarkRunConfig) -> Path:
    bm = config.benchmark
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{bm.name}-{bm.split}-{bm.label_scheme}class-{timestamp}"
    return Path(DEFAULT_OUT_DIR) / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full benchmark evaluation from a YAML config.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", metavar="PATH", help="Path to benchmark run config YAML.")
    group.add_argument("--resume", metavar="RUN_DIR", help="Path to a previous run directory to resume.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.resume:
        run_dir = Path(args.resume)
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.yaml found in resume dir: {run_dir}")
        config = BenchmarkRunConfig.from_yaml(config_path)
        skip_ids = logger.load_completed_ids() if (run_dir / logger.results_filename).exists() else set()
        logger.info(f"[Runner] Resuming run at {run_dir} ({len(skip_ids)} samples already done)")
    else:
        config = BenchmarkRunConfig.from_yaml(args.config)
        run_dir = _make_run_dir(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, run_dir / "config.yaml")
        skip_ids = set()
        logger.info(f"[Runner] Starting new run at {run_dir}")

    logger.set_experiment_dir(run_dir)
    logger.set_log_level(config.run.log_level.lower())  # type: ignore[arg-type]

    run_benchmark(config, run_dir, skip_ids=skip_ids or None)


if __name__ == "__main__":
    main()
