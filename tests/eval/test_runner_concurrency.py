"""Tests for the thread-local agent isolation in run_benchmark and
the shared-event-loop guarantee in ScrapeMMRetriever.

These tests verify that concurrent benchmark runs do not share a single
FactCheckAgent (and therefore a single ScrapeMMRetriever / asyncio event loop)
across worker threads, which was the root cause of segfaults on macOS.

Three levels of testing:
  1. Pure mechanism — threading.local() behaves correctly independent of runner.
  2. Sequential path — concurrency=1 builds exactly one agent, reused for all samples.
  3. Concurrent path — concurrency=N builds one agent per thread, never shared.
"""

from __future__ import annotations

import threading
import time
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch


from mafc.tools.web_search.integrations.scrapemm_retriever import ScrapeMMRetriever

from mafc.eval.run_config import (
    AgentModelConfig,
    AgentsConfig,
    BenchmarkConfig,
    BlueprintsConfig,
    BenchmarkRunConfig,
    FactCheckAgentConfig,
    RunConfig,
    WebSearchAgentConfig,
)
from mafc.eval.runner import run_benchmark

# ── shared fixtures / helpers ─────────────────────────────────────────────────

_MODEL_SPEC = "OPENAI:gpt-5-mini-2025-08-07"


def _make_config(concurrency: int) -> BenchmarkRunConfig:
    return BenchmarkRunConfig(
        benchmark=BenchmarkConfig(name="veritas", split="test", data_path="unused"),
        agents=AgentsConfig(
            fact_check=FactCheckAgentConfig(model=_MODEL_SPEC),
            web_search=WebSearchAgentConfig(model=_MODEL_SPEC),
            media=AgentModelConfig(model=_MODEL_SPEC),
            judge=AgentModelConfig(model=_MODEL_SPEC),
        ),
        blueprints=BlueprintsConfig(selector_model=_MODEL_SPEC),
        run=RunConfig(concurrency=concurrency, traces=False),
    )


def _make_sample(sample_id: str) -> MagicMock:
    s = MagicMock()
    s.id = sample_id
    s.label.value = "true"
    return s


def _dummy_result(sample_id: str) -> dict:
    return {
        "claim_id": sample_id,
        "ground_truth": "true",
        "predicted": "true",
        "correct": True,
        "errors": [],
        "duration_ms": 1,
        "cost": {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


# Minimal summary that satisfies the final log line in run_benchmark
# (needs 'correct', 'completed', 'accuracy', 'cost').
_STUB_SUMMARY = {
    "total": 0,
    "completed": 0,
    "correct": 0,
    "errored": 0,
    "accuracy": None,
    "run_duration_s": 0.0,
    "cost": {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
}


def _make_benchmark_mock(samples: list) -> MagicMock:
    mock = MagicMock()
    mock.__iter__ = MagicMock(side_effect=lambda: iter(samples))
    mock.compute_metrics.return_value = {}
    mock.format_metrics_report.return_value = ""
    mock.save_metric_plots.return_value = []
    return mock


def _runner_patches(benchmark_mock, build_side_effect, run_side_effect) -> list:
    """Return patch context managers that isolate run_benchmark from all I/O."""
    mock_logger = MagicMock()
    mock_logger.results_path.exists.return_value = False
    return [
        patch("mafc.eval.runner.VeriTaS", return_value=benchmark_mock),
        patch("mafc.eval.runner._build_fact_check_agent", side_effect=build_side_effect),
        patch("mafc.eval.runner._run_sample", side_effect=run_side_effect),
        # Bypass the post-run file read + summary computation entirely so we
        # don't need a real results file on disk.
        patch("mafc.eval.runner._compute_summary", return_value=_STUB_SUMMARY),
        patch("mafc.eval.runner.logger", mock_logger),
        patch("mafc.eval.runner.tqdm", return_value=MagicMock()),
    ]


# ── 1. Pure mechanism test ────────────────────────────────────────────────────


def test_thread_local_gives_one_object_per_thread():
    """threading.local() must give each thread its own object.

    With N worker threads processing M > N tasks, there should be exactly N
    distinct objects created — one per thread, reused across that thread's tasks.
    """
    thread_local = threading.local()
    seen: dict[int, object] = {}
    lock = threading.Lock()

    def worker():
        if not hasattr(thread_local, "obj"):
            thread_local.obj = object()
        tid = threading.current_thread().ident
        assert tid is not None
        with lock:
            if tid in seen:
                # Same thread must always get the same object back.
                assert seen[tid] is thread_local.obj
            else:
                seen[tid] = thread_local.obj

    n_threads = 4
    n_tasks = 12  # more tasks than threads → threads are reused

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for f in [ex.submit(worker) for _ in range(n_tasks)]:
            f.result()

    # ThreadPoolExecutor is lazy — it may use fewer than n_threads for fast tasks.
    # The invariant is that every thread that DID run got its own distinct object.
    assert len(seen) >= 1
    assert len(seen) <= n_threads
    assert len({id(v) for v in seen.values()}) == len(
        seen
    ), "Some threads shared an object — thread isolation broken"


# ── 2. Sequential path ────────────────────────────────────────────────────────


def test_sequential_builds_exactly_one_agent(tmp_path):
    """concurrency=1: _build_fact_check_agent called once; every sample uses it."""
    samples = [_make_sample(f"s{i}") for i in range(3)]
    config = _make_config(concurrency=1)

    agents_built: list[MagicMock] = []

    def fake_build(*_args, **_kwargs):
        agent = MagicMock()
        agents_built.append(agent)
        return agent

    agents_used: list[MagicMock] = []

    def fake_run(_config, _bm, sample, _trace_dir, agent=None):
        agents_used.append(agent)
        return _dummy_result(sample.id)

    with ExitStack() as stack:
        for p in _runner_patches(_make_benchmark_mock(samples), fake_build, fake_run):
            stack.enter_context(p)
        run_benchmark(config, tmp_path)

    assert len(agents_built) == 1, f"Expected 1 agent for sequential run, got {len(agents_built)}"
    assert all(a is agents_built[0] for a in agents_used), "Not all samples used the single shared agent"


# ── 3. Concurrent path ───────────────────────────────────────────────────────


def test_concurrent_builds_one_agent_per_thread(tmp_path):
    """concurrency=N: exactly N agents created, one per worker thread."""
    n_workers = 4
    samples = [_make_sample(f"s{i}") for i in range(n_workers * 3)]
    config = _make_config(concurrency=n_workers)

    lock = threading.Lock()
    thread_to_agent: dict[int, MagicMock] = {}
    agents_built: list[MagicMock] = []

    def fake_build(*_args, **_kwargs):
        agent = MagicMock()
        with lock:
            agents_built.append(agent)
            ident = threading.current_thread().ident
            assert ident is not None
            thread_to_agent[ident] = agent
        return agent

    def fake_run(_config, _bm, sample, _trace_dir, agent=None):
        tid = threading.current_thread().ident
        assert tid is not None
        # Sleep long enough that the executor must spin up all n_workers threads
        # to process n_workers*3 tasks concurrently, rather than one thread
        # racing through all tasks before others are created.
        time.sleep(0.02)
        with lock:
            # The agent used must be the one that was built in this thread.
            assert (
                thread_to_agent.get(tid) is agent
            ), f"Thread {tid} used agent {id(agent)!r} but built {id(thread_to_agent.get(tid))!r}"
        return _dummy_result(sample.id)

    with ExitStack() as stack:
        for p in _runner_patches(_make_benchmark_mock(samples), fake_build, fake_run):
            stack.enter_context(p)
        run_benchmark(config, tmp_path)

    assert (
        len(agents_built) == n_workers
    ), f"Expected {n_workers} agents (one per thread), got {len(agents_built)}"
    assert len({id(a) for a in agents_built}) == n_workers, "Some threads share the same agent instance"


def test_concurrent_no_agent_crosses_thread_boundary(tmp_path):
    """Each thread must use only one agent, and no agent may appear in two threads."""
    n_workers = 4
    samples = [_make_sample(f"s{i}") for i in range(n_workers * 3)]
    config = _make_config(concurrency=n_workers)

    lock = threading.Lock()
    # thread_id → set of agent ids used by that thread
    usage: dict[int, set[int]] = {}

    def fake_build(*_args, **_kwargs):
        return MagicMock()

    def fake_run(_config, _bm, sample, _trace_dir, agent=None):
        tid = threading.current_thread().ident
        assert tid is not None
        with lock:
            usage.setdefault(tid, set()).add(id(agent))
        return _dummy_result(sample.id)

    with ExitStack() as stack:
        for p in _runner_patches(_make_benchmark_mock(samples), fake_build, fake_run):
            stack.enter_context(p)
        run_benchmark(config, tmp_path)

    # Each thread used exactly one agent id throughout its lifetime.
    for tid, agent_ids in usage.items():
        assert len(agent_ids) == 1, f"Thread {tid} switched agents mid-run: {agent_ids}"

    # No agent id appears in more than one thread.
    all_agent_ids = [next(iter(ids)) for ids in usage.values()]
    assert len(set(all_agent_ids)) == len(all_agent_ids), "An agent was shared across threads"


# ── 4. ScrapeMMRetriever shared event loop ────────────────────────────────────


def test_scrapemm_retriever_all_instances_share_one_loop():
    """All ScrapeMMRetriever instances — including those created in different
    threads — must share the same asyncio event loop and loop thread.

    Multiple independent event loops doing concurrent SSL from separate OS
    threads causes segfaults on macOS / Python 3.13.
    """
    # Reset class-level state so this test is self-contained.
    ScrapeMMRetriever._shared_loop = None
    ScrapeMMRetriever._shared_loop_thread = None

    instances: list[ScrapeMMRetriever] = []
    lock = threading.Lock()

    def make_instance():
        r = ScrapeMMRetriever(n_workers=2)
        with lock:
            instances.append(r)

    with ThreadPoolExecutor(max_workers=4) as ex:
        for f in [ex.submit(make_instance) for _ in range(8)]:
            f.result()

    assert len(instances) == 8
    loops = {id(r._loop) for r in instances}
    threads = {id(r._loop_thread) for r in instances}
    assert len(loops) == 1, f"Expected 1 shared loop, got {len(loops)}"
    assert len(threads) == 1, f"Expected 1 shared loop thread, got {len(threads)}"
