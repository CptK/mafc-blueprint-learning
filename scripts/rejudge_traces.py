#!/usr/bin/env python3
"""Re-run ONLY the judge over the recorded judge prompts of an existing run.

For each trace in <run_dir>/traces, rebuilds the judge messages from the
recorded prompt (media references re-resolved from <run_dir>/temp) and asks
the judge model again, in one or two arms:

  control  – the system prompt exactly as recorded in the trace
  tuned    – the current EXTRA_JUDGE_RULES_7 from mafc.eval.veritas.labels

The investigation evidence is untouched, so any metric delta is attributable
to the judge prompt (control arm quantifies resampling noise at temp>0).

Usage:
  python scripts/rejudge_traces.py --run-dir out/<run> --out out/<name>.jsonl \
      [--arms control,tuned] [--workers 16] [--first-n N]

Output: JSONL with one record per (claim_id, arm); resumes automatically.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ezmm
from ezmm.common.registry import ItemRegistry

import mafc  # noqa: F401 — loads config/.env

from mafc.agents.judge.agent import JudgeDecisionPayload
from mafc.common.modeling import Message, MessageRole, Prompt, make_model
from mafc.eval.veritas.labels import EXTRA_JUDGE_RULES_7
from mafc.utils.media import deduplicate_media
from mafc.utils.parsing import extract_json_object, strip_json_fences

# ezmm's ItemRegistry shares one SQLite cursor across threads; without a lock
# concurrent media resolution segfaults (same patch as scripts/run_benchmark.py).
_registry_lock = threading.RLock()
for _name in ("get", "get_by_path", "add_item", "get_cached", "update_file_path", "contains"):
    _orig = getattr(ItemRegistry, _name)

    def _make_locked(m):
        def _locked(self, *args, **kwargs):
            with _registry_lock:
                return m(self, *args, **kwargs)

        return _locked

    setattr(ItemRegistry, _name, _make_locked(_orig))

ALLOWED_LABELS = {
    "intact (certain)",
    "intact (rather certain)",
    "intact (rather uncertain)",
    "unknown",
    "compromised (rather uncertain)",
    "compromised (rather certain)",
    "compromised (certain)",
}

TUNED_SYSTEM_TEXT = (
    "You are a benchmark judging agent.\n"
    "Predict exactly one allowed benchmark label.\n"
    "Use only the accepted evidence provided below.\n"
    "If evidence is limited or mixed, prefer the appropriate uncertainty label.\n"
    f"\nAdditional benchmark rules:\n{EXTRA_JUDGE_RULES_7.strip()}\n"
)

_write_lock = threading.Lock()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arms", default="control,tuned")
    p.add_argument("--model", default="gemini_3_flash")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-response-length", type=int, default=64000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--first-n", type=int, default=None)
    p.add_argument("--only-ids", default=None, help="Comma-separated claim_ids or a JSON file with a list.")
    return p.parse_args()


def load_trace_prompt(trace_path: Path) -> tuple[str, str] | None:
    """Return (system_text, user_text) from a strategy trace's judge_run, or None."""
    with open(trace_path) as f:
        trace = json.load(f)
    jr = trace.get("judge_run")
    if not jr or not jr.get("prompt_messages"):
        return None
    sys_text = user_text = None
    for m in jr["prompt_messages"]:
        content = m.get("content")
        text = content.get("text") if isinstance(content, dict) else str(content)
        if m.get("role") == "system":
            sys_text = text
        elif m.get("role") == "user":
            user_text = text
    if sys_text is None or user_text is None:
        return None
    return sys_text, user_text


def parse_judge_response(text: str) -> JudgeDecisionPayload | None:
    try:
        return JudgeDecisionPayload.model_validate(
            json.loads(extract_json_object(strip_json_fences(text.strip())))
        )
    except Exception:
        return None


def judge_once(model, system_text: str, user_text: str) -> dict:
    messages = [
        Message(role=MessageRole.SYSTEM, content=Prompt(text=system_text)),
        Message(role=MessageRole.USER, content=deduplicate_media(Prompt(text=user_text))),
    ]
    resp = model.generate(messages)
    raw = resp.text.strip()
    parsed = parse_judge_response(raw)
    if parsed is None:
        # one repair attempt, mirroring JudgeAgent
        repair = (
            "Convert the following judge response to strict JSON with schema:\n"
            '{"label": "one allowed label", "justification": "short grounded justification"}\n'
            f"Allowed labels: {', '.join(sorted(ALLOWED_LABELS))}\n"
            "Only return JSON.\n\nResponse:\n" + raw
        )
        raw2 = model.generate([Message(role=MessageRole.USER, content=Prompt(text=repair))]).text.strip()
        parsed = parse_judge_response(raw2)
    if parsed is None:
        return {"predicted": None, "justification": None, "raw": raw[:2000], "error": "parse_failed"}
    label = parsed.label.strip().lower()
    if label not in ALLOWED_LABELS:
        return {
            "predicted": None,
            "justification": parsed.justification,
            "raw": raw[:2000],
            "error": f"unknown_label:{parsed.label}",
        }
    return {"predicted": label, "justification": parsed.justification, "raw": None, "error": None}


def process(job, model, out_path: Path) -> str:
    claim_id, arm, system_text, user_text = job
    rec = {"claim_id": claim_id, "arm": arm}
    for attempt in range(3):
        try:
            rec.update(judge_once(model, system_text, user_text))
            break
        except Exception as e:  # API/transient errors
            if attempt == 2:
                rec.update({"predicted": None, "justification": None, "raw": None, "error": f"exception:{e}"})
            else:
                time.sleep(5 * (attempt + 1))
    with _write_lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return f"{claim_id}/{arm}: {rec.get('predicted') or rec.get('error')}"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    ezmm.set_ezmm_path(run_dir / "temp")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, str]] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("predicted") is not None:
                    done.add((r["claim_id"], r["arm"]))

    only_ids: set[str] | None = None
    if args.only_ids:
        if Path(args.only_ids).exists():
            only_ids = set(json.load(open(args.only_ids)))
        else:
            only_ids = {x.strip() for x in args.only_ids.split(",")}

    # Both agents record a judge_run with the same prompt_messages shape, so either
    # trace kind can be replayed: StrategyAgent writes .strategy_trace.json,
    # FactCheckAgent (blueprint runs) writes .fact_check_trace.json.
    trace_paths = sorted(
        [
            *run_dir.glob("traces/benchmark_*.strategy_trace.json"),
            *run_dir.glob("traces/benchmark_*.fact_check_trace.json"),
        ]
    )
    if args.first_n:
        trace_paths = trace_paths[: args.first_n]

    jobs = []
    skipped_no_prompt = 0
    for tp in trace_paths:
        claim_id = tp.name.split("_")[1].split(".")[0]
        if only_ids is not None and claim_id not in only_ids:
            continue
        loaded = load_trace_prompt(tp)
        if loaded is None:
            skipped_no_prompt += 1
            continue
        system_text, user_text = loaded
        for arm in arms:
            if (claim_id, arm) in done:
                continue
            # arms named control* replay the recorded system prompt; all others use current labels.py rules
            jobs.append(
                (claim_id, arm, system_text if arm.startswith("control") else TUNED_SYSTEM_TEXT, user_text)
            )

    print(
        f"traces={len(trace_paths)} no_judge_prompt={skipped_no_prompt} already_done={len(done)} jobs={len(jobs)}"
    )
    if not jobs:
        return

    model = make_model(args.model, temperature=args.temperature, max_response_length=args.max_response_length)

    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process, job, model, out_path) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            msg = fut.result()
            n_ok += 1
            if i % 50 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {msg}", flush=True)

    print(f"finished {n_ok} judge calls -> {out_path}")


if __name__ == "__main__":
    main()
