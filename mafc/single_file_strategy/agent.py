"""Standalone Strategy.md fact-checker.

A self-contained fact-check agent whose entire configuration is:
  * a ``Strategy.md`` playbook (free text),
  * the claim, and
  * two investigation tools, each a sub-agent: ``web_search`` and ``media``.

It has no connection to the blueprint machinery — no blueprint selection, graph,
routing, or blueprint session state. Each planning round the model reads the
strategy and the evidence so far and either calls one or both tools or declares
it is done; then a judge maps the gathered evidence to a verdict label (the judge
is the shared scoring step used by every benchmark agent, not an investigation
tool). One round by default ("one-shot": plan, call tools in parallel, judge);
raise ``max_rounds`` to let the model react to evidence before finalizing.

Usage and verdict
-----------------
The agent emits a trace (:class:`mafc.strategy.tracing.StrategyRunTrace`) whose
keys match what the benchmark's result extractors read, so it drops into the
existing eval unchanged.
"""

from __future__ import annotations

import json
import traceback as _traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from ezmm import MultimodalSequence
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mafc.agents.agent import Agent, AgentResult, format_evidence_block
from mafc.agents.common import AgentSession
from mafc.common.claim import Claim
from mafc.common.evidence import Evidence
from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt
from mafc.single_file_strategy.tracing import StrategyRunTrace
from mafc.utils.parsing import extract_json_object, strip_json_fences

_TOOLS = ("web_search", "media")

_DECISION_SCHEMA = (
    '{"reasoning":"string",'
    '"tool_calls":[{"tool":"web_search|media","instruction":"string"}],'
    '"done":true|false}'
)


class _ToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tool: Literal["web_search", "media"]
    instruction: str


class _Decision(BaseModel):
    model_config = ConfigDict(extra="ignore")
    reasoning: str = ""
    tool_calls: list[_ToolCall] = Field(default_factory=list)
    done: bool = False


def _parse_decision(text: str) -> _Decision:
    """Parse the planner's JSON decision. Raises ValueError on any failure."""
    cleaned = strip_json_fences(text).strip()
    try:
        raw = json.loads(extract_json_object(cleaned))
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Response is not valid JSON: {e}")
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level JSON must be an object, got {type(raw).__name__}.")
    try:
        return _Decision.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Decision does not match schema {_DECISION_SCHEMA}: {e}")


class StrategyAgent(Agent):
    """Fact-checks a claim using only a strategy playbook and two tool sub-agents.

    Args:
        model: Planner model.
        strategy_md: The playbook handed to the planner as guidance.
        web_search_agent: Sub-agent invoked for ``web_search`` tool calls.
        media_agent: Sub-agent invoked for ``media`` tool calls.
        judge_agent: Maps gathered evidence to the verdict label (shared scoring step).
        max_rounds: Planning rounds. 1 = one-shot.
        n_workers: Parallel tool calls within one round.
        trace_dir: Where per-claim traces are written (None disables file output).
    """

    name = "StrategyAgent"
    description = "Fact-checks a claim using a free-text strategy playbook plus web_search and media tools."
    allowed_tools: list = []

    def __init__(
        self,
        model: Model,
        strategy_md: str,
        web_search_agent: Agent,
        media_agent: Agent,
        judge_agent: Agent | None = None,
        max_rounds: int = 1,
        n_workers: int = 1,
        agent_id: str | None = None,
        trace_dir: str | Path | None = None,
    ) -> None:
        super().__init__(model, n_workers=n_workers, agent_id=agent_id)
        self.strategy_md = strategy_md
        self.tools: dict[str, Agent] = {"web_search": web_search_agent, "media": media_agent}
        self.judge_agent = judge_agent
        self.max_rounds = max(1, max_rounds)
        self.trace_dir = trace_dir

    # ------------------------------------------------------------------

    def run(self, session: AgentSession, trace_scope=None, true_label: str | None = None) -> AgentResult:
        """Investigate the claim under the strategy, then judge the evidence."""
        self._mark_running(session)
        claim = self._resolve_claim(session)
        trace = StrategyRunTrace(
            self.trace_dir,
            session_id=session.id,
            claim_text=str(claim) if claim is not None else str(session.goal),
            strategy_word_count=len(self.strategy_md.split()),
            true_label=true_label,
        )

        if claim is None:
            self._mark_failed(session)
            msg = "Strategy session requires a claim or non-empty goal."
            trace.record_error(msg)
            trace.finalize(status=session.status.value, result_text=None, evidence_count=0)
            return AgentResult(
                session=session, result=None, errors=[msg], status=session.status, trace=trace.trace
            )

        session.claim = claim
        evidences: list[Evidence] = list(session.evidences)
        action_history: list[str] = []
        errors: list[str] = []

        try:
            for round_idx in range(1, self.max_rounds + 1):
                last_round = round_idx >= self.max_rounds
                done = self._run_round(
                    session, claim, evidences, action_history, errors, trace, round_idx, last_round
                )
                if done:
                    break

            final_answer = self._synthesize(claim, evidences, trace)
            session.evidences = list(evidences)
            self._judge(session, claim, evidences, errors, trace)

            result_text = MultimodalSequence(final_answer) if final_answer else None
            self._mark_completed(session)
            trace.finalize(
                status=session.status.value,
                result_text=str(result_text) if result_text else None,
                evidence_count=len(evidences),
            )
            return AgentResult(
                session=session,
                result=result_text,
                evidences=list(evidences),
                errors=errors,
                status=session.status,
                trace=trace.trace,
            )
        except Exception as exc:
            self._mark_failed(session)
            msg = f"{type(exc).__name__}: {exc}"
            errors.append(msg)
            trace.record_error(msg)
            logger.error(f"[StrategyAgent] Exception in run():\n{_traceback.format_exc()}")
            trace.finalize(status=session.status.value, result_text=None, evidence_count=len(evidences))
            raise

    # ------------------------------------------------------------------

    def _run_round(
        self,
        session: AgentSession,
        claim: Claim,
        evidences: list[Evidence],
        action_history: list[str],
        errors: list[str],
        trace: StrategyRunTrace,
        round_idx: int,
        last_round: bool,
    ) -> bool:
        """Plan one round; run any tool calls. Returns True when investigation should stop."""
        if self._should_stop:
            errors.append("Agent execution stopped early by stop signal.")
            return True

        system_prompt = self._system_prompt()
        user_prompt = self._round_prompt(claim, evidences, action_history, round_idx, last_round)
        messages = [
            Message(role=MessageRole.SYSTEM, content=Prompt(text=system_prompt)),
            Message(role=MessageRole.USER, content=Prompt(text=user_prompt)),
        ]

        resp = self.model.generate(messages)
        response_text = resp.text.strip()
        trace.add_usage(resp, self.model.name)

        decision = self._parse_with_repair(response_text, trace)
        if decision is None:
            msg = f"Strategy planner returned unparseable output in round {round_idx}."
            errors.append(msg)
            trace.record_round(
                round_index=round_idx,
                prompt=user_prompt,
                response_text=response_text,
                reasoning="",
                tool_calls=[],
                done=True,
                evidence_count_after=len(evidences),
            )
            return True

        # Drop unknown tools defensively (schema already restricts, but be safe).
        calls = [tc for tc in decision.tool_calls if tc.tool in self.tools]
        done = decision.done or not calls or last_round and not calls

        trace.record_round(
            round_index=round_idx,
            prompt=user_prompt,
            response_text=response_text,
            reasoning=decision.reasoning,
            tool_calls=[tc.model_dump() for tc in calls],
            done=decision.done,
            evidence_count_after=len(evidences),
        )

        if calls:
            self._dispatch(session, claim, evidences, action_history, errors, trace, calls, round_idx)

        # Stop if the model said done, or it asked for nothing, or this was the last round.
        return decision.done or not calls or last_round

    def _dispatch(
        self,
        session: AgentSession,
        claim: Claim,
        evidences: list[Evidence],
        action_history: list[str],
        errors: list[str],
        trace: StrategyRunTrace,
        calls: list[_ToolCall],
        round_idx: int,
    ) -> None:
        """Run the round's tool calls (in parallel) and fold their evidence in.

        A failing tool call (e.g. an invalid media reference hallucinated by the
        planner, or a sub-agent crash) must not kill the whole claim: it is turned
        into an error + history entry the planner can react to in later rounds.
        """

        def _invoke(tc: _ToolCall, index: int) -> AgentResult | Exception:
            try:
                child = self._child_session(session, claim, evidences, tc, index, round_idx)
                return self.tools[tc.tool].run(child)
            except Exception as exc:  # noqa: BLE001 — isolate per-call failures
                logger.warning(f"[StrategyAgent] Tool call {tc.tool} failed: {type(exc).__name__}: {exc}")
                return exc

        if len(calls) == 1 or self.n_workers <= 1:
            results = [(tc, _invoke(tc, i)) for i, tc in enumerate(calls)]
        else:
            with ThreadPoolExecutor(max_workers=min(len(calls), self.n_workers)) as pool:
                futures = [(tc, pool.submit(_invoke, tc, i)) for i, tc in enumerate(calls)]
                results = [(tc, fut.result()) for tc, fut in futures]

        for tc, result in results:
            if isinstance(result, Exception):
                msg = f"Tool call '{tc.tool}' failed: {type(result).__name__}: {result}"
                if "does not exist" in str(result):
                    msg += " (use the exact media reference tag shown under claim modalities)"
                errors.append(msg)
                action_history.append(f"{tc.tool}: {tc.instruction[:80]} -> FAILED: {msg}")
                continue
            evidences.extend(result.evidences)
            errors.extend(result.errors)
            trace.absorb_child(result.trace)
            action_history.append(
                f"{tc.tool}: {tc.instruction[:80]} -> {len(result.evidences)} evidence, {len(result.errors)} errors"
            )

    def _child_session(
        self,
        parent: AgentSession,
        claim: Claim,
        evidences: list[Evidence],
        tc: _ToolCall,
        index: int,
        round_idx: int,
    ) -> AgentSession:
        sid = f"{parent.id}:{tc.tool}:{round_idx}:{index}"
        goal = MultimodalSequence(tc.instruction) if tc.tool == "media" else Prompt(text=tc.instruction)
        return AgentSession(
            id=sid,
            goal=goal,
            claim=claim,
            cutoff_date=parent.cutoff_date,
            parent_session_id=parent.id,
            evidences=list(evidences),
        )

    def _judge(
        self,
        session: AgentSession,
        claim: Claim,
        evidences: list[Evidence],
        errors: list[str],
        trace: StrategyRunTrace,
    ) -> None:
        if self.judge_agent is None or not evidences:
            return
        seen: set[str] = set()
        deduped: list[Evidence] = []
        for ev in evidences:
            if ev.source not in seen:
                seen.add(ev.source)
                deduped.append(ev)
        judge_session = AgentSession(
            id=f"{session.id}:judge",
            goal=Prompt(text="Judge the claim using accepted evidence."),
            claim=claim,
            evidences=deduped,
            parent_session_id=session.id,
        )
        judge_result = self.judge_agent.run(judge_session)
        errors.extend(judge_result.errors)
        trace.record_judge(judge_result.trace)

    def _synthesize(self, claim: Claim, evidences: list[Evidence], trace: StrategyRunTrace) -> str:
        """Write a short human-readable verdict synthesis from the evidence."""
        if not evidences:
            return ""
        return self.synthesize_from_evidences(
            f"Summarise the fact-check of this claim and state the verdict:\n{claim.describe()}",
            evidences,
            trace=trace,
        )

    def synthesize_from_evidences(
        self, instruction: str, evidences: list[Evidence], trace: StrategyRunTrace | None = None
    ) -> str:
        evidence_lines = [block for ev in evidences if (block := format_evidence_block(ev)) is not None]
        if not evidence_lines:
            return ""
        prompt = (
            "Use only the evidence below to answer the task. Be concise and explicit about "
            f"uncertainty.\n\nTask:\n{instruction}\n\nEvidence:\n{chr(10).join(evidence_lines)}"
        )
        resp = self.model.generate([Message(role=MessageRole.USER, content=Prompt(text=prompt))])
        if trace is not None:
            trace.add_usage(resp, self.model.name)
        return resp.text.strip()

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        return (
            "You are a fact-checking agent. You verify a claim by following the strategy playbook "
            "below and investigating with two tools. You are an internal controller, not a "
            "user-facing assistant.\n\n"
            "Tools (each runs as a sub-agent that returns evidence):\n"
            "- web_search: searches the web and reads sources. Use for text claims, quotes, "
            "statistics, events, documents, and any verifiable fact.\n"
            "- media: inspects an attached image or video (reverse image search, geolocation, "
            "manipulation/AI checks). Handles ONE media item per call — to target a specific item, "
            "start the instruction with its reference tag (the <type:N> token shown in the claim "
            "modalities).\n\n"
            "Each round, decide what to investigate next given the evidence so far, or set done=true "
            "to stop and reach a verdict. Tool calls in one response run in parallel — request "
            "independent angles together. Request nothing (or done=true) once the evidence is "
            "sufficient or no further investigation would help.\n\n"
            "Respond with strict JSON only:\n"
            f"{_DECISION_SCHEMA}\n\n"
            "--- BEGIN STRATEGY.md ---\n"
            f"{self.strategy_md}\n"
            "--- END STRATEGY.md ---"
        )

    def _round_prompt(
        self,
        claim: Claim,
        evidences: list[Evidence],
        action_history: list[str],
        round_idx: int,
        last_round: bool,
    ) -> str:
        n_images = len(claim.images)
        n_videos = len(claim.videos)
        image_tags = ", ".join(img.reference[1:-1] for img in claim.images) if n_images else "none"
        video_tags = ", ".join(vid.reference[1:-1] for vid in claim.videos) if n_videos else "none"
        finalize_directive = (
            "This is the FINAL round — gather any last evidence now, or set done=true.\n\n"
            if last_round
            else ""
        )
        history = "\n".join(f"- {h}" for h in action_history) if action_history else "None"
        return (
            f"Claim:\n{claim.describe()}\n\n"
            f"Round: {round_idx} / {self.max_rounds}\n\n"
            f"Claim modalities:\n"
            f"- images: {n_images} ({image_tags})\n"
            f"- videos: {n_videos} ({video_tags})\n"
            f"- media tool usable: {n_images > 0 or n_videos > 0}\n\n"
            f"Evidence gathered so far:\n{self._render_evidence(evidences)}\n\n"
            f"Investigation history:\n{history}\n\n"
            f"{finalize_directive}"
            "Following the strategy, decide your tool calls (or done=true). Return strict JSON only:\n"
            f"{_DECISION_SCHEMA}"
        )

    @staticmethod
    def _render_evidence(evidences: list[Evidence]) -> str:
        lines: list[str] = []
        for ev in evidences:
            takeaways = getattr(ev, "takeaways", None)
            summary = str(takeaways).strip() if takeaways else ""
            if summary:
                lines.append(f"- Source: {ev.source}\n  {summary}")
        return "\n".join(lines) if lines else "None"

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_claim(session: AgentSession) -> Claim | None:
        if session.claim is not None:
            return session.claim
        goal_text = str(session.goal).strip()
        if not goal_text and not session.goal.images and not session.goal.videos:
            return None
        return Claim(*session.goal.data)

    def _parse_with_repair(self, response_text: str, trace: StrategyRunTrace) -> _Decision | None:
        try:
            return _parse_decision(response_text)
        except ValueError as e:
            logger.debug(f"[StrategyAgent] Decision parse failed: {e}; issuing repair call.")
        repair = (
            "Convert your previous response to strict JSON matching this schema, nothing else:\n"
            f"{_DECISION_SCHEMA}\n\nPrevious response:\n{response_text}"
        )
        resp = self.model.generate([Message(role=MessageRole.USER, content=Prompt(text=repair))])
        trace.add_usage(resp, self.model.name)
        try:
            return _parse_decision(resp.text.strip())
        except ValueError as e:
            logger.warning(f"[StrategyAgent] Decision unparseable after repair: {e}")
            return None
