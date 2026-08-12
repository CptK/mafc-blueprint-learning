from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
from pathlib import Path
import time
import traceback as _traceback

from ezmm import MultimodalSequence

from mafc.agents.agent import Agent, AgentResult, format_evidence_block
from mafc.agents.common import AgentSession
from mafc.agents.fact_check.models import (
    CheckStatus,
    DelegatedTaskRecord,
    DelegationTask,
    FactCheckSessionState,
    PlannerCheckUpdate,
    PlannerDecisionType,
)
from mafc.agents.fact_check.parsing import (
    try_parse_check_update_decision,
    try_parse_planner_decision,
    try_parse_routing_decision,
)
from mafc.utils.parsing import try_parse_with_repair
from mafc.agents.fact_check.prompts import (
    build_action_node_prompt,
    build_check_update_prompt,
    build_final_synthesis_prompt,
    build_refine_node_prompt,
    build_routing_prompt,
    build_system_prompt,
    render_available_sub_agents,
)
from mafc.blueprints.models import (
    Blueprint,
    BlueprintActionNode,
    BlueprintNode,
    BlueprintSynthesisNode,
    BlueprintTransition,
)
from mafc.agents.fact_check.tracing import FactCheckTraceRecorder
from mafc.common.trace import TraceScope
from mafc.blueprints.topology import analyze_blueprint_topology
from mafc.blueprints.selector import BlueprintSelector
from mafc.common.claim import Claim
from mafc.common.evidence import Evidence
from mafc.common.logger import logger
from mafc.common.media_referent import annotate_evidence_referents
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt

REFINE_NODE_ID = "refine"
"""Id of the synthetic node injected when the graph would finalize with budget left.

Synthetic rather than declared in the blueprints: it applies uniformly to every
blueprint, and blueprints are loaded once and shared across worker threads, so
mutating their graphs at runtime would be a data race.
"""

_REFINE_NODE = BlueprintActionNode(
    id=REFINE_NODE_ID,
    type="actions",
    actions=[],
    activates_checks=[],
    transition=[BlueprintTransition(**{"if": "continue", "to": "finalize"})],
)
"""Immutable prototype. Its single exit means execution leaves after one pass
unless the routing layer redirects back into it while budget remains."""


class FactCheckAgent(Agent):
    """Top-level orchestration agent that uses a selected blueprint as guidance."""

    name = "FactCheckAgent"
    description = "Guides fact-checking with a selected blueprint and delegates to worker agents."
    allowed_tools = []

    def __init__(
        self,
        model: Model,
        blueprint_selector: BlueprintSelector,
        delegation_agents: dict[str, list[Agent]] | None = None,
        judge_agent: Agent | None = None,
        n_workers: int = 1,
        agent_id: str | None = None,
        trace_dir: str | Path | None = None,
        enable_refine_node: bool = False,
    ):
        """Initialize the top-level fact-check agent with its selector and delegation pools.

        Args:
            enable_refine_node: Route to a synthetic ``refine`` node instead of
                finalizing when iteration budget remains and checks on the taken
                path are still open. Off by default: it changes the stopping
                behaviour of every blueprint, so runs opt in rather than shift
                under existing comparisons.
        """
        super().__init__(model, n_workers=n_workers, agent_id=agent_id)
        self.blueprint_selector = blueprint_selector
        self.delegation_agents = delegation_agents or {}
        self.judge_agent = judge_agent
        self.trace_dir = trace_dir
        self.enable_refine_node = enable_refine_node
        self._agent_type_aliases = {
            "media": "media",
            "media_agent": "media",
            "web": "web_search",
            "web_search": "web_search",
            "web_search_agent": "web_search",
        }

    def run(
        self, session: AgentSession, trace_scope: TraceScope | None = None, true_label: str | None = None
    ) -> AgentResult:
        """Run the blueprint-guided orchestration loop for one session."""
        self._mark_running(session)
        root_scope = trace_scope or TraceScope.root(
            scope_type="fact_check_run",
            trace_id=session.id,
            trace_dir=self.trace_dir,
            key=session.id,
            metadata={"agent": self.name},
        )
        trace = FactCheckTraceRecorder(
            self.trace_dir, session, self.name, trace_scope=root_scope, true_label=true_label
        )
        claim = self._resolve_claim(session)
        trace.set_claim(claim)
        logger.debug(f"[FactCheckAgent] Starting verification for claim: '{claim}'")
        state: FactCheckSessionState | None = None
        errors: list[str] = []
        result: AgentResult | None = None
        if claim is None:
            self._mark_failed(session)
            result = AgentResult(
                session=session,
                result=None,
                errors=["Fact-check session requires a claim or non-empty goal."],
                status=session.status,
                trace=trace.trace,
            )
            errors.extend(result.errors)
            trace.record_error(phase="resolve_claim", message=result.errors[0])
            trace.finalize(session=session, state=state, result=result, errors=errors)
            return result

        session.claim = claim
        selection_result = self.blueprint_selector.select(claim)
        trace.set_blueprint(selection_result)
        logger.debug(f"[FactCheckAgent] Running with blueprint '{selection_result.selected_blueprint.name}'")
        state = self._initialize_state(selection_result.selected_blueprint, session.evidences)
        max_iterations = selection_result.selected_blueprint.policy_constraints.max_iterations
        try:
            for iteration in range(1, max_iterations + 1):
                should_stop = self._execute_iteration(
                    session=session,
                    claim=claim,
                    state=state,
                    iteration=iteration,
                    errors=errors,
                    trace=trace,
                )
                if should_stop:
                    break

            result = self._finalize_run(session=session, state=state, errors=errors, trace=trace)
            return result
        except Exception as exc:
            self._mark_failed(session)
            error_message = f"{type(exc).__name__}: {exc}"
            errors.append(error_message)
            trace.record_error(
                phase="run", message=error_message, iteration=state.iteration if state else None
            )
            logger.error(f"[FactCheckAgent] Exception in run():\n{_traceback.format_exc()}")
            raise
        finally:
            trace.finalize(session=session, state=state, result=result, errors=errors)

    def synthesize_from_evidences(self, instruction: str, evidences: list[Evidence]) -> str:
        """Synthesize an answer from evidence using the top-level model."""
        evidence_lines = [
            block for evidence in evidences if (block := format_evidence_block(evidence)) is not None
        ]
        if not evidence_lines:
            return ""

        prompt = (
            "Use only the evidence below to answer the task.\n"
            "Be concise and explicit about uncertainty.\n\n"
            f"Task:\n{instruction}\n\n"
            f"Evidence:\n{chr(10).join(evidence_lines)}"
        )
        return self.model.generate([Message(role=MessageRole.USER, content=Prompt(text=prompt))]).text.strip()

    def _resolve_claim(self, session: AgentSession) -> Claim | None:
        """Resolve the session claim or construct one from the current goal."""
        if session.claim is not None:
            return session.claim
        logger.warning(
            "[FactCheckAgent] No explicit claim provided in session; attempting to resolve from goal."
        )
        goal_text = str(session.goal).strip()
        if not goal_text and not session.goal.images and not session.goal.videos:
            logger.error("[FactCheckAgent] Unable to resolve claim: session goal is empty.")
            return None
        return Claim(*session.goal.data)

    def _initialize_state(self, blueprint: Blueprint, evidences: list[Evidence]) -> FactCheckSessionState:
        """Initialize top-level orchestration state for the selected blueprint."""
        topology = analyze_blueprint_topology(blueprint)
        # Checks referenced by a node activate when the path reaches that node;
        # only unreferenced (global) checks are active from the start.
        node_scoped = blueprint.node_scoped_check_ids()
        global_checks = [check for check in blueprint.required_checks if check.id not in node_scoped]
        return FactCheckSessionState(
            selected_blueprint=blueprint,
            current_node_id=blueprint.verification_graph.start_node,
            node_layers=topology.node_layers,
            max_layer=topology.max_layer,
            required_check_status={check.id: CheckStatus.UNCHECKED for check in global_checks},
            required_check_defs={check.id: check for check in global_checks},
            evidences=list(evidences),
        )

    def _system_message(self, state: FactCheckSessionState) -> Message:
        """Build the system message containing the full blueprint and current position."""
        available_sub_agents = render_available_sub_agents(self._available_sub_agent_descriptions())
        return Message(
            role=MessageRole.SYSTEM,
            content=Prompt(text=build_system_prompt(state, available_sub_agents)),
        )

    def _available_sub_agent_descriptions(self) -> dict[str, str]:
        """Return one planner-facing capability description per configured agent type."""
        descriptions: dict[str, str] = {}
        for agent_type, agents in self.delegation_agents.items():
            if not agents:
                continue
            descriptions[agent_type] = getattr(agents[0], "description", agents[0].__class__.__name__)
        return descriptions

    def _execute_iteration(
        self,
        session: AgentSession,
        claim: Claim,
        state: FactCheckSessionState,
        iteration: int,
        errors: list[str],
        trace: FactCheckTraceRecorder,
    ) -> bool:
        """Execute one iteration: node execution phase followed by routing phase."""
        if self._should_stop:
            message = "Agent execution stopped early by stop signal."
            errors.append(message)
            trace.record_error(phase="stop_signal", message=message, iteration=iteration)
            return True

        state.iteration = iteration
        errors_before = len(errors)
        node_before = state.current_node_id
        trace.start_iteration(iteration, node_before, len(state.evidences))

        current_node = self._get_current_node(state)

        # Node-attached checks become active when the path reaches their node.
        added_checks = state.activate_node_checks(current_node)
        if added_checks:
            logger.debug(
                f"[FactCheckAgent] Node '{current_node.id}' activated checks: {', '.join(added_checks)}"
            )

        # Phase 1: execute the current node
        should_stop = self._execute_node(session, claim, state, current_node, errors, trace)

        # Phase 2: resolve the next node (skipped if execution already finalized)
        if not should_stop:
            should_stop = self._resolve_next_node(session, state, current_node, errors, trace)

        trace.record_node_transition(
            iteration=iteration,
            from_node=node_before,
            to_node=state.current_node_id,
            requested_target=None,
        )
        trace.finish_iteration(
            iteration=iteration,
            evidence_count_after=len(state.evidences),
            new_errors=errors[errors_before:],
        )
        return should_stop

    def _get_current_node(self, state: FactCheckSessionState) -> BlueprintNode:
        """Return the blueprint node matching the current state position."""
        if state.current_node_id == REFINE_NODE_ID:
            return _REFINE_NODE
        return next(
            node
            for node in state.selected_blueprint.verification_graph.nodes
            if node.id == state.current_node_id
        )

    def _refine_budget_remaining(self, state: FactCheckSessionState) -> int:
        """Iterations still available once the blueprint's own path is complete.

        Refine is only ever entered where the graph would otherwise finalize, so
        the mandatory path is already behind us and its remaining length is zero —
        every unspent iteration is free. Entering refine anywhere else would need
        the longest remaining path subtracted here, which is only well-defined
        because refine is never added to the blueprint's own (acyclic) graph.
        """
        return max(state.selected_blueprint.policy_constraints.max_iterations - state.iteration, 0)

    def _should_refine(self, state: FactCheckSessionState) -> bool:
        """Whether to divert to (or stay in) the refine node instead of finalizing.

        Requires open checks, not merely spare budget: with nothing unresolved on
        the taken path there is no stated question left for more search to answer,
        and diverting would spend budget on claims the planner correctly finished.

        Applies to re-entry as well, so refine can spend the whole remaining budget
        across several passes. Three things bound it: the run loop's own
        ``range(1, max_iterations + 1)``, the budget term reaching zero, and the
        planner's own ``finalize`` — which the refine prompt explicitly invites
        whenever more search would not change the verdict.
        """
        if not self.enable_refine_node:
            return False
        return self._refine_budget_remaining(state) > 0 and bool(state.open_check_ids())

    def position_at_refine(self, state: FactCheckSessionState) -> None:
        """Move state onto the refine node without touching the trace.

        Split out from ``_enter_refine`` because the trace recorder asserts an
        active iteration, which holds inside the run loop but not for a session
        rebuilt from a recorded trace (see ``mafc.agents.fact_check.resume``).
        """
        # Register the synthetic node in the layer map before moving there: the
        # system prompt indexes node_layers by the current node every iteration,
        # and refine is not part of the blueprint topology that built the map.
        # One past the deepest layer, so the prompt's remaining-layers term is
        # zero and its `stay_allowed` reads as "budget is all that is left".
        state.node_layers.setdefault(REFINE_NODE_ID, state.max_layer + 1)
        state.current_node_id = REFINE_NODE_ID
        state.node_history.append(REFINE_NODE_ID)
        state.last_synthesis = None

    def _enter_refine(self, state: FactCheckSessionState, trace: FactCheckTraceRecorder) -> bool:
        """Divert to the refine node; returns False so the loop keeps running."""
        self.position_at_refine(state)
        trace.record_auto_routing(REFINE_NODE_ID, state.iteration)
        logger.debug(
            f"[FactCheckAgent] Diverting to refine at iteration {state.iteration}; "
            f"{self._refine_budget_remaining(state)} iterations left, "
            f"open checks: {', '.join(state.open_check_ids())}"
        )
        return False

    def _execute_node(
        self,
        session: AgentSession,
        claim: Claim,
        state: FactCheckSessionState,
        current_node: BlueprintNode,
        errors: list[str],
        trace: FactCheckTraceRecorder,
    ) -> bool:
        """Dispatch to the node-type-specific execution handler."""
        if isinstance(current_node, BlueprintActionNode):
            trace.record_execution_type("action_node", state.iteration)
            return self._execute_action_node(session, claim, state, errors, trace)
        if isinstance(current_node, BlueprintSynthesisNode):
            trace.record_execution_type("synthesis_node", state.iteration)
            self._execute_synthesis_node(session, state, trace)
            return False
        return False

    def _execute_action_node(
        self,
        session: AgentSession,
        claim: Claim,
        state: FactCheckSessionState,
        errors: list[str],
        trace: FactCheckTraceRecorder,
    ) -> bool:
        """Run the LLM execution call for an action node and dispatch delegated tasks."""
        prompt_text = (
            build_refine_node_prompt(session, state)
            if state.current_node_id == REFINE_NODE_ID
            else build_action_node_prompt(session, state)
        )
        messages = [
            self._system_message(state),
            Message(role=MessageRole.USER, content=Prompt(text=prompt_text)),
        ]
        trace.record_planner_messages(messages, state.iteration)
        logger.debug(f"[FactCheckAgent] Iteration {state.iteration} action node messages:")
        for msg in messages:
            logger.debug(f"- {msg.role.value} message:\n{msg.content}\n")

        _resp = self.model.generate(messages)
        response_text = _resp.text.strip()
        trace.add_usage(_resp, self.model.name)
        trace.add_timing("llm_action_planner_ms", _resp.duration_ms or 0.0)
        trace.record_planner_response(response_text, state.iteration)
        logger.debug(f"[FactCheckAgent] Iteration {state.iteration} action node response:\n{response_text}")

        _action_repair_prefix = (
            "Convert the following response to strict JSON with schema:\n"
            '{"decision_type":"delegate|finalize","rationale":"string",'
            '"tasks":[{"task_id":"string","agent_type":"string","instruction":"string",'
            '"follow_up_to":"string|null","rationale":"string|null"}],'
            '"final_answer":"string|null"}\n'
            "Only return JSON."
        )
        decision, repair_text = try_parse_with_repair(
            response_text, try_parse_planner_decision, self.model, _action_repair_prefix, trace
        )
        if repair_text is not None:
            trace.record_repair(
                phase="action_planner_parse",
                prompt=f"{_action_repair_prefix}\n\nResponse:\n{response_text}",
                response_text=repair_text,
                iteration=state.iteration,
            )
        if decision is None:
            message = f"Action node planner returned invalid output in iteration {state.iteration}."
            errors.append(message)
            trace.record_error(
                phase="action_planner_parse",
                message=message,
                iteration=state.iteration,
                raw_response=response_text,
            )
            return True

        trace.record_decision(decision, state.iteration)
        state.action_history.append(f"{decision.decision_type.value}: {decision.rationale}")

        if decision.decision_type == PlannerDecisionType.FINALIZE:
            # Finalizing here skips the routing phase entirely, so settle the checks
            # first — the synthesis renders the ledger into the verdict.
            self._maybe_update_checks(session, state, errors, trace, stage="action_finalize")
            state.final_answer = decision.final_answer or self._synthesize_findings(
                session, state, None, trace=trace, stage="action_finalize"
            )
            return True

        # DELEGATE
        self._delegate_tasks(session, claim, state, decision.tasks, errors, trace)
        return False

    def _execute_synthesis_node(
        self,
        session: AgentSession,
        state: FactCheckSessionState,
        trace: FactCheckTraceRecorder,
    ) -> None:
        """Auto-execute synthesis and store the result for use in the routing phase."""
        synthesis = self._synthesize_findings(session, state, None, trace=trace, stage="synthesis_node")
        state.last_synthesis = synthesis

    def _resolve_next_node(
        self,
        session: AgentSession,
        state: FactCheckSessionState,
        current_node: BlueprintNode,
        errors: list[str],
        trace: FactCheckTraceRecorder,
    ) -> bool:
        """Determine the next node: auto-advance if only one option, else ask the LLM."""
        options = self._get_routing_options(current_node)

        if not options:
            # No outgoing transitions — end of graph. No routing call happens here,
            # so the checks get their last chance to be settled before the verdict.
            self._maybe_update_checks(session, state, errors, trace, stage="terminal_node")
            # Checked after the update so refine sees the settled ledger: this is
            # the path the graph-exhausted claims take, and their open checks are
            # only known once this call has run.
            if self._should_refine(state):
                return self._enter_refine(state, trace)
            trace.record_auto_routing("finalize", state.iteration)
            return True

        if len(options) == 1:
            # Single path — advance without a routing LLM call, but still let the
            # model settle whatever this node's evidence now resolves.
            self._maybe_update_checks(session, state, errors, trace, stage="single_exit_node")
            target = options[0].to
            if target == "finalize" and self._should_refine(state):
                return self._enter_refine(state, trace)
            trace.record_auto_routing(target, state.iteration)
            if target == "finalize":
                return True
            state.current_node_id = target
            state.node_history.append(target)
            state.last_synthesis = None
            return False

        # Multiple paths — ask the LLM
        return self._llm_routing_call(session, state, options, errors, trace)

    def _get_routing_options(self, node: BlueprintNode) -> list[BlueprintTransition]:
        """Return the routing options for the current node."""
        return list(node.transition)

    def _llm_routing_call(
        self,
        session: AgentSession,
        state: FactCheckSessionState,
        options: list[BlueprintTransition],
        errors: list[str],
        trace: FactCheckTraceRecorder,
    ) -> bool:
        """Ask the LLM to choose among multiple routing options and advance the node."""
        messages = [
            self._system_message(state),
            Message(
                role=MessageRole.USER, content=Prompt(text=build_routing_prompt(session, state, options))
            ),
        ]
        _resp = self.model.generate(messages)
        response_text = _resp.text.strip()
        trace.add_usage(_resp, self.model.name)
        trace.add_timing("llm_router_ms", _resp.duration_ms or 0.0)
        logger.debug(f"[FactCheckAgent] Iteration {state.iteration} routing response:\n{response_text}")

        _routing_repair_prefix = (
            "Convert the following response to strict JSON with schema:\n"
            '{"next_node_id":"string","rationale":"string","final_answer":"string|null",'
            '"check_updates":[{"id":"string","status":"supported|refuted|unclear","reason":"string"}]}\n'
            "Only return JSON."
        )
        routing, repair_text = try_parse_with_repair(
            response_text, try_parse_routing_decision, self.model, _routing_repair_prefix, trace
        )
        if repair_text is not None:
            trace.record_repair(
                phase="routing_parse",
                prompt=f"{_routing_repair_prefix}\n\nResponse:\n{response_text}",
                response_text=repair_text,
                iteration=state.iteration,
            )
        if routing is None:
            message = f"Routing decision could not be parsed in iteration {state.iteration}."
            errors.append(message)
            trace.record_error(
                phase="routing_parse",
                message=message,
                iteration=state.iteration,
                raw_response=response_text,
            )
            return True

        for warning in routing.coercion_warnings:
            trace.record_error(
                phase="routing_parse_coercion",
                message=warning,
                iteration=state.iteration,
            )
        trace.record_routing_call(
            messages=messages,
            response_text=response_text,
            routing=routing,
            iteration=state.iteration,
        )
        self._apply_check_updates(state, routing.check_updates)

        if routing.next_node_id == "finalize":
            # Checked after _apply_check_updates so this routing call's own updates
            # count: a call that just closed the last open check must be allowed to
            # finalize rather than be diverted on a stale ledger.
            if self._should_refine(state):
                return self._enter_refine(state, trace)
            state.final_answer = routing.final_answer or self._synthesize_findings(
                session, state, None, trace=trace, stage="routing_finalize"
            )
            return True

        valid_targets = {opt.to for opt in options if opt.to != "finalize"}
        if routing.next_node_id not in valid_targets:
            message = (
                f"Routing selected unknown node '{routing.next_node_id}' in iteration {state.iteration}."
            )
            errors.append(message)
            trace.record_error(phase="routing_invalid_target", message=message, iteration=state.iteration)
            return True

        state.current_node_id = routing.next_node_id
        state.node_history.append(routing.next_node_id)
        state.last_synthesis = None
        return False

    def _maybe_update_checks(
        self,
        session: AgentSession,
        state: FactCheckSessionState,
        errors: list[str],
        trace: FactCheckTraceRecorder,
        stage: str,
    ) -> None:
        """Re-evaluate unresolved checks wherever the routing phase makes no LLM call.

        Check updates normally ride along with the routing decision, but routing
        only calls the LLM when a node offers two or more transitions. Without this
        pass, a check activated on a single-exit chain would stay 'unchecked' for
        the rest of the run and reach the verdict as unresolved uncertainty —
        penalising the deeper path for investigating further.

        The call is a cheap, focused one (no blueprint system prompt, no routing
        decision) and is skipped whenever it could not learn anything: no
        unresolved checks, no evidence to judge them against, or nothing changed
        since the checks were last evaluated.
        """
        if not state.open_check_ids() or not state.evidences:
            return
        if state.check_evaluation_signature() == state.last_check_evaluation:
            return

        messages = [
            Message(role=MessageRole.USER, content=Prompt(text=build_check_update_prompt(session, state)))
        ]
        _resp = self.model.generate(messages)
        response_text = _resp.text.strip()
        trace.add_usage(_resp, self.model.name)
        trace.add_timing("llm_check_update_ms", _resp.duration_ms or 0.0)
        logger.debug(f"[FactCheckAgent] Iteration {state.iteration} check update response:\n{response_text}")

        _check_repair_prefix = (
            "Convert the following response to strict JSON with schema:\n"
            '{"check_updates":[{"id":"string","status":"supported|refuted|unclear","reason":"string"}]}\n'
            "Only return JSON."
        )
        decision, repair_text = try_parse_with_repair(
            response_text, try_parse_check_update_decision, self.model, _check_repair_prefix, trace
        )
        if repair_text is not None:
            trace.record_repair(
                phase="check_update_parse",
                prompt=f"{_check_repair_prefix}\n\nResponse:\n{response_text}",
                response_text=repair_text,
                iteration=state.iteration,
            )
        if decision is None:
            message = f"Check update returned invalid output in iteration {state.iteration}."
            errors.append(message)
            trace.record_error(
                phase="check_update_parse",
                message=message,
                iteration=state.iteration,
                raw_response=response_text,
            )
            # Unparseable, but the evidence was still put to the model: stamp the
            # state so the next single-exit node does not re-ask the same question.
            self._apply_check_updates(state, [])
            return

        for warning in decision.coercion_warnings:
            trace.record_error(
                phase="check_update_parse_coercion",
                message=warning,
                iteration=state.iteration,
            )
        trace.record_check_update_call(
            messages=messages,
            response_text=response_text,
            decision=decision,
            iteration=state.iteration,
            stage=stage,
        )
        self._apply_check_updates(state, decision.check_updates)

    def _apply_check_updates(
        self, state: FactCheckSessionState, check_updates: list[PlannerCheckUpdate]
    ) -> None:
        """Merge planner-provided required-check updates into the session state.

        Also stamps the state the checks were evaluated against, so the standalone
        check-update pass never re-asks about an unchanged state.
        """
        for update in check_updates:
            if update.id not in state.required_check_status:
                continue
            state.required_check_status[update.id] = update.status
            state.required_check_reasons[update.id] = update.reason
        state.last_check_evaluation = state.check_evaluation_signature()

    def _delegate_tasks(
        self,
        session: AgentSession,
        claim: Claim,
        state: FactCheckSessionState,
        tasks: list[DelegationTask],
        errors: list[str],
        trace: FactCheckTraceRecorder,
    ) -> None:
        """Delegate planner-created mid-level tasks to worker pools and track task lineage."""
        if not tasks:
            message = "Planner requested delegation without any tasks."
            errors.append(message)
            trace.record_error(phase="delegate_tasks", message=message, iteration=state.iteration)
            return

        delegated_calls: list[tuple[str, Agent, AgentSession, TraceScope | None]] = []
        for task in tasks:
            normalized_agent_type = self._normalize_agent_type(task.agent_type)
            logger.debug(
                f"[FactCheckAgent] Delegating task '{task.task_id}' of type '{task.agent_type}' "
                f"(normalized='{normalized_agent_type}') with instruction: {task.instruction}"
            )
            worker_agents = self.delegation_agents.get(normalized_agent_type, [])
            if not worker_agents:
                message = f"No worker agents configured for agent type '{task.agent_type}'."
                errors.append(message)
                trace.record_error(phase="delegate_tasks", message=message, iteration=state.iteration)
                continue

            worker_index = len(delegated_calls) % len(worker_agents)
            worker_agent = worker_agents[worker_index]
            if (
                trace.trace_dir is not None
                and hasattr(worker_agent, "trace_dir")
                and getattr(worker_agent, "trace_dir") is None
            ):
                setattr(worker_agent, "trace_dir", trace.trace_dir)
            task_scope = None
            if trace.scope is not None:
                task_scope = trace.scope.child_scope(
                    "delegated_task",
                    key=task.task_id,
                    metadata={
                        "agent_type": normalized_agent_type,
                        "instruction": task.instruction,
                        "iteration": state.iteration,
                    },
                )
                task_scope.append_event(
                    "task_created",
                    {
                        "task_id": task.task_id,
                        "agent_type": normalized_agent_type,
                        "instruction": task.instruction,
                        "iteration": state.iteration,
                    },
                )
            try:
                child_session = self._build_child_session(
                    parent_session=session,
                    claim=claim,
                    state=state,
                    task_id=task.task_id,
                    agent_type=normalized_agent_type,
                    instruction=task.instruction,
                    follow_up_to=task.follow_up_to,
                )
            except ValueError as exc:
                message = f"Failed to build session for task '{task.task_id}': {exc}"
                errors.append(message)
                trace.record_error(phase="delegate_tasks", message=message, iteration=state.iteration)
                continue
            state.delegated_tasks[task.task_id] = DelegatedTaskRecord(
                task_id=task.task_id,
                agent_type=normalized_agent_type,
                child_session_id=child_session.id,
                instruction=task.instruction,
                iteration=state.iteration,
                follow_up_to=task.follow_up_to,
                rationale=task.rationale,
            )
            trace.record_delegated_task(
                iteration=state.iteration,
                task_id=task.task_id,
                agent_type=normalized_agent_type,
                instruction=task.instruction,
                follow_up_to=task.follow_up_to,
                rationale=task.rationale,
                child_session_id=child_session.id,
            )
            delegated_calls.append((task.task_id, worker_agent, child_session, task_scope))

        if not delegated_calls:
            return

        _delegation_t0 = time.monotonic()
        if len(delegated_calls) == 1 or self.n_workers <= 1:
            results = [
                (task_id, self._run_worker_agent(worker_agent, child_session, task_scope))
                for task_id, worker_agent, child_session, task_scope in delegated_calls
            ]
        else:
            max_workers = min(len(delegated_calls), self.n_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    (
                        task_id,
                        executor.submit(self._run_worker_agent, worker_agent, child_session, task_scope),
                    )
                    for task_id, worker_agent, child_session, task_scope in delegated_calls
                ]
                results = []
                for task_id, future in futures:
                    try:
                        results.append((task_id, future.result()))
                    except Exception as _worker_exc:
                        logger.error(
                            f"[FactCheckAgent] Worker thread exception for task '{task_id}':\n"
                            f"{_traceback.format_exc()}"
                        )
                        raise
        trace.add_timing("delegation_ms", (time.monotonic() - _delegation_t0) * 1000)

        for task_id, result in results:
            session.messages.extend(result.messages)
            state.evidences.extend(result.evidences)
            errors.extend(result.errors)
            trace.record_delegated_task_result(iteration=state.iteration, task_id=task_id, result=result)
            child_trace = result.trace
            if child_trace is not None:
                trace.record_delegated_task_trace(
                    iteration=state.iteration,
                    task_id=task_id,
                    child_trace=child_trace,
                )
            for error in result.errors:
                trace.record_error(
                    phase="delegated_task",
                    message=f"Task {task_id}: {error}",
                    iteration=state.iteration,
                )
            state.action_history.append(
                f"task {task_id} completed with {len(result.evidences)} evidences and {len(result.errors)} errors"
            )
        session.evidences = list(state.evidences)

    def _build_child_session(
        self,
        parent_session: AgentSession,
        claim: Claim,
        state: FactCheckSessionState,
        task_id: str,
        agent_type: str,
        instruction: str,
        follow_up_to: str | None,
    ) -> AgentSession:
        """Build one child session for a delegated worker-agent call."""
        existing_task = state.delegated_tasks.get(task_id)
        follow_up_task = state.delegated_tasks.get(follow_up_to) if follow_up_to is not None else None
        if existing_task is not None:
            child_session_id = existing_task.child_session_id
        elif follow_up_task is not None:
            child_session_id = follow_up_task.child_session_id
        else:
            child_session_id = f"{parent_session.id}:{agent_type}:{task_id}:{state.iteration}"

        if agent_type == "media":
            goal = MultimodalSequence(instruction or str(claim))
            return AgentSession(
                id=child_session_id,
                goal=goal,
                claim=claim,
                cutoff_date=parent_session.cutoff_date,
                blocked_urls=parent_session.blocked_urls,
                parent_session_id=parent_session.id,
                evidences=list(state.evidences),
            )

        return AgentSession(
            id=child_session_id,
            goal=Prompt(text=instruction or str(claim)),
            claim=claim,
            cutoff_date=parent_session.cutoff_date,
            blocked_urls=parent_session.blocked_urls,
            parent_session_id=parent_session.id,
            evidences=list(state.evidences),
        )

    def _run_worker_agent(
        self, worker_agent: Agent, child_session: AgentSession, trace_scope: TraceScope | None
    ) -> AgentResult:
        run_signature = inspect.signature(worker_agent.run)
        if "trace_scope" in run_signature.parameters:
            return worker_agent.run(child_session, trace_scope=trace_scope)
        return worker_agent.run(child_session)

    def _synthesize_findings(
        self,
        session: AgentSession,
        state: FactCheckSessionState,
        instruction: str | None,
        trace: FactCheckTraceRecorder | None = None,
        stage: str = "finalize",
    ) -> str:
        """Produce a synthesis from accumulated evidence when finalizing or summarizing."""
        if not state.evidences:
            return ""
        prompt_text = build_final_synthesis_prompt(session, state)
        if instruction:
            prompt_text += f"\n\nAdditional instruction:\n{instruction}"
        _synth_resp = self.model.generate([Message(role=MessageRole.USER, content=Prompt(text=prompt_text))])
        answer = _synth_resp.text.strip()
        if trace is not None:
            trace.add_usage(_synth_resp, self.model.name)
            trace.add_timing("llm_synthesis_ms", _synth_resp.duration_ms or 0.0)
        if trace is not None:
            trace.record_synthesis(
                iteration=state.iteration or None,
                stage=stage,
                instruction=instruction,
                answer=answer,
                evidence_count=len(state.evidences),
            )
        return answer

    def _normalize_agent_type(self, agent_type: str) -> str:
        """Map planner-facing agent type aliases onto configured delegation pool keys."""
        normalized = agent_type.strip().lower()
        return self._agent_type_aliases.get(normalized, normalized)

    def _finalize_run(
        self,
        session: AgentSession,
        state: FactCheckSessionState,
        errors: list[str],
        trace: FactCheckTraceRecorder | None = None,
    ) -> AgentResult:
        """Finalize the top-level run from accumulated evidence and synthesized answer."""
        if not state.final_answer:
            state.final_answer = self._synthesize_findings(
                session, state, None, trace=trace, stage="run_finalize"
            )

        if not state.final_answer.strip():
            self._mark_failed(session)
            return AgentResult(
                session=session,
                result=None,
                evidences=list(state.evidences),
                errors=errors,
                status=session.status,
                trace=trace.trace if trace is not None else None,
            )

        session.evidences = list(state.evidences)

        if self.judge_agent is not None and session.evidences:
            seen_sources: set[str] = set()
            deduped_evidences: list[Evidence] = []
            for ev in session.evidences:
                if ev.source not in seen_sources:
                    seen_sources.add(ev.source)
                    deduped_evidences.append(ev)
            # Resolve referent status across the assembled set before judging: the
            # page that carries a debunk and the reverse-image-search hit that
            # identifies the media are different evidence items, so the link
            # between them only exists at this level. Stamped onto the items here,
            # it is serialized with them and the judge stays a pure function.
            if session.claim is not None:
                annotate_evidence_referents(session.claim, deduped_evidences)
            judge_session = AgentSession(
                id=f"{session.id}:judge",
                goal=Prompt(text="Judge the claim using accepted evidence."),
                claim=session.claim,
                evidences=deduped_evidences,
                parent_session_id=session.id,
            )
            judge_scope = (
                trace.scope.child_scope("judge_run", key=judge_session.id, metadata={"agent": "JudgeAgent"})
                if trace is not None and trace.scope is not None
                else None
            )
            judge_result = self._run_worker_agent(self.judge_agent, judge_session, judge_scope)
            errors.extend(judge_result.errors)
            if trace is not None and judge_result.trace is not None:
                trace.record_judge_run(judge_result.trace)

        result_text = MultimodalSequence(state.final_answer)
        result_message = self.make_result_message(session, result_text, list(state.evidences))
        session.messages.append(result_message)
        self._mark_completed(session)
        return AgentResult(
            session=session,
            result=result_text,
            messages=[result_message],
            evidences=list(state.evidences),
            errors=errors,
            status=session.status,
            trace=trace.trace if trace is not None else None,
        )
