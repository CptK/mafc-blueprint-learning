from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ezmm import MultimodalSequence

from mafc.agents.agent import Agent, AgentResult
from mafc.agents.common import AgentSession
from mafc.agents.fact_check.models import (
    CheckStatus,
    DelegatedTaskRecord,
    FactCheckSessionState,
    PlannerDecisionType,
)
from mafc.agents.fact_check.parsing import try_parse_planner_decision
from mafc.agents.fact_check.prompts import (
    build_blueprint_reminder,
    build_final_synthesis_prompt,
    build_initial_system_prompt,
    build_iteration_prompt,
    render_available_sub_agents,
)
from mafc.agents.fact_check.tracing import FactCheckTraceRecorder
from mafc.blueprints.models import Blueprint
from mafc.blueprints.topology import analyze_blueprint_topology
from mafc.blueprints.selector import BlueprintSelector
from mafc.common.claim import Claim
from mafc.common.evidence import Evidence
from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt


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
        n_workers: int = 1,
        agent_id: str | None = None,
        trace_dir: str | Path | None = None,
    ):
        """Initialize the top-level fact-check agent with its selector and delegation pools."""
        super().__init__(model, n_workers=n_workers, agent_id=agent_id)
        self.blueprint_selector = blueprint_selector
        self.delegation_agents = delegation_agents or {}
        self.trace_dir = trace_dir
        self._agent_type_aliases = {
            "media": "media",
            "media_agent": "media",
            "web": "web_search",
            "web_search": "web_search",
            "web_search_agent": "web_search",
        }

    def run(self, session: AgentSession) -> AgentResult:
        """Run the blueprint-guided orchestration loop for one session."""
        self._mark_running(session)
        trace = FactCheckTraceRecorder(self.trace_dir, session, self.name)
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
            )
            errors.extend(result.errors)
            trace.record_error(phase="resolve_claim", message=result.errors[0])
            trace.finalize(session=session, state=state, result=result, errors=errors)
            return result

        session.claim = claim
        selection_result = self.blueprint_selector.select(claim)
        trace.set_blueprint(
            selection_result.selected_blueprint.name,
            selection_result.selected_blueprint.policy_constraints.max_iterations,
            selection_result.selected_blueprint.verification_graph.start_node,
        )
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
            raise
        finally:
            trace.finalize(session=session, state=state, result=result, errors=errors)

    def synthesize_from_evidences(self, instruction: str, evidences: list[Evidence]) -> str:
        """Synthesize an answer from evidence using the top-level model."""
        evidence_lines = []
        for evidence in evidences:
            summary = (
                str(evidence.takeaways).strip()
                if evidence.takeaways is not None
                else str(evidence.raw).strip()
            )
            if summary:
                evidence_lines.append(f"- Source: {evidence.source}\n  Summary: {summary}")
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
        return FactCheckSessionState(
            selected_blueprint=blueprint,
            current_node_id=blueprint.verification_graph.start_node,
            node_layers=topology.node_layers,
            max_layer=topology.max_layer,
            required_check_status={check.id: CheckStatus.UNCHECKED for check in blueprint.required_checks},
            evidences=list(evidences),
        )

    def _build_planner_messages(self, session: AgentSession, state: FactCheckSessionState) -> list[Message]:
        """Build planner messages with a strategy system message and a user-state message."""
        available_sub_agents = render_available_sub_agents(self._available_sub_agent_descriptions())
        if not state.system_context_initialized:
            state.system_context_initialized = True
            return [
                Message(
                    role=MessageRole.SYSTEM,
                    content=Prompt(text=build_initial_system_prompt(state, available_sub_agents)),
                ),
                Message(
                    role=MessageRole.USER,
                    content=Prompt(text=build_iteration_prompt(session, state)),
                ),
            ]
        return [
            Message(
                role=MessageRole.SYSTEM,
                content=Prompt(text=build_blueprint_reminder(state, available_sub_agents)),
            ),
            Message(
                role=MessageRole.USER,
                content=Prompt(text=build_iteration_prompt(session, state)),
            ),
        ]

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
        """Execute one planner/delegation iteration and return whether the loop should stop."""
        if self._should_stop:
            message = "Agent execution stopped early by stop signal."
            errors.append(message)
            trace.record_error(phase="stop_signal", message=message, iteration=iteration)
            return True

        state.iteration = iteration
        errors_before = len(errors)
        node_before = state.current_node_id
        trace.start_iteration(iteration, node_before, len(state.evidences))
        planner_messages = self._build_planner_messages(session, state)
        trace.record_planner_messages(planner_messages, iteration)
        logger.info(f"[FactCheckAgent] Iteration {iteration} planner messages:")
        for msg in planner_messages:
            logger.info(f"- {msg.role.value} message:\n{msg.content}\n")
        planner_response = self.model.generate(planner_messages).text.strip()
        trace.record_planner_response(planner_response, iteration)
        logger.info(f"[FactCheckAgent] Iteration {iteration} planner response:\n{planner_response}")
        decision = try_parse_planner_decision(planner_response)
        if decision is None:
            message = f"Planner returned invalid output in iteration {iteration}."
            errors.append(message)
            trace.record_error(phase="planner_parse", message=message, iteration=iteration)
            trace.finish_iteration(
                iteration=iteration,
                evidence_count_after=len(state.evidences),
                new_errors=errors[errors_before:],
            )
            return True

        trace.record_decision(decision, iteration)
        self._apply_check_updates(state, decision.check_updates)
        self._apply_node_progression(state, decision, errors)
        trace.record_node_transition(
            iteration=iteration,
            from_node=node_before,
            to_node=state.current_node_id,
            requested_target=decision.target_node_id,
        )
        state.action_history.append(f"{decision.decision_type.value}: {decision.rationale}")

        if decision.decision_type == PlannerDecisionType.DELEGATE:
            self._delegate_tasks(
                session=session,
                claim=claim,
                state=state,
                tasks=decision.tasks,
                errors=errors,
                trace=trace,
            )
            trace.finish_iteration(
                iteration=iteration,
                evidence_count_after=len(state.evidences),
                new_errors=errors[errors_before:],
            )
            return False

        if decision.decision_type == PlannerDecisionType.SYNTHESIZE:
            state.final_answer = self._synthesize_findings(
                session,
                state,
                decision.instruction,
                trace=trace,
                stage="iteration_synthesize",
            )
            trace.finish_iteration(
                iteration=iteration,
                evidence_count_after=len(state.evidences),
                new_errors=errors[errors_before:],
            )
            return False

        if decision.decision_type == PlannerDecisionType.FINALIZE:
            state.final_answer = decision.final_answer or self._synthesize_findings(
                session,
                state,
                decision.instruction,
                trace=trace,
                stage="iteration_finalize",
            )
            if decision.final_answer:
                trace.record_synthesis(
                    iteration=iteration,
                    stage="planner_finalize",
                    instruction=decision.instruction,
                    answer=state.final_answer,
                    evidence_count=len(state.evidences),
                )
            trace.finish_iteration(
                iteration=iteration,
                evidence_count_after=len(state.evidences),
                new_errors=errors[errors_before:],
            )
            return True

        trace.finish_iteration(
            iteration=iteration,
            evidence_count_after=len(state.evidences),
            new_errors=errors[errors_before:],
        )
        return False

    def _apply_check_updates(self, state: FactCheckSessionState, check_updates) -> None:
        """Merge planner-provided required-check updates into the session state."""
        for update in check_updates:
            if update.id not in state.required_check_status:
                continue
            state.required_check_status[update.id] = update.status
            state.required_check_reasons[update.id] = update.reason

    def _apply_node_progression(self, state: FactCheckSessionState, decision, errors: list[str]) -> None:
        """Apply planner-selected node movement while enforcing layer-budget constraints."""
        current_layer = state.node_layers[state.current_node_id]
        remaining_layers = max(state.max_layer - current_layer, 0)
        remaining_budget = max(
            state.selected_blueprint.policy_constraints.max_iterations - state.iteration + 1, 0
        )
        stay_allowed = remaining_budget > remaining_layers
        allowed_next_nodes = {
            node.id
            for node in state.selected_blueprint.verification_graph.nodes
            if state.node_layers[node.id] == current_layer + 1
        }

        requested_target = decision.target_node_id
        if requested_target is None:
            if stay_allowed or not allowed_next_nodes:
                return
            chosen_target = sorted(allowed_next_nodes)[0]
            errors.append(
                f"Planner attempted to stay on node '{state.current_node_id}' without remaining layer slack; "
                f"auto-advancing to '{chosen_target}'."
            )
            state.current_node_id = chosen_target
            state.node_history.append(chosen_target)
            return

        requested_layer = state.node_layers.get(requested_target)
        if requested_layer is None:
            errors.append(f"Planner selected unknown target node '{requested_target}'.")
            return

        if requested_layer == current_layer:
            if stay_allowed or not allowed_next_nodes:
                state.current_node_id = requested_target
                state.node_history.append(requested_target)
                return
            chosen_target = sorted(allowed_next_nodes)[0]
            errors.append(
                f"Planner attempted to stay on node '{requested_target}' without remaining layer slack; "
                f"auto-advancing to '{chosen_target}'."
            )
            state.current_node_id = chosen_target
            state.node_history.append(chosen_target)
            return

        if requested_layer == current_layer + 1 and requested_target in allowed_next_nodes:
            state.current_node_id = requested_target
            state.node_history.append(requested_target)
            return

        errors.append(
            f"Planner selected invalid target node '{requested_target}' from layer {current_layer}. "
            f"Allowed next nodes: {sorted(allowed_next_nodes)}."
        )

    def _delegate_tasks(
        self,
        session: AgentSession,
        claim: Claim,
        state: FactCheckSessionState,
        tasks,
        errors: list[str],
        trace: FactCheckTraceRecorder,
    ) -> None:
        """Delegate planner-created mid-level tasks to worker pools and track task lineage."""
        if not tasks:
            message = "Planner requested delegation without any tasks."
            errors.append(message)
            trace.record_error(phase="delegate_tasks", message=message, iteration=state.iteration)
            return

        delegated_calls: list[tuple[str, Agent, AgentSession]] = []
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
            child_session = self._build_child_session(
                parent_session=session,
                claim=claim,
                state=state,
                task_id=task.task_id,
                agent_type=normalized_agent_type,
                instruction=task.instruction,
                follow_up_to=task.follow_up_to,
            )
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
            delegated_calls.append((task.task_id, worker_agent, child_session))

        if not delegated_calls:
            return

        if len(delegated_calls) == 1 or self.n_workers <= 1:
            results = [
                (task_id, worker_agent.run(child_session))
                for task_id, worker_agent, child_session in delegated_calls
            ]
        else:
            max_workers = min(len(delegated_calls), self.n_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    (task_id, executor.submit(worker_agent.run, child_session))
                    for task_id, worker_agent, child_session in delegated_calls
                ]
                results = [(task_id, future.result()) for task_id, future in futures]

        for task_id, result in results:
            session.messages.extend(result.messages)
            state.evidences.extend(result.evidences)
            errors.extend(result.errors)
            trace.record_delegated_task_result(iteration=state.iteration, task_id=task_id, result=result)
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
            goal = MultimodalSequence(instruction or str(claim), *claim.data[1:])
            return AgentSession(
                id=child_session_id,
                goal=goal,
                claim=claim,
                parent_session_id=parent_session.id,
                evidences=list(state.evidences),
            )

        return AgentSession(
            id=child_session_id,
            goal=Prompt(text=instruction or str(claim)),
            claim=claim,
            parent_session_id=parent_session.id,
            evidences=list(state.evidences),
        )

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
        answer = self.model.generate(
            [Message(role=MessageRole.USER, content=Prompt(text=prompt_text))]
        ).text.strip()
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
            )

        session.evidences = list(state.evidences)
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
        )
