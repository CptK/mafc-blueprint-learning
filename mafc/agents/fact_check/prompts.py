from __future__ import annotations

from mafc.agents.common import AgentSession
from mafc.agents.fact_check.models import FactCheckSessionState
from mafc.agents.web_search.actions import InspectWebSource


def build_planner_system_instructions() -> str:
    """Build the stable orchestration contract that must be present every iteration."""
    return (
        "You are a fact-checking orchestration agent.\n"
        "Use the selected blueprint as strategic guidance, not as a rigid program.\n"
        "Delegate to worker agents when that is the best next step.\n"
        "Avoid redundant work. Track which required checks are satisfied, refuted, or still unclear.\n"
        "Only finalize when the evidence is sufficient or the budget is exhausted.\n"
        "You are an internal controller, not a user-facing assistant.\n"
        "Return strict JSON only with schema:\n"
        '{"decision_type":"delegate|synthesize|finalize|advance_node","rationale":"string",'
        '"tasks":[{"task_id":"string","agent_type":"string","instruction":"string","follow_up_to":"string|null","rationale":"string|null"}],'
        '"instruction":"string|null","target_node_id":"string|null",'
        '"final_answer":"string|null","check_updates":[{"id":"string","status":"unchecked|supported|refuted|unclear","reason":"string"}]}'
    )


def build_system_prompt(state: FactCheckSessionState, available_sub_agents: str) -> str:
    """Build the system prompt for one orchestration iteration.

    Always includes the full blueprint (graph, checks, policy) and the current
    runtime state (position, budget, open checks) so the planner has complete
    forward visibility regardless of which iteration it is.
    """
    blueprint = state.selected_blueprint
    policy = blueprint.policy_constraints
    progression = _progression_summary(state)
    open_checks = [
        check_id
        for check_id, check_status in state.required_check_status.items()
        if check_status.value in {"unchecked", "unclear"}
    ]
    return (
        f"{build_planner_system_instructions()}\n\n"
        f"Available sub-agents:\n{available_sub_agents}\n\n"
        f"Selected blueprint: {blueprint.name}\n"
        f"Blueprint description: {blueprint.description}\n\n"
        f"Policy constraints:\n"
        f"- allowed_actions: {', '.join(policy.allowed_actions) if policy.allowed_actions else 'None specified'}\n"
        f"- max_iterations: {policy.max_iterations}\n"
        f"- require_counterevidence_search: {policy.require_counterevidence_search}\n\n"
        f"Blueprint layers:\n"
        f"- max_layer: {state.max_layer}\n\n"
        f"Required checks:\n{_render_required_checks(state)}\n\n"
        f"Blueprint graph:\n{_render_full_graph(state)}\n\n"
        f"Current position:\n"
        f"- current node: {state.current_node_id}\n"
        f"- current layer: {state.node_layers[state.current_node_id]}\n"
        f"- remaining budget: {max(policy.max_iterations - state.iteration, 0)}\n"
        f"- stay_allowed: {progression['stay_allowed']}\n"
        f"- allowed_next_nodes: {_render_allowed_transitions(state, progression['allowed_next_nodes'])}\n"
        f"- unresolved required checks: {', '.join(open_checks) if open_checks else 'None'}\n"
        f"- delegated tasks: {_render_delegated_tasks(state)}"
    )


def build_iteration_prompt(
    session: AgentSession,
    state: FactCheckSessionState,
) -> str:
    """Build the working prompt for one orchestration iteration."""
    blueprint = state.selected_blueprint
    current_node = next(
        node for node in blueprint.verification_graph.nodes if node.id == state.current_node_id
    )
    progression = _progression_summary(state)
    check_status_lines = [
        f"- {check_id}: {status.value}"
        + (f" — {state.required_check_reasons[check_id]}" if check_id in state.required_check_reasons else "")
        for check_id, status in state.required_check_status.items()
    ]
    current_node_lines = [
        f"- id: {current_node.id}",
        f"- type: {current_node.type}",
    ]
    if hasattr(current_node, "actions"):
        current_node_lines.append(
            "- actions: "
            + (
                ", ".join(action.action for action in current_node.actions)
                if current_node.actions
                else "None"
            )
        )
    if hasattr(current_node, "transition"):
        current_node_lines.append(
            "- transitions: "
            + (
                " | ".join(f"{transition.if_} -> {transition.to}" for transition in current_node.transition)
                if current_node.transition
                else "None"
            )
        )

    claim_has_image = bool(session.claim.images) if session.claim is not None else bool(session.goal.images)
    claim_has_video = bool(session.claim.videos) if session.claim is not None else bool(session.goal.videos)
    return (
        f"Claim:\n{session.claim.describe() if session.claim is not None else str(session.goal).strip()}\n\n"
        f"Iteration: {state.iteration}\n"
        f"Claim modalities:\n"
        f"- has_image: {claim_has_image}\n"
        f"- has_video: {claim_has_video}\n"
        f"- media_delegation_allowed: {claim_has_image or claim_has_video}\n\n"
        f"Current node details:\n{chr(10).join(current_node_lines)}\n\n"
        f"Progression constraints:\n"
        f"- current_layer: {progression['current_layer']}\n"
        f"- max_layer: {progression['max_layer']}\n"
        f"- remaining_layers: {progression['remaining_layers']}\n"
        f"- remaining_budget: {progression['remaining_budget']}\n"
        f"- stay_allowed: {progression['stay_allowed']}\n"
        f"- allowed_next_nodes: {_render_allowed_transitions(state, progression['allowed_next_nodes'])}\n\n"
        f"Accepted evidence summaries:\n{_render_planner_evidence_summaries(state)}\n\n"
        f"Action history:\n"
        f"{chr(10).join(f'- {item}' for item in state.action_history) if state.action_history else 'None'}\n\n"
        f"Delegated task history:\n{_render_delegated_tasks_block(state)}\n\n"
        f"Required check status:\n{chr(10).join(check_status_lines) if check_status_lines else 'None'}\n\n"
        "Decide the best next step."
    )


def build_final_synthesis_prompt(session: AgentSession, state: FactCheckSessionState) -> str:
    """Build a final synthesis prompt when the planner did not provide a final answer."""
    evidence_lines = []
    for evidence in state.evidences:
        summary = (
            str(evidence.takeaways).strip() if evidence.takeaways is not None else str(evidence.raw).strip()
        )
        if summary:
            evidence_lines.append(f"- Source: {evidence.source}\n  Summary: {summary}")

    check_lines = [
        f"- {check_id}: {status.value}" for check_id, status in state.required_check_status.items()
    ]
    return (
        "Provide a concise fact-check synthesis using only the accepted evidence and check statuses below.\n"
        "Be explicit about unresolved uncertainty.\n\n"
        f"Claim:\n{session.claim.describe() if session.claim is not None else str(session.goal).strip()}\n\n"
        f"Required checks:\n{chr(10).join(check_lines) if check_lines else 'None'}\n\n"
        f"Evidence:\n{chr(10).join(evidence_lines) if evidence_lines else 'None'}"
    )


def _render_required_checks(state: FactCheckSessionState) -> str:
    """Render the full required-check list for the initial prompt."""
    checks = state.selected_blueprint.required_checks
    if not checks:
        return "- None"
    return "\n".join(f"- {check.id}: {check.description}" for check in checks)


def _render_full_graph(state: FactCheckSessionState) -> str:
    """Render the blueprint graph in a compact but complete textual form."""
    lines: list[str] = []
    for node in state.selected_blueprint.verification_graph.nodes:
        lines.append(f"- node {node.id} ({node.type})")
        if hasattr(node, "actions") and node.actions:
            lines.append("  actions: " + ", ".join(action.action for action in node.actions))
        if hasattr(node, "transition") and node.transition:
            lines.append(
                "  transitions: "
                + " | ".join(f"{transition.if_} -> {transition.to}" for transition in node.transition)
            )
        if hasattr(node, "rules"):
            lines.append(
                "  rules: "
                f"support={node.rules.support_conditions}, "
                f"refute={node.rules.refute_conditions}, "
                f"if_fail={node.rules.if_fail}"
            )
    return "\n".join(lines) if lines else "- None"


def _render_delegated_tasks(state: FactCheckSessionState) -> str:
    """Render a compact one-line summary of delegated tasks."""
    if not state.delegated_tasks:
        return "None"
    return ", ".join(f"{task.task_id}({task.agent_type})" for task in state.delegated_tasks.values())


def _render_delegated_tasks_block(state: FactCheckSessionState) -> str:
    """Render delegated task lineage for later-turn planner prompts."""
    if not state.delegated_tasks:
        return "None"
    return "\n".join(
        f"- {task.task_id}: {task.agent_type}, session={task.child_session_id}, "
        f"iteration={task.iteration}, follow_up_to={task.follow_up_to or 'None'}, "
        f"instruction={task.instruction}"
        for task in state.delegated_tasks.values()
    )


def _progression_summary(state: FactCheckSessionState) -> dict[str, int | bool | list[str]]:
    """Summarize layer/budget slack and allowed forward movement."""
    current_layer = state.node_layers[state.current_node_id]
    remaining_layers = max(state.max_layer - current_layer, 0)
    remaining_budget = max(
        state.selected_blueprint.policy_constraints.max_iterations - state.iteration + 1, 0
    )
    allowed_next_nodes = [
        node.id
        for node in state.selected_blueprint.verification_graph.nodes
        if state.node_layers[node.id] == current_layer + 1
    ]
    return {
        "current_layer": current_layer,
        "max_layer": state.max_layer,
        "remaining_layers": remaining_layers,
        "remaining_budget": remaining_budget,
        "stay_allowed": remaining_budget > remaining_layers,
        "allowed_next_nodes": allowed_next_nodes,
    }


def _render_allowed_transitions(state: FactCheckSessionState, allowed_next_nodes: list[str]) -> str:
    """Render allowed next nodes annotated with the transition condition that leads to each."""
    if not allowed_next_nodes:
        return "None"
    current_node = next(
        (
            node
            for node in state.selected_blueprint.verification_graph.nodes
            if node.id == state.current_node_id
        ),
        None,
    )
    transitions = getattr(current_node, "transition", None) or [] if current_node is not None else []
    condition_by_target = {t.to: t.if_ for t in transitions}
    parts = []
    for node_id in allowed_next_nodes:
        condition = condition_by_target.get(node_id)
        parts.append(f"{node_id} (if: {condition})" if condition else node_id)
    return ", ".join(parts)


def render_available_sub_agents(agent_descriptions: dict[str, str]) -> str:
    """Render configured worker agent capabilities for the planner."""
    if not agent_descriptions:
        return "- None"
    return "\n".join(
        f"- {agent_type}: {description}" for agent_type, description in sorted(agent_descriptions.items())
    )


def _render_planner_evidence_summaries(state: FactCheckSessionState) -> str:
    """Render concise planner-facing evidence summaries without falling back to raw page text."""
    lines: list[str] = []
    for evidence in state.evidences:
        summary = _planner_summary_for_evidence(evidence)
        if summary is None:
            continue
        lines.append(f"- Source: {evidence.source}\n  Summary: {summary}")
    return "\n".join(lines) if lines else "None"


def _planner_summary_for_evidence(evidence) -> str | None:
    """Choose the safest short evidence summary for planner context."""
    if evidence.takeaways is not None:
        takeaway_text = str(evidence.takeaways).strip()
        if takeaway_text:
            return takeaway_text

    action = evidence.action
    if isinstance(action, InspectWebSource):
        preview = str(evidence.raw).strip()
        if preview and _looks_like_web_preview(preview):
            return preview

    return None


def _looks_like_web_preview(text: str) -> bool:
    """Heuristically accept short web previews and reject long raw scrape payloads."""
    if not text or len(text) > 500:
        return False

    blocked_fragments = (
        "cookie settings",
        "accept all cookies",
        "strictly necessary cookies",
        "manage consent preferences",
        "please paste the exact claim",
        "reply with the exact claim",
        "if you want,",
        "if you prefer",
    )
    lowered = text.lower()
    return not any(fragment in lowered for fragment in blocked_fragments)
