from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BlueprintBaseModel(BaseModel):
    """Shared base model for blueprint schema types with strict validation."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ClaimFeatures(BlueprintBaseModel):
    """Deterministic feature set extracted from a claim for blueprint selection."""

    has_claim_text: bool
    text_length: int
    has_image: bool
    image_count: int
    has_video: bool
    video_count: int
    is_multimodal: bool
    has_url: bool
    has_date: bool
    has_question: bool
    claim_has_author: bool = False
    claim_has_origin: bool = False
    claim_has_meta_info: bool = False
    claim_has_date_metadata: bool = False


class BlueprintCondition(BlueprintBaseModel):
    """One feature predicate used in blueprint entry or selection logic."""

    feature: str
    op: str
    value: Any


class BlueprintEntryConditions(BlueprintBaseModel):
    """Conditions that define whether a blueprint is eligible for a claim."""

    all: list[BlueprintCondition] = Field(default_factory=list)
    any: list[BlueprintCondition] = Field(default_factory=list)


class BlueprintSelectorHintSection(BlueprintBaseModel):
    """Positive or negative hints that help rank eligible blueprints."""

    features: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list, validation_alias="exaples")


class BlueprintSelectorHints(BlueprintBaseModel):
    """Optional ranking hints used by a blueprint selector."""

    positive: BlueprintSelectorHintSection = Field(default_factory=BlueprintSelectorHintSection)
    negative: BlueprintSelectorHintSection = Field(default_factory=BlueprintSelectorHintSection)


class BlueprintPolicyConstraints(BlueprintBaseModel):
    """Execution policy limits that constrain how the blueprint may run."""

    allowed_actions: list[str] = Field(default_factory=list)
    max_iterations: int = 3
    require_counterevidence_search: bool = False


class BlueprintRequiredCheck(BlueprintBaseModel):
    """One verification condition that should be satisfied during execution."""

    id: str
    description: str


class BlueprintTransition(BlueprintBaseModel):
    """Conditional edge between verification graph nodes."""

    if_: str = Field(alias="if")
    to: str


class BlueprintAction(BlueprintBaseModel):
    """One delegated action to perform inside an action node."""

    action: str
    intent: str | None = None
    query_guidance: str | None = None


class BlueprintActionNode(BlueprintBaseModel):
    """Graph node that executes one or more delegated actions."""

    id: str
    type: Literal["actions"]
    actions: list[BlueprintAction] = Field(default_factory=list)
    transition: list[BlueprintTransition] = Field(default_factory=list)


class BlueprintSynthesisNode(BlueprintBaseModel):
    """Graph node that synthesizes accumulated evidence into intermediate state."""

    id: str
    type: Literal["synthesis"]
    transition: list[BlueprintTransition] = Field(default_factory=list)


class GateRules(BlueprintBaseModel):
    """Decision rules for a gate node."""

    support_conditions: list[str] = Field(default_factory=list)
    refute_conditions: list[str] = Field(default_factory=list)
    if_fail: str


class BlueprintGateNode(BlueprintBaseModel):
    """Graph node that evaluates support or refutation conditions."""

    id: str
    type: Literal["gate"]
    rules: GateRules


BlueprintNode = Annotated[
    BlueprintActionNode | BlueprintSynthesisNode | BlueprintGateNode,
    Field(discriminator="type"),
]


class BlueprintVerificationGraph(BlueprintBaseModel):
    """Directed verification workflow declared by a blueprint."""

    start_node: str
    nodes: list[BlueprintNode] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> "BlueprintVerificationGraph":
        """Ensure the graph is internally consistent before execution."""
        node_ids = [node.id for node in self.nodes]
        duplicate_ids = {node_id for node_id in node_ids if node_ids.count(node_id) > 1}
        if duplicate_ids:
            duplicates = ", ".join(sorted(duplicate_ids))
            raise ValueError(f"Duplicate node ids found: {duplicates}")

        if self.start_node not in node_ids:
            raise ValueError(f"start_node '{self.start_node}' does not exist in verification_graph.nodes")

        valid_targets = set(node_ids)
        for node in self.nodes:
            transitions = getattr(node, "transition", [])
            for transition in transitions:
                if transition.to not in valid_targets:
                    raise ValueError(
                        f"Node '{node.id}' has transition target '{transition.to}' "
                        "which does not exist in verification_graph.nodes"
                    )

            if (
                isinstance(node, BlueprintGateNode)
                and node.rules.if_fail not in {"return unknown"} | valid_targets
            ):
                raise ValueError(
                    f"Gate node '{node.id}' has if_fail target '{node.rules.if_fail}' "
                    "which is neither a known node id nor 'return unknown'"
                )
        return self


class BlueprintTerminationRule(BlueprintBaseModel):
    """Rule describing when blueprint execution should return a final outcome."""

    if_: str = Field(alias="if")
    return_: str = Field(alias="return")


class Blueprint(BlueprintBaseModel):
    """Top-level declarative specification for one fact-checking workflow."""

    name: str
    description: str
    entry_conditions: BlueprintEntryConditions = Field(default_factory=BlueprintEntryConditions)
    selector_hints: BlueprintSelectorHints = Field(default_factory=BlueprintSelectorHints)
    policy_constraints: BlueprintPolicyConstraints = Field(default_factory=BlueprintPolicyConstraints)
    required_checks: list[BlueprintRequiredCheck] = Field(default_factory=list)
    verification_graph: BlueprintVerificationGraph
    termination: list[BlueprintTerminationRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_required_checks(self) -> "Blueprint":
        """Reject duplicate required check identifiers within one blueprint."""
        check_ids = [check.id for check in self.required_checks]
        duplicate_ids = {check_id for check_id in check_ids if check_ids.count(check_id) > 1}
        if duplicate_ids:
            duplicates = ", ".join(sorted(duplicate_ids))
            raise ValueError(f"Duplicate required check ids found: {duplicates}")
        return self
