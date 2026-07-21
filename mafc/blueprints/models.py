from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


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
    op: str = Field(validation_alias=AliasChoices("op", "operator"))
    value: Any


def _parse_condition_string(s: str) -> dict:
    """Convert a plain-string condition like 'has_image' or 'text_length > 50' to a dict."""
    s = s.strip()
    match = re.match(r"^(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)$", s)
    if match:
        feature, op, value_str = match.groups()
        value_str = value_str.strip()
        if value_str.lower() == "true":
            value: Any = True
        elif value_str.lower() == "false":
            value = False
        else:
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str
        return {"feature": feature, "op": op, "value": value}
    return {"feature": s, "op": "==", "value": True}


class BlueprintEntryConditions(BlueprintBaseModel):
    """Conditions that define whether a blueprint is eligible for a claim."""

    all: list[BlueprintCondition] = Field(default_factory=list)
    any: list[BlueprintCondition] = Field(default_factory=list)

    @field_validator("all", "any", mode="before")
    @classmethod
    def normalize_condition_list(cls, v: Any) -> Any:
        """Convert plain-string conditions to structured dicts (LLM sometimes omits op/value)."""
        if not isinstance(v, list):
            return v
        return [_parse_condition_string(item) if isinstance(item, str) else item for item in v]


class BlueprintSelectorHintSection(BlueprintBaseModel):
    """Positive or negative hints that help rank eligible blueprints."""

    features: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list, validation_alias="exaples")


class BlueprintSelectorHints(BlueprintBaseModel):
    """Optional ranking hints used by a blueprint selector."""

    positive: BlueprintSelectorHintSection = Field(default_factory=BlueprintSelectorHintSection)
    negative: BlueprintSelectorHintSection = Field(default_factory=BlueprintSelectorHintSection)

    @model_validator(mode="before")
    @classmethod
    def normalize_hint_structure(cls, data: Any) -> Any:
        """Repair common LLM structural mistakes in selector_hints output.

        Handles:
        - 'features' placed at the top-level of selector_hints instead of inside
          positive/negative sections.
        - positive/negative given as a flat list of strings (treat as examples).
        - positive/negative given as a single-element list wrapping a dict.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        top_features: list[str] = data.pop("features", None) or []
        for key in ("positive", "negative"):
            v = data.get(key)
            if v is None:
                continue
            if isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict):
                v = v[0]
            if isinstance(v, list):
                v = {"features": list(top_features), "examples": [s for s in v if isinstance(s, str)]}
            elif isinstance(v, dict) and top_features and "features" not in v:
                v = dict(v)
                v["features"] = list(top_features)
            data[key] = v
        return data


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
    activates_checks: list[str] = Field(default_factory=list)
    """Ids of root-level required_checks that become active when execution first
    reaches this node. Check DEFINITIONS always live in Blueprint.required_checks
    (the contract every consumer relies on); nodes only reference them, letting a
    merged strategy tree scope lane-specific checks to the paths that enter the
    lane. Root checks referenced by no node are global (active from the start)."""


class BlueprintSynthesisNode(BlueprintBaseModel):
    """Graph node that synthesizes accumulated evidence into intermediate state."""

    id: str
    type: Literal["synthesis"]
    transition: list[BlueprintTransition] = Field(default_factory=list)
    activates_checks: list[str] = Field(default_factory=list)
    """Ids of root-level required_checks activated at this node (see BlueprintActionNode)."""


BlueprintNode = Annotated[
    BlueprintActionNode | BlueprintSynthesisNode,
    Field(discriminator="type"),
]


class BlueprintVerificationGraph(BlueprintBaseModel):
    """Directed verification workflow declared by a blueprint."""

    start_node: str
    nodes: list[BlueprintNode] = Field(default_factory=list)

    @field_validator("nodes", mode="before")
    @classmethod
    def normalize_nodes(cls, v: Any) -> Any:
        """Flatten nodes where the LLM wrapped fields under the type key.

        E.g. {synthesis: {id: x, transition: [...]}, type: synthesis}
          -> {id: x, transition: [...], type: synthesis}
        """
        if not isinstance(v, list):
            return v
        result = []
        for item in v:
            if isinstance(item, dict):
                node_type = item.get("type")
                if node_type and node_type in item and isinstance(item[node_type], dict):
                    merged = dict(item[node_type])
                    merged["type"] = node_type
                    for k, val in item.items():
                        if k != node_type:
                            merged.setdefault(k, val)
                    result.append(merged)
                else:
                    result.append(item)
            else:
                result.append(item)
        return result

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
            for transition in node.transition:
                if transition.to not in valid_targets | {"finalize"}:
                    raise ValueError(
                        f"Node '{node.id}' has transition target '{transition.to}' "
                        "which is neither a known node id nor 'finalize'"
                    )
        return self


class Blueprint(BlueprintBaseModel):
    """Top-level declarative specification for one fact-checking workflow."""

    name: str
    description: str
    entry_conditions: BlueprintEntryConditions = Field(default_factory=BlueprintEntryConditions)
    selector_hints: BlueprintSelectorHints = Field(default_factory=BlueprintSelectorHints)
    policy_constraints: BlueprintPolicyConstraints = Field(default_factory=BlueprintPolicyConstraints)
    required_checks: list[BlueprintRequiredCheck] = Field(default_factory=list)
    verification_graph: BlueprintVerificationGraph

    @model_validator(mode="before")
    @classmethod
    def migrate_embedded_node_checks(cls, data: Any) -> Any:
        """Hoist legacy embedded node ``checks`` to root definitions + id refs.

        A transitional merge-pipeline format embedded check OBJECTS on nodes.
        The contract is: definitions live in root ``required_checks``; nodes
        carry only ``activates_checks`` id references. This migrates such files
        on load — id collisions with differing descriptions are renamed.
        """
        if not isinstance(data, dict):
            return data
        graph = data.get("verification_graph")
        nodes = graph.get("nodes") if isinstance(graph, dict) else None
        if not isinstance(nodes, list) or not any(
            isinstance(n, dict) and n.get("checks") for n in nodes
        ):
            return data

        checks: list = [c for c in (data.get("required_checks") or []) if isinstance(c, dict)]
        by_id = {c.get("id"): c for c in checks}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            embedded = node.pop("checks", None) or []
            refs = list(node.get("activates_checks") or [])
            for check in embedded:
                if not isinstance(check, dict) or not check.get("id"):
                    continue
                cid = check["id"]
                existing = by_id.get(cid)
                if existing is None:
                    by_id[cid] = check
                    checks.append(check)
                elif existing.get("description") != check.get("description"):
                    base = cid
                    i = 2
                    while cid in by_id:
                        cid = f"{base}_{i}"
                        i += 1
                    check = {**check, "id": cid}
                    by_id[cid] = check
                    checks.append(check)
                if cid not in refs:
                    refs.append(cid)
            if refs:
                node["activates_checks"] = refs
        data["required_checks"] = checks
        return data

    @model_validator(mode="after")
    def validate_required_checks(self) -> "Blueprint":
        """Reject duplicate check ids and node references to undefined checks."""
        check_ids = [check.id for check in self.required_checks]
        duplicate_ids = {check_id for check_id in check_ids if check_ids.count(check_id) > 1}
        if duplicate_ids:
            duplicates = ", ".join(sorted(duplicate_ids))
            raise ValueError(f"Duplicate required check ids found: {duplicates}")
        known = set(check_ids)
        for node in self.verification_graph.nodes:
            unknown = [cid for cid in node.activates_checks if cid not in known]
            if unknown:
                raise ValueError(
                    f"Node '{node.id}' activates undefined check(s): {', '.join(sorted(unknown))}"
                )
        return self

    def node_scoped_check_ids(self) -> set[str]:
        """Ids of required checks activated by some node (vs. global from start)."""
        return {cid for node in self.verification_graph.nodes for cid in node.activates_checks}
