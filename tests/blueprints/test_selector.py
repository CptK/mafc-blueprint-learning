from __future__ import annotations

from pathlib import Path

from ezmm import Image, MultimodalSequence
from ezmm.common.registry import item_registry

from mafc.blueprints import BlueprintRegistry, BlueprintSelector, extract_claim_features
from mafc.blueprints.selector import BlueprintSelectionMode
from mafc.common.modeling.model import Model, Response
from mafc.common.modeling.prompt import Prompt

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


class SequencedModel(Model):
    def __init__(self, outputs: list[str]):
        super().__init__(specifier="OPENAI:gpt-5-mini-2025-08-07")
        self.outputs = outputs
        self.calls: list[str] = []

    def generate(self, prompt: Prompt) -> Response:
        self.calls.append(str(prompt))
        text = self.outputs.pop(0) if self.outputs else ""
        return Response(text=text, total_cost=0.0)


def _load_registry(tmp_path) -> BlueprintRegistry:
    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        """
name: default
description: Catch-all fallback blueprint.
verification_graph:
  start_node: synth
  nodes:
    - id: synth
      type: synthesis
      transition: []
""".strip(),
        encoding="utf-8",
    )
    media_path = tmp_path / "media.yaml"
    media_path.write_text(
        """
name: media_location
description: Investigate location-oriented image claims.
entry_conditions:
  all:
    - feature: has_image
      op: "=="
      value: true
selector_hints:
  positive:
    features: [has_image]
    examples:
      - "Where was this image taken?"
verification_graph:
  start_node: synth
  nodes:
    - id: synth
      type: synthesis
      transition: []
""".strip(),
        encoding="utf-8",
    )
    web_path = tmp_path / "web.yaml"
    web_path.write_text(
        """
name: web_general
description: Investigate text-only web claims.
entry_conditions:
  all:
    - feature: has_image
      op: "=="
      value: false
verification_graph:
  start_node: synth
  nodes:
    - id: synth
      type: synthesis
      transition: []
""".strip(),
        encoding="utf-8",
    )
    hybrid_path = tmp_path / "hybrid.yaml"
    hybrid_path.write_text(
        """
name: hybrid_media
description: Investigate multimodal claims that may need mixed evidence.
entry_conditions:
  all:
    - feature: has_image
      op: "=="
      value: true
selector_hints:
  positive:
    features: [has_image]
    examples:
      - "This image is from Athens."
  negative:
    features: [has_video]
    examples:
      - "This video was filmed in Berlin."
verification_graph:
  start_node: synth
  nodes:
    - id: synth
      type: synthesis
      transition: []
""".strip(),
        encoding="utf-8",
    )
    return BlueprintRegistry.from_path(tmp_path)


def _registered_image() -> Image:
    image = Image(file_path=ASSETS_DIR / "Greece.jpeg")
    item_registry.add_item(image)
    return image


def test_extract_claim_features_for_multimodal_claim() -> None:
    image = _registered_image()
    claim = MultimodalSequence("Where was this image taken in 2024? https://example.com/source", image)

    features = extract_claim_features(claim)

    assert features.has_claim_text is True
    assert features.has_image is True
    assert features.has_video is False
    assert features.is_multimodal is True
    assert features.has_url is True
    assert features.has_date is True


def test_selector_returns_rule_based_match_when_only_one_survives(tmp_path) -> None:
    registry = _load_registry(tmp_path)
    selector = BlueprintSelector(
        model=SequencedModel(outputs=[]),
        registry=registry,
        default_blueprint_name="default",
    )

    result = selector.select("Who won the election?")

    assert result.selected_blueprint.name == "web_general"
    assert result.selection_mode == BlueprintSelectionMode.RULE_BASED
    assert result.surviving_blueprints == ["web_general"]


def test_selector_returns_default_when_no_blueprint_survives(tmp_path) -> None:
    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        """
name: default
description: Catch-all fallback blueprint.
verification_graph:
  start_node: synth
  nodes:
    - id: synth
      type: synthesis
      transition: []
""".strip(),
        encoding="utf-8",
    )
    video_path = tmp_path / "video_only.yaml"
    video_path.write_text(
        """
name: video_only
description: Blueprint that only handles video claims.
entry_conditions:
  all:
    - feature: has_video
      op: "=="
      value: true
verification_graph:
  start_node: synth
  nodes:
    - id: synth
      type: synthesis
      transition: []
""".strip(),
        encoding="utf-8",
    )
    registry = BlueprintRegistry.from_path(tmp_path)
    selector = BlueprintSelector(
        model=SequencedModel(outputs=[]),
        registry=registry,
        default_blueprint_name="default",
    )

    result = selector.select("This claim mentions 2024 but includes no media.")

    assert result.selected_blueprint.name == "default"
    assert result.selection_mode == BlueprintSelectionMode.DEFAULT_FALLBACK
    assert result.surviving_blueprints == []


def test_selector_uses_llm_tiebreak_when_multiple_blueprints_survive(tmp_path) -> None:
    registry = _load_registry(tmp_path)
    model = SequencedModel(outputs=["""
{
  "selected_blueprint": "media_location",
  "reason": "The claim is specifically about image location.",
  "rejected_blueprints": [
    {"name": "hybrid_media", "reason": "Broader than necessary for this claim."}
  ]
}
""".strip()])
    selector = BlueprintSelector(model=model, registry=registry, default_blueprint_name="default")
    image = _registered_image()

    result = selector.select(MultimodalSequence("Where was this image taken?", image))

    assert result.selected_blueprint.name == "media_location"
    assert result.selection_mode == BlueprintSelectionMode.LLM_TIEBREAK
    assert set(result.surviving_blueprints) == {"media_location", "hybrid_media"}
    assert "Where was this image taken?" in model.calls[0]


def test_selector_falls_back_to_default_on_invalid_llm_response(tmp_path) -> None:
    registry = _load_registry(tmp_path)
    selector = BlueprintSelector(
        model=SequencedModel(outputs=['{"selected_blueprint":"does_not_exist"}']),
        registry=registry,
        default_blueprint_name="default",
    )
    image = _registered_image()

    result = selector.select(MultimodalSequence("Where was this image taken?", image))

    assert result.selected_blueprint.name == "default"
    assert result.selection_mode == BlueprintSelectionMode.DEFAULT_FALLBACK
