from __future__ import annotations

import json

from ezmm import Image
from ezmm.common.registry import item_registry

from mafc.agents import AgentSession
from mafc.agents.fact_check.agent import FactCheckAgent
from mafc.agents.fact_check.models import CheckStatus, FactCheckSessionState
from mafc.agents.fact_check.prompts import build_runtime_state_block, build_system_prompt
from mafc.common.claim import Claim
from mafc.common.modeling.prompt import Prompt

from tests.agents.fact_check.test_refine_node import _blueprint

from tests.agents.fact_check.helpers import (
    ASSETS_DIR,
    FakeWorkerAgent,
    SequencedModel,
    make_registry,
    make_selector,
    registered_image,
)


def test_prompt_shows_actual_image_references(tmp_path) -> None:
    registry = make_registry(tmp_path)
    planner = SequencedModel(
        outputs=[
            json.dumps(
                {
                    "decision_type": "delegate",
                    "rationale": "Analyze both images.",
                    "tasks": [{"task_id": "media_0", "agent_type": "media", "instruction": "Investigate."}],
                }
            ),
            json.dumps(
                {
                    "next_node_id": "finalize",
                    "rationale": "Done.",
                    "final_answer": "Both images analyzed.",
                    "check_updates": [{"id": "location_checked", "status": "supported", "reason": "done"}],
                }
            ),
        ]
    )
    image_a = registered_image()
    image_b = Image(file_path=ASSETS_DIR / "Greece.jpeg")
    item_registry.add_item(image_b)
    media_agent = FakeWorkerAgent("Evidence.", "image://result")
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=make_selector(registry),
        delegation_agents={"media": [media_agent]},
    )
    claim = Claim("Check these images.", image_a, image_b)
    session = AgentSession(id="fact-check:refs", goal=Prompt(text="Fact-check claim"), claim=claim)

    agent.run(session)

    assert image_a.reference in planner.calls[0]
    assert image_b.reference in planner.calls[0]
    assert "images: 2" in planner.calls[0]


def _state_at(iteration: int, node_id: str, checks: dict[str, CheckStatus]) -> FactCheckSessionState:
    return FactCheckSessionState(
        selected_blueprint=_blueprint(),
        current_node_id=node_id,
        node_layers={"n0": 0, "n1": 1},
        max_layer=1,
        iteration=iteration,
        required_check_status=dict(checks),
    )


def test_system_prompt_is_byte_identical_across_iterations() -> None:
    """The caching contract: the system half must not move when state moves.

    Prompt caching is a prefix match, so one volatile byte here turns every planner
    call into a cache write instead of a read. Anything iteration-dependent belongs
    in the runtime state block, which rides on the user message.
    """
    early = _state_at(1, "n0", {"on_path": CheckStatus.UNCHECKED})
    late = _state_at(7, "n1", {"on_path": CheckStatus.SUPPORTED, "other_lane": CheckStatus.UNCLEAR})

    assert build_system_prompt(early, "web_search") == build_system_prompt(late, "web_search")
    assert build_runtime_state_block(early) != build_runtime_state_block(late)


def test_runtime_state_reaches_the_planner_on_the_user_message() -> None:
    """The split must relocate state, not drop it."""
    agent = FactCheckAgent.__new__(FactCheckAgent)
    agent.delegation_agents = {}
    state = _state_at(3, "n0", {"on_path": CheckStatus.UNCHECKED})

    system_msg, user_msg = agent._planner_messages(state, "TASK BODY")

    assert "remaining budget" in str(user_msg.content)
    assert "current node: n0" in str(user_msg.content)
    assert str(user_msg.content).endswith("TASK BODY")
    assert "remaining budget" not in str(system_msg.content)
    assert "Blueprint graph:" in str(system_msg.content)


def test_planner_media_is_attached_once_but_still_named() -> None:
    """A repeated media item uploads once; later mentions survive as text tags.

    The planner names the same item in the claim, in evidence summaries, and in the
    delegated-task history it is told to build. Uploading each mention cost a video
    its five sampled frames every time. Dropping the tags outright would instead
    lose track of which item a past task investigated, so they stay as text.
    """
    from ezmm import Image

    from mafc.agents.fact_check.agent import _prepare_planner_media
    from mafc.common.modeling.anthropic_model import format_input

    image = registered_image()
    prompt = Prompt(
        text=(
            f"Claim: {image.reference}\n"
            f"Accepted evidence summaries:\n- Source: {image.reference} Summary: x\n"
            f"Delegated task history:\n- media_origin: {image.reference} Investigate.\n"
        )
    )
    assert sum(isinstance(b, Image) for b in prompt.to_list()) == 3

    deduped = _prepare_planner_media(prompt)
    blocks = format_input(deduped, context_window=100_000)
    rendered = "".join(b.get("text", "") for b in blocks if b["type"] == "text")

    assert sum(1 for b in blocks if b["type"] == "image") == 1
    # One caption on the surviving attachment, plus the two later mentions.
    assert rendered.count(image.reference) == 3


def test_every_attached_media_item_is_captioned_with_its_reference() -> None:
    """Each attachment is immediately preceded by its own tag, in order.

    Without this the payload carries bare image blocks, and the only naming is a
    separate modality line listing claim media in attachment order. Evidence media
    never reaches that line at all, so the planner had no way to name it in a
    delegated instruction.
    """
    from ezmm import Image
    from ezmm.common.registry import item_registry

    from mafc.agents.fact_check.agent import _prepare_planner_media
    from mafc.common.modeling.anthropic_model import format_input

    # Two distinct files: same-file images collapse under the dedup pass by hash.
    first = registered_image()
    second = Image(file_path=ASSETS_DIR / "ai-generated-city-scene.jpeg")
    item_registry.add_item(second)
    prompt = Prompt(text=f"Claim: {first.reference} and {second.reference} show X")

    blocks = format_input(_prepare_planner_media(prompt), context_window=100_000)

    captions = [
        blocks[i - 1]["text"]
        for i, b in enumerate(blocks)
        if b["type"] == "image" and i > 0 and blocks[i - 1]["type"] == "text"
    ]
    assert captions == [first.reference, second.reference]
