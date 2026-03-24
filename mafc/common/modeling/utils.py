import numpy as np
import pandas as pd
from ezmm import MultimodalSequence
from ezmm.common.items import Image, Video

from mafc.common.modeling.message import Message

AVAILABLE_MODELS = pd.read_csv("config/available_models.csv", skipinitialspace=True)


def model_specifier_to_shorthand(specifier: str) -> tuple[str, str]:
    """Returns model shorthand and name for the given specifier."""
    try:
        platform, model_name = specifier.split(":")
    except Exception as e:
        print(e)
        raise ValueError(
            f'Invalid model specification "{specifier}". Check "config/available_models.csv" for available\
                          models. Standard format "<PLATFORM>:<Specifier>".'
        )

    match = (AVAILABLE_MODELS["Platform"] == platform) & (AVAILABLE_MODELS["Name"] == model_name)
    if not np.any(match):
        raise ValueError(f"Specified model '{specifier}' not available.")
    shorthand = AVAILABLE_MODELS[match]["Shorthand"].iloc[0]
    return shorthand, model_name


def model_shorthand_to_full_specifier(shorthand: str) -> str:
    match = AVAILABLE_MODELS["Shorthand"] == shorthand
    platform = AVAILABLE_MODELS["Platform"][match].iloc[0]
    model_name = AVAILABLE_MODELS["Name"][match].iloc[0]
    return f"{platform}:{model_name}"


def get_model_context_window(name: str) -> int:
    """Returns the number of tokens that fit into the context of the model at most."""
    if name not in AVAILABLE_MODELS["Shorthand"].to_list():
        name, _ = model_specifier_to_shorthand(name)
    return int(AVAILABLE_MODELS["Context window"][AVAILABLE_MODELS["Shorthand"] == name].iloc[0])


def get_model_api_pricing(name: str) -> tuple[float, float]:
    """Returns the cost per 1M input tokens and the cost per 1M output tokens for the
    specified model."""
    if name not in AVAILABLE_MODELS["Shorthand"].to_list():
        name, _ = model_specifier_to_shorthand(name)
    input_cost = float(
        AVAILABLE_MODELS["Cost per 1M input tokens"][AVAILABLE_MODELS["Shorthand"] == name].iloc[0]
    )
    output_cost = float(
        AVAILABLE_MODELS["Cost per 1M output tokens"][AVAILABLE_MODELS["Shorthand"] == name].iloc[0]
    )
    return input_cost, output_cost


def with_videos_as_frames(content: MultimodalSequence, n_frames: int = 5) -> MultimodalSequence:
    """Return content with videos replaced by sampled image frames."""
    from mafc.common.modeling.prompt import Prompt

    if isinstance(content, Prompt):
        return content.with_videos_as_frames(n_frames)
    new_data = []
    for item in content.data:
        if isinstance(item, Video):
            frames = item.sample_frames(n_frames, format="jpeg")
            for frame_bytes in frames:
                new_data.append(Image(binary_data=frame_bytes))
        else:
            new_data.append(item)
    return MultimodalSequence(*new_data)


def messages_with_videos_as_frames(messages: list[Message], n_frames: int = 5) -> list[Message]:
    """Return messages with all video content normalized into sampled frames."""
    return [
        Message(role=message.role, content=with_videos_as_frames(message.content, n_frames))
        for message in messages
    ]
