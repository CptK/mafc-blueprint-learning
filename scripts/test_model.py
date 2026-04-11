"""This script is for testing model loading and reachability."""

from mafc.common.modeling import make_model
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.prompt import Prompt

PROMPT = "What is the capital of France?"
MODEL_NAME = "gemma_31b"


def test_model():
    model = make_model(MODEL_NAME)
    prompt = Prompt(PROMPT)
    response = model.generate([Message(role=MessageRole.USER, content=prompt)])
    print(response)


if __name__ == "__main__":
    test_model()
