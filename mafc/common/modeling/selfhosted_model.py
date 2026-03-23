from openai import OpenAI
import openai
from typing import cast

from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from mafc.common.modeling.model import API, APIResponse, Model, Response
from mafc.common.modeling.message import Message
from mafc.common.modeling.utils import messages_with_videos_as_frames
from mafc.common.logger import logger
import config.globals as globals
from mafc.common.modeling.openai_model import format_input


class SelfhostedAPI(API):
    def __init__(self, model: str, context_window: int):
        self.model = model
        self.context_window = context_window
        if not globals.selfhosted_url:
            raise ValueError(
                "Missing self-hosted model URL. Set selfhosted_url in the environment or config/.env."
            )
        self.client = OpenAI(base_url=globals.selfhosted_url, api_key="none", timeout=300)

    def __call__(self, messages: list[Message], **kwargs) -> APIResponse:
        provider_messages = cast(
            list[ChatCompletionMessageParam],
            [
                {
                    "role": message.role.value,
                    "content": cast(
                        list[ChatCompletionContentPartParam],
                        format_input(message.content, context_window=self.context_window),
                    ),
                }
                for message in messages
            ],
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=provider_messages,
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            max_tokens=kwargs.get("max_response_length", 2048),
        )

        content = response.choices[0].message.content
        if not content:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            logger.error(
                f"[Selfhosted] Empty response from model '{self.model}': "
                f"finish_reason={finish_reason!r}, max_response_length={kwargs.get('max_response_length', 2048)}"
            )
            content = "Failed to generate a response."
        return APIResponse(
            text=content,
            input_token_count=response.usage.prompt_tokens if response.usage else None,
            output_token_count=response.usage.completion_tokens if response.usage else None,
            total_token_count=response.usage.total_tokens if response.usage else None,
        )


class SelfhostedModel(Model):

    def __init__(
        self,
        specifier: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        max_response_length: int = 2048,
        video_frames_to_sample: int = 5,
    ):
        super().__init__(
            specifier=specifier,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_response_length=max_response_length,
            video_frames_to_sample=video_frames_to_sample,
        )
        self.api = SelfhostedAPI(model=self.model, context_window=self.context_window)

    def generate(self, messages: list[Message]) -> Response:
        try:
            api_response = self.api(
                messages_with_videos_as_frames(messages, self.video_frames_to_sample),
                temperature=self.temperature,
                top_p=self.top_p,
                max_response_length=self.max_response_length,
            )
        except openai.RateLimitError as e:
            logger.error("Rate limit exceeded on self-hosted model.")
            raise e
        except openai.AuthenticationError as e:
            logger.error("Authentication failed for self-hosted model.")
            raise e
        except Exception as e:
            logger.error(
                f"An error occurred while communicating with the OpenAI API: {e}\nInput: {messages_with_videos_as_frames(messages, self.video_frames_to_sample)}"
            )
            raise e

        return Response(
            text=api_response.text,
            input_token_count=api_response.input_token_count,
            output_token_count=api_response.output_token_count,
            total_token_count=api_response.total_token_count,
            total_cost=self.compute_cost(api_response),
        )


if __name__ == "__main__":
    # Quick test to verify connectivity to the self-hosted model
    api = SelfhostedAPI(model="Qwen/Qwen3.5-122B-A10B-FP8", context_window=4096)

    from ezmm import MultimodalSequence
    from mafc.common.modeling import Message, MessageRole

    messages = [Message(role=MessageRole.USER, content=MultimodalSequence("What is the capital of France?"))]

    response = api(messages)
    print("Response from self-hosted model:", response.text)
