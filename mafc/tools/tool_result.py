from pydantic import BaseModel, ConfigDict
from ezmm import MultimodalSequence

from mafc.common.action import Action
from mafc.common.results import Results


class ToolResult(BaseModel):
    """Structured output of one tool action before it is promoted to evidence.

    This is typically a batch result such as a search response, API payload,
    or model-assisted summary produced directly by a tool invocation.
    """

    raw: Results
    action: Action
    takeaways: MultimodalSequence | None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def is_useful(self) -> bool:
        """Return True when the tool output yielded any takeaways."""
        return self.takeaways is not None

    def __str__(self) -> str:
        header = f"### Result from `{self.action.name}`\n"
        body = str(self.takeaways if self.takeaways else self.raw)
        return header + body
