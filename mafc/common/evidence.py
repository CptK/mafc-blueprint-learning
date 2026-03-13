from pydantic import BaseModel, ConfigDict
from ezmm import MultimodalSequence

from mafc.common.action import Action


class Evidence(BaseModel):
    """Source-backed evidence item derived from retrieved content.

    Unlike `ToolResult`, this represents one concrete piece of information that
    can be attributed to a specific source, such as a document, URL, or image.
    """

    raw: MultimodalSequence  # The source contents or relevant excerpt.
    action: Action  # The action that produced or extracted this evidence.
    source: str  # The originating source, usually a URL, file, or tool-specific reference.
    preview: str | None = None  # The original search-engine snippet for the source, if available.
    takeaways: MultimodalSequence | None = None  # Helpful distilled information extracted from the source.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def is_useful(self) -> bool:
        """Return True when the evidence yielded any useful takeaways."""
        return self.takeaways is not None

    def __str__(self) -> str:
        header = f"### Evidence from `{self.action.name}`\n"
        body = str(self.takeaways if self.takeaways else self.raw)
        return header + body
