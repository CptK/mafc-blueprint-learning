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

    referent: str | None = None
    """This source's relation to the claim's media: 'exact', 'local', 'partial', or None.

    'exact' — reverse image search confirmed an identical copy of the claim's media
    on this page; 'local' — the same, established by direct frame comparison
    (mafc.common.referent_verifier); 'partial' — only a similar/edited version was
    found; None — unverified, which is NOT evidence of different media.

    The action records which media was *searched*; this records what the *source*
    actually contains, which is what verdicts about origin and debunking depend on.
    Populated at evidence construction for RIS results and at evidence assembly for
    every other source (see media_referent.annotate_evidence_referents), so the
    judge reads it rather than re-deriving it from rendered text.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def is_useful(self) -> bool:
        """Return True when the evidence yielded any useful takeaways."""
        return self.takeaways is not None

    def __str__(self) -> str:
        header = f"### Evidence from `{self.action.name}`\n"
        body = str(self.takeaways if self.takeaways else self.raw)
        return header + body
