from ezmm import MultimodalSequence
from datetime import datetime

from mafc.common.label import BaseLabel


class Claim(MultimodalSequence):
    """A single fact-checkable statement with optional metadata.

    This version does not depend on a separate Content object; metadata
    such as author/date/origin/meta_info are stored directly on the Claim.
    """

    id: str | None
    dataset: str | None
    scope: tuple[int, int] | None
    verdict: BaseLabel | None
    justification: MultimodalSequence | None

    def __init__(
        self,
        *args,
        id: str | int | None = None,
        scope: tuple[int, int] | None = None,
        dataset: str | None = None,
        author: str | None = None,
        date: datetime | None = None,
        origin: str | None = None,
        meta_info: str | None = None,
        **kwargs,
    ):
        self.id = str(id) if id is not None else None
        self.scope = scope
        self.dataset = dataset
        self._author = author
        self._date = date
        self._origin = origin
        self._meta_info = meta_info
        self.verdict = None
        self.justification = None
        super().__init__(*args)

    @property
    def author(self):
        return self._author

    @property
    def date(self):
        return self._date

    @property
    def origin(self):
        return self._origin

    @property
    def meta_info(self):
        return self._meta_info

    def describe(self) -> str:
        """Human-friendly representation including optional metadata.

        Returns a normal str (not LiteralString) to avoid tightening __str__'s
        return type from the MultimodalSequence base class.
        """
        base = super().__str__()
        out = f'Claim: "{base}"'
        if author := self.author:
            out += f"\nAuthor: {author}"
        if date := self.date:
            out += f"\nDate: {date.strftime('%B %d, %Y')}"
        if origin := self.origin:
            out += f"\nOrigin: {origin}"
        if meta_info := self.meta_info:
            out += f"\nMeta info: {meta_info}"
        return out
    
    def __repr__(self):
        return (
            f"Claim(str_len={len(self.__str__())}, "
            f"author={self._author}, date={self._date}, origin={self._origin})"
        )
