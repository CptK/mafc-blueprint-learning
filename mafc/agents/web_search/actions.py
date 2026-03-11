from __future__ import annotations

from mafc.common.action import Action


class InspectWebSource(Action):
    """Inspect one retrieved web source for potentially useful evidence."""

    name = "inspect_web_source"

    def __init__(self, query_text: str, source_url: str, source_title: str | None = None):
        """Create a provenance action for evidence extracted from one web source.

        Args:
            query_text: Search query that surfaced the source.
            source_url: URL of the inspected source.
            source_title: Optional title of the inspected source.
        """
        self._save_parameters(locals())
        self.query_text = query_text
        self.source_url = source_url
        self.source_title = source_title
