from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from ezmm import MultimodalSequence
from ezmm.common.items import Image, Video
from ezmm.common.registry import item_registry

from mafc.common.action import Action, MediaRequirement
from mafc.common.logger import logger
from mafc.tools.tool import Tool
from mafc.tools.web_search.common import Query
from mafc.tools.web_search.google_vision import GoogleRisResults, GoogleVisionAPI, google_vision_api


class ReverseImageSearch(Action):
    """Perform reverse image search for a registered image or video."""

    name = "reverse_image_search"
    media_requirement = MediaRequirement.IMAGE_OR_VIDEO

    def __init__(self, image: str):
        """Args:
        image: reference to the image or video to inspect (must exist in the item registry)
        """
        self._save_parameters(locals())
        try:
            media = item_registry.get(reference=image)
        except ValueError:
            logger.error(f"[Action:{self.name}] Invalid media reference: {image}")
            media = None
        if media is None:
            logger.error(f"[Action:{self.name}] Media not found in registry for reference: {image}")
            self.image = None
        elif not isinstance(media, Image | Video):
            logger.error(
                f"[Action:{self.name}] Item found for reference {image} is not an image/video: "
                f"{type(media).__name__}"
            )
            self.image = None
        else:
            self.image = cast(Image | Video, media)


@dataclass
class ReverseImageSearchTool(Tool[ReverseImageSearch, GoogleRisResults]):
    """Tool wrapper around Google Cloud Vision reverse image search."""

    name = "reverse_image_search"
    actions = [ReverseImageSearch]

    api: GoogleVisionAPI

    def __init__(self, api: GoogleVisionAPI | None = None, **kwargs):
        super().__init__(**kwargs)
        self.api = api or google_vision_api

    def _perform(self, action: ReverseImageSearch) -> GoogleRisResults:
        if not isinstance(action, ReverseImageSearch):
            raise TypeError(f"Invalid action type: {type(action).__name__}")
        if action.image is None:
            return GoogleRisResults(
                sources=[],
                query=Query(text="missing media"),
                entities={},
                best_guess_labels=[],
            )
        return self.api.search(Query(media=action.image))

    def _summarize(self, result: GoogleRisResults, **kwargs) -> MultimodalSequence | None:
        if not result.sources and not result.entities and not result.best_guess_labels:
            return None
        return MultimodalSequence(str(result))
