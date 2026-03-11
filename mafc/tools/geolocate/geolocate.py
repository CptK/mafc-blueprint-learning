import base64
from dataclasses import dataclass
from ezmm import MultimodalSequence
from ezmm.common.items import Image, Video
from ezmm.common.registry import item_registry
import io
from PIL import Image as PIL
from PIL.Image import Image as PILImage
import requests
from typing import Any, cast

from config.globals import geolocator_default_countries, geolocator_url
from mafc.common.action import Action, MediaRequirement
from mafc.common.logger import logger
from mafc.common.results import Results
from mafc.tools.tool import Tool


class Geolocate(Action):
    """Performs geolocation to determine the country where an image or video was taken."""

    name = "geolocate"
    media_requirement = MediaRequirement.IMAGE_OR_VIDEO

    def __init__(self, media: str, top_k: int = 5):
        """Args:
        media: reference to the image or video to geolocate (must be in the item registry)
        top_k: number of top locations to return (default: 5)
        """
        self._save_parameters(locals())
        item = item_registry.get(reference=media)
        if item is None:
            logger.error(f"[Action:{self.name}] Media not found in registry for reference: {media}")
            self.media = None
        elif not isinstance(item, Image | Video):
            logger.error(
                f"[Action:{self.name}] Item found for reference {media} is not an Image/Video: "
                f"{type(item).__name__}"
            )
            self.media = None
        else:
            self.media = cast(Image | Video, item)
        self.top_k = top_k

    def __eq__(self, other):
        return isinstance(other, Geolocate) and self.media == other.media

    def __hash__(self):
        return hash((self.name, self.media))


@dataclass
class GeolocationResults(Results):
    text: str
    most_likely_location: str
    top_k_locations: list[str]
    model_output: Any | None = None

    def __str__(self):
        locations_str = ", ".join(self.top_k_locations)
        text = (
            f"Most likely location: {self.most_likely_location}\n"
            f"Top {len(self.top_k_locations)} locations: {locations_str}"
        )
        return text

    def is_useful(self) -> bool | None:
        return self.model_output is not None


class Geolocator(Tool[Geolocate, GeolocationResults]):
    """Localizes a given photo."""

    name = "geolocator"
    actions = [Geolocate]
    summarize = False

    def __init__(self, top_k=10, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.server_url = geolocator_url
        self.default_choices = geolocator_default_countries

    def _perform(self, action: Geolocate) -> GeolocationResults:
        if not isinstance(action, Geolocate):
            logger.error(f"[Tool:{self.name}] Invalid action type: {type(action).__name__}")
            return GeolocationResults(text="Invalid action type", most_likely_location="", top_k_locations=[])
        if not action.media:
            logger.error(f"[Tool:{self.name}] Media not found for reference: {action.media}")
            return GeolocationResults(text="Media not found", most_likely_location="", top_k_locations=[])

        if isinstance(action.media, Video):
            logger.warning(
                f"[Tool:{self.name}] Received video input {action.media.reference}; sampling one frame for geolocation."
            )
            frame = action.media.sample_frames(1, format="jpeg")[0]
            return self.locate(PIL.open(io.BytesIO(frame)))

        return self.locate(action.media.image)

    def locate(self, image: PILImage, choices: list[str] | None = None) -> GeolocationResults:
        """
        Perform geolocation on an image.

        :param image: A PIL image.
        :param choices: A list of location choices. If None, uses a default list of countries.
        :return: A GeoLocationResult object containing location predictions and their probabilities.
        """
        if choices is None:
            choices = self.default_choices

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()
        payload = {"image_b64": image_b64, "top_k": self.top_k}
        if choices is not None:
            payload["choices"] = choices
        response = requests.post(f"{self.server_url}/geolocate", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        result = GeolocationResults(
            text=data["text"],
            most_likely_location=data["most_likely_location"],
            top_k_locations=data["top_k_locations"],
        )
        logger.debug(f"[Tool:{self.name}] Geolocation result: {str(result)}")
        return result

    def _summarize(self, result: GeolocationResults, **kwargs) -> MultimodalSequence | None:
        return MultimodalSequence(result.text)  # TODO: Improve summary w.r.t. uncertainty
