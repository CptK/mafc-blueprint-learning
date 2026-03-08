import base64
from dataclasses import dataclass
from ezmm import MultimodalSequence
from ezmm.common.items import Image
from ezmm.common.registry import item_registry
import io
from PIL.Image import Image as PILImage
import requests
from typing import Any, cast

from config.globals import geolocator_default_countries, geolocator_url
from mafc.common.action import Action
from mafc.common.logger import logger
from mafc.common.results import Results
from mafc.tools.tool import Tool


class Geolocate(Action):
    """Performs geolocation to determine the country where an image was taken."""

    name = "geolocate"
    requires_image = True

    def __init__(self, image: str, top_k: int = 5):
        """Args:
        image: reference to the image to geolocate (must be in the item registry)
        top_k: number of top locations to return (default: 5)
        """
        self._save_parameters(locals())
        img = item_registry.get(reference=image)
        if img is None:
            logger.error(f"[Action:{self.name}] Image not found in registry for reference: {image}")
            self.image = None
        elif not isinstance(img, Image):
            logger.error(
                f"[Action:{self.name}] Item found for reference {image} is not an Image: {type(img).__name__}"
            )
            self.image = None
        else:
            self.image = cast(Image, img)
        self.top_k = top_k

    def __eq__(self, other):
        return isinstance(other, Geolocate) and self.image == other.image

    def __hash__(self):
        return hash((self.name, self.image))


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
        if not action.image:
            logger.error(f"[Tool:{self.name}] Image not found for reference: {action.image}")
            return GeolocationResults(text="Image not found", most_likely_location="", top_k_locations=[])

        return self.locate(action.image.image)

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
