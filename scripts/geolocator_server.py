"""
Geolocator model server. Loads the model once per worker and serves geolocation
requests over HTTP so multiple DEFAME workers can share model instances without
CUDA/fork issues.

Two backends. `llm` (the default) asks a multimodal LLM for its top countries and
may abstain when the image shows no location cues (`abstained=true`, empty
locations); it beat the CLIP classifier in our ablation. `clip` zero-shot scores
the image against the fixed country list and always names a country.

Usage:
    python -m scripts.geolocator_server --port 5555 --workers 5
    python -m scripts.geolocator_server --model claude_5_sonnet --port 5555 --workers 5
    python -m scripts.geolocator_server --backend clip --port 5555 --workers 5
"""

import argparse
import base64
from contextlib import asynccontextmanager
import io
import os
from typing import Any

import torch
import uvicorn
from PIL import Image as PILImage
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoModel

from config.globals import geolocator_default_countries

# Populated in lifespan startup after fork, so each worker process has its own copy.
processor: Any | None = None
model: Any | None = None
device: torch.device | None = None
llm: Any | None = None

# The tool asks for top_k=10 by default, which is meaningless for an LLM shortlist.
LLM_MAX_COUNTRIES = int(os.environ.get("GEOLOCATOR_LLM_TOP_K", "3"))

LLM_PROMPT = """You are a geolocation expert. Given an image, name the {top_k} most likely \
countries where the photo was taken, most likely first. Judge only from visible cues \
(architecture, vegetation, terrain, vehicles, signage and script, clothing, and the like).

If the image shows no usable location cues (a studio shot, a close-up, a screenshot, a plain \
interior, a text graphic) or is not a photo of a real place, answer exactly UNKNOWN. Abstaining \
is a valid answer; a confidently named wrong country is worse than UNKNOWN.

Answer with one line of country names in English, separated by commas, most likely first. No \
explanation, no numbering.
Example: Greece, Italy, Turkey"""


class GeolocateRequest(BaseModel):
    image_b64: str
    top_k: int = 10
    choices: list[str] = Field(default_factory=list)


class GeolocateResponse(BaseModel):
    most_likely_location: str
    top_k_locations: list[str]
    text: str
    # Set by the LLM backend when it declines to name a country. Defaults to False,
    # so existing clients are unaffected.
    abstained: bool = False


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model inside each worker process after forking.

    Loading here (rather than in __main__) avoids inheriting a CUDA context
    across a fork, which can cause hangs or errors.
    """
    global processor, model, device, llm
    backend = os.environ.get("GEOLOCATOR_BACKEND", "llm")
    default_model = "geolocal/StreetCLIP" if backend == "clip" else "gemini_3_flash"
    model_name = os.environ.get("GEOLOCATOR_MODEL", default_model)
    if backend != "clip":
        from mafc.common.modeling import make_model

        # Short answers only: one line of country names.
        llm = make_model(model_name, max_response_length=128)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PID {os.getpid()}] Loading {model_name} on {device}...", flush=True)
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    print(f"[PID {os.getpid()}] Geolocator ready ({backend}: {model_name}).", flush=True)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


def locate_with_llm(image: PILImage.Image, top_k: int) -> GeolocateResponse:
    """Ask a multimodal LLM for the top-k countries, allowing it to abstain."""
    from ezmm import Image, MultimodalSequence
    from ezmm.common.registry import item_registry

    from mafc.common.modeling import Message, MessageRole

    item = Image(pillow_image=image)
    try:
        response = llm.generate(
            [
                Message(role=MessageRole.SYSTEM, content=MultimodalSequence(LLM_PROMPT.format(top_k=top_k))),
                Message(role=MessageRole.USER, content=MultimodalSequence(item)),
            ]
        )
    finally:
        # The registry caches every item it sees, so a long-running server would keep
        # one decoded image per request. Drop this one now the model has consumed it.
        item_registry.cache.pop((item.kind, item.id), None)
        item.file_path.unlink(missing_ok=True)

    answer = response.text.strip().splitlines()[-1] if response.text.strip() else ""
    countries = [c.strip(" .\"'") for c in answer.split(",")]
    countries = [c for c in countries if c][:top_k]

    if not countries or "unknown" in countries[0].lower():
        return GeolocateResponse(
            most_likely_location="",
            top_k_locations=[],
            text="The image shows no location cues, so the country where it was taken cannot be determined from the image alone.",
            abstained=True,
        )

    return GeolocateResponse(
        most_likely_location=countries[0],
        top_k_locations=countries,
        text=(
            "The most likely countries where the image was taken are, from most to least likely: "
            f"{', '.join(countries)}. This is a visual estimate without probabilities."
        ),
    )


@app.post("/geolocate", response_model=GeolocateResponse)
def geolocate(req: GeolocateRequest):
    if llm is None and (processor is None or model is None or device is None):
        raise HTTPException(status_code=503, detail="Geolocator model is not ready.")

    choices = req.choices or geolocator_default_countries
    image = PILImage.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")

    if llm is not None:
        return locate_with_llm(image, min(req.top_k, LLM_MAX_COUNTRIES))

    inputs = processor(text=choices, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    prediction = outputs.logits_per_image.softmax(dim=1)
    confidences = {choices[i]: round(float(prediction[0][i].item()), 2) for i in range(len(choices))}
    top_k = dict(sorted(confidences.items(), key=lambda x: x[1], reverse=True)[: req.top_k])
    most_likely = max(top_k.items(), key=lambda item: item[1])[0] if top_k else ""

    return GeolocateResponse(
        most_likely_location=most_likely,
        top_k_locations=list(top_k.keys()),
        text=f"The most likely countries where the image was taken are: {top_k}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["llm", "clip"], default="llm")
    parser.add_argument(
        "--model",
        default=None,
        help="Shorthand from config/available_models.csv for --backend llm (default: "
        "gemini_3_flash), or a HuggingFace model for --backend clip (default: geolocal/StreetCLIP).",
    )
    parser.add_argument(
        "--llm-top-k", type=int, default=3, help="Max countries the LLM backend returns."
    )
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    # Pass the config to worker processes via environment variables.
    os.environ["GEOLOCATOR_BACKEND"] = args.backend
    os.environ["GEOLOCATOR_MODEL"] = args.model or (
        "gemini_3_flash" if args.backend == "llm" else "geolocal/StreetCLIP"
    )
    os.environ["GEOLOCATOR_LLM_TOP_K"] = str(args.llm_top_k)

    uvicorn.run(
        "scripts.geolocator_server:app",
        host="0.0.0.0",
        port=args.port,
        workers=args.workers,
    )
