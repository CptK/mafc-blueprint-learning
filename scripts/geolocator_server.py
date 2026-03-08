"""
Geolocator model server. Loads StreetCLIP once per worker and serves
geolocation requests over HTTP so multiple DEFAME workers can share model
instances without CUDA/fork issues.

Usage:
    python scripts/geolocator_server.py --model geolocal/StreetCLIP --port 5555 --workers 5
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


class GeolocateRequest(BaseModel):
    image_b64: str
    top_k: int = 10
    choices: list[str] = Field(default_factory=list)


class GeolocateResponse(BaseModel):
    most_likely_location: str
    top_k_locations: list[str]
    text: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model inside each worker process after forking.

    Loading here (rather than in __main__) avoids inheriting a CUDA context
    across a fork, which can cause hangs or errors.
    """
    global processor, model, device
    model_name = os.environ.get("GEOLOCATOR_MODEL", "geolocal/StreetCLIP")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PID {os.getpid()}] Loading {model_name} on {device}...", flush=True)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    print(f"[PID {os.getpid()}] Geolocator ready.", flush=True)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/geolocate", response_model=GeolocateResponse)
def geolocate(req: GeolocateRequest):
    if processor is None or model is None or device is None:
        raise HTTPException(status_code=503, detail="Geolocator model is not ready.")

    choices = req.choices or geolocator_default_countries
    image = PILImage.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")

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
    parser.add_argument("--model", default="geolocal/StreetCLIP")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    # Pass model name to worker processes via environment variable.
    os.environ["GEOLOCATOR_MODEL"] = args.model

    uvicorn.run(
        "scripts.geolocator_server:app",
        host="0.0.0.0",
        port=args.port,
        workers=args.workers,
    )
