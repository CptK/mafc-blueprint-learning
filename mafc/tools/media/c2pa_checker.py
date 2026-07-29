"""Reads C2PA provenance metadata from an image or video and reports whether
the file's own manifest declares it as AI-generated, edited, or captured.

Absence of a manifest is NOT evidence of authenticity: most media circulating
online carries no C2PA data at all, and platform re-encoding strips manifests
from legitimate camera-signed files. A verdict pulled from a manifest that
fails validation is a claim, not a fact.
"""

from dataclasses import dataclass, field
import json
from typing import cast

from c2pa import Reader
from ezmm import MultimodalSequence
from ezmm.common.items import Image, Video
from ezmm.common.registry import item_registry

from mafc.common.action import Action, MediaRequirement
from mafc.common.logger import logger
from mafc.common.results import Results
from mafc.tools.tool import Tool

# IPTC DigitalSourceType vocabulary. The manifest stores full URIs under
# http://cv.iptc.org/newscodes/digitalsourcetype/ ; we key on the last segment.
SOURCE_TYPES = {
    # model-generated
    "trainedAlgorithmicMedia": ("ai_generated", "created by a generative model"),
    "compositeWithTrainedAlgorithmicMedia": ("ai_partial", "composite including model-generated content"),
    "trainedAlgorithmicData": ("ai_generated", "model-generated data"),
    # algorithmic but not necessarily a trained model
    "algorithmicMedia": ("synthetic", "algorithmically generated, not necessarily a trained model"),
    "algorithmicallyEnhanced": ("edited", "algorithmically enhanced capture"),
    "dataDrivenMedia": ("synthetic", "generated from data"),
    "compositeSynthetic": ("synthetic", "synthetic composite"),
    # camera / real-world origin
    "digitalCapture": ("captured", "captured by a camera or recording device"),
    "computationalCapture": ("captured", "computational photography capture"),
    "negativeFilm": ("captured", "scanned from film"),
    "positiveFilm": ("captured", "scanned from film"),
    "print": ("captured", "scanned from print"),
    "screenCapture": ("captured", "screen capture"),
    "virtualRecording": ("synthetic", "recording of a virtual scene"),
    # human authored / mixed
    "digitalCreation": ("human_created", "created with a digital tool by a person"),
    "humanEdits": ("edited", "edited by a person"),
    "composite": ("edited", "composite of multiple assets"),
    "compositeCapture": ("edited", "composite of captured assets"),
}

# categories that mean "the file itself says AI was involved"
AI_CATEGORIES = {"ai_generated", "ai_partial"}


def _short(uri: str | None) -> str | None:
    """Reduce an IPTC digitalSourceType URI to its final segment."""
    return uri.rstrip("/").rsplit("/", 1)[-1] if uri else None


def _actions(manifest: dict):
    """Yield every action dict across c2pa.actions and c2pa.actions.v2."""
    for assertion in manifest.get("assertions") or []:
        if assertion.get("label", "").startswith("c2pa.actions"):
            for action in (assertion.get("data") or {}).get("actions") or []:
                yield action


class CheckC2PAProvenance(Action):
    """Reads C2PA provenance metadata embedded in an image or video and reports
    whether the file's own manifest declares it as AI-generated, edited, or
    captured. Absence of a manifest is common and is NOT evidence that the
    media is authentic; presence of an invalid/unvalidated manifest means the
    declared provenance is a claim, not a verified fact."""

    name = "check_c2pa_provenance"
    media_requirement = MediaRequirement.IMAGE_OR_VIDEO

    def __init__(self, media: str):
        """Args:
        media: reference to the image or video to check (must be in the item registry)
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

    def __eq__(self, other):
        return isinstance(other, CheckC2PAProvenance) and self.media == other.media

    def __hash__(self):
        return hash((self.name, self.media))


@dataclass
class SourceTypeEntry:
    source_type: str
    category: str
    description: str
    action: str | None
    software_agent: str | None
    active_manifest: bool


@dataclass
class C2PAProvenanceResults(Results):
    ai_generated: bool | None = None  # True / False / None (unknown)
    verdict: str = "unknown"
    provenance: str = "absent"  # absent | valid | invalid
    validation_state: str | None = None
    source_types: list[SourceTypeEntry] = field(default_factory=list)
    signer: str | None = None
    signed_at: str | None = None
    generator: str | None = None
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def __str__(self) -> str:
        if self.error:
            return f"C2PA check failed: {self.error}"
        lines = [f"C2PA provenance: **{self.verdict}** (manifest {self.provenance})."]
        if self.generator:
            lines.append(f"Claim generator: {self.generator}.")
        if self.signer:
            signed = f" at {self.signed_at}" if self.signed_at else ""
            lines.append(f"Signed by: {self.signer}{signed}.")
        for s in self.source_types:
            tag = "" if s.active_manifest else " (non-active manifest)"
            lines.append(f"- {s.source_type} -> {s.category}: {s.description}{tag}")
        for n in self.notes:
            lines.append(f"Note: {n}")
        return "\n".join(lines)

    def is_useful(self) -> bool | None:
        return self.error is None and self.provenance != "absent"


class C2PAChecker(Tool[CheckC2PAProvenance, C2PAProvenanceResults]):
    """Reads C2PA (Content Authenticity Initiative) provenance manifests from
    media files via https://github.com/contentauth/c2pa-python."""

    name = "c2pa_checker"
    description = (
        "Reads embedded C2PA provenance manifests to check whether an image or "
        "video declares itself AI-generated, edited, or captured."
    )
    actions = [CheckC2PAProvenance]

    def _perform(self, action: CheckC2PAProvenance) -> C2PAProvenanceResults:
        if action.media is None:
            return C2PAProvenanceResults(error="media not found in the item registry")

        path = action.media.file_path
        try:
            reader = Reader.try_create(str(path))
        except Exception as e:
            logger.error(f"[Tool:{self.name}] Failed to read {action.media.reference}: {e}")
            return C2PAProvenanceResults(error=str(e))

        if reader is None:
            return C2PAProvenanceResults(
                notes=[
                    "no C2PA manifest; this is the common case and says nothing about authenticity or if AI was involved"
                ]
            )

        try:
            return self._inspect(reader)
        except Exception as e:
            logger.error(f"[Tool:{self.name}] Failed to parse manifest for {action.media.reference}: {e}")
            return C2PAProvenanceResults(error=str(e))
        finally:
            reader.close()

    def _inspect(self, reader: Reader) -> C2PAProvenanceResults:
        result = C2PAProvenanceResults()

        store = json.loads(reader.json())
        state = reader.get_validation_state()
        result.validation_state = state
        result.provenance = "valid" if state == "Valid" else "invalid"
        if result.provenance == "invalid":
            result.notes.append(
                f"manifest present but validation state is {state!r}; "
                "signature broken, untrusted issuer, or asset modified after signing"
            )

        manifests = store.get("manifests") or {}
        active_label = store.get("active_manifest")

        # Walk every manifest in the store, not just the active one: an edit chain
        # can carry an AI-generated ingredient under a human-signed top manifest.
        for label, manifest in manifests.items():
            is_active = label == active_label
            for action in _actions(manifest):
                key = _short(action.get("digitalSourceType"))
                if not key:
                    continue
                category, description = SOURCE_TYPES.get(
                    key, ("unrecognized", "source type not in the known vocabulary")
                )
                agent = action.get("softwareAgent")
                if isinstance(agent, dict):
                    agent = " ".join(filter(None, [agent.get("name"), agent.get("version")]))
                result.source_types.append(
                    SourceTypeEntry(
                        source_type=key,
                        category=category,
                        description=description,
                        action=action.get("action"),
                        software_agent=agent,
                        active_manifest=is_active,
                    )
                )

        active = manifests.get(active_label) or {}
        sig = active.get("signature_info") or {}
        result.signer = sig.get("issuer") or sig.get("common_name")
        result.signed_at = sig.get("time")
        gens = active.get("claim_generator_info") or []
        result.generator = ", ".join(g.get("name", "") for g in gens if isinstance(g, dict)) or None

        categories = {s.category for s in result.source_types}
        if categories & AI_CATEGORIES:
            result.ai_generated = True
            result.verdict = "ai_partial" if categories == {"ai_partial"} else "ai_generated"
        elif categories & {"captured", "human_created"}:
            result.ai_generated = False
            result.verdict = "not_ai_declared"
        elif categories:
            result.verdict = sorted(categories)[0]
            result.notes.append("source type present but not a clear AI or capture declaration")
        else:
            result.notes.append("manifest present but declares no digitalSourceType")

        # A verdict from a manifest that failed validation is a claim, not a fact.
        if result.provenance == "invalid" and result.ai_generated is not None:
            result.notes.append("verdict comes from an unvalidated manifest; treat as a lead only")

        return result

    def _summarize(self, result: C2PAProvenanceResults, **kwargs) -> MultimodalSequence | None:
        if result.error is not None:
            return None
        if not result.is_useful():
            return None
        return MultimodalSequence(str(result))
