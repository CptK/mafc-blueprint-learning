"""An oracle manipulation detector: returns the ground-truth integrity label.

This is an evaluation instrument, not a detector. It exists to answer one
question before any budget is spent on real detectors:

    if manipulation detection were PERFECT, how much better would the
    end-to-end pipeline actually be?

That number is the ceiling. If the oracle barely beats a run with no detector
at all, then no amount of Sightengine spend can pay for itself, and the whole
line of work is closed by a cheap experiment instead of an expensive one.

Labels come from media_integrity_labels.json (see
scripts/ablations/detector_comparison/label_media_integrity.py).

WHAT THIS DELIBERATELY DOES NOT RETURN
--------------------------------------
The labels were derived from the fact-checker's own free-text justification,
and those texts contain the entire answer — "reverse image searches confirm
Maduro's face was superimposed onto the Medvedchuk arrest photo" names the
manipulation, the source image, and the verdict. Handing that to the pipeline
would measure a perfect *researcher*, not a perfect *detector*, and would
inflate the ceiling beyond anything a real tool could reach.

So this returns only what a flawless detector could emit: a verdict, a
manipulation type, and a confidence. The justification and the stored evidence
quote are never exposed.

Even so, treat the result as a generous upper bound. The labels are drawn from
the same fact-check text that produced the benchmark's gold veracity label, so
the oracle agrees with the grader's reasoning by construction — an advantage no
real detector has.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import cast

from ezmm import MultimodalSequence
from ezmm.common.items import Image, Video
from ezmm.common.registry import item_registry

from config.globals import oracle_labels_path
from mafc.common.action import Action, MediaRequirement
from mafc.common.logger import logger
from mafc.common.results import Results
from mafc.tools.tool import Tool

# Human-readable gloss per manipulation type. Describes the *kind* of alteration
# only — never the specific content, which would leak the fact-check's finding.
TYPE_DESCRIPTIONS = {
    "ai_generated": "fully synthesized by a generative model",
    "deepfake": "a face or likeness was swapped or synthesized",
    "splice_composite": "two or more real images were combined",
    "fabricated_screenshot": "a forged rendering of a post, article, or document",
    "graphic_edit": "a real base image altered by overlay, retouch, or crop",
    "temporal_edit": "real footage selectively cut, slowed, or sped up",
    "other_manipulation": "altered after capture",
    "none": "no alteration",
}


class CheckOracleManipulation(Action):
    """Reports whether an image or video was digitally manipulated, using a
    perfect detector. Returns the verdict, the kind of manipulation, and a
    confidence. Some media is genuinely unassessed, in which case no verdict is
    available."""

    name = "check_oracle_manipulation"
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
        return isinstance(other, CheckOracleManipulation) and self.media == other.media

    def __hash__(self):
        return hash((self.name, self.media))


@dataclass
class OracleManipulationResults(Results):
    """Ground-truth integrity verdict for one media item."""

    label: str | None = None  # manipulated | authentic | unknown
    manipulation_type: str | None = None
    misleading_but_authentic: bool = False
    found: bool = False  # whether the media had a label at all
    error: str | None = None

    def __str__(self) -> str:
        if self.error:
            return f"Manipulation check failed: {self.error}"
        if not self.found or self.label == "unknown":
            return (
                "Manipulation analysis is inconclusive for this media: its provenance could not "
                "be established either way. This is not evidence that it is authentic."
            )
        if self.label == "authentic":
            text = "Manipulation analysis: the media is an UNALTERED capture, with no signs of editing or synthesis."
            if self.misleading_but_authentic:
                text += (
                    " Note this concerns the file only — an unaltered recording can still be presented "
                    "misleadingly, e.g. a staged event or a false caption, date, or location."
                )
            return text
        gloss = TYPE_DESCRIPTIONS.get(self.manipulation_type or "", "altered after capture")
        return f"Manipulation analysis: the media is MANIPULATED — {gloss}."

    def is_useful(self) -> bool | None:
        # An unassessed item is not a finding, exactly as with a real detector
        # that returns nothing.
        return self.error is None and self.found and self.label != "unknown"


class OracleManipulationChecker(Tool[CheckOracleManipulation, OracleManipulationResults]):
    """A perfect manipulation detector, for ceiling experiments only.

    NEVER enable this in a real run: it reads the answer key. It exists to
    measure how much a perfect detector would be worth, so that the value of
    real (paid) detectors can be judged against that ceiling.
    """

    name = "oracle_manipulation"
    description = (
        "Analyzes whether an image or video has been digitally manipulated, edited, or "
        "synthetically generated, and reports what kind of manipulation was found."
    )
    actions = [CheckOracleManipulation]

    def __init__(self, labels_path: str | Path | None = None, **kwargs):
        """Args:
        labels_path: media_integrity_labels.json. Defaults to the
            `oracle_labels_path` env var.
        """
        super().__init__(**kwargs)
        path = Path(labels_path) if labels_path else oracle_labels_path
        if path is None:
            raise ValueError(
                "OracleManipulationChecker needs a labels file; set oracle_labels_path or pass labels_path"
            )
        self.labels_path = Path(path)
        self._labels: dict[str, dict] | None = None
        logger.warning(
            f"[Tool:{self.name}] ORACLE ENABLED — reading ground-truth labels from {self.labels_path}. "
            "This is a ceiling experiment; results are not valid pipeline performance."
        )

    @property
    def labels(self) -> dict[str, dict]:
        if self._labels is None:
            raw = json.loads(self.labels_path.read_text())
            self._labels = raw.get("labels", {})
            logger.info(f"[Tool:{self.name}] loaded {len(self._labels)} ground-truth labels")
        return self._labels

    def _perform(self, action: CheckOracleManipulation) -> OracleManipulationResults:
        if action.media is None:
            return OracleManipulationResults(error="media not found in the item registry")

        # Dataset media is stored as <media_id>.<ext>, so the stem is the id the
        # labels are keyed by.
        media_id = Path(action.media.file_path).stem
        record = self.labels.get(media_id)
        if record is None:
            logger.warning(f"[Tool:{self.name}] no ground-truth label for media id {media_id!r}")
            return OracleManipulationResults(found=False)

        return OracleManipulationResults(
            label=record.get("label"),
            manipulation_type=record.get("manipulation_type"),
            misleading_but_authentic=bool(record.get("misleading_but_authentic")),
            found=True,
        )

    def _summarize(self, result: OracleManipulationResults, **kwargs) -> MultimodalSequence | None:
        if result.error is not None:
            return None
        return MultimodalSequence(str(result))
