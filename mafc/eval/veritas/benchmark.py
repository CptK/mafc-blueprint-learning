import json
from datetime import datetime
import math
from pathlib import Path
from typing import Any

from ezmm import Image, Video

from mafc.common.logger import logger
from mafc.common.claim import Claim
from mafc.eval.benchmark import Benchmark
from mafc.common.label import BaseLabel
from mafc.eval.veritas.types import ClaimEntry, MediaScore
from mafc.eval.veritas.types import VeriTaSBenchmarkSample
from mafc.eval.metrics import save_confusion_matrix_png
from mafc.eval.veritas.metrics import compute_veritas_metrics, format_veritas_metrics_report
from mafc.eval.veritas.labels import (
    Veritas3Label,
    Veritas7Label,
    CLASS_MAPPING_3,
    CLASS_DEFINITIONS_3,
    CLASS_MAPPING_7,
    CLASS_DEFINITIONS_7,
    EXTRA_JUDGE_RULES_3,
    EXTRA_JUDGE_RULES_7,
    THRESHOLDS_7,
)


def _classify_integrity_7(score: float) -> Veritas7Label:
    """Classify an integrity score into a 7-class label using threshold bins."""
    for threshold, label in THRESHOLDS_7:
        if score < threshold:
            return label
    return Veritas7Label.INTACT_CERTAIN


class VeriTaS(Benchmark[VeriTaSBenchmarkSample]):
    name = "VeriTaS"
    shorthand = "veritas"

    is_multimodal = True

    # Defaults (overridden in __init__ based on label_scheme)
    class_mapping = CLASS_MAPPING_3
    class_definitions = CLASS_DEFINITIONS_3

    extra_prepare_rules = """**Multimodal Integrity Assessment**: Evaluate the overall integrity of the claim by considering:
    - **Veracity**: Is the textual claim factually accurate?
    - **Media Authenticity**: Are any images/videos genuine or manipulated?
    - **Media Contextualization**: Is the media used in the proper context or taken out of context?
    The overall integrity combines all these factors."""

    extra_plan_rules = """* **Comprehensive Verification**: For each claim, verify:
    1. The factual accuracy of the text claim (use web search)
    2. The authenticity of any referenced media (use reverse image search)
    3. The proper contextualization of media (verify the original context)
    * **Multimodal Claims**: Pay special attention to claims with media - verify both text and visual content.
    """

    extra_judge_rules = EXTRA_JUDGE_RULES_3

    available_actions = []

    # Thresholds for three-class classification
    INTACT_THRESHOLD = 0.33
    COMPROMISED_THRESHOLD = -0.33

    def __init__(self, data_path: str, variant: str = "q1_2024", label_scheme: int = 3):
        """
        Initialize VeriTaS benchmark.

        Args:
            variant: The quarter to use, e.g., 'q1_2024', 'q2_2024', 'q3_2024', 'q4_2024',
                     or 'longitudinal' for the new longitudinal format
            data_path: Optional explicit path to a claims.json file or directory containing it.
                       If provided, this overrides the default path resolution.
            label_scheme: Number of classes for the label scheme (3 or 7). Default: 3.
        """
        if label_scheme not in (3, 7):
            raise ValueError(f"Unsupported label_scheme: {label_scheme}. Choose 3 or 7.")

        self.label_scheme = label_scheme

        # Configure class mapping, definitions, and judge rules based on label scheme
        if label_scheme == 7:
            self.class_mapping = CLASS_MAPPING_7
            self.class_definitions = CLASS_DEFINITIONS_7
            self.extra_judge_rules = EXTRA_JUDGE_RULES_7
        else:
            self.class_mapping = CLASS_MAPPING_3
            self.class_definitions = CLASS_DEFINITIONS_3
            self.extra_judge_rules = EXTRA_JUDGE_RULES_3

        self.variant = variant

        if (path := Path(data_path)).is_file():
            self.file_path = path
            self.data_dir = path.parent
        else:
            self.data_dir = path
            self.file_path = path / "claims.json"

        if not self.file_path.exists():
            raise ValueError(f"Claims file not found at {self.file_path}")

        # Skip parent __init__ file_path handling since we set it directly
        self.full_name = f"{self.name} ({variant})"
        self.data = self._load_data()

    def sample_extra_fields(self, sample: VeriTaSBenchmarkSample) -> dict[str, Any]:
        return {
            "gt_integrity_score": sample.gt_score,
            "gt_veracity": sample.gt_veracity,
            "gt_context_coverage": sample.gt_context_coverage,
        }

    def compute_metrics(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        return compute_veritas_metrics(results, label_scheme=self.label_scheme)

    def format_metrics_report(self, metrics: dict[str, Any]) -> str:
        return format_veritas_metrics_report(metrics, label_scheme=self.label_scheme)

    def save_metric_plots(self, metrics: dict[str, Any], run_dir: Path) -> list[Path]:
        written: list[Path] = []
        cm = metrics.get("confusion_matrix") or {}
        if cm:
            n = self.label_scheme
            path = run_dir / f"confusion_matrix_{n}class.png"
            save_confusion_matrix_png(
                cm,
                path,
                title=f"Confusion Matrix ({n}-class)",
                subtitle=f"accuracy={metrics.get('accuracy', 0):.1%}",
            )
            written.append(path)
        coarsened = metrics.get("coarsened_3class") or {}
        cm3 = coarsened.get("confusion_matrix") or {}
        if cm3:
            path3 = run_dir / "confusion_matrix_3class_coarsened.png"
            save_confusion_matrix_png(
                cm3,
                path3,
                title="Confusion Matrix (3-class coarsened)",
                subtitle=f"accuracy={coarsened.get('accuracy', 0):.1%}",
            )
            written.append(path3)
        return written

    def _get_integrity_score(self, claim_entry: ClaimEntry) -> float | None:
        """Extract integrity score from claim entry, handling both formats."""
        integrity_obj = claim_entry.get("integrity")
        if integrity_obj is None:
            return None
        raw_score = integrity_obj.get("score") if isinstance(integrity_obj, dict) else integrity_obj
        if raw_score is None:
            return None
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            claim_id = claim_entry.get("id", "<unknown>")
            logger.warning(
                f"[VeriTaS] Invalid integrity score for claim {claim_id}: {raw_score!r}. Falling back to UNKNOWN."
            )
            return None
        if not math.isfinite(score):
            claim_id = claim_entry.get("id", "<unknown>")
            logger.warning(
                f"[VeriTaS] Non-finite integrity score for claim {claim_id}: {score!r}. Falling back to UNKNOWN."
            )
            return None
        return score

    def _get_claim_text(self, claim_entry: ClaimEntry) -> str:
        """Extract claim text from claim entry, handling both formats."""
        return claim_entry.get("text", "")

    def _build_claim_text_with_media(self, claim_entry: ClaimEntry, claim_id: str) -> str:
        """
        Build claim text with inline media references for the longitudinal format.

        The longitudinal format has media as a separate array, but DEFAME expects
        inline references like <image:ID> in the claim text.
        """
        claim_text = self._get_claim_text(claim_entry)

        # Build media references from the media array
        media_list = claim_entry.get("media", [])
        media_refs = []

        for media_item in media_list:
            media_type = media_item.get("type")
            media_id = media_item.get("id")

            if media_type and media_id:
                media_refs.append(f"<{media_type}:{media_id}>")

        # Prepend media references to the claim text
        if media_refs:
            claim_text = " ".join(media_refs) + " " + claim_text

        return claim_text

    def _get_media_path(self, media_type: str, media_id: str) -> Path:
        """Get the path to a media file for the longitudinal format."""
        extension = "jpg" if media_type == "image" else "mp4"
        return self.data_dir / f"{media_type}s" / f"{media_id}.{extension}"

    def _get_justification(self, claim_entry: ClaimEntry) -> dict:
        """Extract justification/ground truth info from claim entry."""
        integrity_obj: float | MediaScore | None = claim_entry.get("integrity")
        integrity_score = integrity_obj.get("score") if isinstance(integrity_obj, dict) else integrity_obj

        # Build media verdicts from the new format
        media_verdicts = []
        for media_item in claim_entry.get("media", []):
            authenticity: float | MediaScore | None = media_item.get("authenticity")
            contextualization: float | MediaScore | None = media_item.get("contextualization")
            media_verdicts.append(
                {
                    "media_id": media_item.get("id"),
                    "media_type": media_item.get("type"),
                    "authenticity": (
                        authenticity.get("score") if isinstance(authenticity, dict) else authenticity
                    ),
                    "contextualization": (
                        contextualization.get("score")
                        if isinstance(contextualization, dict)
                        else contextualization
                    ),
                }
            )

        veracity_obj = claim_entry.get("veracity")
        context_obj = claim_entry.get("context_coverage")

        return {
            "veracity": veracity_obj.get("score") if isinstance(veracity_obj, dict) else veracity_obj,
            "context_coverage": context_obj.get("score") if isinstance(context_obj, dict) else context_obj,
            "integrity": integrity_score,
            "media_verdicts": media_verdicts,
        }

    def _load_data(self) -> list[VeriTaSBenchmarkSample]:
        """Load claims from the VeriTaS dataset."""
        logger.info(f"[VeriTaS] Opening claims file: {self.file_path}")
        with open(self.file_path, "r") as f:
            data_raw = json.load(f)

        metadata = data_raw.get("metadata", {})
        claims = data_raw.get("claims", [])

        logger.info(
            f"[VeriTaS] Loading {metadata.get('total_claims', len(claims))} claims from VeriTaS {self.variant}..."
        )

        data: list[VeriTaSBenchmarkSample] = []
        skipped_claims = 0
        for claim_entry in claims:
            claim_id = str(claim_entry["id"])

            # Build claim text with inline media references
            claim_text = self._build_claim_text_with_media(claim_entry, claim_id)

            # IMPORTANT: Register media files BEFORE creating the Claim object
            # This ensures media references are resolvable when Claim validates them
            try:
                claim_text = self._register_media(claim_text, claim_id, claim_entry)
            except Exception as e:
                logger.error(f"[VeriTaS] ERROR registering media for claim {claim_id}: {e}")
                skipped_claims += 1
                continue

            # Create Claim object (will now validate media references successfully)
            date_str = claim_entry.get("date")
            try:
                date = datetime.fromisoformat(date_str.replace("Z", "+00:00")) if date_str else None
            except Exception as e:
                logger.error(f"[VeriTaS] ERROR parsing date '{date_str}' for claim {claim_id}: {e}")
                date = None

            try:
                claim = Claim(claim_text, date=date, id=claim_id)
            except Exception as e:
                logger.error(f"[VeriTaS] ERROR creating Claim object for claim {claim_id}: {e}")
                logger.error(f"[VeriTaS] Claim text: {claim_text}")
                skipped_claims += 1
                continue

            # Label based on integrity score (3-class or 7-class)
            integrity = self._get_integrity_score(claim_entry)
            if integrity is None:
                logger.warning(f"[VeriTaS] Skipping claim {claim_id}: missing integrity score.")
                skipped_claims += 1
                continue
            elif self.label_scheme == 7:
                label: Veritas7Label | Veritas3Label = _classify_integrity_7(integrity)
            elif integrity >= self.INTACT_THRESHOLD:
                label = Veritas3Label.INTACT
            elif integrity <= self.COMPROMISED_THRESHOLD:
                label = Veritas3Label.COMPROMISED
            else:
                label = Veritas3Label.UNKNOWN

            gt = self._get_justification(claim_entry)

            data.append(
                VeriTaSBenchmarkSample(
                    id=claim_id,
                    input=claim,
                    label=label,
                    justification=gt,
                    gt_score=integrity,
                    gt_veracity=gt["veracity"],
                    gt_context_coverage=gt["context_coverage"],
                    gt_media_verdicts=gt["media_verdicts"],
                )
            )

        logger.info(f"[VeriTaS] Successfully loaded {len(data)} claims")
        logger.info("[VeriTaS] Label distribution:")
        label_counts: dict[BaseLabel, int] = {}
        for item in data:
            lbl = item.label
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        for lbl, count in label_counts.items():
            logger.info(f"[VeriTaS]   {lbl.value}: {count}")
        if skipped_claims:
            logger.warning(f"[VeriTaS] Skipped {skipped_claims} claim(s) due to media/claim parsing errors.")

        return data

    def _register_media(self, claim_text: str, claim_id: str, claim_entry: ClaimEntry | None = None) -> str:
        """
        Register media files referenced in the claim text with the media registry.
        MUST be called BEFORE creating the Claim object.

        Args:
            claim_text: The claim text potentially containing <image:ID> or <video:ID> references
            claim_id: The claim ID for logging purposes
            claim_entry: Optional claim entry dict (used for format-specific path resolution)

        Returns:
            The claim text with media properly registered
        """
        import re

        # Find all media references in the text and enforce numeric IDs only.
        media_pattern = r"<(image|video):([^>]+)>"
        matches = re.findall(media_pattern, claim_text)

        registered_refs = []
        for media_type, media_id in matches:
            if not media_id.isdigit():
                msg = (
                    f"Non-numeric {media_type} ID in claim {claim_id}: '{media_id}'. "
                    "Expected numeric IDs in <image:...>/<video:...> references."
                )
                logger.error(f"[VeriTaS] {msg}")
                raise ValueError(msg)

            # Construct path to media file (longitudinal format)
            media_path = self._get_media_path(media_type, media_id)

            if media_path.exists():
                try:
                    # Register the media with the global media registry
                    # This creates the media object and assigns it a registry ID
                    if media_type == "image":
                        media_obj = Image(media_path)
                    elif media_type == "video":
                        media_obj = Video(media_path)
                    else:
                        logger.warning(f"[VeriTaS]   WARNING: Unknown media type '{media_type}'")
                        logger.warning(f"Unknown media type '{media_type}' in claim {claim_id}")
                        continue

                    # Store the reference for replacement if needed
                    old_ref = f"<{media_type}:{media_id}>"
                    new_ref = media_obj.reference
                    registered_refs.append((old_ref, new_ref))
                except Exception as e:
                    logger.error(f"[VeriTaS] ERROR registering media {media_path}: {e}")
                    logger.error(f"Error registering media {media_path} for claim {claim_id}: {e}")
                    raise
            else:
                logger.warning(f"[VeriTaS] WARNING: Media file not found for claim {claim_id}: {media_path}")
                logger.warning(f"Media file not found for claim {claim_id}: {media_path}")

        # Replace old references with new registry references if different
        for old_ref, new_ref in registered_refs:
            if old_ref != new_ref:
                claim_text = claim_text.replace(old_ref, new_ref)

        return claim_text
