from __future__ import annotations

import json

from ezmm import MultimodalSequence
from pydantic import BaseModel, ConfigDict

from mafc.common.claim import Claim
from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt
from mafc.utils.parsing import extract_json_object

_EXTRACTION_PROMPT = """\
Classify what this fact-checking claim is ABOUT. The claim may be in any language; \
judge the meaning, not the wording.

Answer each question with exactly one of: "yes", "no", "unknown". Use "unknown" \
whenever the claim does not give you enough to decide — a wrong "yes" or "no" \
misroutes the investigation, while "unknown" is always safe.

- asserts_place_or_date: does the claim assert WHERE or WHEN the attached media was \
captured, or when the depicted event happened?
- asserts_identity: does the claim assert WHO is depicted or involved?
- asserts_synthetic_origin: is the claim about the media being AI-generated, edited, \
staged, or otherwise manipulated?
- asserts_recontextualization: is the claim about media being old, unrelated, or taken \
from a different event than the one it is presented as showing?
- is_document_screenshot: is the attached media a screenshot of a document, post, \
headline, poll, chart, or announcement (rather than a photo/video of a scene)?
- is_quote_attribution: does the claim assert that a person said, wrote, or stated \
something?
- is_statistical: is the core of the claim a number, rate, share, or statistic?
- is_scientific_medical: is the claim about health, medicine, or a scientific finding?

Return strict JSON only, no other text:
{"asserts_place_or_date":"yes|no|unknown","asserts_identity":"...",\
"asserts_synthetic_origin":"...","asserts_recontextualization":"...",\
"is_document_screenshot":"...","is_quote_attribution":"...",\
"is_statistical":"...","is_scientific_medical":"..."}

Claim:
"""

_TRISTATE = {"yes": True, "no": False, "unknown": None, "true": True, "false": False}

_MAX_EXTRACTION_ATTEMPTS = 2


class SemanticFeatureResponse(BaseModel):
    """Validated tri-state payload returned by the semantic feature extractor."""

    model_config = ConfigDict(extra="ignore")

    asserts_place_or_date: str | None = None
    asserts_identity: str | None = None
    asserts_synthetic_origin: str | None = None
    asserts_recontextualization: str | None = None
    is_document_screenshot: str | None = None
    is_quote_attribution: str | None = None
    is_statistical: str | None = None
    is_scientific_medical: str | None = None

    def to_feature_values(self) -> dict[str, bool | None]:
        """Map the raw yes/no/unknown strings onto tri-state booleans."""
        return {
            key: _TRISTATE.get((value or "").strip().lower())
            for key, value in self.model_dump().items()
        }


class SemanticFeatureExtractor:
    """Single cheap LLM pass that labels what a claim is about, in any language."""

    def __init__(self, model: Model):
        """Initialize the extractor with the model used for the classification call."""
        self.model = model

    def extract(self, claim: Claim | MultimodalSequence | str) -> dict[str, bool | None]:
        """Return tri-state semantic features, or an empty map if extraction fails.

        Failure is never fatal: an empty result leaves every semantic entry condition
        undetermined and therefore non-eliminating, so selection degrades to the LLM
        tie-break. It is still retried, because silently losing the gating signal
        pushes claims back onto the tie-break that entry conditions could have decided.
        """
        claim_text = str(claim).strip()
        if not claim_text:
            return {}

        prompt = Prompt(text=_EXTRACTION_PROMPT + claim_text)
        for attempt in range(1, _MAX_EXTRACTION_ATTEMPTS + 1):
            try:
                raw = self.model.generate(
                    [Message(role=MessageRole.USER, content=prompt)]
                ).text.strip()
                if raw.startswith("```"):
                    raw = "\n".join(
                        line for line in raw.splitlines() if not line.startswith("```")
                    ).strip()
                payload = json.loads(extract_json_object(raw))
                return SemanticFeatureResponse.model_validate(payload).to_feature_values()
            except (json.JSONDecodeError, ValueError, AttributeError) as exc:
                if attempt >= _MAX_EXTRACTION_ATTEMPTS:
                    logger.warning(
                        f"Semantic feature extraction failed after {attempt} attempts; "
                        f"all features stay undetermined. {type(exc).__name__}: {exc}"
                    )
                    return {}
        return {}
