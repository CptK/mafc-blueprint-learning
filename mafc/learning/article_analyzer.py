from __future__ import annotations

import json
import re

from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt
from mafc.learning.models import ActionEvidenceLink, ArticleAnalysis
from mafc.utils.parsing import extract_json_object, strip_json_fences, try_parse_with_repair

_SYSTEM_PROMPT = """\
You are an expert analyst specializing in misinformation research and fact-checking methodology.
Your task is to analyze fact-check articles and extract structured information about the \
claim investigated, the evidence found, and — where described — the investigative process used.

Be precise and grounded: only extract what is actually present in the article. \
Do not invent steps or evidence that are not mentioned or strongly implied.\
"""

_RECTIFICATION_NOTE = """\

Note: the claim above is a corrected version of the original claim that the article debunks. \
The original (false) claim was: "{original_claim}"
The article investigates the original claim; the evidence and reasoning it contains still \
apply — just in the direction of confirming the corrected claim rather than refuting it.\
"""

_USER_PROMPT_TEMPLATE = """\
Analyze the following fact-check article with respect to the specific claim under investigation.

---CLAIM---
{claim_text}
---END CLAIM---

---ARTICLE---
{article_content}
---END ARTICLE---

The article may cover multiple claims or a broader topic. Focus your analysis \
only on the evidence and reasoning that is directly relevant to the specific claim above. \
Ignore parts of the article that address other claims.
{rectification_note}
Return a JSON object with these fields:

{{
  "claim_type": string,
  // High-level category of the claim being checked.
  // Use one of: "media_authenticity", "quote_attribution", "event_claim",
  // "statistic", "identity_claim", "context_manipulation", "other".

  "verdict_summary": string,
  // 1-2 sentences explaining why the claim is true / false / misleading.

  "key_evidence": [string, ...],
  // Concrete pieces of evidence cited. E.g.:
  // "Video geotag matches Ghana school fire from 2019, not a Nigerian church"
  // "AFP photographer confirmed image was taken before the event in question"

  "evidence_types": [string, ...],
  // Abstract action types that produced the evidence. Infer these from conclusions
  // when the process is not explicitly described. Use values like:
  // "reverse_image_search", "reverse_video_search", "geolocation",
  // "source_lookup", "metadata_analysis", "expert_consultation",
  // "official_records", "social_media_search", "web_search",
  // "date_verification", "quote_verification", "context_check"
  //
  // Critical distinctions — classify by HOW the evidence was obtained, not by WHAT was verified:
  //
  // "expert_consultation" — the fact-checker ACTIVELY CONTACTED a person or institution
  //   (official, expert, spokesperson, or any named individual) to request a statement,
  //   confirmation, denial, or opinion. Use this whenever someone was reached out to,
  //   regardless of whether the subject is a quote, an event, an official matter, or a
  //   health claim. Examples: emailing an embassy, calling a police officer, messaging a
  //   UNICEF spokesperson on WhatsApp, interviewing a doctor.
  //
  // "official_records" — the fact-checker LOOKED UP an existing published document,
  //   database, or registry WITHOUT active outreach to a person. Examples: checking court
  //   filings, searching a government statistics portal, reviewing a published official
  //   advisory, consulting a public register.
  //
  // "quote_verification" — the fact-checker located and examined a PRIMARY SOURCE
  //   (transcript, archived article, video recording) to check whether a specific quote
  //   or statement actually appears in it, without contacting anyone. Examples: finding
  //   the original speech on YouTube, pulling the archived newspaper article, reading the
  //   original interview transcript.
  //
  // "source_lookup" — finding the original source of a piece of media or information
  //   (image, video, news story) to establish provenance. Distinct from expert_consultation
  //   (no active outreach) and from reverse_image_search (no image search tool used).

  "action_evidence_links": null | [
    {{
      "action": string,          // same vocabulary and definitions as evidence_types above
      "finding": string,         // what the action revealed
      "query_or_input": string | null,  // search query or input used, if mentioned
      "was_decisive": boolean    // did this directly determine the verdict?
    }},
    ...
  ],
  // Ordered mapping from actions to findings. Set to null if the article does not
  // describe the investigative process at all (result_only).

  "investigative_steps": null | [string, ...],
  // High-level ordered steps taken. E.g.:
  // ["Ran reverse image search on the video thumbnail",
  //  "Cross-referenced location metadata with known footage databases"]
  // Set to null if no process is described.

  "search_queries": null | [string, ...],
  // Specific search queries or lookup terms mentioned. Set to null if none mentioned.

  "process_richness": "full" | "partial" | "result_only",
  // "full"         — step-by-step process is described
  // "partial"      — some steps mentioned but incomplete
  // "result_only"  — only conclusions and evidence are stated

  "notes": null | string
  // Anything unusual, ambiguous, or hard to categorize.
}}

Return only the JSON object, no additional text.\
"""

_REPAIR_PROMPT = """\
The previous response was not valid JSON. \
Please return only a valid JSON object matching the required schema, with no additional text.\
"""


def _parse_article_analysis(text: str) -> ArticleAnalysis | None:
    try:
        raw = json.loads(extract_json_object(strip_json_fences(text)))
    except (json.JSONDecodeError, ValueError):
        return None

    links_raw = raw.get("action_evidence_links")
    links = None
    if links_raw is not None:
        try:
            links = [
                ActionEvidenceLink(
                    action=link["action"],
                    finding=link["finding"],
                    query_or_input=link.get("query_or_input"),
                    was_decisive=bool(link.get("was_decisive", False)),
                )
                for link in links_raw
            ]
        except (KeyError, TypeError):
            links = None

    process_richness = raw.get("process_richness", "result_only")
    if process_richness not in ("full", "partial", "result_only"):
        process_richness = "result_only"

    return ArticleAnalysis(
        claim_type=raw.get("claim_type", "other"),
        verdict_summary=raw.get("verdict_summary", ""),
        key_evidence=raw.get("key_evidence") or [],
        evidence_types=raw.get("evidence_types") or [],
        action_evidence_links=links,
        investigative_steps=raw.get("investigative_steps"),
        search_queries=raw.get("search_queries"),
        process_richness=process_richness,
        notes=raw.get("notes"),
    )


class ArticleAnalyzer:
    """Extracts structured information from a fact-check article using an LLM."""

    def __init__(self, model: Model):
        self.model = model

    def analyze(
        self,
        article_content: str,
        claim_text: str,
        original_claim: str | None = None,
        claim_id: str | None = None,
    ) -> ArticleAnalysis | None:
        """Analyze a fact-check article with respect to a specific claim.

        Args:
            article_content: The full text of the fact-check article.
            claim_text: The specific claim being investigated. For rectified claims
                this is the corrected version; pass original_claim as well in that case.
            original_claim: The original (pre-rectification) claim text, if the dataset
                claim was rectified. The article will be about this claim, not claim_text.
            claim_id: Optional identifier for logging.

        Returns None if the LLM response cannot be parsed even after repair.
        """
        label = f"[ArticleAnalyzer{f' claim={claim_id}' if claim_id else ''}]"
        # Strip <image:...> / <video:...> tokens that are scraping artifacts from
        # fact-check sites — they are not registered media items and would cause
        # Prompt (MultimodalSequence) to raise on construction.
        clean_content = re.sub(r"<(?:image|video):[^>]+>", "", article_content)
        rectification_note = (
            _RECTIFICATION_NOTE.format(original_claim=original_claim) if original_claim else ""
        )
        prompt = _USER_PROMPT_TEMPLATE.format(
            claim_text=claim_text,
            article_content=clean_content,
            rectification_note=rectification_note,
        )
        messages = [
            Message(role=MessageRole.SYSTEM, content=Prompt(text=_SYSTEM_PROMPT)),
            Message(role=MessageRole.USER, content=Prompt(text=prompt)),
        ]

        response = self.model.generate(messages)
        raw_text = response.text.strip()
        result, repair_text = try_parse_with_repair(
            response_text=raw_text,
            parse_fn=_parse_article_analysis,
            model=self.model,
            repair_prompt_prefix=_REPAIR_PROMPT,
        )

        if result is None:
            logger.warning(
                f"{label} Failed to parse article analysis after repair.\n"
                f"  Initial response ({len(raw_text)} chars) tail: {raw_text[-200:]!r}\n"
                f"  Repair response ({len(repair_text or '')} chars) tail: {(repair_text or '')[-200:]!r}"
            )
            return None

        if repair_text is not None:
            logger.debug(f"{label} Repaired JSON parse.")

        logger.debug(
            f"{label} process_richness={result.process_richness} "
            f"claim_type={result.claim_type} "
            f"evidence_types={result.evidence_types}"
        )
        return result
