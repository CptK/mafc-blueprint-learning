"""Sightengine AI-generated / deepfake / AI-speech detection, wired into the
mafc tool interface.

Runs images through Sightengine's `genai` and `deepfake` models
(https://sightengine.com/docs/) via the synchronous check endpoint, and videos
through the matching video endpoint plus (best-effort) an `ai_speech` check on
the video's extracted audio track. All three are pixel/waveform-based
classifiers: they ignore metadata, EXIF tags, C2PA provenance and invisible
watermarks, so their verdicts are an independent signal from (and can
disagree with) the C2PA and TruFor checkers. Their published threshold (0.5)
is a tunable starting point, not proof of authenticity either way:
  * `genai` targets fully synthetic media — it will not reliably flag real
    footage/images that were only edited.
  * `deepfake` targets identity-level face manipulation in photographic media
    — it will not reliably flag fully AI-generated faces, drawings, or
    cartoons, and is only meaningful when a face is present in the frame.
  * `ai_speech` targets fully synthetic speech in the audio track — it is
    gated to enterprise Sightengine accounts, so it may fail with an
    access-denied error on other plans; that failure is reported as a note,
    not treated as a fatal error for the rest of the check.
    Note: the `ai_speech` check requires Sightengine Enterprise Plan and is therefore not activated by default

Videos up to ~60s are scored via Sightengine's synchronous video endpoint in
one request. Longer videos use Sightengine's async video endpoint instead
(same per-frame scoring; the video file is submitted whole rather than us
extracting/sampling frames locally first). Either way, Sightengine samples
frames from the video at its own server-side rate — not every frame — and
returns a score per sampled frame; the result reports how many frames it
scored. For the async job specifically: the video is submitted
without a callback_url, and results are retrieved by polling
`video/byid.json` until the job finishes. If the account has no access to
that async feature (e.g. "not available on the free plan"), this falls back
to trimming the video to its first MAX_SYNC_VIDEO_SECONDS via ffmpeg and
scoring that via the sync endpoint instead — noted explicitly in the result,
since the remainder of the video then goes unchecked. The `ai_speech` check
requires the `ffmpeg` binary on PATH to extract the audio track; if it's
missing, or the video has no audio track, the score is simply omitted with
a note.

Because every check is a paid remote API call, results are looked up in
precomputed stores first and only fetched when missing; newly fetched scores
are cached, so scoring a whole dataset is a one-time cost (see precompute.py).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time
from typing import cast, NamedTuple

import requests
from ezmm import MultimodalSequence
from ezmm.common.items import Image, Video
from ezmm.common.registry import item_registry

from config.globals import sightengine_cache_dir, sightengine_stores
from mafc.common.action import Action, MediaRequirement
from mafc.common.logger import logger
from mafc.common.results import Results
from mafc.tools.tool import Tool

from .store import SightengineRecord, SightengineStore, file_sha256

IMAGE_CHECK_URL = "https://api.sightengine.com/1.0/check.json"
VIDEO_CHECK_SYNC_URL = "https://api.sightengine.com/1.0/video/check-sync.json"
VIDEO_CHECK_ASYNC_URL = "https://api.sightengine.com/1.0/video/check.json"
VIDEO_BY_ID_URL = "https://api.sightengine.com/1.0/video/byid.json"
AUDIO_CHECK_URL = "https://api.sightengine.com/1.0/audio/check.json"

# Sightengine's own documented starting point for all three models ("scores
# above 0.5 typically indicate ..."); tune to taste, not a hard cutoff.
DEFAULT_AI_GENERATED_THRESHOLD = 0.5
DEFAULT_DEEPFAKE_THRESHOLD = 0.5
DEFAULT_AI_SPEECH_THRESHOLD = 0.5

# The sync video endpoint is documented for videos under ~1 minute; longer
# videos are dispatched to the async endpoint + polling instead (see
# `_score_video_async`).
MAX_SYNC_VIDEO_SECONDS = 60

FFMPEG_TIMEOUT_SECONDS = 60

# Defaults for polling the async video job to completion.
DEFAULT_ASYNC_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_ASYNC_MAX_WAIT_SECONDS = 600.0

# Score below which a generator match is treated as noise and dropped.
MIN_GENERATOR_SCORE = 0.05

# How per-frame video scores collapse into one score for the whole video.
_AGGREGATIONS: dict[str, Callable[[list[float]], float]] = {
    "max": max,
    "mean": lambda s: sum(s) / len(s),
    "median": lambda s: statistics.median(s),
}


class _CredentialsMissing(RuntimeError):
    """Raised when no Sightengine API credentials are configured."""


class _VideoScores(NamedTuple):
    """Intermediate result of scoring a video's visual frames, before the
    ai_speech score (checked separately) is folded in."""

    ai_generated_score: float | None
    deepfake_score: float | None
    n_frames: int
    note: str | None
    error: str | None


class _SightengineScores(NamedTuple):
    """Raw model outputs for one media file, before thresholds/verdict are
    applied. This is what gets persisted (as a SightengineRecord) and what the
    result presentation is rebuilt from, so a cached result is identical to a
    freshly fetched one."""

    ai_generated_score: float | None
    deepfake_score: float | None
    ai_speech_score: float | None
    ai_generators: dict[str, float] | None
    n_frames: int | None
    notes: list[str]  # dynamic notes only (truncation, async, ai_speech failure)
    error: str | None


def _top_generator(ai_generators: dict[str, float] | None) -> tuple[str | None, float | None]:
    """Highest-scoring generator match, or (None, None) if there is none above
    the noise floor."""
    if not ai_generators:
        return None, None
    name, score = max(ai_generators.items(), key=lambda kv: kv[1])
    return (name, score) if score > MIN_GENERATOR_SCORE else (None, None)


class SightengineDetectionAction(Action):
    """Runs Sightengine's genai/deepfake/ai_speech pixel- and waveform-based
    classifiers against an image or short video. Says nothing about metadata,
    C2PA provenance, or whether content is used in a misleading context."""

    name = "sightengine_detection"
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
        return isinstance(other, SightengineDetectionAction) and self.media == other.media

    def __hash__(self):
        return hash((self.name, self.media))


@dataclass
class SightengineDetectionResults(Results):
    """All scores (ai_generated, deepfake, ai_speech) are on a 0-1 scale, where
    higher means more likely AI-generated/deepfaked/synthetic-speech; they are
    not probabilities. `ai_involved` and `verdict` are derived, read-only
    properties (not stored fields) so they can never drift from the scores and
    thresholds they're computed from."""

    ai_generated_score: float | None = None  # 0-1, higher = more likely AI-generated
    deepfake_score: float | None = None  # 0-1, higher = more likely a deepfake
    ai_speech_score: float | None = None  # 0-1, higher = more likely synthetic speech; videos only, None if not checked
    ai_generated_threshold: float = DEFAULT_AI_GENERATED_THRESHOLD
    deepfake_threshold: float = DEFAULT_DEEPFAKE_THRESHOLD
    ai_speech_threshold: float = DEFAULT_AI_SPEECH_THRESHOLD
    top_generator: str | None = None  # highest-scoring entry in ai_generators, if Sightengine returned one
    top_generator_score: float | None = None  # 0-1
    n_frames: int | None = None  # videos: number of frames scored
    aggregation: str | None = None  # videos: how frame scores were combined
    from_cache: bool = False  # served from a precomputed store / cache rather than a live API call
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def is_useful(self) -> bool:
        return self.error is None and (
            self.ai_generated_score is not None
            or self.deepfake_score is not None
            or self.ai_speech_score is not None
        )

    def _triggered_flags(self) -> list[str]:
        """Which of the checked models crossed their threshold, in a fixed order."""
        flags = []
        if self.ai_generated_score is not None and self.ai_generated_score >= self.ai_generated_threshold:
            flags.append("ai_generated")
        if self.deepfake_score is not None and self.deepfake_score >= self.deepfake_threshold:
            flags.append("deepfake")
        if self.ai_speech_score is not None and self.ai_speech_score >= self.ai_speech_threshold:
            flags.append("ai_speech")
        return flags

    @property
    def ai_involved(self) -> bool | None:
        """True if any checked model (ai_generated, deepfake, ai_speech) crossed its
        threshold; False if scores exist but none did; None if nothing was scored."""
        if not self.is_useful():
            return None
        return bool(self._triggered_flags())

    @property
    def verdict(self) -> str:
        if not self.is_useful():
            return "unknown"
        flags = self._triggered_flags()
        return "+".join(flags) if flags else "no_signal"

    def __str__(self) -> str:
        if self.error is not None:
            return f"Sightengine check failed: {self.error}"
        if not self.is_useful():
            return "No useful results"
        parts = [f"Sightengine verdict: **{self.verdict}**. All scores are on a 0-1 scale (higher = more likely)."]
        if self.ai_generated_score is not None:
            parts.append(
                f"AI-generated score: {self.ai_generated_score:.3f} (threshold {self.ai_generated_threshold:.2f})."
            )
        if self.deepfake_score is not None:
            parts.append(f"Deepfake score: {self.deepfake_score:.3f} (threshold {self.deepfake_threshold:.2f}).")
        if self.ai_speech_score is not None:
            parts.append(
                f"AI-speech score: {self.ai_speech_score:.3f} (threshold {self.ai_speech_threshold:.2f})."
            )
        if self.top_generator and self.top_generator_score is not None:
            parts.append(f"Closest generator match: {self.top_generator} ({self.top_generator_score:.3f}).")
        if self.n_frames:
            parts.append(
                f"Sightengine's video API scored {self.n_frames} frames it sampled server-side "
                f"from the video; the score is the {self.aggregation} over those frames."
            )
        for n in self.notes:
            parts.append(f"\nNote: {n}")
        return " ".join(parts)


class SightengineChecker(Tool[SightengineDetectionAction, SightengineDetectionResults]):
    """Runs Sightengine's `genai`, `deepfake` and (for video) `ai_speech`
    classifiers (https://sightengine.com/docs/) against an image or short video.

    Scores are looked up in precomputed stores first and only fetched from the
    (paid) API when missing. Newly fetched scores are cached, so repeated runs
    over the same dataset cost nothing.
    """

    name = "sightengine_checker"
    description = (
        "Runs Sightengine's genai, deepfake and ai_speech models to check whether an image or short "
        "video is likely AI-generated, a deepfake, or has AI-generated speech, purely from pixel/audio "
        "content (no metadata)."
    )
    actions = [SightengineDetectionAction]

    def __init__(
        self,
        stores: list[str | Path] | None = None,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        ai_generated_threshold: float = DEFAULT_AI_GENERATED_THRESHOLD,
        deepfake_threshold: float = DEFAULT_DEEPFAKE_THRESHOLD,
        ai_speech_threshold: float = DEFAULT_AI_SPEECH_THRESHOLD,
        check_ai_speech: bool = False,
        video_aggregation: str = "mean",
        timeout: float = 30.0,
        async_poll_interval: float = DEFAULT_ASYNC_POLL_INTERVAL_SECONDS,
        async_max_wait: float = DEFAULT_ASYNC_MAX_WAIT_SECONDS,
        **kwargs,
    ):
        """Args:
        stores: read-only score stores to consult before calling the API (in order).
            Defaults to the `sightengine_stores` env var.
        cache_dir: writable store for newly fetched scores. Defaults to temp/sightengine.
        use_cache: set False to score without reading/writing the writable cache
            (the read-only stores are still consulted).
        ai_generated_threshold: score above which the genai model counts as a positive verdict.
        deepfake_threshold: score above which the deepfake model counts as a positive verdict.
        ai_speech_threshold: score above which the ai_speech model counts as a positive verdict.
        check_ai_speech: also extract the audio track of videos (via ffmpeg) and run the ai_speech
            model on it. Requires ffmpeg on PATH and an ai_speech-enabled (Enterprise) Sightengine
            plan, so it defaults to off; failures (missing ffmpeg, no audio track, access denied)
            degrade to a note rather than an error when enabled.
        video_aggregation: how per-frame video scores collapse into one score — "mean"
            (default; matches the video-level score shown in Sightengine's own web viewer),
            "max" (most sensitive to a short synthetic/deepfaked segment inside an otherwise
            real video, but diverges sharply from Sightengine's own displayed score and is
            easily swayed by a single noisy frame) or "median". Applied when the video is
            scored; because frames come from the API rather than a local sampler, changing
            this requires re-precomputing videos.
        timeout: HTTP timeout in seconds for each Sightengine API call.
        async_poll_interval: seconds between polls of the async video job (videos over
            MAX_SYNC_VIDEO_SECONDS only).
        async_max_wait: give up waiting for the async video job after this many seconds.
        """
        if video_aggregation not in _AGGREGATIONS:
            raise ValueError(f"video_aggregation must be one of {sorted(_AGGREGATIONS)}")
        super().__init__(**kwargs)
        store_paths = [Path(p) for p in stores] if stores is not None else list(sightengine_stores)
        self.stores = [SightengineStore(p, writable=False) for p in store_paths]
        self.cache: SightengineStore | None = (
            SightengineStore(Path(cache_dir) if cache_dir else Path(sightengine_cache_dir))
            if use_cache
            else None
        )
        self.ai_generated_threshold = ai_generated_threshold
        self.deepfake_threshold = deepfake_threshold
        self.ai_speech_threshold = ai_speech_threshold
        self.check_ai_speech = check_ai_speech
        self.video_aggregation = video_aggregation
        self.timeout = timeout
        self.async_poll_interval = async_poll_interval
        self.async_max_wait = async_max_wait

    # --- lookup / caching -----------------------------------------------------

    def _lookup(self, sha256: str) -> SightengineRecord | None:
        for store in [*self.stores, *(s for s in [self.cache] if s is not None)]:
            record = store.get(sha256)
            if record is not None:
                return record
        return None

    def _record_from_scores(self, source_name: str, scores: _SightengineScores) -> SightengineRecord:
        top_generator, top_generator_score = _top_generator(scores.ai_generators)
        return SightengineRecord(
            ai_generated_score=scores.ai_generated_score,
            deepfake_score=scores.deepfake_score,
            ai_speech_score=scores.ai_speech_score,
            top_generator=top_generator,
            top_generator_score=top_generator_score,
            source_name=source_name,
            n_frames=scores.n_frames,
            aggregation=self.video_aggregation if scores.n_frames else None,
            notes=list(scores.notes),
        )

    def _result_from_record(
        self, record: SightengineRecord, from_cache: bool
    ) -> SightengineDetectionResults:
        result = self._build_result(
            ai_generated_score=record.ai_generated_score,
            deepfake_score=record.deepfake_score,
            ai_speech_score=record.ai_speech_score,
        )
        result.top_generator = record.top_generator
        result.top_generator_score = record.top_generator_score
        result.n_frames = record.n_frames
        result.aggregation = record.aggregation
        result.from_cache = from_cache
        result.notes.extend(record.notes)
        return result

    def compute_record(self, media: Image | Video) -> SightengineRecord:
        """Fetches Sightengine scores for one media item and returns them as a
        record. Consults nothing and caches nothing — this is the pure "call the
        API" step, used both by the tool and by precompute.py. Raises on missing
        credentials, a request failure, or an API-level error."""
        api_user, api_secret = self._resolve_key()
        if not api_user or not api_secret:
            raise _CredentialsMissing(
                "Sightengine API credentials not configured "
                "(set SIGHTENGINE_API_USER / SIGHTENGINE_API_SECRET)"
            )
        if isinstance(media, Video):
            scores = self._compute_video_scores(media, api_user, api_secret)
        else:
            scores = self._compute_image_scores(cast(Image, media), api_user, api_secret)
        if scores.error:
            raise RuntimeError(scores.error)
        return self._record_from_scores(Path(media.file_path).name, scores)

    # --- Tool interface -------------------------------------------------------

    def _perform(self, action: SightengineDetectionAction) -> SightengineDetectionResults:
        if action.media is None:
            return SightengineDetectionResults(error="media not found in the item registry")

        sha = file_sha256(action.media.file_path)
        record = self._lookup(sha)
        if record is not None:
            return self._result_from_record(record, from_cache=True)

        try:
            record = self.compute_record(action.media)
        except _CredentialsMissing as e:
            return SightengineDetectionResults(error=str(e))
        except requests.RequestException as e:
            logger.error(f"[Tool:{self.name}] Request to Sightengine failed for {action.media.reference}: {e}")
            return SightengineDetectionResults(error=f"request failed: {e}")
        except Exception as e:
            logger.error(f"[Tool:{self.name}] Failed to check {action.media.reference}: {e}")
            return SightengineDetectionResults(error=str(e))

        if self.cache is not None:
            self.cache.put(sha, record)
            self.cache.save()
        return self._result_from_record(record, from_cache=False)

    # --- score fetching -------------------------------------------------------

    def _compute_image_scores(
        self, media: Image, api_user: str, api_secret: str
    ) -> _SightengineScores:
        params = {"models": "genai,deepfake", "api_user": api_user, "api_secret": api_secret}
        with open(media.file_path, "rb") as f:
            r = requests.post(IMAGE_CHECK_URL, files={"media": f}, data=params, timeout=self.timeout)
        output = r.json()
        error = self._extract_error(output, r.status_code)
        if error:
            return _SightengineScores(None, None, None, None, None, [], error)
        type_data = output.get("type") or {}
        return _SightengineScores(
            ai_generated_score=type_data.get("ai_generated"),
            deepfake_score=type_data.get("deepfake"),
            ai_speech_score=None,
            ai_generators=type_data.get("ai_generators"),
            n_frames=None,
            notes=[],
            error=None,
        )

    def _compute_video_scores(
        self, media: Video, api_user: str, api_secret: str
    ) -> _SightengineScores:
        duration = media.duration
        if duration and duration > MAX_SYNC_VIDEO_SECONDS:
            visual = self._score_video_async(media, api_user, api_secret)
            if visual.error and self._is_plan_restricted(visual.error):
                visual = self._score_video_truncated(media, duration, api_user, api_secret)
        else:
            visual = self._score_video_sync(media.file_path, api_user, api_secret)
        if visual.error:
            return _SightengineScores(None, None, None, None, 0, [], visual.error)

        notes: list[str] = []
        if visual.note:
            notes.append(visual.note)

        ai_speech_score: float | None = None
        if self.check_ai_speech:
            ai_speech_score, ai_speech_note = self._check_ai_speech(media, api_user, api_secret)
            if ai_speech_note:
                notes.append(ai_speech_note)

        return _SightengineScores(
            ai_generated_score=visual.ai_generated_score,
            deepfake_score=visual.deepfake_score,
            ai_speech_score=ai_speech_score,
            ai_generators=None,
            n_frames=visual.n_frames,
            notes=notes,
            error=None,
        )

    def _score_video_sync(self, file_path: Path, api_user: str, api_secret: str) -> _VideoScores:
        params = {"models": "genai,deepfake", "api_user": api_user, "api_secret": api_secret}
        with open(file_path, "rb") as f:
            r = requests.post(VIDEO_CHECK_SYNC_URL, files={"media": f}, data=params, timeout=self.timeout)
        output = r.json()
        error = self._extract_error(output, r.status_code)
        if error:
            return _VideoScores(None, None, 0, None, error)

        frames = ((output.get("data") or {}).get("frames")) or []
        if not frames:
            return _VideoScores(None, None, 0, None, "Sightengine returned no video frames to score")
        return self._aggregate_frames(frames, note=None)

    def _score_video_truncated(
        self, media: Video, duration: float, api_user: str, api_secret: str
    ) -> _VideoScores:
        """Falls back to the sync endpoint on the first MAX_SYNC_VIDEO_SECONDS of the
        video. Used when the account has no access to the async video job feature —
        the remainder of the video past the cutoff is simply not checked."""
        trimmed_path = self._trim_video(media.file_path)
        if trimmed_path is None:
            return _VideoScores(
                None, None, 0, None,
                "async video analysis is not available on this Sightengine plan, and trimming the "
                "video for a fallback sync check failed (ffmpeg missing or errored)",
            )
        try:
            result = self._score_video_sync(trimmed_path, api_user, api_secret)
        finally:
            trimmed_path.unlink(missing_ok=True)
        if result.error:
            return result
        note = (
            f"async video analysis is not available on this Sightengine plan; scored only the first "
            f"{MAX_SYNC_VIDEO_SECONDS}s of this {duration:.0f}s video via the synchronous endpoint — "
            "the remainder was not checked"
        )
        return result._replace(note=note)

    def _trim_video(self, video_path: Path) -> Path | None:
        """Cuts the first MAX_SYNC_VIDEO_SECONDS of a video via ffmpeg. Tries a fast
        stream copy first, falls back to re-encoding if that fails (some codecs/containers
        don't cut cleanly on copy). Returns None (never raises) on any failure."""
        if shutil.which("ffmpeg") is None:
            logger.warning(f"[Tool:{self.name}] ffmpeg not found on PATH; cannot trim video for fallback")
            return None
        fd, tmp_name = tempfile.mkstemp(suffix=video_path.suffix or ".mp4")
        os.close(fd)
        out_path = Path(tmp_name)
        try:
            for extra_args in (["-c", "copy"], []):
                proc = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(video_path), "-t", str(MAX_SYNC_VIDEO_SECONDS),
                     *extra_args, str(out_path)],
                    capture_output=True,
                    timeout=FFMPEG_TIMEOUT_SECONDS,
                )
                if proc.returncode == 0 and out_path.stat().st_size > 0:
                    return out_path
            return None
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning(f"[Tool:{self.name}] Video trim failed for {video_path}: {e}")
            return None
        finally:
            if out_path.exists() and out_path.stat().st_size == 0:
                out_path.unlink(missing_ok=True)

    def _is_plan_restricted(self, message: str) -> bool:
        """Heuristic for Sightengine's "Feature not available on the free plan"-style
        errors, distinguished from transient/other errors so we only fall back to the
        truncated sync check when the async endpoint is genuinely inaccessible."""
        lowered = message.lower()
        return "plan" in lowered or "not available" in lowered or "access" in lowered

    def _score_video_async(self, media: Video, api_user: str, api_secret: str) -> _VideoScores:
        """Scores a video over MAX_SYNC_VIDEO_SECONDS via Sightengine's async video job:
        submit without a callback_url, then poll `video/byid.json` for the result.
        """
        params = {"models": "genai,deepfake", "api_user": api_user, "api_secret": api_secret}
        with open(media.file_path, "rb") as f:
            r = requests.post(VIDEO_CHECK_ASYNC_URL, files={"media": f}, data=params, timeout=self.timeout)
        output = r.json()
        error = self._extract_error(output, r.status_code)
        if error:
            return _VideoScores(None, None, 0, None, error)

        media_id = (output.get("media") or {}).get("id")
        if not media_id:
            return _VideoScores(None, None, 0, None, "Sightengine did not return a job id for the async video check")

        poll_params = {"id": media_id, "api_user": api_user, "api_secret": api_secret}
        deadline = time.monotonic() + self.async_max_wait
        while True:
            time.sleep(self.async_poll_interval)
            r = requests.get(VIDEO_BY_ID_URL, params=poll_params, timeout=self.timeout)
            poll_output = r.json()
            error = self._extract_error(poll_output, r.status_code)
            if error:
                return _VideoScores(None, None, 0, None, error)

            job = (poll_output.get("output") or {}).get("data") or {}
            status = job.get("status")
            if status and status != "ongoing":
                frames = job.get("frames") or []
                if not frames:
                    return _VideoScores(
                        None, None, 0, None,
                        f"Sightengine async video job finished with status {status!r} but returned no frames",
                    )
                return self._aggregate_frames(frames, note=f"scored via Sightengine's async video job ({status})")

            if time.monotonic() > deadline:
                return _VideoScores(
                    None, None, 0, None,
                    f"Sightengine async video job did not finish within {self.async_max_wait:.0f}s",
                )

    def _aggregate_frames(self, frames: list[dict], note: str | None) -> _VideoScores:
        ai_scores = [f["type"]["ai_generated"] for f in frames if "ai_generated" in (f.get("type") or {})]
        deepfake_scores = [f["type"]["deepfake"] for f in frames if "deepfake" in (f.get("type") or {})]
        agg = _AGGREGATIONS[self.video_aggregation]
        return _VideoScores(
            agg(ai_scores) if ai_scores else None,
            agg(deepfake_scores) if deepfake_scores else None,
            len(frames),
            note,
            None,
        )

    def _check_ai_speech(
        self, video: Video, api_user: str, api_secret: str
    ) -> tuple[float | None, str | None]:
        """Extracts the video's audio track and runs the ai_speech model on it.

        Never raises: any failure (no ffmpeg, no audio track, request error,
        access denied on a non-enterprise plan) comes back as ``(None, note)``
        so the caller can still report the genai/deepfake scores.
        """
        audio_path = self._extract_audio(video.file_path)
        if audio_path is None:
            return None, "ai_speech not checked: could not extract an audio track from the video"
        try:
            params = {"models": "ai_speech", "api_user": api_user, "api_secret": api_secret}
            with open(audio_path, "rb") as f:
                r = requests.post(AUDIO_CHECK_URL, files={"audio": f}, data=params, timeout=self.timeout)
            output = r.json()
            error = self._extract_error(output, r.status_code)
            if error:
                return None, f"ai_speech not checked: {error}"
            score = (output.get("type") or {}).get("ai_speech")
            if score is None:
                return None, "ai_speech not checked: Sightengine response did not include a score"
            return score, None
        except requests.RequestException as e:
            return None, f"ai_speech not checked: request failed: {e}"
        finally:
            audio_path.unlink(missing_ok=True)

    def _extract_audio(self, video_path: Path) -> Path | None:
        """Extracts the audio track to a temporary mono 16kHz WAV file via ffmpeg."""
        if shutil.which("ffmpeg") is None:
            logger.warning(f"[Tool:{self.name}] ffmpeg not found on PATH; skipping ai_speech check")
            return None
        fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        out_path = Path(tmp_name)
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(out_path)],
                capture_output=True,
                timeout=FFMPEG_TIMEOUT_SECONDS,
            )
            if proc.returncode != 0 or out_path.stat().st_size == 0:
                return None
            return out_path
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning(f"[Tool:{self.name}] Audio extraction failed for {video_path}: {e}")
            return None
        finally:
            if out_path.exists() and out_path.stat().st_size == 0:
                out_path.unlink(missing_ok=True)

    def _build_result(
        self,
        ai_generated_score: float | None,
        deepfake_score: float | None,
        ai_speech_score: float | None = None,
        ai_generators: dict[str, float] | None = None,
    ) -> SightengineDetectionResults:
        result = SightengineDetectionResults(
            ai_generated_score=ai_generated_score,
            deepfake_score=deepfake_score,
            ai_speech_score=ai_speech_score,
            ai_generated_threshold=self.ai_generated_threshold,
            deepfake_threshold=self.deepfake_threshold,
            ai_speech_threshold=self.ai_speech_threshold,
        )
        result.top_generator, result.top_generator_score = _top_generator(ai_generators)

        result.notes.append(
            "these models are purely pixel/waveform-based; they ignore metadata, EXIF tags, C2PA "
            "provenance and watermarks, and a score near the threshold is not proof of authenticity either way"
        )
        if ai_generated_score is not None:
            result.notes.append(
                "genai targets fully synthetic media and will not reliably flag real media that was only edited"
            )
        if deepfake_score is not None:
            result.notes.append(
                "deepfake targets identity-level face manipulation in photographic media; it will not "
                "reliably flag fully AI-generated faces, drawings, or cartoons, and is only meaningful "
                "when a face is present"
            )
        if ai_speech_score is not None:
            result.notes.append(
                "ai_speech targets fully synthetic speech and will not reliably flag real speech that was only edited"
            )
        return result

    def _extract_error(self, output: dict, status_code: int) -> str | None:
        if output.get("status") == "success":
            return None
        err = output.get("error") or {}
        return err.get("message") or f"Sightengine request failed with HTTP {status_code}"

    def _summarize(self, result: SightengineDetectionResults, **kwargs) -> MultimodalSequence | None:
        if result.error is not None:
            return None
        if not result.is_useful():
            return None
        return MultimodalSequence(str(result))

    def _resolve_key(self) -> tuple[str | None, str | None]:
        api_user = os.environ.get("SIGHTENGINE_API_USER") or os.environ.get("sightengine_api_user")
        api_secret = os.environ.get("SIGHTENGINE_API_SECRET") or os.environ.get("sightengine_api_secret")
        return api_user, api_secret
