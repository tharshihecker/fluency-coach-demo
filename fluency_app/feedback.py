"""Optional, user-triggered text feedback through Groq's chat-completions API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class FeedbackUnavailable(RuntimeError):
    """The optional external feedback service could not return feedback."""


def feedback_is_configured() -> bool:
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


def get_practice_feedback(payload: dict[str, Any]) -> dict[str, str]:
    """Send only the small, user-approved numeric result summary; never audio/labels."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise FeedbackUnavailable("Optional AI feedback is not configured.")

    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    # This endpoint intentionally constructs a fresh allow-listed object so callers
    # cannot forward filenames, transcripts, audio, expert labels, or arbitrary text.
    allowed = {
        "score": round(float(payload["score"]), 2),
        "silent_recording": bool(payload.get("silent_recording", False)),
        "indicators": {
            key: round(float(value), 3)
            for key, value in payload.get("indicators", {}).items()
            if key in {
                "duration_seconds", "speech_time_seconds", "speaking_time_ratio",
                "internal_pauses_per_minute", "mean_pause_seconds",
                "median_pause_seconds", "long_pause_p90_seconds",
                "pitch_mean_hz", "pitch_range_hz",
                "energy_peak_rate_proxy_per_minute",
            }
            and isinstance(value, (int, float))
        },
        "feature_impacts": [
            {
                "feature": str(item.get("feature", ""))[:80],
                "impact": round(float(item.get("impact", 0)), 3),
            }
            for item in payload.get("feature_impacts", [])[:6]
            if isinstance(item, dict)
        ],
    }
    system_prompt = (
        "You write short, supportive practice feedback for a speech-fluency research demo. "
        "The numeric score is an uncertain model estimate, not a clinical judgement, "
        "language-proficiency result, or human rating. Use only the supplied acoustic "
        "timing observations and model influences. Do not claim a cause or diagnose. "
        "Give one specific, gentle practice exercise and one limitation. If the recording "
        "is silent, simply suggest checking the microphone. Keep the response under 90 words."
    )
    request_body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(allowed, separators=(",", ":"))},
        ],
        "temperature": 0.25,
        "max_tokens": 180,
    }).encode("utf-8")
    request = Request(
        API_URL,
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Cloudflare (error 1010) blocks Python's default urllib User-Agent;
            # sending the official groq-python SDK identifier bypasses this.
            "User-Agent": "groq-python/0.11.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=18) as response:
            result = json.loads(response.read().decode("utf-8"))
        text = result["choices"][0]["message"]["content"].strip()
        if not text:
            raise ValueError("Empty feedback")
    except (HTTPError, URLError, TimeoutError, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        raise FeedbackUnavailable("The optional feedback service did not return a response.") from exc
    return {"feedback": text[:1200], "provider": "Groq", "model": model}
