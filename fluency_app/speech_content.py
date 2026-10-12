"""Local English intelligibility gate for speech-fluency predictions."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import torch
import whisper
from whisper.tokenizer import LANGUAGES


_FILLERS = {
    "aah", "ah", "aha", "eh", "eee", "er", "erm", "hmm", "huh",
    "mm", "mmm", "oh", "ooh", "uh", "uhh", "um", "umm", "uhm",
}
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
_VOCAL_FILLER_RE = re.compile(r"(?:a+|e+|o+|u+|m+|h+m+|[aeiou]h+)", re.IGNORECASE)


class SpeechContentRejected(ValueError):
    def __init__(self, message: str, language_code: str, language_name: str) -> None:
        super().__init__(message)
        self.language_code = language_code
        self.language_name = language_name


@lru_cache(maxsize=2)
def _load_model(device: str):
    """Load the small multilingual Whisper model once; weights are cached locally."""
    cache = Path.home() / ".cache" / "whisper"
    cache.mkdir(parents=True, exist_ok=True)
    # On a hosted deployment, fetch official Whisper weights on first use and
    # cache them locally. No recording or transcript is sent with this request.
    return whisper.load_model("base", device=device, download_root=str(cache))


def validate_english_content(path: str | Path, device: torch.device) -> tuple[bool, str, str, str, str]:
    """Check language without letting imperfect transcription reject English speech.

    Clear English language detection is enough; a low-confidence transcript is not
    required to contain two recognized words. If auto-detection picks another
    language, try one English-forced transcription before rejecting. The transcript
    is used only for this check and is never returned or stored.
    """
    model = _load_model(device.type)
    result = model.transcribe(
        str(path),
        task="transcribe",
        fp16=device.type == "cuda",
        verbose=False,
        temperature=0,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
    )
    transcript = str(result.get("text", "")).strip()
    language_code = str(result.get("language", "unknown"))
    language_name = LANGUAGES.get(language_code, language_code).title()
    # Whisper can label a single short word as English because it has too little
    # context. Require two recognized content words for automatic confirmation;
    # users can still explicitly confirm English when an accent or stutter hurts
    # transcription quality.
    words = [word.lower() for word in _WORD_RE.findall(transcript)]
    content_words = [
        word for word in words
        if word not in _FILLERS and not _VOCAL_FILLER_RE.fullmatch(word)
    ]
    if language_code == "en" and len(content_words) >= 2:
        return True, "", transcript, language_code, language_name

    # A second pass fixed to English helps when the first language decision was
    # confused by accent or code-switching. Keep the content check permissive.
    forced = model.transcribe(
        str(path),
        task="transcribe",
        language="en",
        fp16=device.type == "cuda",
        verbose=False,
        temperature=0,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.5,
    )
    forced_text = str(forced.get("text", "")).strip()
    words = [word.lower() for word in _WORD_RE.findall(forced_text)]
    content_words = [
        word for word in words
        if word not in _FILLERS and not _VOCAL_FILLER_RE.fullmatch(word)
    ]
    if len(content_words) >= 2:
        return True, "", forced_text, "en", LANGUAGES["en"].title()

    reason = (
        "There wasn't enough clear speech to identify the language reliably. "
        "If you are speaking English, select the English confirmation option and keep speaking."
        if language_code == "en"
        else f"The language check detected {language_name}, but accents and stutters can confuse it. "
        "If you are speaking English, select the English confirmation option and try again."
    )
    return False, reason, transcript, language_code, language_name
