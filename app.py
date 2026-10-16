from __future__ import annotations

import logging
import tempfile
import os
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
import whisper

from fluency_app.inference import FluencyPredictor
from fluency_app.feedback import feedback_is_configured, get_practice_feedback
from fluency_app.speech_content import SpeechContentRejected

logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parent
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = str(st.secrets["GROQ_API_KEY"])
if "GROQ_MODEL" in st.secrets:
    os.environ["GROQ_MODEL"] = str(st.secrets["GROQ_MODEL"])

st.set_page_config(
    page_title="Speech Fluency Estimator",
    page_icon="🎙️",
    layout="wide",
)


def predictor_cache_key() -> str:
    """Change the resource-cache key when the artifact or inference code changes."""
    metadata_path = ROOT / "outputs" / "selected_fluency_model_metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    artifact_path = ROOT / "outputs" / metadata["artifact"]
    digest = hashlib.sha256(metadata_bytes)
    for source_path in (
        Path(__file__).resolve(),
        Path(__import__("fluency_app.inference", fromlist=["__file__"]).__file__).resolve(),
        Path(__import__("fluency_app.explainability", fromlist=["__file__"]).__file__).resolve(),
        Path(__import__("fluency_app.audio_features", fromlist=["__file__"]).__file__).resolve(),
    ):
        digest.update(source_path.read_bytes())
    artifact_stat = artifact_path.stat()
    digest.update(f"{artifact_stat.st_size}:{artifact_stat.st_mtime_ns}".encode("ascii"))
    return digest.hexdigest()


@st.cache_resource(show_spinner="Loading the fluency model…")
def load_predictor(cache_key: str) -> FluencyPredictor:
    # cache_key fingerprints the model artifact and the prediction/explanation code.
    return FluencyPredictor(ROOT / "outputs")


def get_recording_result(audio_file, confirm_english: bool) -> dict:
    predictor = load_predictor(predictor_cache_key())
    suffix = Path(audio_file.name or "recording.wav").suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(prefix="fluency-upload-", suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(audio_file.getvalue())
    try:
        predictor.speech_device = torch.device("cpu")
        return predictor.predict_file(
            temp_path,
            original_name=Path(audio_file.name or "recording.wav").name,
            explain=True,
            confirm_english=confirm_english,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def render_feature_chart(impacts: list[dict]) -> None:
    if not impacts:
        st.info("A feature influence chart is unavailable for this recording.")
        return
    ordered = list(reversed(impacts))
    figure, axis = plt.subplots(figsize=(8, max(3, 0.45 * len(ordered))))
    colors = ["#2878c8" if item["impact"] >= 0 else "#db705b" for item in ordered]
    axis.barh([item["feature"] for item in ordered],
              [item["impact"] for item in ordered], color=colors)
    axis.axvline(0, color="#64748b", linewidth=0.8)
    axis.set_xlabel("SHAP contribution (score points)")
    axis.set_title("Feature influence on this estimate")
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)


def render_speech_profile(indicators: dict) -> None:
    """Show interpretable timing measures on sensible, labeled display scales."""
    duration = float(indicators.get("duration_seconds", 0))
    rows = [
        ("Speaking share", float(indicators.get("speaking_time_ratio", 0)), 1.0,
         f"{float(indicators.get('speaking_time_ratio', 0)):.0%}"),
        ("Pauses per minute", float(indicators.get("internal_pauses_per_minute", 0)), 20.0,
         f"{float(indicators.get('internal_pauses_per_minute', 0)):.1f} / min"),
        ("Average pause", float(indicators.get("mean_pause_seconds", 0)), 2.0,
         f"{float(indicators.get('mean_pause_seconds', 0)):.2f} s"),
        ("Long pause estimate", float(indicators.get("long_pause_p90_seconds", 0)), 3.0,
         f"{float(indicators.get('long_pause_p90_seconds', 0)):.2f} s"),
    ]
    speech_seconds = float(indicators.get("speech_time_seconds", 0))
    st.caption(f"Speech detected for {speech_seconds:.1f} of {duration:.1f} seconds.")
    for label, value, scale, display_value in rows:
        st.markdown(f"**{label}** · {display_value}")
        st.progress(min(1.0, max(0.0, value / scale)))
    st.caption("Pause bars use display scales (20 pauses/min, 2 s average pause, 3 s long pause).")


def render_voice_graphs(graphs: dict | None) -> None:
    """Render the recording's downsampled energy and pitch time series."""
    if not graphs:
        st.info("Voice graphs are unavailable for this recording.")
        return
    panels = st.columns(2)
    for panel, key, title, y_label in [
        (panels[0], "energy", "Energy over time", "Relative energy (dB)"),
        (panels[1], "pitch", "Pitch over time", "Pitch (Hz)"),
    ]:
        series = graphs.get(key) or {}
        times, values = series.get("times", []), series.get("values", [])
        if times and values:
            chart_data = pd.DataFrame({"Time (seconds)": times, y_label: values})
            panel.line_chart(chart_data, x="Time (seconds)", y=y_label, height=260)
            panel.caption(title)
        else:
            panel.info(f"{title} is unavailable for this recording.")


st.title("🎙️ Speech Fluency Estimator")
st.write(
    "Record a sample on your phone or upload an audio file. The app returns a "
    "research estimate with a speech profile, voice graphs, and a local SHAP explanation."
)

with st.sidebar:
    st.subheader("About this estimate")
    st.write(
        "The selected model is a weighted ensemble of Extra Trees, Random Forest, "
        "and XGBoost. It produces a 0–10 estimate."
    )
    st.caption(
        "This is a research estimate, not a human expert rating, formal language "
        "certification, or clinical assessment."
    )
    st.caption("Recordings are processed on this hosted app and are not saved as project data.")

left, right = st.columns([1, 1], gap="large")
with left:
    recorded_audio = st.audio_input("Record speech with your microphone")
    uploaded_audio = st.file_uploader(
        "Or upload a recording",
        type=["wav", "mp3", "m4a", "mp4", "ogg", "opus", "flac", "webm"],
    )
    confirm_english = st.checkbox(
        "I am speaking English (use this if an accent or stutter confuses the language check)",
        value=False,
    )
    predict_clicked = st.button("Estimate fluency", type="primary", use_container_width=True)
    st.caption("Use a clear sample of at least 20 seconds for a useful estimate.")

audio_file = recorded_audio or uploaded_audio
if predict_clicked:
    st.session_state.pop("fluency_result", None)
    if audio_file is None:
        st.warning("Record or upload a speech sample first.")
    else:
        with st.spinner("Processing the recording… the first run may take longer while the speech-check model downloads."):
            try:
                result = get_recording_result(audio_file, confirm_english)
                st.session_state["fluency_result"] = result
            except SpeechContentRejected as exc:
                st.error(str(exc))
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                st.error(str(exc))
            except Exception as exc:
                # Keep browser output concise; record the traceback in the
                # hosted logs so unexpected inference failures can be diagnosed.
                logger.exception("Audio fluency prediction failed")
                st.error(
                    "Prediction failed while processing this recording "
                    f"({type(exc).__name__}). Please try again."
                )

result = st.session_state.get("fluency_result")
if result:
    with right:
        st.metric("Estimated fluency", f"{result['score']:.2f} / 10")
        if result.get("silent_recording"):
            st.info("No speech was detected. The model's score is 0.")
        else:
            st.caption(result.get("message", ""))
        indicators = result.get("indicators", {})
        if indicators:
            st.subheader("Your speech profile")
            render_speech_profile(indicators)
            with st.expander("Detailed speech indicators"):
                st.dataframe(
                    pd.DataFrame(
                        [{"Indicator": key.replace("_", " ").title(), "Value": value}
                         for key, value in indicators.items()]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

    st.subheader("How your voice changed over time")
    render_voice_graphs(result.get("voice_graphs"))
    impacts = result.get("feature_impacts", [])
    if impacts:
        st.subheader("What influenced this estimate")
        render_feature_chart(impacts)
        st.caption(result.get("explanation_note", ""))
        st.caption(
            "Positive bars pushed the model estimate up; negative bars pushed it down. "
            "They describe model behavior, not causes or expert judgments."
        )
    else:
        st.subheader("Model explanation")
        st.info(result.get("explanation_note", "Per-recording feature attribution is unavailable."))

    st.subheader("Practice feedback")
    feedback_source = "the configured AI provider" if feedback_is_configured() else "built-in guidance"
    st.caption(
        f"On request, feedback uses {feedback_source}. Audio and transcripts are not sent to the feedback provider."
    )
    if st.button("Generate practice feedback"):
        try:
            payload = {
                "score": result["score"],
                "silent_recording": result.get("silent_recording", False),
                "indicators": result.get("indicators", {}),
                "feature_impacts": [
                    {"feature": item["feature"], "impact": item["impact"]}
                    for item in result.get("feature_impacts", [])
                ],
            }
            feedback = get_practice_feedback(payload)
            st.write(feedback["feedback"])
            st.caption(f"Feedback source: {feedback['provider']}")
        except Exception as exc:
            st.error(f"Feedback is temporarily unavailable: {exc}")
