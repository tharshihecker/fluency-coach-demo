from __future__ import annotations

import logging
import tempfile
import os
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


@st.cache_resource(show_spinner="Loading the fluency model…")
def load_predictor() -> FluencyPredictor:
    return FluencyPredictor(ROOT / "outputs")


def get_recording_result(audio_file, confirm_english: bool) -> dict:
    predictor = load_predictor()
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
    axis.set_xlabel("Change relative to the model's background average")
    axis.set_title("Feature influence for this estimate")
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)


st.title("🎙️ Speech Fluency Estimator")
st.write(
    "Record a sample on your phone or upload an audio file. The app returns a "
    "research estimate with speech indicators and a feature influence chart."
)

with st.sidebar:
    st.subheader("About this estimate")
    st.write("The selected tuned Extra Trees model produces a 0–10 estimate.")
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
            st.subheader("Speech indicators")
            st.dataframe(
                pd.DataFrame(
                    [{"Indicator": key.replace("_", " ").title(), "Value": value}
                     for key, value in indicators.items()]
                ),
                hide_index=True,
                use_container_width=True,
            )
    st.subheader("What influenced the estimate")
    render_feature_chart(result.get("feature_impacts", []))
    st.caption(result.get("explanation_note", ""))
    st.caption(
        "Positive bars pushed the model estimate up; negative bars pushed it down. "
        "They describe model behavior, not causes or expert judgments."
    )

    if feedback_is_configured():
        if st.button("Generate one practice suggestion"):
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
                st.write(get_practice_feedback(payload))
            except Exception as exc:
                st.error(f"Feedback is temporarily unavailable: {exc}")
