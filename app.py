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

import streamlit.components.v1 as components

visualizer_html = """
<!DOCTYPE html>
<html>
<head>
<style>
  body { font-family: sans-serif; color: #334155; margin: 0; padding: 0; display: flex; flex-direction: column; justify-content: center; height: 100vh; }
  .box { border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; background-color: #f8fafc; }
  .metric { font-size: 1.5rem; font-weight: bold; color: #168d82; }
  .label { font-size: 0.85rem; color: #64748b; text-transform: uppercase; font-weight: bold; margin-bottom: 2px;}
  .bar { width: 100%; height: 8px; background: #e2e8f0; border-radius: 4px; margin-top: 5px; overflow: hidden; }
  .fill { height: 100%; background: #168d82; width: 0%; transition: width 0.05s linear; }
  button { padding: 8px 16px; background: #168d82; color: white; border: none; border-radius: 4px; cursor: pointer; margin-bottom: 10px; width: 100%; font-weight: bold;}
  button:hover { background: #13776d; }
  .flex-row { display: flex; justify-content: space-between; margin-bottom: 10px; }
  .flex-col { display: flex; flex-direction: column; width: 45%; }
</style>
</head>
<body>
  <div class="box">
    <button id="startBtn">🎤 Start Live Mic Preview</button>
    <div class="flex-row">
        <div class="flex-col">
            <div class="label">Time</div>
            <div class="metric" id="timer">0.0s</div>
        </div>
        <div class="flex-col">
            <div class="label">Pitch</div>
            <div class="metric" id="pitch">-- Hz</div>
        </div>
    </div>
    <div>
        <div class="label" style="display:flex; justify-content: space-between;">
            <span>Energy</span>
            <span id="energyVal">0%</span>
        </div>
        <div class="bar"><div class="fill" id="energyFill"></div></div>
    </div>
  </div>
  <script>
    let audioContext;
    let analyser;
    let microphone;
    let startTime;
    let isRecording = false;
    let streamRef;

    const startBtn = document.getElementById('startBtn');
    const timerEl = document.getElementById('timer');
    const pitchEl = document.getElementById('pitch');
    const energyFill = document.getElementById('energyFill');
    const energyVal = document.getElementById('energyVal');

    function autoCorrelate(buf, sampleRate) {
        let SIZE = buf.length;
        let rms = 0;

        for (let i = 0; i < SIZE; i++) {
            rms += buf[i] * buf[i];
        }
        rms = Math.sqrt(rms / SIZE);
        if (rms < 0.01) return -1;

        let r1 = 0, r2 = SIZE - 1, thres = 0.2;
        for (let i = 0; i < SIZE / 2; i++)
            if (Math.abs(buf[i]) < thres) { r1 = i; break; }
        for (let i = 1; i < SIZE / 2; i++)
            if (Math.abs(buf[SIZE - i]) < thres) { r2 = SIZE - i; break; }

        buf = buf.slice(r1, r2);
        SIZE = buf.length;

        let c = new Array(SIZE).fill(0);
        for (let i = 0; i < SIZE; i++)
            for (let j = 0; j < SIZE - i; j++)
                c[i] = c[i] + buf[j] * buf[j + i];

        let d = 0; while (c[d] > c[d + 1]) d++;
        let maxval = -1, maxpos = -1;
        for (let i = d; i < SIZE; i++) {
            if (c[i] > maxval) {
                maxval = c[i];
                maxpos = i;
            }
        }
        let T0 = maxpos;
        let x1 = c[T0 - 1], x2 = c[T0], x3 = c[T0 + 1];
        let a = (x1 + x3 - 2 * x2) / 2;
        let b = (x3 - x1) / 2;
        if (a) T0 = T0 - b / (2 * a);
        return sampleRate / T0;
    }

    function update() {
        if (!isRecording) return;
        requestAnimationFrame(update);

        let bufferLength = analyser.fftSize;
        let buffer = new Float32Array(bufferLength);
        analyser.getFloatTimeDomainData(buffer);

        let rms = 0;
        for (let i = 0; i < bufferLength; i++) {
            rms += buffer[i] * buffer[i];
        }
        rms = Math.sqrt(rms / bufferLength);

        let energyPercent = Math.min(100, Math.round(rms * 400));
        energyFill.style.width = energyPercent + "%";
        energyVal.innerText = energyPercent + "%";

        let pitch = autoCorrelate(buffer, audioContext.sampleRate);
        if (pitch !== -1 && pitch > 50 && pitch < 1000) {
            pitchEl.innerText = Math.round(pitch) + " Hz";
        } else {
            pitchEl.innerText = "-- Hz";
        }

        let elapsed = (Date.now() - startTime) / 1000;
        timerEl.innerText = elapsed.toFixed(1) + "s";
    }

    startBtn.addEventListener('click', async () => {
        if (isRecording) {
            isRecording = false;
            startBtn.innerText = "🎤 Start Live Mic Preview";
            if (audioContext) audioContext.close();
            if (streamRef) streamRef.getTracks().forEach(t => t.stop());
            timerEl.innerText = "0.0s";
            pitchEl.innerText = "-- Hz";
            energyFill.style.width = "0%";
            energyVal.innerText = "0%";
            return;
        }

        try {
            streamRef = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 2048;
            microphone = audioContext.createMediaStreamSource(streamRef);
            microphone.connect(analyser);

            isRecording = true;
            startBtn.innerText = "⏹ Stop Preview";
            startTime = Date.now();
            update();
        } catch (err) {
            console.error("Error accessing mic: ", err);
            alert("Could not access microphone.");
        }
    });
  </script>
</body>
</html>
"""

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
    digest.update(artifact_path.read_bytes())
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



def render_radar_chart(indicators: dict) -> None:
    """Render a radar chart summarizing key fluency timing metrics."""
    import numpy as np

    # Extract features focusing on timing/fluency, ignore pitch/energy
    speaking_ratio = float(indicators.get("speaking_time_ratio", 0))
    # Normalize pauses per minute (lower is better, let's say 0-30 range -> 0-1)
    # We invert it so further out is "better" or just represent the value
    pauses_per_min = float(indicators.get("internal_pauses_per_minute", 0))
    # Inverse map pauses/min to 0-1 range (0 pauses = 1.0, 30 pauses = 0.0)
    norm_pauses = max(0.0, 1.0 - (pauses_per_min / 30.0))

    avg_pause = float(indicators.get("mean_pause_seconds", 0))
    # Inverse map avg pause to 0-1 (0s = 1.0, 2s = 0.0)
    norm_avg_pause = max(0.0, 1.0 - (avg_pause / 2.0))

    # Energy peak rate proxy (articulation rate proxy)
    # Let's say 0-200 peaks/min is the range
    articulation_proxy = float(indicators.get("energy_peak_rate_proxy_per_minute", 0))
    norm_articulation = min(1.0, articulation_proxy / 200.0)

    categories = ['Speaking Share', 'Few Pauses', 'Short Pauses', 'Articulation Rate']
    values = [speaking_ratio, norm_pauses, norm_avg_pause, norm_articulation]

    # Number of variables
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]

    # Initialise the spider plot
    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='#334155', size=9, weight='bold')

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25","0.50","0.75"], color="#94a3b8", size=7)
    plt.ylim(0, 1)

    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', color="#168d82")

    # Fill area
    ax.fill(angles, values, color="#168d82", alpha=0.35)
    ax.set_title("Fluency Characteristics", size=11, color="#172a40", y=1.15, weight='bold')

    # Adjust padding to prevent labels from being cut off
    plt.tight_layout(pad=2.0)

    st.pyplot(fig, clear_figure=True)


def render_speech_profile(indicators: dict) -> None:
    """Show model-related timing measures and flag coaching cues in red."""
    duration = float(indicators.get("duration_seconds", 0))
    speech_seconds = float(indicators.get("speech_time_seconds", 0))
    share = min(1.0, max(0.0, float(indicators.get("speaking_time_ratio", 0))))
    visual_col, measures_col = st.columns([1, 2])
    with visual_col:
        figure, axis = plt.subplots(figsize=(2.3, 2.3))
        axis.pie(
            [share, max(0.0, 1 - share)],
            startangle=90,
            counterclock=False,
            colors=["#168d82", "#e7edf4"],
            wedgeprops={"width": 0.25, "edgecolor": "white"},
        )
        axis.text(0, 0.06, f"{share:.0%}", ha="center", va="center", fontsize=20, weight="bold", color="#172a40")
        axis.text(0, -0.2, "speaking", ha="center", va="center", fontsize=9, color="#6f8092")
        axis.set_title("Speaking share", fontsize=10, color="#172a40")
        st.pyplot(figure, clear_figure=True)
    with measures_col:
        st.caption(f"{speech_seconds:.1f} s speech in a {duration:.1f} s recording")
        rows = [
            ("Speaking share", float(indicators.get("speaking_time_ratio", 0)), 1.0,
             f"{float(indicators.get('speaking_time_ratio', 0)):.0%}", float(indicators.get("speaking_time_ratio", 0)) < 0.5),
            ("Pauses per minute", float(indicators.get("internal_pauses_per_minute", 0)), 20.0,
             f"{float(indicators.get('internal_pauses_per_minute', 0)):.1f} / min", float(indicators.get("internal_pauses_per_minute", 0)) > 12),
            ("Average pause", float(indicators.get("mean_pause_seconds", 0)), 2.0,
             f"{float(indicators.get('mean_pause_seconds', 0)):.2f} s", float(indicators.get("mean_pause_seconds", 0)) > 0.8),
            ("Long pause estimate", float(indicators.get("long_pause_p90_seconds", 0)), 3.0,
             f"{float(indicators.get('long_pause_p90_seconds', 0)):.2f} s", float(indicators.get("long_pause_p90_seconds", 0)) > 1.5),
        ]
        for label, value, scale, display_value, weak in rows:
            color = "#d64545" if weak else "#168d82"
            flag = " · Practice cue" if weak else ""
            st.markdown(f"**{label}** · {display_value}{flag}")
            width = min(100.0, max(0.0, value / scale * 100))
            st.markdown(
                f'<div role="meter" aria-label="{label}" aria-valuenow="{value:.3f}" '
                f'style="height:8px;border-radius:8px;background:#e7edf4;margin:0 0 8px;">'
                f'<div style="width:{width:.1f}%;height:8px;border-radius:8px;background:{color};"></div></div>',
                unsafe_allow_html=True,
            )
        st.caption("Red cues use coaching heuristics, not model cutoffs or clinical standards.")


def render_voice_graphs(graphs: dict | None) -> None:
    """Render speech activity and pause timing, the fluency-related time series."""
    if not graphs:
        st.info("Speech timing graphs are unavailable for this recording.")
        return
    figure = plt.figure(figsize=(12, 6.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 0.82])
    activity_axis = figure.add_subplot(grid[0, 0])
    pause_axis = figure.add_subplot(grid[0, 1])
    distribution_axis = figure.add_subplot(grid[1, :])
    activity = graphs.get("activity") or {}
    activity_times, activity_values = activity.get("times", []), activity.get("values", [])
    if activity_times and activity_values:
        activity_axis.step(activity_times, activity_values, where="post", color="#168d82", linewidth=2)
        activity_axis.fill_between(activity_times, activity_values, step="post", alpha=0.12, color="#168d82")
        activity_axis.set_yticks([0, 1], ["Pause", "Speaking"])
        activity_axis.set_ylim(-0.15, 1.15)
        activity_axis.set_xlabel("Time (seconds)")
        activity_axis.set_title("Speaking and pause timeline")
        activity_axis.grid(axis="x", alpha=0.2)
    else:
        activity_axis.text(0.5, 0.5, "No speech activity was available", ha="center", va="center", transform=activity_axis.transAxes)
        activity_axis.set_axis_off()

    pauses = graphs.get("pauses") or {}
    pause_times, pause_values = pauses.get("times", []), pauses.get("values", [])
    if pause_times and pause_values:
        colors = ["#d64545" if value > 1.5 else "#527bb5" for value in pause_values]
        pause_axis.vlines(pause_times, 0, pause_values, color=colors, alpha=0.7, linewidth=1.5)
        pause_axis.scatter(pause_times, pause_values, c=colors, s=32, zorder=3)
        pause_axis.axhline(1.5, color="#d64545", linestyle="--", linewidth=1, label="1.5 s practice cue")
        pause_axis.set_xlabel("Time (seconds)")
        pause_axis.set_ylabel("Pause length (seconds)")
        pause_axis.set_title("Pause lengths across the recording")
        pause_axis.legend(frameon=False, fontsize=8)
        pause_axis.grid(alpha=0.2)
    else:
        pause_axis.text(0.5, 0.5, "No internal pauses of at least 0.25 seconds were detected", ha="center", va="center", transform=pause_axis.transAxes)
        pause_axis.set_axis_off()

    ranges = [(0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, float("inf"))]
    counts = [sum(low <= float(value) < high for value in pause_values) for low, high in ranges]
    labels = ["0.25–0.49 s", "0.5–0.99 s", "1.0–1.49 s", "1.5 s +"]
    colors = ["#4d88bd", "#4d88bd", "#4d88bd", "#d64545"]
    distribution_axis.bar(labels, counts, color=colors, width=0.62)
    distribution_axis.set_title("Pause duration groups")
    distribution_axis.set_ylabel("Number of pauses")
    distribution_axis.grid(axis="y", alpha=0.2)
    st.pyplot(figure, clear_figure=True)


st.title("🎙️ Speech Fluency Estimator")
st.write(
    "Record a sample on your phone or upload an audio file. The app returns a "
    "research estimate with fluency timing graphs and a local SHAP explanation."
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
    st.subheader("Live Audio Feedback")
    st.write("Use this live preview to see your speaking energy and pitch while you prepare!")
    components.html(visualizer_html, height=180)
    st.divider()

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
        tab_summary, tab_timing, tab_details, tab_model, tab_feedback = st.tabs([
            "📊 Summary", "⏱️ Timing Analysis", "📋 Detailed Metrics", "🧠 Model Info", "💡 Feedback"
        ])

        with tab_summary:
            st.subheader("Your speech profile")
            col1, col2 = st.columns([1, 1])
            with col1:
                render_speech_profile(indicators)
            with col2:
                render_radar_chart(indicators)

            st.subheader("Key fluency measures")
            metric_columns = st.columns(5)
            metric_rows = [
                ("Speaking share", indicators.get("speaking_time_ratio", 0), "{:.0%}", float(indicators.get("speaking_time_ratio", 0)) < 0.5),
                ("Pauses / minute", indicators.get("internal_pauses_per_minute", 0), "{:.1f}", float(indicators.get("internal_pauses_per_minute", 0)) > 12),
                ("Average pause", indicators.get("mean_pause_seconds", 0), "{:.2f} s", float(indicators.get("mean_pause_seconds", 0)) > 0.8),
                ("Long pause", indicators.get("long_pause_p90_seconds", 0), "{:.2f} s", float(indicators.get("long_pause_p90_seconds", 0)) > 1.5),
                ("Syllables / min (est)", indicators.get("energy_peak_rate_proxy_per_minute", 0), "{:.0f}", float(indicators.get("energy_peak_rate_proxy_per_minute", 0)) < 100),
            ]
            for column, (label, value, format_string, weak) in zip(metric_columns, metric_rows):
                column.metric(label, format_string.format(float(value)))
                column.caption("🔴 Practice cue" if weak else "Timing measure")

        with tab_timing:
            st.subheader("Speech rhythm across your recording")
            render_voice_graphs(result.get("voice_graphs"))

        with tab_details:
            st.subheader("Detailed speech indicators")
            st.dataframe(
                pd.DataFrame(
                    [{"Indicator": key.replace("_", " ").title(), "Value": value}
                     for key, value in indicators.items()]
                ),
                hide_index=True,
                use_container_width=True,
            )

        with tab_model:
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

        with tab_feedback:
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
