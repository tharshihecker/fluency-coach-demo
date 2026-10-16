"""Shared audio feature extraction used by both the notebook and API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy import signal

TARGET_SR = 16_000


def _mean_std(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"{prefix}_mean": float("nan"), f"{prefix}_std": float("nan")}
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
    }


def extract_audio_features(
    y: np.ndarray,
    sr: int = TARGET_SR,
    filename: str = "audio",
    include_voice_graphs: bool = False,
) -> dict[str, Any]:
    """Extract the same 51 numeric predictors used to train the final model."""
    y = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq")
        sr = TARGET_SR
    if y.size < sr:
        raise ValueError("Audio must be at least one second long.")

    if not np.any(np.abs(y) > 1e-8):
        # Preserve duration metadata, zero every speech/acoustic feature, and let
        # inference enforce the user-requested zero score for complete silence.
        zero: dict[str, float | str] = {
            "file": filename,
            "duration_s": float(y.size / sr),
            "silent_recording": 1.0,
        }
        for key in [
            "speech_time_s", "speaking_time_ratio", "pause_count_per_min",
            "pause_mean_s", "pause_median_s", "pause_p90_s", "pause_total_s",
            "vad_threshold_db", "speech_rms_mean", "zcr_mean", "zcr_std",
            "centroid_hz_mean", "centroid_hz_std", "bandwidth_hz_mean",
            "bandwidth_hz_std", "rolloff_hz_mean", "rolloff_hz_std",
            "flatness_mean", "flatness_std", "f0_hz_mean", "f0_hz_std",
            "f0_range_hz", "energy_peak_rate_proxy_min",
        ]:
            zero[key] = 0.0
        for i in range(1, 14):
            zero[f"mfcc{i:02d}_mean"] = 0.0
            zero[f"mfcc{i:02d}_std"] = 0.0
        if include_voice_graphs:
            zero["_voice_graphs"] = {"duration_seconds": round(y.size / sr, 2), "energy": [], "pitch": []}
        return zero

    frame = int(0.030 * sr)
    hop = int(0.010 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=True)[0]
    floor_db = 20 * np.log10(max(float(np.percentile(rms, 20)), 1e-7))
    threshold_db = max(floor_db + 10.0, -48.0)
    threshold = 10 ** (threshold_db / 20)
    active = rms >= threshold
    active = signal.medfilt(active.astype(np.uint8), kernel_size=11).astype(bool)
    idx = np.flatnonzero(active)
    if idx.size == 0:
        raise ValueError("No speech activity was detected in this recording.")

    first, last = idx[0], idx[-1]
    speech_seconds = float(active.sum() * hop / sr)
    total_seconds = float(y.size / sr)
    frame_times = librosa.frames_to_time(np.arange(rms.size), sr=sr, hop_length=hop)
    gaps = np.diff(np.r_[False, ~active, False].astype(np.int8))
    starts = np.flatnonzero(gaps == 1)
    ends = np.flatnonzero(gaps == -1)
    pause_s = ((ends - starts) * hop / sr).astype(float)
    inside = (starts > first) & (ends < last)
    pauses = pause_s[inside]
    pauses = pauses[pauses >= 0.25]
    speech = y[max(0, int(frame_times[first] * sr)):min(len(y), int(frame_times[last] * sr) + frame)]
    if speech.size < frame:
        speech = y

    feats: dict[str, float | str] = {
        "file": filename,
        "duration_s": total_seconds,
        "silent_recording": 0.0,
        "speech_time_s": speech_seconds,
        "speaking_time_ratio": speech_seconds / total_seconds,
        "pause_count_per_min": float(len(pauses) / max(total_seconds, 1e-6) * 60),
        "pause_mean_s": float(pauses.mean()) if len(pauses) else 0.0,
        "pause_median_s": float(np.median(pauses)) if len(pauses) else 0.0,
        "pause_p90_s": float(np.percentile(pauses, 90)) if len(pauses) else 0.0,
        "pause_total_s": float(pauses.sum()),
        "vad_threshold_db": float(threshold_db),
        "speech_rms_mean": float(np.sqrt(np.mean(speech**2))),
    }

    zcr = librosa.feature.zero_crossing_rate(speech, frame_length=frame, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=speech, sr=sr, n_fft=frame, hop_length=hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=speech, sr=sr, n_fft=frame, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(y=speech, sr=sr, n_fft=frame, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=speech, n_fft=frame, hop_length=hop)[0]
    for arr, name in [
        (zcr, "zcr"), (centroid, "centroid_hz"), (bandwidth, "bandwidth_hz"),
        (rolloff, "rolloff_hz"), (flatness, "flatness"),
    ]:
        feats.update(_mean_std(arr, name))

    mfcc = librosa.feature.mfcc(y=speech, sr=sr, n_mfcc=13, n_fft=frame, hop_length=hop)
    for i, row in enumerate(mfcc, 1):
        feats[f"mfcc{i:02d}_mean"] = float(np.mean(row))
        feats[f"mfcc{i:02d}_std"] = float(np.std(row))

    f0 = librosa.yin(speech, fmin=75, fmax=500, sr=sr, frame_length=1024, hop_length=hop)
    feats.update(_mean_std(f0, "f0_hz"))
    feats["f0_range_hz"] = float(np.nanpercentile(f0, 90) - np.nanpercentile(f0, 10))

    if include_voice_graphs:
        # Keep visualization payloads bounded while preserving the recording's
        # overall shape. Energy is relative to the clip peak; pitch gaps indicate
        # pauses or frames where a stable voiced pitch was not found.
        max_points = 160
        energy_db = 20 * np.log10(np.maximum(rms, 1e-6) / max(float(rms.max()), 1e-6))
        energy_db = np.clip(energy_db, -45, 0)
        pitch_times = frame_times[first] + librosa.frames_to_time(
            np.arange(f0.size), sr=sr, hop_length=hop
        )
        pitch_indices = np.rint(pitch_times * sr / hop).astype(int)
        pitch_active = np.zeros(f0.size, dtype=bool)
        valid_indices = (pitch_indices >= 0) & (pitch_indices < active.size)
        pitch_active[valid_indices] = active[pitch_indices[valid_indices]]
        pitch_values: list[float | None] = [
            round(float(value), 1) if is_active and 75 <= value <= 500 else None
            for value, is_active in zip(f0, pitch_active)
        ]

        def downsample(values: np.ndarray | list[float | None], times: np.ndarray) -> tuple[list[float], list[float | None]]:
            if len(values) <= max_points:
                indices = np.arange(len(values))
            else:
                indices = np.linspace(0, len(values) - 1, max_points).astype(int)
            return (
                [round(float(times[index]), 2) for index in indices],
                [float(values[index]) if values[index] is not None else None for index in indices],
            )

        energy_times, energy_values = downsample(energy_db, frame_times)
        sampled_pitch_times, sampled_pitch_values = downsample(pitch_values, pitch_times)
        feats["_voice_graphs"] = {
            "duration_seconds": round(total_seconds, 2),
            "energy": {"times": energy_times, "values": energy_values},
            "pitch": {"times": sampled_pitch_times, "values": sampled_pitch_values},
        }

    env = librosa.feature.rms(y=speech, frame_length=frame, hop_length=hop)[0]
    peaks, _ = signal.find_peaks(
        env,
        distance=int(0.10 * sr / hop),
        prominence=max(float(env.std()) * 0.35, 1e-4),
    )
    feats["energy_peak_rate_proxy_min"] = float(len(peaks) / max(speech_seconds, 1e-6) * 60)
    return feats


def extract_processed_file(
    path: str | Path,
    filename: str | None = None,
    include_voice_graphs: bool = False,
) -> dict[str, Any]:
    """Load an already-enhanced 16 kHz mono file and extract model features."""
    audio_path = Path(path)
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    return extract_audio_features(
        y, sr=sr, filename=filename or audio_path.name, include_voice_graphs=include_voice_graphs
    )
