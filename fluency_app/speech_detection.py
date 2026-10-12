"""Conservative speech-presence gate for noisy recordings.

This lightweight gate looks for voiced, periodic speech frames after the normal
denoising pass. It is not a transcript or a replacement for trained VAD; it
prevents steady broadband/room noise from being treated as speech activity.
"""

from __future__ import annotations

import librosa
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from fluency_app.audio_features import TARGET_SR


def detect_speech(path: str) -> tuple[bool, float]:
    """Return whether the processed clip contains enough voiced speech.

    Requires at least 0.24 seconds of voiced frames, including one 0.15-second
    continuous run. Voicing is estimated by normalized autocorrelation over
    human speech pitch periods (75–500 Hz), in addition to an adaptive energy
    floor. This rejects most stationary background noise after denoising.
    """
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    if sr != TARGET_SR or y.size < int(0.25 * TARGET_SR):
        return False, 0.0

    frame_length = 480  # 30 ms at 16 kHz
    hop = 160  # 10 ms
    frames = sliding_window_view(y, frame_length)[::hop]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    floor_db = 20 * np.log10(max(float(np.percentile(rms, 20)), 1e-7))
    active = rms >= 10 ** (max(floor_db + 10.0, -48.0) / 20)
    if not np.any(active):
        return False, 0.0

    # FFT-based autocorrelation in batches avoids a large temporary matrix for
    # long recordings. Normalize each lag by the overlapping signal energy.
    fft_size = 1 << (2 * frame_length - 1).bit_length()
    low_lag = int(TARGET_SR / 500)
    high_lag = int(TARGET_SR / 75)
    periodic = np.zeros(len(frames), dtype=bool)
    active_indices = np.flatnonzero(active)
    for start in range(0, len(active_indices), 128):
        selected = active_indices[start : start + 128]
        batch = frames[selected].astype(np.float32, copy=False)
        spectrum = np.fft.rfft(batch, n=fft_size, axis=1)
        ac = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size, axis=1)
        energy = np.sum(batch * batch, axis=1)
        best = np.zeros(len(selected), dtype=np.float32)
        for lag in range(low_lag, high_lag + 1):
            left = np.sum(batch[:, :-lag] ** 2, axis=1)
            right = np.sum(batch[:, lag:] ** 2, axis=1)
            denom = np.sqrt(left * right) + 1e-10
            best = np.maximum(best, ac[:, lag] / denom)
        periodic[selected] = (best >= 0.38) & (energy > 1e-9)

    # Bridge tiny gaps from stop consonants, then require enough voicing and a
    # sustained segment; isolated tonal noise cannot trigger speech by itself.
    bridged = periodic.copy()
    for offset in (1, 2):
        bridged[offset:-offset] |= periodic[:-2 * offset] & periodic[2 * offset:]
    changes = np.diff(np.r_[False, bridged, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    longest_seconds = float(np.max(ends - starts) * hop / TARGET_SR) if len(starts) else 0.0
    voiced_seconds = float(np.count_nonzero(periodic) * hop / TARGET_SR)
    speech_present = voiced_seconds >= 0.24 and longest_seconds >= 0.15
    return speech_present, voiced_seconds if speech_present else 0.0
