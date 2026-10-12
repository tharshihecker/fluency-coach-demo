"""Speech-safe preprocessing for fluency-assessment recordings.

This pipeline intentionally preserves timing (including pauses and repetitions).
It applies only gentle enhancement: DC/rumble removal, adaptive spectral
denoising, a speech-band low-pass, and optional peak normalisation.

Example:
    .\\.venv\\Scripts\\python.exe preprocess_fluency_audio.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import librosa
import subprocess
import torch
from tqdm import tqdm


TARGET_SR = 16_000
PEAK_DBFS = -1.0
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}


def dbfs(value: float) -> float:
    """Convert a linear amplitude to dBFS, safely."""
    return 20.0 * np.log10(max(float(value), 1e-12))


def get_device(device_name: str) -> torch.device:
    """Resolve the requested device and fail clearly when CUDA is not available."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no compatible GPU is available in this environment.")
    return device


def process_one(path: Path, output: Path, normalize: bool, device: torch.device | str = "cpu") -> dict[str, float | str]:
    # Decode once for the input audit metrics. FFmpeg does the enhancement in C,
    # making the full dataset practical to process while retaining timing.
    audio, sample_rate = librosa.load(path, sr=TARGET_SR, mono=True, res_type="soxr_hq")
    audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    device = torch.device(device)
    audio_tensor = torch.as_tensor(audio, dtype=torch.float32, device=device)
    input_peak = torch.max(torch.abs(audio_tensor)).item()
    input_rms = torch.sqrt(torch.mean(audio_tensor ** 2)).item()

    # `afftdn` tracks changing room noise; nr=12 dB is deliberately modest.
    # The limiter is a -1 dBFS ceiling, not automatic gain control, so prosody
    # and within-recording amplitude changes remain intact.
    filters = "highpass=f=60:p=2,lowpass=f=7600:p=2,afftdn=nr=12:nf=-45:tn=1:ad=0.5:gs=6"
    if normalize:
        filters += f",alimiter=limit={10 ** (PEAK_DBFS / 20.0):.6f}:level=0:latency=1"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(TARGET_SR), "-af", filters,
        "-c:a", "pcm_s16le", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "FFmpeg preprocessing failed")
    processed, output_sr = librosa.load(output, sr=None, mono=True)
    if output_sr != TARGET_SR:
        raise RuntimeError(f"Unexpected output sample rate: {output_sr}")
    return {
        "source_file": path.name,
        "processed_file": output.name,
        "sample_rate_hz": sample_rate,
        "duration_seconds": round(len(processed) / output_sr, 4),
        "input_peak_dbfs": round(dbfs(input_peak), 2),
        "output_peak_dbfs": round(dbfs(np.max(np.abs(processed))), 2),
        "input_rms_dbfs": round(dbfs(input_rms), 2),
        "output_rms_dbfs": round(dbfs(np.sqrt(np.mean(processed**2))), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preserve timing while gently denoising speech.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw_audio"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_audio_advanced"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Use CUDA when available, otherwise fall back to CPU.")
    parser.add_argument("--no-normalize", action="store_true", help="Keep original recording gain.")
    args = parser.parse_args()

    device = get_device(args.device)
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU: CUDA is not available on this machine.")

    source_files = sorted((p for p in args.input_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS), key=lambda p: p.name)
    if not source_files:
        raise FileNotFoundError(f"No supported audio files found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Several extensions with the same stem would otherwise overwrite the same
    # <stem>.wav. Prefer the only non-silent source when the collision is a
    # digital-silence duplicate; fail loudly if multiple distinct recordings
    # share one stem because the label key cannot disambiguate them.
    groups: dict[str, list[Path]] = {}
    for path in source_files:
        groups.setdefault(path.stem.casefold(), []).append(path)
    files: list[Path] = []
    duplicates: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    for same_stem in groups.values():
        if len(same_stem) == 1:
            files.append(same_stem[0])
            continue
        source_rms = {}
        for path in same_stem:
            audio, _ = librosa.load(path, sr=TARGET_SR, mono=True, res_type="soxr_hq")
            source_rms[path] = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        non_silent = [path for path in same_stem if source_rms[path] > 1e-7]
        if len(non_silent) == 1:
            selected = non_silent[0]
            files.append(selected)
            for ignored in same_stem:
                if ignored != selected:
                    duplicates.append({
                        "stem": selected.stem,
                        "selected_source": selected.name,
                        "ignored_source": ignored.name,
                        "reason": "Digital-silence duplicate; kept the only non-silent source.",
                    })
        else:
            failed.append({
                "source_file": ";".join(path.name for path in same_stem),
                "error": "Duplicate filename stem maps to one output/label key and cannot be resolved automatically.",
            })
    files.sort(key=lambda p: p.name)

    completed = []
    for path in tqdm(files, desc="Enhancing recordings", unit="file"):
        try:
            completed.append(process_one(path, args.output_dir / f"{path.stem}.wav", not args.no_normalize, device=device))
        except Exception as exc:  # retain a complete audit trail instead of silently skipping a file
            failed.append({"source_file": path.name, "error": str(exc)})

    columns = list(completed[0]) if completed else ["source_file"]
    with (args.output_dir / "preprocessing_report.csv").open("w", newline="", encoding="utf-8") as report:
        writer = csv.DictWriter(report, fieldnames=columns)
        writer.writeheader()
        writer.writerows(completed)
    with (args.output_dir / "preprocessing_errors.csv").open("w", newline="", encoding="utf-8") as report:
        writer = csv.DictWriter(report, fieldnames=["source_file", "error"])
        writer.writeheader()
        writer.writerows(failed)
    with (args.output_dir / "preprocessing_duplicates.csv").open("w", newline="", encoding="utf-8") as report:
        writer = csv.DictWriter(report, fieldnames=["stem", "selected_source", "ignored_source", "reason"])
        writer.writeheader()
        writer.writerows(duplicates)

    print(f"Completed: {len(completed)}/{len(source_files)} source files ({len(duplicates)} silent duplicate(s) ignored)")
    print(f"Output: {args.output_dir.resolve()}")
    print(f"Failures: {len(failed)}")
    if failed:
        raise RuntimeError("Some files could not be processed; see preprocessing_errors.csv")


if __name__ == "__main__":
    main()
