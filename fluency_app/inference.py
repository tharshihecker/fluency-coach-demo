"""Load the trained model and return one prediction from an audio file."""

from __future__ import annotations

import tempfile
import json
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import torch
from torch import nn

from fluency_app.audio_features import TARGET_SR, extract_processed_file
from fluency_app.explainability import LocalShapExplainer, LocalTreeShapExplainer
from fluency_app.models import ResidualMLP
from fluency_app.speech_detection import detect_speech
from fluency_app.speech_content import SpeechContentRejected, validate_english_content
from preprocess_fluency_audio import process_one


class FluencyMLP(nn.Module):
    """The exact compact MLP architecture used in the final training cell."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FluencyPredictor:
    """Audio preprocessing, shared feature extraction, and model inference."""

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        metadata_path = self.artifact_dir / "selected_fluency_model_metadata.json"
        self.model_name = "Compact MLP"
        self.research_deep_model = False
        self.classical_model = False
        self.ensemble_model = False
        self.ensemble_weights: np.ndarray | None = None
        self.speech_device = torch.device("cpu")
        self.project_root = self.artifact_dir.resolve().parent
        self._shap_explainer: LocalShapExplainer | None = None
        self._tree_shap_explainer: LocalTreeShapExplainer | None = None
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            artifact_name = metadata.get("artifact", "")
            if artifact_name == "selected_fluency_deep_model.pt":
                self._load_research_deep(metadata)
                return
            if artifact_name == "selected_fluency_model.joblib":
                self._load_classical_model(metadata)
                return
            if artifact_name == "selected_fluency_ensemble.joblib":
                self._load_ensemble_model(metadata)
                return

        weights_path = self.artifact_dir / "selected_fluency_mlp_weights.pt"
        preprocessing_path = self.artifact_dir / "selected_fluency_mlp_preprocessing.joblib"
        if not weights_path.is_file() or not preprocessing_path.is_file():
            raise FileNotFoundError(
                "Model artifacts are missing. Run the final model training cells in "
                "speech_fluency_pipeline.ipynb first."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # weights_only=False: PyTorch 2.6 changed the default to True, but these
        # .pt files contain numpy arrays (opcode 149) saved during local training
        # and are trusted artifacts — False is safe here.
        self.weights = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.preprocessing = joblib.load(preprocessing_path)
        self.feature_columns = list(self.weights["feature_columns"])
        if self.feature_columns != list(self.preprocessing["feature_columns"]):
            raise ValueError("Saved model and preprocessing feature lists do not match.")

        architecture = self.weights.get("architecture", {})
        hidden = architecture.get("hidden_dims", [64, 32])
        if hidden != [64, 32]:
            raise ValueError(f"Unsupported saved MLP architecture: {hidden}")
        self.model = FluencyMLP(len(self.feature_columns))
        self.model.load_state_dict(self.weights["state_dict"])
        self.model.to(self.device).eval()
        self.target_mean = float(self.weights["target_mean"])
        self.target_std = float(self.weights["target_std"])

    def _load_classical_model(self, metadata: dict) -> None:
        """Load the fitted sklearn pipeline exported by notebook 04."""
        model_path = self.artifact_dir / metadata["artifact"]
        if not model_path.is_file():
            raise FileNotFoundError(f"The selected model artifact is missing: {model_path}")
        self.model = joblib.load(model_path)
        self.feature_columns = list(metadata["feature_columns"])
        if not hasattr(self.model, "predict"):
            raise TypeError("The selected classical artifact does not provide predict().")
        self.device = torch.device("cpu")
        self.speech_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = metadata.get("model", "Selected tree model")
        self.classical_model = True

    def _load_ensemble_model(self, metadata: dict) -> None:
        """Load the weighted sklearn ensemble exported by notebook 04."""
        model_path = self.artifact_dir / metadata["artifact"]
        if not model_path.is_file():
            raise FileNotFoundError(f"The selected ensemble artifact is missing: {model_path}")
        artifact = joblib.load(model_path)
        self.models = artifact["models"]
        self.ensemble_members = list(artifact["members"])
        self.ensemble_weights = np.asarray(
            artifact.get("weights", np.ones(len(self.ensemble_members)) / len(self.ensemble_members)),
            dtype=float,
        )
        if (len(self.ensemble_members) != len(self.ensemble_weights)
                or not np.isclose(self.ensemble_weights.sum(), 1.0)
                or np.any(self.ensemble_weights < 0)):
            raise ValueError("Saved ensemble members and weights are inconsistent.")
        self.feature_columns = list(metadata["feature_columns"])
        if set(self.ensemble_members) != set(self.models):
            raise ValueError("Saved ensemble member names do not match its models.")
        self.device = torch.device("cpu")
        self.speech_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = metadata.get("model", "Weighted Extra Trees ensemble")
        self.classical_model = True
        self.ensemble_model = True

    def _load_research_deep(self, metadata: dict) -> None:
        """Load the feature-selected residual MLP produced by notebook 04."""
        weights_path = self.artifact_dir / metadata["artifact"]
        preprocessing_path = self.artifact_dir / "selected_fluency_deep_preprocessing.joblib"
        if not weights_path.is_file() or not preprocessing_path.is_file():
            raise FileNotFoundError("The selected research model or its preprocessing artifact is missing.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # weights_only=False: trusted local artifact; contains numpy-backed data
        # incompatible with PyTorch 2.6's new weights_only=True default.
        self.weights = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.preprocessing = joblib.load(preprocessing_path)
        self.feature_columns = list(self.preprocessing["feature_columns"])
        self.selected_indices = np.asarray(self.preprocessing["selected_indices"], dtype=int)
        if len(self.selected_indices) != int(self.weights["n_selected_features"]):
            raise ValueError("Saved model and selected-feature preprocessing do not match.")
        self.model = ResidualMLP(len(self.selected_indices))
        self.model.load_state_dict(self.weights["state_dict"])
        self.model.to(self.device).eval()
        self.model_name = metadata.get("model", "Residual MLP")
        self.research_deep_model = True

    @property
    def gpu_name(self) -> str | None:
        return torch.cuda.get_device_name(0) if self.device.type == "cuda" else None

    def predict_file(
        self,
        audio_path: str | Path,
        original_name: str = "audio",
        explain: bool = True,
        language_only: bool = False,
        confirm_english: bool = False,
    ) -> dict:
        """Enhance one input, run the shared feature extractor, and predict fluency."""
        with tempfile.TemporaryDirectory(prefix="fluency-") as work_dir:
            processed_path = Path(work_dir) / "enhanced.wav"
            preprocessing_report = process_one(Path(audio_path), processed_path, normalize=True)
            speech_present, voiced_seconds = detect_speech(str(processed_path))
            duration = float(preprocessing_report["duration_seconds"])
            is_digital_silence = float(preprocessing_report["input_peak_dbfs"]) <= -110.0
            if is_digital_silence:
                speech_present = False
            if language_only:
                if is_digital_silence:
                    return {
                        "language_status": "silence", "detected_language": None,
                        "detected_language_code": None, "english_words_confirmed": False,
                        "message": "Silence detected. Start speaking to identify the language.",
                    }
                if not speech_present:
                    return {
                        "language_status": "unclear", "detected_language": None,
                        "detected_language_code": None, "english_words_confirmed": False,
                        "message": "No clear speech detected yet. Background noise is not treated as language.",
                    }
                if confirm_english:
                    return {
                        "language_status": "english", "detected_language": "English",
                        "detected_language_code": "en", "english_words_confirmed": True,
                        "language_check_overridden": True,
                        "message": "English confirmed by you. Keep speaking for a score.",
                    }
                accepted, reason, _transcript, language_code, language_name = validate_english_content(
                    processed_path, self.speech_device
                )
                return {
                    "language_status": "english" if language_code == "en" else "non_english",
                    "detected_language": language_name,
                    "detected_language_code": language_code,
                    "english_words_confirmed": accepted,
                    "message": reason if not accepted else "English speech detected.",
                }

            if not is_digital_silence:
                if not speech_present:
                    raise ValueError(
                        "No clear voiced speech was detected. Please record a spoken English sentence; background noise is not scored."
                    )
                if not confirm_english:
                    accepted, rejection_reason, _transcript, language_code, language_name = validate_english_content(
                        processed_path, self.speech_device
                    )
                    if not accepted:
                        raise SpeechContentRejected(rejection_reason, language_code, language_name)
                features = extract_processed_file(processed_path, filename=original_name)
                # The fitted model only saw complete recordings of at least 20 s,
                # with at least 10 s of speech and a 20% speaking-time ratio.
                # Keep short/noisy clips outside that training range from receiving
                # a misleading regression score (fan hum can pass the lightweight VAD).
                if (
                    float(features["duration_s"]) < 20.0
                    or float(features["speech_time_s"]) < 10.0
                    or float(features["speaking_time_ratio"]) < 0.20
                ):
                    raise ValueError(
                        "Not enough clear speech for a reliable score. This model needs a recording "
                        "of at least 20 seconds with about 10 seconds of speech. Reduce background "
                        "noise and speak in complete sentences, then try again."
                    )

        is_silent = not speech_present
        if is_silent:
            indicators = self._zero_speech_indicators(duration)
        else:
            indicators = self._indicators(features)
            indicators["voiced_speech_seconds"] = round(voiced_seconds, 2)
        if is_silent:
            # Explicit product behavior: complete digital silence receives 0.
            score = 0.0
        elif self.classical_model:
            row = pd.DataFrame([{key: features[key] for key in self.feature_columns}])
            if self.ensemble_model:
                member_predictions = np.asarray([
                    float(np.asarray(self.models[name].predict(row)).reshape(-1)[0])
                    for name in self.ensemble_members
                ])
                raw = float(np.dot(self.ensemble_weights, member_predictions))
            else:
                raw = float(np.asarray(self.model.predict(row)).reshape(-1)[0])
            score = float(np.clip(raw, 0.0, 10.0))
        else:
            row = pd.DataFrame([{key: features[key] for key in self.feature_columns}])
            transformed = self.preprocessing["imputer"].transform(row)
            if self.research_deep_model:
                transformed = transformed[:, self.selected_indices]
                transformed = self.preprocessing["scaler"].transform(transformed).astype(np.float32)
            else:
                transformed = self.preprocessing["scaler"].transform(transformed).astype(np.float32)
            with torch.no_grad():
                model_input = torch.as_tensor(transformed, device=self.device)
                raw = float(self.model(model_input).item())
                if not self.research_deep_model:
                    raw = raw * self.target_std + self.target_mean
            score = float(np.clip(raw, 0.0, 10.0))

        explanation: dict = {
            "baseline_score": None,
            "feature_impacts": [],
            "explanation_method": None,
            "explanation_note": "A local SHAP explanation is calculated for the final estimate.",
        }
        if explain and not is_silent and self.ensemble_model:
            explanation["explanation_note"] = (
                "The estimate uses a weighted ensemble; local feature attribution is unavailable for this ensemble."
            )
        elif explain and not is_silent and self.classical_model:
            try:
                if self._tree_shap_explainer is None:
                    self._tree_shap_explainer = LocalTreeShapExplainer(
                        model=self.model,
                        feature_columns=self.feature_columns,
                        project_root=self.project_root,
                    )
                explanation = self._tree_shap_explainer.explain(features)
            except Exception:
                explanation["explanation_note"] = (
                    "The estimate is ready, but its local SHAP explanation could not be calculated."
                )
        elif explain and not is_silent and self.research_deep_model:
            try:
                if self._shap_explainer is None:
                    self._shap_explainer = LocalShapExplainer(
                        model=self.model,
                        device=self.device,
                        preprocessing=self.preprocessing,
                        feature_columns=self.feature_columns,
                        selected_indices=self.selected_indices,
                        project_root=self.project_root,
                    )
                explanation = self._shap_explainer.explain(features)
            except Exception:
                # A SHAP dependency/background issue should never block a score.
                explanation["explanation_note"] = (
                    "The estimate is ready, but its local SHAP explanation could not be calculated."
                )

        return {
            "score": round(score, 2),
            "score_scale": {"minimum": 0, "maximum": 10},
            "silent_recording": is_silent,
            "speech_detected": not is_silent,
            "detected_language": "English" if not is_silent else None,
            "detected_language_code": "en" if not is_silent else None,
            "english_words_confirmed": not is_silent,
            "language_status": "english" if not is_silent else "silence",
            "language_check_overridden": bool(confirm_english and not is_silent),
            "indicators": indicators,
            **explanation,
            "model": self.model_name,
            "device": str(self.device),
            "feedback_status": "optional_ai_feedback",
            "message": (
                "No speech was detected. The score is set to 0."
                if is_silent
                else "English confirmed by you. Research estimate based on acoustic and speech-timing indicators."
                if confirm_english
                else "Research estimate based on acoustic and speech-timing indicators."
            ),
        }

    @staticmethod
    def _indicators(features: dict) -> dict[str, float | int]:
        """Expose understandable indicators, not the full model feature vector."""
        return {
            "duration_seconds": round(float(features["duration_s"]), 2),
            "speech_time_seconds": round(float(features["speech_time_s"]), 2),
            "speaking_time_ratio": round(float(features["speaking_time_ratio"]), 3),
            "internal_pauses_per_minute": round(float(features["pause_count_per_min"]), 2),
            "mean_pause_seconds": round(float(features["pause_mean_s"]), 3),
            "median_pause_seconds": round(float(features["pause_median_s"]), 3),
            "long_pause_p90_seconds": round(float(features["pause_p90_s"]), 3),
            "pitch_mean_hz": round(float(features["f0_hz_mean"]), 2),
            "pitch_range_hz": round(float(features["f0_range_hz"]), 2),
            "energy_peak_rate_proxy_per_minute": round(float(features["energy_peak_rate_proxy_min"]), 2),
        }

    @staticmethod
    def _zero_speech_indicators(duration: float) -> dict[str, float | int]:
        """Avoid reporting noise-derived pitch/pauses as speech indicators."""
        return {
            "duration_seconds": round(float(duration), 2),
            "speech_time_seconds": 0.0,
            "speaking_time_ratio": 0.0,
            "internal_pauses_per_minute": 0.0,
            "mean_pause_seconds": 0.0,
            "median_pause_seconds": 0.0,
            "long_pause_p90_seconds": 0.0,
            "pitch_mean_hz": 0.0,
            "pitch_range_hz": 0.0,
            "energy_peak_rate_proxy_per_minute": 0.0,
        }
