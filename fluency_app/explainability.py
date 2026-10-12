"""Local SHAP explanations for the selected fluency estimator.

Only acoustic predictor columns are loaded as the explanation background. The
expert score/target column is never read by this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


FEATURE_LABELS = {
    "duration_s": "Recording duration",
    "speech_time_s": "Time detected as speech",
    "speaking_time_ratio": "Share of clip spent speaking",
    "pause_count_per_min": "Internal pauses per minute",
    "pause_mean_s": "Average pause length",
    "pause_median_s": "Typical pause length",
    "pause_p90_s": "Long pause length (90th percentile)",
    "pause_total_s": "Total pause time",
    "vad_threshold_db": "Speech detection threshold",
    "speech_rms_mean": "Average speech energy",
    "zcr_mean": "Average zero-crossing rate",
    "centroid_hz_mean": "Average spectral brightness",
    "bandwidth_hz_mean": "Average spectral bandwidth",
    "rolloff_hz_mean": "Average spectral rolloff",
    "flatness_mean": "Average spectral flatness",
    "f0_hz_mean": "Average pitch",
    "f0_range_hz": "Pitch range",
    "energy_peak_rate_proxy_min": "Energy-peak rate estimate",
}


def display_feature_name(feature: str) -> str:
    """Turn a model column into a short, human-readable chart label."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.startswith("mfcc"):
        number, statistic = feature[4:].split("_", maxsplit=1)
        stat_label = "average" if statistic == "mean" else "variation"
        return f"Speech spectrum pattern {int(number)} ({stat_label})"
    for suffix, label in (("_std", "variation"), ("_mean", "average")):
        if feature.endswith(suffix):
            stem = feature[: -len(suffix)].replace("_", " ")
            return f"{stem.title()} ({label})"
    return feature.replace("_", " ").title()


class LocalTreeShapExplainer:
    """Explain the selected tree-ensemble pipeline against an acoustic background."""

    def __init__(self, *, model: Any, feature_columns: list[str], project_root: str | Path) -> None:
        import shap

        self.model = model
        self.feature_columns = list(feature_columns)
        self.imputer = model.named_steps["impute"]
        self.selector = model.named_steps["select"]
        self.estimator = model.named_steps["model"]
        self.selected_names = [self.feature_columns[i] for i in self.selector.indices_]
        self.project_root = Path(project_root)
        background_path = self.project_root / "outputs" / "explainability" / "shap_background.csv"
        if background_path.is_file():
            frame = pd.read_csv(background_path, usecols=self.selected_names)
            background = frame[self.selected_names].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        else:
            feature_file = self.project_root / "data" / "features" / "acoustic_features.csv"
            frame = pd.read_csv(feature_file, usecols=self.feature_columns)[self.feature_columns]
            frame = frame.sample(n=min(32, len(frame)), random_state=42)
            background = self.selector.transform(self.imputer.transform(frame))
        if not len(background):
            raise ValueError("The SHAP background has no acoustic feature rows.")
        self.explainer = shap.TreeExplainer(
            self.estimator,
            data=background,
            feature_perturbation="interventional",
            model_output="raw",
        )

    def explain(self, features: dict[str, Any], top_n: int = 6) -> dict[str, Any]:
        row = pd.DataFrame([{name: features.get(name, np.nan) for name in self.feature_columns}])
        selected = self.selector.transform(self.imputer.transform(row))
        explanation = self.explainer(selected, check_additivity=False)
        impacts = np.asarray(explanation.values, dtype=float).reshape(-1)
        base = np.asarray(explanation.base_values, dtype=float).reshape(-1)[0]
        order = np.argsort(np.abs(impacts))[::-1][:top_n]
        return {
            "baseline_score": round(float(base), 3),
            "feature_impacts": [
                {
                    "feature": display_feature_name(self.selected_names[i]),
                    "feature_key": self.selected_names[i],
                    "value": round(float(selected[0, i]), 4),
                    "impact": round(float(impacts[i]), 4),
                }
                for i in order
            ],
            "explanation_method": "SHAP TreeExplainer",
            "explanation_note": (
                "Positive bars raised this estimate relative to the model's background average; "
                "negative bars lowered it. These show model influences, not causes or expert judgments."
            ),
        }


class LocalShapExplainer:
    """Explain the saved Residual MLP against an acoustic-only background set."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        preprocessing: dict[str, Any],
        feature_columns: list[str],
        selected_indices: np.ndarray,
        project_root: str | Path,
    ) -> None:
        import shap

        self.model = model
        self.device = device
        self.preprocessing = preprocessing
        self.feature_columns = list(feature_columns)
        self.selected_indices = np.asarray(selected_indices, dtype=int)
        self.selected_names = [self.feature_columns[i] for i in self.selected_indices]
        self.project_root = Path(project_root)
        self.impute_values = np.asarray(
            preprocessing["imputer"].statistics_, dtype=float
        )[self.selected_indices]

        saved_background = self.project_root / "outputs" / "explainability" / "shap_background.csv"
        if saved_background.is_file():
            background_frame = pd.read_csv(saved_background, usecols=self.selected_names)
        else:
            # usecols intentionally excludes the expert score and file-name columns.
            feature_file = self.project_root / "data" / "features" / "acoustic_features.csv"
            background_frame = pd.read_csv(feature_file, usecols=self.feature_columns)
            background_frame = background_frame[self.selected_names]
            background_frame = background_frame.sample(
                n=min(32, len(background_frame)), random_state=42
            )
        background = background_frame.replace([np.inf, -np.inf], np.nan).to_numpy(float)
        if not len(background):
            raise ValueError("The SHAP background has no acoustic feature rows.")
        self.background = background
        self.explainer = shap.PermutationExplainer(
            self._predict,
            shap.maskers.Independent(background, max_samples=len(background)),
            feature_names=[display_feature_name(name) for name in self.selected_names],
            seed=42,
        )

    def _predict(self, raw_selected: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_selected, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        values = np.where(np.isfinite(values), values, self.impute_values)
        transformed = self.preprocessing["scaler"].transform(values).astype(np.float32)
        with torch.no_grad():
            tensor = torch.as_tensor(transformed, device=self.device)
            result = self.model(tensor).detach().cpu().numpy().reshape(-1)
        return np.clip(result, 0.0, 10.0)

    def explain(self, features: dict[str, Any], top_n: int = 6) -> dict[str, Any]:
        row = np.asarray([features.get(name, np.nan) for name in self.selected_names], dtype=float)
        row = np.where(np.isfinite(row), row, self.impute_values)
        # Three forward/backward permutations provide a useful latency/variance tradeoff.
        explanation = self.explainer(
            row.reshape(1, -1), max_evals=6 * len(self.selected_names) + 1
        )
        impacts = np.asarray(explanation.values, dtype=float).reshape(-1)
        order = np.argsort(np.abs(impacts))[::-1][:top_n]
        return {
            "baseline_score": round(float(np.asarray(explanation.base_values).reshape(-1)[0]), 3),
            "feature_impacts": [
                {
                    "feature": display_feature_name(self.selected_names[i]),
                    "feature_key": self.selected_names[i],
                    "value": round(float(row[i]), 4),
                    "impact": round(float(impacts[i]), 4),
                }
                for i in order
            ],
            "explanation_method": "SHAP permutation values",
            "explanation_note": "Positive bars push this estimate above the background average; negative bars push it below. These are model influences, not causes or expert judgments.",
        }
