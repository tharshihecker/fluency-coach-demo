# Public deployment bundle

This folder is the self-contained app repository for Streamlit Community Cloud.
It contains the app, the selected inference model, and numeric-only SHAP
background features. It excludes the expert-score workbook, training notebooks,
source recordings, filenames, and evaluation score tables.

The hosted app accepts audio on Streamlit Community Cloud, computes the score
there, and deletes its temporary local copy after inference. The score is a
research estimate from the selected weighted Extra Trees, Random Forest, and
XGBoost ensemble. Results include a speech profile, speech-activity and pause-
duration graphs, and a local permutation-SHAP explanation for the ensemble.
Red marks identify timing practice cues using coaching heuristics; they are not
model cutoffs or clinical standards. The speech profile is for the current
recording; it is not a saved user account or history.

To deploy, publish the contents of this folder in a public GitHub repository,
then select `app.py` as the Streamlit Community Cloud entrypoint.

Practice feedback is available on request. Without an AI provider key, the app
uses built-in local guidance. If a provider key is configured in Streamlit
Secrets, clicking the feedback button sends the score, numeric indicators, and
SHAP impacts to that provider—not the audio, file name, transcript, or expert
scores.
