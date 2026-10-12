# Public deployment bundle

This folder is the self-contained app repository for Streamlit Community Cloud.
It contains the app, the selected inference model, and numeric-only SHAP
background features. It excludes the expert-score workbook, training notebooks,
source recordings, filenames, and evaluation score tables.

The hosted app accepts audio on Streamlit Community Cloud, computes the score
there, and deletes its temporary local copy after inference. The score is a
research estimate from the selected tuned Extra Trees model.

To deploy, publish the contents of this folder in a public GitHub repository,
then select `app.py` as the Streamlit Community Cloud entrypoint.

Optional feedback remains disabled until a `GROQ_API_KEY` is added to the
Streamlit app's Secrets settings. If enabled, it is called only when a visitor
clicks the feedback button and receives the score, selected numeric indicators,
and SHAP impacts—not the audio, file name, transcript, or expert scores.
