# Mars Surface Atlas — Orange Embeddings Edition

This version restores the original Orange + Python feature pipeline.

## Training
1. In Orange, connect **Import Images → Image Embedding (SqueezeNet) → Save Data**.
2. Export the labelled table as CSV, keeping `category` and `n0`–`n999`.
3. In this app, open **01 · Train model** and upload the CSV.
4. The app uses the original Random Forest settings: 300 trees, random state 42, balanced class weights, stratified 75/25 holdout.
5. Download the trained `.joblib` model bundle.

## Prediction
1. Put new images in a separate folder.
2. In Orange, run **Import Images → Image Embedding (SqueezeNet) → Save Data** with the same widget setup.
3. Export the new embedding CSV. It must contain `n0`–`n999`; `category` is not needed.
4. Open **02 · Classify images**, load the trained model, and upload the new CSV.
5. Review/download predictions.

**Do not upload raw JPG/PNG files to the prediction uploader.** Orange must create their SqueezeNet embeddings first.

## Deploy
Upload `app.py`, `requirements.txt`, `README.md`, `LIBRARIES.md`, `.gitignore`, and `.streamlit/config.toml` to the GitHub repository. Set Streamlit's main file to `app.py`; Python 3.12 is recommended.

The app reads Orange CSVs using the original `header=0, skiprows=[1, 2]` format when those metadata rows are present, with a fallback for ordinary CSVs.
