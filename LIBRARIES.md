# Libraries

| Library | Use |
|---|---|
| Streamlit | Web interface, CSV/model uploads, metrics, tables, downloads. |
| Pandas | Reads Orange-exported CSVs and prepares tabular data. |
| NumPy | Converts the 1,000 embedding columns to arrays and validates values. |
| Scikit-learn | Trains Random Forest and calculates evaluation metrics. |
| Joblib | Saves and loads the trained model bundle. |

Orange Data Mining is an external desktop application, not a Python dependency. Its Image Embedding widget performs SqueezeNet feature extraction. This app consumes those exported features and does not recreate them with Torchvision.
