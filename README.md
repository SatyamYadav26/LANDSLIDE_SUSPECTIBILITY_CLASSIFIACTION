# Landslide Susceptibility Classification - Streamlit App

Streamlit dashboard for the provided landslide dataset using Random Forest, Logistic Regression, and SVM.

## Features
- Dataset overview and EDA
- Target distribution and correlation matrix
- Model comparison: Accuracy, Precision, Recall, F1, ROC-AUC
- Classification reports
- Confusion matrices
- ROC curve comparison
- Random Forest feature importance
- Interactive landslide prediction

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Upload `app.py`, `landslide_dataset.csv`, and `requirements.txt` to GitHub, then deploy `app.py` from Streamlit Community Cloud.

## Dataset columns
`Rainfall_mm`, `Slope_Angle`, `Soil_Saturation`, `Vegetation_Cover`, `Earthquake_Activity`, `Proximity_to_Water`, `Landslide`, `Soil_Type_Gravel`, `Soil_Type_Sand`, `Soil_Type_Silt`
