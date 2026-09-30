import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve)

st.set_page_config(page_title="Landslide Susceptibility Classification", page_icon="🌋", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("landslide_dataset.csv")

@st.cache_resource
def train_models(df):
    X = df.drop("Landslide", axis=1)
    y = df["Landslide"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))]),
        "SVM": Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=42))])
    }

    results, predictions, probabilities = [], {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        predictions[name], probabilities[name] = pred, prob
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob)
        })
    return models, X_test, y_test, predictions, probabilities, pd.DataFrame(results)

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Put 'landslide_dataset.csv' in the same folder as app.py.")
    st.stop()

models, X_test, y_test, predictions, probabilities, results = train_models(df)

st.title("🌋 Landslide Susceptibility Classification")
st.write("Machine learning classification using Random Forest, Logistic Regression, and Support Vector Machine (SVM).")

page = st.sidebar.radio("Navigation", ["Dashboard", "Model Comparison", "Confusion Matrices", "ROC Curves", "Feature Importance", "Predict Susceptibility", "Dataset"])

if page == "Dashboard":
    st.header("📊 Project Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", len(df)); c2.metric("Features", df.shape[1]-1)
    c3.metric("Landslide Samples", int(df["Landslide"].sum())); c4.metric("Non-Landslide Samples", int((df["Landslide"] == 0).sum()))
    st.subheader("Dataset Preview"); st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Target Distribution")
    fig, ax = plt.subplots(figsize=(7,4)); sns.countplot(x="Landslide", data=df, ax=ax); ax.set_title("Landslide Class Distribution"); st.pyplot(fig); plt.close(fig)
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(11,7)); sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax); ax.set_title("Correlation Matrix"); st.pyplot(fig); plt.close(fig)

elif page == "Model Comparison":
    st.header("🤖 Model Performance Comparison")
    st.dataframe(results.style.format({c:"{:.4f}" for c in results.columns if c != "Model"}), use_container_width=True)
    fig, ax = plt.subplots(figsize=(11,6)); results.set_index("Model").plot(kind="bar", ax=ax); ax.set_ylim(0,1.05); ax.set_ylabel("Score"); ax.set_title("Model Performance Comparison"); ax.tick_params(axis="x", rotation=0); st.pyplot(fig); plt.close(fig)
    selected = st.selectbox("Classification report", list(models.keys()))
    report = classification_report(y_test, predictions[selected], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.4f}"), use_container_width=True)

elif page == "Confusion Matrices":
    st.header("🔲 Confusion Matrix")
    selected = st.selectbox("Select model", list(models.keys()))
    cm = confusion_matrix(y_test, predictions[selected])
    fig, ax = plt.subplots(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Landslide","Landslide"], yticklabels=["No Landslide","Landslide"], ax=ax); ax.set_title(f"{selected} Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); st.pyplot(fig); plt.close(fig)

elif page == "ROC Curves":
    st.header("📈 ROC Curve Comparison")
    fig, ax = plt.subplots(figsize=(9,6))
    for name in models:
        fpr, tpr, _ = roc_curve(y_test, probabilities[name]); auc = roc_auc_score(y_test, probabilities[name]); ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0,1],[0,1],"--",label="Random Classifier"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve Comparison"); ax.legend(); ax.grid(alpha=.25); st.pyplot(fig); plt.close(fig)

elif page == "Feature Importance":
    st.header("🌳 Random Forest Feature Importance")
    importance = pd.DataFrame({"Feature":X_test.columns,"Importance":models["Random Forest"].feature_importances_}).sort_values("Importance", ascending=False)
    st.dataframe(importance.style.format({"Importance":"{:.4f}"}), use_container_width=True)
    fig, ax = plt.subplots(figsize=(10,6)); sns.barplot(x="Importance", y="Feature", data=importance, ax=ax); ax.set_title("Random Forest Feature Importance"); st.pyplot(fig); plt.close(fig)

elif page == "Predict Susceptibility":
    st.header("🔮 Predict Landslide Susceptibility")
    selected = st.selectbox("Prediction model", list(models.keys()))
    c1,c2 = st.columns(2)
    with c1:
        rainfall=st.number_input("Rainfall (mm)", min_value=0.0, value=100.0, step=1.0)
        slope=st.number_input("Slope Angle", min_value=0.0, value=30.0, step=1.0)
        saturation=st.number_input("Soil Saturation", min_value=0.0, value=0.50, step=0.01)
        vegetation=st.number_input("Vegetation Cover", min_value=0.0, value=0.50, step=0.01)
    with c2:
        earthquake=st.number_input("Earthquake Activity", min_value=0.0, value=0.20, step=0.01)
        water=st.number_input("Proximity to Water", min_value=0.0, value=0.50, step=0.01)
        soil=st.selectbox("Soil Type", ["Gravel","Sand","Silt"])
    data = pd.DataFrame([{
        "Rainfall_mm":rainfall,"Slope_Angle":slope,"Soil_Saturation":saturation,"Vegetation_Cover":vegetation,
        "Earthquake_Activity":earthquake,"Proximity_to_Water":water,
        "Soil_Type_Gravel":int(soil=="Gravel"),"Soil_Type_Sand":int(soil=="Sand"),"Soil_Type_Silt":int(soil=="Silt")
    }])
    if st.button("🚀 Predict Landslide Susceptibility", use_container_width=True):
        model=models[selected]; pred=model.predict(data)[0]; prob=model.predict_proba(data)[0,1]
        if pred==1: st.error("⚠️ Predicted Class: LANDSLIDE")
        else: st.success("✅ Predicted Class: NO LANDSLIDE")
        st.metric("Landslide Probability", f"{prob*100:.2f}%")
        level="Low" if prob<.33 else "Moderate" if prob<.66 else "High"
        st.subheader("Susceptibility Level"); st.progress(float(prob)); st.write(f"### {level} Susceptibility")
        st.subheader("Input Values"); st.dataframe(data, use_container_width=True)
        st.caption("Low/Moderate/High thresholds are illustrative probability categories, not official geological hazard levels.")

elif page == "Dataset":
    st.header("📁 Dataset Information")
    st.write("Dataset shape:", df.shape)
    info=pd.DataFrame({"Column":df.columns,"Data Type":df.dtypes.astype(str).values,"Missing Values":df.isnull().sum().values,"Unique Values":[df[c].nunique() for c in df.columns]})
    st.subheader("Column Information"); st.dataframe(info,use_container_width=True)
    st.subheader("Descriptive Statistics"); st.dataframe(df.describe().T,use_container_width=True)
    st.subheader("Duplicate Rows"); st.write(int(df.duplicated().sum()))
    st.subheader("Full Dataset"); st.dataframe(df,use_container_width=True)

st.sidebar.divider(); st.sidebar.caption("Random Forest • Logistic Regression • SVM")
