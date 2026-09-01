import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("landslide_dataset.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. DATA INFORMATION
# ============================================================

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nTarget Distribution:")
print(df["Landslide"].value_counts())


# ============================================================
# 3. TARGET DISTRIBUTION
# ============================================================

plt.figure(figsize=(6, 4))

sns.countplot(
    x="Landslide",
    data=df
)

plt.title("Landslide Class Distribution")

plt.show()


# ============================================================
# 4. CORRELATION MATRIX
# ============================================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.show()


# ============================================================
# 5. FEATURES AND TARGET
# ============================================================

X = df.drop("Landslide", axis=1)

y = df["Landslide"]


# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 7. RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

y_prob_rf = rf.predict_proba(X_test)[:, 1]


# ============================================================
# 8. LOGISTIC REGRESSION
# ============================================================

lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

y_prob_lr = lr.predict_proba(X_test)[:, 1]


# ============================================================
# 9. SVM
# ============================================================

svm = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        C=1,
        gamma="scale",
        probability=True,
        random_state=42
    ))
])

svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

y_prob_svm = svm.predict_proba(X_test)[:, 1]


# ============================================================
# 10. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, y_test, prediction, probability):

    print("\n===================================")
    print(name)
    print("===================================")

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction
    )

    recall = recall_score(
        y_test,
        prediction
    )

    f1 = f1_score(
        y_test,
        prediction
    )

    auc = roc_auc_score(
        y_test,
        probability
    )

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC-AUC  :", auc)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            prediction
        )
    )

    return [
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]


# ============================================================
# 11. EVALUATE MODELS
# ============================================================

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    y_pred_rf,
    y_prob_rf
)

lr_results = evaluate_model(
    "Logistic Regression",
    y_test,
    y_pred_lr,
    y_prob_lr
)

svm_results = evaluate_model(
    "SVM",
    y_test,
    y_pred_svm,
    y_prob_svm
)


# ============================================================
# 12. COMPARISON TABLE
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Random Forest",
        "Logistic Regression",
        "SVM"
    ],

    "Accuracy": [
        rf_results[0],
        lr_results[0],
        svm_results[0]
    ],

    "Precision": [
        rf_results[1],
        lr_results[1],
        svm_results[1]
    ],

    "Recall": [
        rf_results[2],
        lr_results[2],
        svm_results[2]
    ],

    "F1 Score": [
        rf_results[3],
        lr_results[3],
        svm_results[3]
    ],

    "ROC-AUC": [
        rf_results[4],
        lr_results[4],
        svm_results[4]
    ]
})


print("\nFINAL MODEL COMPARISON")
print(results)


# ============================================================
# 13. MODEL COMPARISON GRAPH
# ============================================================

results.set_index("Model").plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Model Performance Comparison")

plt.ylabel("Score")

plt.ylim(0, 1.05)

plt.xticks(rotation=0)

plt.show()


# ============================================================
# 14. CONFUSION MATRICES
# ============================================================

models = {
    "Random Forest": y_pred_rf,
    "Logistic Regression": y_pred_lr,
    "SVM": y_pred_svm
}

for name, prediction in models.items():

    cm = confusion_matrix(
        y_test,
        prediction
    )

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(name + " Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.show()


# ============================================================
# 15. ROC CURVE
# ============================================================

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    y_prob_rf
)

fpr_lr, tpr_lr, _ = roc_curve(
    y_test,
    y_prob_lr
)

fpr_svm, tpr_svm, _ = roc_curve(
    y_test,
    y_prob_svm
)


plt.figure(figsize=(8, 6))

plt.plot(
    fpr_rf,
    tpr_rf,
    label="Random Forest"
)

plt.plot(
    fpr_lr,
    tpr_lr,
    label="Logistic Regression"
)

plt.plot(
    fpr_svm,
    tpr_svm,
    label="SVM"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.show()


# ============================================================
# 16. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": rf.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance)


plt.figure(figsize=(10, 6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title(
    "Random Forest Feature Importance"
)

plt.show()