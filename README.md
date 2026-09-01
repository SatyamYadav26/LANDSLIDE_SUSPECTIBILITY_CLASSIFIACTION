# Landslide Susceptibility Classification Using Machine Learning

## 📌 Project Overview

This project focuses on **Landslide Susceptibility Classification** using three supervised machine learning algorithms:

* 🌳 Random Forest
* 📈 Logistic Regression
* ⚡ Support Vector Machine (SVM)

The models use environmental and soil-related factors to predict whether a given observation is associated with a landslide.

The target variable is:

* `0` → No Landslide
* `1` → Landslide

The performance of all three models is compared using different classification metrics such as **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze the landslide dataset.
2. Perform data preprocessing and cleaning.
3. Explore relationships between landslide-conditioning factors.
4. Train Random Forest, Logistic Regression, and SVM models.
5. Evaluate the performance of each model.
6. Compare the three machine learning algorithms.
7. Identify important features using Random Forest.
8. Classify observations into landslide and non-landslide classes.
9. Demonstrate landslide susceptibility levels using predicted probabilities.

---

## 📂 Dataset

The dataset used in this project is:

```text
landslide_dataset.csv
```

The dataset contains **2,000 observations and 10 columns**.

### Dataset Features

| Feature               | Description                           |
| --------------------- | ------------------------------------- |
| `Rainfall_mm`         | Rainfall measurement                  |
| `Slope_Angle`         | Slope angle of the terrain            |
| `Soil_Saturation`     | Degree of soil saturation             |
| `Vegetation_Cover`    | Amount of vegetation cover            |
| `Earthquake_Activity` | Earthquake/seismic activity indicator |
| `Proximity_to_Water`  | Proximity to water bodies             |
| `Landslide`           | Target variable                       |
| `Soil_Type_Gravel`    | Gravel soil indicator                 |
| `Soil_Type_Sand`      | Sand soil indicator                   |
| `Soil_Type_Silt`      | Silt soil indicator                   |

### Target Variable

```text
Landslide = 0 → No Landslide
Landslide = 1 → Landslide
```

---

## 🧠 Machine Learning Algorithms

### 1. Random Forest

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to make predictions.

Advantages:

* Handles nonlinear relationships.
* Can model interactions between features.
* Works well with numerical features.
* Provides feature importance.
* Generally does not require feature scaling.

Example:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)
```

---

### 2. Logistic Regression

Logistic Regression is a classification algorithm that estimates the probability of an observation belonging to a particular class.

It is used as an interpretable baseline model.

Feature scaling is applied using `StandardScaler`.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

lr.fit(X_train, y_train)
```

---

### 3. Support Vector Machine

Support Vector Machine (SVM) is a supervised learning algorithm that finds a decision boundary separating different classes.

An **RBF kernel** is used to allow nonlinear classification.

Feature scaling is applied before SVM training.

```python
from sklearn.svm import SVC

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
```

---

## 🔄 Project Workflow

```text
                    Dataset
                       |
                       v
              Data Preprocessing
                       |
                       v
            Exploratory Data Analysis
                       |
                       v
              Feature Selection
                       |
                       v
             Train-Test Split
                       |
          +------------+------------+
          |            |            |
          v            v            v
    Random Forest      LR          SVM
          |            |            |
          +------------+------------+
                       |
                       v
               Model Evaluation
                       |
                       v
       Accuracy / Precision / Recall
             F1-Score / ROC-AUC
                       |
                       v
              Model Comparison
                       |
                       v
             Best Performing Model
                       |
                       v
          Landslide Susceptibility
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**

---

## 📦 Installation

Make sure Python is installed on your system.

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 📁 Project Structure

```text
Landslide-Susceptibility-Classification/
│
├── landslide_dataset(1).csv
│
├── landslide_classification.py
│
├── README.md
│
└── results/
    ├── confusion_matrix_rf.png
    ├── confusion_matrix_lr.png
    ├── confusion_matrix_svm.png
    ├── roc_curve.png
    ├── feature_importance.png
    └── model_comparison.png
```

---

## 🔍 Data Preprocessing

The dataset is first loaded using Pandas:

```python
import pandas as pd

df = pd.read_csv("landslide_dataset(1).csv")
```

### Check dataset information

```python
print(df.info())
print(df.shape)
print(df.head())
```

### Check missing values

```python
print(df.isnull().sum())
```

### Check duplicate values

```python
print(df.duplicated().sum())
```

### Check target distribution

```python
print(df["Landslide"].value_counts())
```

---

## 📊 Exploratory Data Analysis

The project performs exploratory data analysis to understand the dataset.

### Class Distribution

The distribution of landslide and non-landslide observations is visualized using:

```python
sns.countplot(
    x="Landslide",
    data=df
)

plt.title("Landslide Class Distribution")
plt.show()
```

### Correlation Heatmap

A correlation matrix is generated to examine relationships among the variables.

```python
plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")
plt.show()
```

---

## ✂️ Feature and Target Separation

The target variable is separated from the input features:

```python
X = df.drop("Landslide", axis=1)

y = df["Landslide"]
```

Where:

```text
X → Input features
y → Landslide target
```

---

## 🧪 Train-Test Split

The dataset is divided into training and testing data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The split uses:

```text
80% → Training data
20% → Testing data
```

`stratify=y` is used to preserve the approximate class distribution in both sets.

---

## 📏 Feature Scaling

Feature scaling is particularly important for Logistic Regression and SVM.

`StandardScaler` is used through a Pipeline:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
```

Random Forest does not require feature scaling.

---

## 📈 Model Evaluation

The models are evaluated using:

### Accuracy

Measures the percentage of correctly classified observations.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Measures how many observations predicted as landslides were actually landslides.

### Recall

Measures how many actual landslide observations were correctly identified.

Recall is particularly important for landslide classification because false negatives represent actual landslide observations classified as non-landslide.

### F1-Score

F1-score provides a balance between precision and recall.

### ROC-AUC

ROC-AUC measures how well the model distinguishes between the two classes across different classification thresholds.

---

## 📋 Model Comparison

The final results are presented in a table:

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Random Forest       |        — |         — |      — |        — |       — |
| Logistic Regression |        — |         — |      — |        — |       — |
| SVM                 |        — |         — |      — |        — |       — |

The values should be generated by running the model on the dataset rather than manually assuming them.

---

## 📉 Confusion Matrix

A confusion matrix is generated for each model.

```text
                 Predicted
                0        1
Actual  0      TN       FP
        1      FN       TP
```

Where:

* **TN** = True Negative
* **FP** = False Positive
* **FN** = False Negative
* **TP** = True Positive

---

## 📈 ROC Curve

ROC curves are generated to compare the classification performance of all three algorithms.

The graph contains:

* Random Forest ROC curve
* Logistic Regression ROC curve
* SVM ROC curve
* Random classifier reference line

A higher ROC-AUC generally indicates better ability to distinguish the two classes.

---

## 🌳 Feature Importance

Random Forest is used to determine the relative importance of the input features.

```python
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)
```

This allows us to identify which environmental factors contribute most strongly to the Random Forest model's predictions.

> Feature importance describes the model's behavior and should not be interpreted by itself as proof of causal relationships.

---

## ⚙️ Hyperparameter Tuning

The project can be improved using `GridSearchCV`.

### Random Forest

Parameters that can be tuned include:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

### Logistic Regression

The following parameter can be tuned:

```text
C
```

### SVM

Important parameters include:

```text
C
gamma
```

Example:

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    model,
    parameters,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)
```

---

## 🔁 Cross-Validation

Five-fold stratified cross-validation can be used to obtain a more reliable estimate of model performance.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
```

---

## 🗺️ Susceptibility Classification

The probability produced by a classification model can be used to demonstrate susceptibility categories.

Example:

```text
Probability < 0.33
        ↓
       Low

0.33 - 0.66
        ↓
    Moderate

Probability >= 0.66
        ↓
       High
```

Python implementation:

```python
probability = model.predict_proba(new_location)[0][1]

if probability < 0.33:
    susceptibility = "Low"
elif probability < 0.66:
    susceptibility = "Moderate"
else:
    susceptibility = "High"

print("Susceptibility:", susceptibility)
```

> These thresholds are illustrative. For real-world or research-grade landslide susceptibility assessment, thresholds should be validated and scientifically justified.

---

## 🚀 How to Run the Project

### Step 1: Clone/download the project

Place all files in the same project directory.

### Step 2: Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Step 3: Run the Python program

```bash
python landslide_classification.py
```

### Step 4: Analyze the output

The program will generate:

* Dataset information
* Missing-value analysis
* Duplicate analysis
* Class distribution
* Correlation matrix
* Model predictions
* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrices
* ROC curve
* Feature importance
* Model comparison

---

## 📌 Limitations

* The dataset does not contain latitude and longitude.
* The dataset does not contain GIS raster layers.
* Therefore, the current project performs **tabular landslide classification**, not a true geographic susceptibility mapping.
* The dataset is not associated with a documented geographic study area.
* Model performance depends on the quality and representativeness of the available data.
* Probability thresholds for Low/Moderate/High susceptibility require validation before practical use.

---

## 🔮 Future Scope

The project can be extended by integrating real-world GIS and remote-sensing data such as:

* Digital Elevation Model (DEM)
* Slope
* Aspect
* Curvature
* Rainfall
* Lithology
* Soil type
* Land-use/Land-cover
* NDVI
* Distance from roads
* Distance from rivers
* Fault proximity
* Historical landslide inventory
* Latitude and longitude

With spatial data, the trained machine learning models can be applied to GIS raster layers to generate an actual **Landslide Susceptibility Map**.

---

## 📝 Conclusion

This project demonstrates the use of **Random Forest, Logistic Regression, and Support Vector Machine** for landslide susceptibility classification.

Environmental and soil-related factors are used as input variables, while the `Landslide` column is used as the target variable. The models are evaluated using multiple performance metrics, allowing their classification capabilities to be compared.

Random Forest additionally provides feature-importance information, which can be used to understand which input variables are most influential in the model's predictions.

The project provides a foundation for extending machine-learning-based landslide classification toward a complete **GIS-based landslide susceptibility mapping system** using real-world spatial datasets.

---

## 👨‍💻 Author

**Landslide Susceptibility Classification Project**

### Algorithms Used

```text
Random Forest
Logistic Regression
Support Vector Machine (SVM)
```

### Developed Using

```text
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
```
