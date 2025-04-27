# Data manipulation and analysis
import numpy as np
import pandas as pd

# Statistical modeling
import statsmodels.api as sm

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning models and utilities
from sklearn.preprocessing import scale, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import LocalOutlierFactor

# Ignore warnings for cleaner output
import warnings
warnings.simplefilter(action="ignore")

# Load the Pima Indians Diabetes Dataset
df = pd.read_csv("data/diabetes.csv")   # Make sure you have the dataset in a "data" folder

# Display the first few rows
df.head()

# Checking basic information about the dataset
df.info()

# Summary statistics
df.describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

# Checking target class distribution
print(df["Outcome"].value_counts())
print(df["Outcome"].value_counts(normalize=True) * 100)

# Visualizing the distribution of features
fig, ax = plt.subplots(4, 2, figsize=(16, 16))

sns.histplot(df.Age, bins=20, ax=ax[0,0])
sns.histplot(df.Pregnancies, bins=20, ax=ax[0,1])
sns.histplot(df.Glucose, bins=20, ax=ax[1,0])
sns.histplot(df.BloodPressure, bins=20, ax=ax[1,1])
sns.histplot(df.SkinThickness, bins=20, ax=ax[2,0])
sns.histplot(df.Insulin, bins=20, ax=ax[2,1])
sns.histplot(df.DiabetesPedigreeFunction, bins=20, ax=ax[3,0])
sns.histplot(df.BMI, bins=20, ax=ax[3,1])

plt.tight_layout()
plt.show()

# Save figure
fig.savefig("images/feature_distributions.png")

plt.figure(figsize=(20, 15))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="magma")
plt.title("Correlation Matrix")
plt.show()

# Save heatmap
plt.savefig("images/correlation_matrix.png")

# Some features cannot be zero (e.g., Glucose, BloodPressure, etc.)
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.NaN)

# Fill missing values based on the median grouped by Outcome

def median_target(var):
    temp = df[df[var].notnull()]
    return temp.groupby('Outcome')[var].median()

for col in cols_with_zero:
    medians = median_target(col)
    df.loc[(df['Outcome'] == 0) & (df[col].isnull()), col] = medians[0]
    df.loc[(df['Outcome'] == 1) & (df[col].isnull()), col] = medians[1]

# Handle outliers in Insulin using IQR
Q1 = df['Insulin'].quantile(0.25)
Q3 = df['Insulin'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df['Insulin'] = np.where(df['Insulin'] > upper_bound, upper_bound, df['Insulin'])

lof = LocalOutlierFactor(n_neighbors=10)
outlier_pred = lof.fit_predict(df)
outlier_scores = lof.negative_outlier_factor_

threshold = np.sort(outlier_scores)[7]  # Pick 7th lowest score
outliers = outlier_scores > threshold

# Keep only non-outliers
df = df[outliers]

# Create new BMI categories
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi <= 24.9:
        return "Normal"
    elif bmi <= 29.9:
        return "Overweight"
    elif bmi <= 34.9:
        return "Obesity 1"
    elif bmi <= 39.9:
        return "Obesity 2"
    else:
        return "Obesity 3"

df['NewBMI'] = df['BMI'].apply(bmi_category)

def insulin_score(insulin):
    return "Normal" if 16 <= insulin <= 166 else "Abnormal"

df['NewInsulinScore'] = df['Insulin'].apply(insulin_score)

def glucose_category(glucose):
    if glucose <= 70:
        return "Low"
    elif glucose <= 99:
        return "Normal"
    elif glucose <= 126:
        return "Overweight"
    else:
        return "Secret"

df['NewGlucose'] = df['Glucose'].apply(glucose_category)

# One-hot encode the new categorical features
df = pd.get_dummies(df, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)

# Target and Features
y = df["Outcome"]
X = df.drop(columns=["Outcome"])

# Scale numeric features
transformer = RobustScaler().fit(X)
X_scaled = pd.DataFrame(transformer.transform(X), columns=X.columns)

X = X_scaled

# List of models
models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('KNN', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('SVM', SVC(gamma='auto', random_state=42)),
    ('XGBoost', GradientBoostingClassifier(random_state=42)),
    ('LightGBM', LGBMClassifier(random_state=42))
]

# Compare models
results, names = [], []

for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

# Visual comparison
plt.figure(figsize=(15,10))
plt.boxplot(results)
plt.xticks(range(1, len(names)+1), names)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.show()

# Save the plot
plt.savefig("images/model_comparison_boxplot.png")

rf_params = {
    "n_estimators": [100, 200],
    "max_features": [3, 5],
    "min_samples_split": [2, 10],
    "max_depth": [3, 5, None]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=10, n_jobs=-1, verbose=2)
rf_grid.fit(X, y)

rf_best = rf_grid.best_estimator_
print("Tuned Random Forest Accuracy:", cross_val_score(rf_best, X, y, cv=10).mean())

lgbm_params = {
    "learning_rate": [0.01, 0.1],
    "n_estimators": [500, 1000],
    "max_depth": [3, 5]
}

lgbm_grid = GridSearchCV(LGBMClassifier(random_state=42), lgbm_params, cv=10, n_jobs=-1, verbose=2)
lgbm_grid.fit(X, y)

lgbm_best = lgbm_grid.best_estimator_
print("Tuned LightGBM Accuracy:", cross_val_score(lgbm_best, X, y, cv=10).mean())

xgb_params = {
    "learning_rate": [0.01, 0.1],
    "min_samples_split": [0.1, 0.2],
    "max_depth": [3, 5],
    "subsample": [0.9, 1.0],
    "n_estimators": [100, 500]
}

xgb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), xgb_params, cv=10, n_jobs=-1, verbose=2)
xgb_grid.fit(X, y)

xgb_best = xgb_grid.best_estimator_
print("Tuned XGBoost Accuracy:", cross_val_score(xgb_best, X, y, cv=10).mean())
