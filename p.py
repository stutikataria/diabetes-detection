# Installation of required libraries
import numpy as np
import pandas as pd 
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
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
import warnings
warnings.simplefilter(action = "ignore") 

# Reading the dataset
df = pd.read_csv("diabetes.csv")


# Initial analysis
df.head()
df.info()
df.describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T
df["Outcome"].value_counts()*100/len(df)
df["Outcome"].value_counts()

# Histograms
fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.histplot(df.Age, bins=20, ax=ax[0,0]) 
sns.histplot(df.Pregnancies, bins=20, ax=ax[0,1]) 
sns.histplot(df.Glucose, bins=20, ax=ax[1,0]) 
sns.histplot(df.BloodPressure, bins=20, ax=ax[1,1]) 
sns.histplot(df.SkinThickness, bins=20, ax=ax[2,0])
sns.histplot(df.Insulin, bins=20, ax=ax[2,1])
sns.histplot(df.DiabetesPedigreeFunction, bins=20, ax=ax[3,0]) 
sns.histplot(df.BMI, bins=20, ax=ax[3,1]) 

# Correlation heatmap
plt.figure(figsize=(20,15))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="magma")
plt.title("Correlation Matrix")
plt.show()

# Replace zeros with NaN for specific columns
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Impute missing values by median grouped by Outcome
def median_target(var):
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp

columns = df.columns.drop("Outcome")
for i in columns:
    median_vals = median_target(i)
    df.loc[(df['Outcome'] == 0) & (df[i].isnull()), i] = median_vals[i][0]
    df.loc[(df['Outcome'] == 1) & (df[i].isnull()), i] = median_vals[i][1]

# Outlier handling with IQR for Insulin
Q1 = df.Insulin.quantile(0.25)
Q3 = df.Insulin.quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
df.loc[df["Insulin"] > upper, "Insulin"] = upper

# LOF Outlier detection
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=10)
df_scores = lof.fit_predict(df)
outlier_scores = lof.negative_outlier_factor_
threshold = np.sort(outlier_scores)[7]
outliers = outlier_scores > threshold
df = df[outliers]

# Feature engineering for BMI
NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype="category")
df["NewBMI"] = NewBMI

conditions = [
    (df["BMI"] < 18.5),
    (df["BMI"] <= 24.9),
    (df["BMI"] <= 29.9),
    (df["BMI"] <= 34.9),
    (df["BMI"] <= 39.9),
    (df["BMI"] > 39.9)
]
choices = NewBMI
for i, choice in enumerate(choices):
    df.loc[conditions[i], "NewBMI"] = choice

# Feature engineering for Insulin
def set_insulin(row):
    return "Normal" if 16 <= row["Insulin"] <= 166 else "Abnormal"
df = df.assign(NewInsulinScore=df.apply(set_insulin, axis=1))

# Feature engineering for Glucose
NewGlucose = pd.Series(["Low", "Normal", "Overweight", "Secret"], dtype="category")
df["NewGlucose"] = NewGlucose

df.loc[df["Glucose"] <= 70, "NewGlucose"] = "Low"
df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = "Normal"
df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = "Overweight"
df.loc[df["Glucose"] > 126 ,"NewGlucose"] = "Secret"

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=["NewBMI","NewInsulinScore", "NewGlucose"], drop_first=True)

# Final dataset prep
y = df["Outcome"]
categorical_df = df[[col for col in df.columns if col.startswith("New")]]
X = df.drop(["Outcome"] + list(categorical_df.columns), axis=1)

transformer = RobustScaler().fit(X)
X = pd.DataFrame(transformer.transform(X), columns=X.columns, index=X.index)
X = pd.concat([X, categorical_df], axis=1)

# Model comparisons
models = [
    ('LR', LogisticRegression(random_state=12345)),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier(random_state=12345)),
    ('RF', RandomForestClassifier(random_state=12345)),
    ('SVM', SVC(gamma='auto', random_state=12345)),
    ('XGB', GradientBoostingClassifier(random_state=12345)),
    ("LightGBM", LGBMClassifier(random_state=12345))
]

results, names = [], []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=12345, shuffle=True)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

plt.figure(figsize=(15,10))
plt.boxplot(results)
plt.xticks(range(1, len(names)+1), names)
plt.title("Model Comparison")
plt.show()

# Best Models - Tuning and Evaluation
rf_model = RandomForestClassifier(random_state=12345)
rf_params = {"n_estimators":[100,200], "max_features": [3,5], "min_samples_split": [2,10], "max_depth": [3,5,None]}
gs_rf = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=2).fit(X, y)
rf_tuned = RandomForestClassifier(**gs_rf.best_params_).fit(X, y)
print("RF Tuned Accuracy:", cross_val_score(rf_tuned, X, y, cv=10).mean())

lgbm = LGBMClassifier(random_state=12345)
lgbm_params = {"learning_rate": [0.01, 0.1], "n_estimators": [500, 1000], "max_depth": [3,5]}
gs_lgbm = GridSearchCV(lgbm, lgbm_params, cv=10, n_jobs=-1, verbose=2).fit(X, y)
lgbm_tuned = LGBMClassifier(**gs_lgbm.best_params_).fit(X, y)
print("LightGBM Tuned Accuracy:", cross_val_score(lgbm_tuned, X, y, cv=10).mean())

xgb = GradientBoostingClassifier(random_state=12345)
xgb_params = {"learning_rate": [0.01, 0.1], "min_samples_split": [0.1, 0.2], "max_depth": [3,5], "subsample": [0.9, 1.0], "n_estimators": [100,500]}
gs_xgb = GridSearchCV(xgb, xgb_params, cv=10, n_jobs=-1, verbose=2).fit(X, y)
xgb_tuned = GradientBoostingClassifier(**gs_xgb.best_params_).fit(X, y)
print("XGBoost Tuned Accuracy:", cross_val_score(xgb_tuned, X, y, cv=10).mean())
