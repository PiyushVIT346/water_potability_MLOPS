import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow.sklearn


mlflow.set_experiment("water_exp2")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

data = pd.read_csv(r"D:\projects\DVC project2\Experiment_using_MLFLOW\data\water_potability.csv")

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# --- Modular Preprocessing Functions ---

def get_imputation_params(df: pd.DataFrame, exclude_cols: list) -> dict:
    medians = {}
    cols_to_process = [col for col in df.columns if col not in exclude_cols]
    for col in cols_to_process:
        medians[col] = df[col].median()
    return medians

def impute_missing(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    df_out = df.copy()
    for col, median_val in medians.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(median_val)
    return df_out

def get_outlier_bounds(df: pd.DataFrame, exclude_cols: list) -> dict:
    bounds = {}
    cols_to_process = [col for col in df.columns if col not in exclude_cols]
    for col in cols_to_process:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return bounds

def cap_outliers(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    df_out = df.copy()
    for col, (lower, upper) in bounds.items():
        if col in df_out.columns:
            df_out[col] = np.clip(df_out[col], lower, upper)
    return df_out

def get_scaling_params(df: pd.DataFrame, exclude_cols: list) -> dict:
    params = {}
    cols_to_process = [col for col in df.columns if col not in exclude_cols]
    for col in cols_to_process:
        params[col] = {'mean': df[col].mean(), 'std': df[col].std()}
    return params

def scale_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_out = df.copy()
    for col, stats in params.items():
        if col in df_out.columns and stats['std'] != 0:
            df_out[col] = (df_out[col] - stats['mean']) / stats['std']
    return df_out

# --- Apply Preprocessing ---

exclude_cols = ['Potability']

# 1. Fit parameters on Training data only
medians = get_imputation_params(train_data, exclude_cols)

# Temporarily impute train to calculate accurate bounds and scaling params
train_temp = impute_missing(train_data, medians)
bounds = get_outlier_bounds(train_temp, exclude_cols)
train_temp = cap_outliers(train_temp, bounds)
scaling_params = get_scaling_params(train_temp, exclude_cols)

# 2. Transform Train Data
train_processed_data = impute_missing(train_data, medians)
train_processed_data = cap_outliers(train_processed_data, bounds)
train_processed_data = scale_features(train_processed_data, scaling_params)

# 3. Transform Test Data
test_processed_data = impute_missing(test_data, medians)
test_processed_data = cap_outliers(test_processed_data, bounds)
test_processed_data = scale_features(test_processed_data, scaling_params)


# --- Original ML Workflow ---

x_train = train_processed_data.iloc[:, :-1].values
y_train = train_processed_data.iloc[:, -1].values


n_estimators=500

with mlflow.start_run():

    clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(x_train, y_train)

    with open(r"model.pkl", 'wb') as f:
        pickle.dump(clf, f)

    x_test = test_processed_data.iloc[:, :-1].values
    y_test = test_processed_data.iloc[:, -1].values

    with open(r"model.pkl", 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1 score", f1)
    mlflow.log_param("n_estimators", n_estimators)

    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    mlflow.sklearn.log_model(clf,"GradientBoostingClassifier")

    mlflow.log_artifact(__file__) 

    mlflow.set_tag("author","data_thinkers")
    mlflow.set_tag("model","GB")

    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
