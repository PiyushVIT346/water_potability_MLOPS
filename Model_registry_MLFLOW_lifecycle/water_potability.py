import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("water_potability_hp")

data_path = r"D:\projects\DVC project2\Experiment_using_MLFLOW\data\water_potability.csv"
data = pd.read_csv(data_path)

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

# Prepare training data
# x_train = train_processed_data.iloc[:, :-1].values
# y_train = train_processed_data.iloc[:, -1].values

x_train = train_processed_data.drop('Potability', axis=1)
y_train = train_processed_data['Potability']

rf = RandomForestClassifier(random_state=42)
param_dist = {
    "n_estimators": [100, 200, 300, 400, 500, 1000],
    "max_depth": [None, 10, 20, 30, 40],
}

random_search = RandomizedSearchCV(
    estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, n_jobs=-1, verbose=1, random_state=42
)

with mlflow.start_run(run_name="Random forest tuning") as parent_run:
    random_search.fit(x_train, y_train)

    for i in range(len(random_search.cv_results_["params"])):
        with mlflow.start_run(run_name=f"Combination{i+1}", nested=True) as child_run:
            mlflow.log_params(random_search.cv_results_["params"][i])
            mlflow.log_metric("mean_test_score", random_search.cv_results_["mean_test_score"][i])

    best_params = random_search.best_params_
    print("Best parameters found: ", best_params)

    mlflow.log_params(best_params)

    best_rf = random_search.best_estimator_
    best_rf.fit(x_train, y_train)

    pickle.dump(best_rf, open("model.pkl", "wb"))

    x_test = test_processed_data.iloc[:, :-1].values
    y_test = test_processed_data.iloc[:, -1].values

    model = pickle.load(open("model.pkl", "rb"))
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metrics({"accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1})

    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)

    mlflow.log_input(train_df, "train")
    mlflow.log_input(test_df, "test")

    mlflow.log_artifact(__file__)

    signature = infer_signature(x_test, random_search.best_estimator_.predict(x_test))

    mlflow.sklearn.log_model(best_rf, "Best Model", signature=signature)

    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
