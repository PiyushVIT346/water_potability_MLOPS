import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

# Set up MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("water_potability_hp")

# Load data (Fix path issue)
data_path = r"D:\projects\DVC project2\Experiment_using_MLFLOW\data\water_potability.csv"
data = pd.read_csv(data_path)

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Function to fill missing values with median
def fill_missing_with_median(df):
    return df.fillna(df.median())

train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# Prepare training data
x_train = train_processed_data.iloc[:, :-1].values
y_train = train_processed_data.iloc[:, -1].values

# Define Random Forest and Hyperparameter Grid
rf = RandomForestClassifier(random_state=42)
param_dist = {
    "n_estimators": [100, 200, 300, 400, 500, 1000],
    "max_depth": [None, 10, 20, 30, 40],
}

# Hyperparameter tuning with RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf, param_distributions=param_dist, cv=5, n_jobs=-1, verbose=1, random_state=42
)

with mlflow.start_run(run_name="Random forest tuning") as parent_run:
    # Fit model
    random_search.fit(x_train, y_train)

    # Log all hyperparameter combinations
    for i in range(len(random_search.cv_results_["params"])):
        with mlflow.start_run(run_name=f"Combination{i+1}", nested=True) as child_run:
            mlflow.log_params(random_search.cv_results_["params"][i])
            mlflow.log_metric("mean_test_score", random_search.cv_results_["mean_test_score"][i])

    # Get best parameters
    best_params = random_search.best_params_
    print("Best parameters found: ", best_params)

    # Log best parameters
    mlflow.log_params(best_params)

    # Train the best model
    best_rf = random_search.best_estimator_
    best_rf.fit(x_train, y_train)

    # Save the model
    model_path = "model.pkl"
    pickle.dump(best_rf, open(model_path, "wb"))

    # Prepare test data
    x_test = test_processed_data.iloc[:, :-1].values
    y_test = test_processed_data.iloc[:, -1].values

    # Load model and make predictions
    model = pickle.load(open(model_path, "rb"))
    y_pred = model.predict(x_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metrics({"accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1})

    # Save processed data for logging
    train_data_path = "train_processed.csv"
    test_data_path = "test_processed.csv"
    train_processed_data.to_csv(train_data_path, index=False)
    test_processed_data.to_csv(test_data_path, index=False)

    # Log processed data
    mlflow.log_artifact(train_data_path)
    mlflow.log_artifact(test_data_path)

    # Log the script file (Check if __file__ exists)
    if "__file__" in globals():
        mlflow.log_artifact(__file__)

    # Log the best model
    mlflow.sklearn.log_model(best_rf, "Best Model")

    # Print metrics
    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
