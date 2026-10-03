import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

# Set up MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("water_potability_hp")

# Load data
data = pd.read_csv("D:\projects\DVC project2\Experiment_using_MLFLOW\data\water_potability.csv")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Function to fill missing values with median
def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
    return df

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
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, cv=5, n_jobs=-1, verbose=2, random_state=42)

with mlflow.start_run():
    # Fit model
    random_search.fit(x_train, y_train)

    # Get best parameters
    best_params = random_search.best_params_
    print("Best parameters found: ", best_params)

    # Log best parameters correctly
    for param, value in best_params.items():
        mlflow.log_param(param, value)

    # Train the best model
    best_rf = random_search.best_estimator_
    best_rf.fit(x_train, y_train)

    # Save the model
    pickle.dump(best_rf, open("model.pkl", 'wb'))

    # Prepare test data
    x_test = test_processed_data.iloc[:, :-1].values
    y_test = test_processed_data.iloc[:, -1].values

    # Load model and make predictions
    model = pickle.load(open("model.pkl", 'rb'))
    y_pred = model.predict(x_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log training and test data
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)

    mlflow.log_input(train_df, "train")
    mlflow.log_input(test_df, "test")

    # Log the script file
    mlflow.log_artifact(__file__)

    # Log the best model
    mlflow.sklearn.log_model(best_rf, "Best Model")

    # Print metrics
    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
