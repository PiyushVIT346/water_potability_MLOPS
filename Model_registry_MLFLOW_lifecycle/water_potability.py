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

def fill_missing_with_median(df):
    return df.fillna(df.median())

train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

#Prepare trainign data
#x_train = train_processed_data.iloc[:, :-1].values
#y_train = train_processed_data.iloc[:, -1].values

x_train=train_processed_data.drop('Potability',axis=1)
y_train=train_processed_data['Potability']

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

    train_df=mlflow.data.from_pandas(train_processed_data)
    test_df=mlflow.data.from_pandas(test_processed_data)

    mlflow.log_input(train_df,"train")
    mlflow.log_input(test_df,"test")

    mlflow.log_artifact(__file__)

    signature=infer_signature(x_test, random_search.best_estimator_.predict(x_test))

    mlflow.sklearn.log_model(best_rf, "Best Model",signature=signature)

    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
