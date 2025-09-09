import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow.sklearn
import dagshub
dagshub.init(repo_owner='PiyushVIT346', repo_name='mlflow_exp_dagshub', mlflow=True)


mlflow.set_experiment("water_exp_rf")
mlflow.set_tracking_uri("https://dagshub.com/PiyushVIT346/mlflow_exp_dagshub.mlflow")

data = pd.read_csv("D:\projects\DVC project2\Experiment_using_MLFLOW\data\water_potability.csv")

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
    return df

train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

x_train = train_processed_data.iloc[:, :-1].values
y_train = train_processed_data.iloc[:, -1].values


n_estimators=500
max_depth=10
with mlflow.start_run():

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
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
    mlflow.log_param("max_depth", max_depth)

    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    mlflow.sklearn.log_model(clf,"RandomForestClassifier")

    mlflow.log_artifact(__file__) 

    mlflow.set_tag("author","anil")
    mlflow.set_tag("model","RF")

    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")