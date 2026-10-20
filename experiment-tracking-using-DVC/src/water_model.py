import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from dvclive import Live
import yaml

n_estimators = yaml.safe_load(open(r'D:\projects\DVC project2\experiment-tracking-using-DVC\dvclive\params.yaml'))["n_estimators"]

data = pd.read_csv("data/water_potability.csv")

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
