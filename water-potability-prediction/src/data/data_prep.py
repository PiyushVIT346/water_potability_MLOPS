import os
import numpy as np
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")


def get_imputation_params(df: pd.DataFrame, exclude_cols: list) -> dict:
    """Calculates median for imputation using only training data."""
    medians = {}
    cols_to_process = [col for col in df.columns if col not in exclude_cols]
    for col in cols_to_process:
        medians[col] = df[col].median()
    return medians


def impute_missing(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """Fills missing values using the calculated medians."""
    df_out = df.copy()
    for col, median_val in medians.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(median_val)
    return df_out


def get_outlier_bounds(df: pd.DataFrame, exclude_cols: list) -> dict:
    """Calculates Interquartile Range (IQR) boundaries for outlier capping."""
    bounds = {}
    cols_to_process = [col for col in df.columns if col not in exclude_cols]
    for col in cols_to_process:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return bounds


def cap_outliers(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Caps extreme outliers to the IQR boundaries to stabilize models."""
    df_out = df.copy()
    for col, (lower, upper) in bounds.items():
        if col in df_out.columns:
            df_out[col] = np.clip(df_out[col], lower, upper)
    return df_out


def get_scaling_params(df: pd.DataFrame, exclude_cols: list) -> dict:
    """Calculates Mean and Standard Deviation for Z-Score scaling."""
    params = {}
    cols_to_process = [col for col in df.columns if col not in exclude_cols]
    for col in cols_to_process:
        params[col] = {"mean": df[col].mean(), "std": df[col].std()}
    return params


def scale_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Standardizes features (Mean=0, Std=1) without adding new columns."""
    df_out = df.copy()
    for col, stats in params.items():
        if col in df_out.columns and stats["std"] != 0:
            df_out[col] = (df_out[col] - stats["mean"]) / stats["std"]
    return df_out


def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}: {e}")


def main():
    try:
        raw_data_path = "./data/raw"
        processed_data_path = "./data/processed"

        train_data = load_data(os.path.join(raw_data_path, "train.csv"))
        test_data = load_data(os.path.join(raw_data_path, "test.csv"))

        # We do not want to transform the target variable
        exclude_cols = ["Potability"]

        # 1. FIT (Calculate parameters ONLY on Training data to prevent leakage)
        medians = get_imputation_params(train_data, exclude_cols)

        # Temporarily impute train to calculate accurate bounds and scaling params
        train_temp = impute_missing(train_data, medians)

        bounds = get_outlier_bounds(train_temp, exclude_cols)
        train_temp = cap_outliers(train_temp, bounds)

        scaling_params = get_scaling_params(train_temp, exclude_cols)

        # 2. TRANSFORM (Apply the calculated parameters to Train and Test data)
        # Process Train
        train_processed = impute_missing(train_data, medians)
        train_processed = cap_outliers(train_processed, bounds)
        train_processed = scale_features(train_processed, scaling_params)

        # Process Test
        test_processed = impute_missing(test_data, medians)
        test_processed = cap_outliers(test_processed, bounds)
        test_processed = scale_features(test_processed, scaling_params)

        # 3. SAVE
        save_data(
            train_processed,
            os.path.join(processed_data_path, "train_processed.csv"),
        )
        save_data(
            test_processed,
            os.path.join(processed_data_path, "test_processed.csv"),
        )

        print("Data processing completed successfully!")

    except Exception as e:
        raise Exception(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
