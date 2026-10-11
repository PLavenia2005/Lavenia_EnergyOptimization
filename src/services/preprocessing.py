from pathlib import Path

import numpy as np
import pandas as pd


def load_raw_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the original energy consumption dataset.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}"
        )

    df = pd.read_csv(file_path)

    print(f"Loaded dataset: {file_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names so that the rest of the
    pipeline can work consistently.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


def identify_timestamp_column(df: pd.DataFrame) -> str:
    """
    Identify the timestamp/date column used by the dataset.
    """

    possible_columns = [
        "date",
        "datetime",
        "timestamp",
        "time"
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    raise ValueError(
        "No timestamp/date column was found in the dataset. "
        f"Available columns: {list(df.columns)}"
    )


def identify_energy_column(df: pd.DataFrame) -> str:
    """
    Identify the primary energy consumption column.

    The Appliances Energy Prediction dataset normally contains
    an 'Appliances' column.
    """

    possible_columns = [
        "appliances",
        "energy_consumption",
        "energy",
        "global_active_power"
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    raise ValueError(
        "No primary energy consumption column was found. "
        f"Available columns: {list(df.columns)}"
    )


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning.
    """

    df = df.copy()

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Identify timestamp and energy columns
    timestamp_column = identify_timestamp_column(df)
    energy_column = identify_energy_column(df)

    # Convert timestamp to datetime
    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce"
    )

    # Convert energy column to numeric
    df[energy_column] = pd.to_numeric(
        df[energy_column],
        errors="coerce"
    )

    # Remove rows where timestamp or energy is invalid
    df = df.dropna(
        subset=[
            timestamp_column,
            energy_column
        ]
    )

    # Remove impossible negative energy values
    df = df[df[energy_column] >= 0]

    # Sort chronologically
    df = df.sort_values(
        by=timestamp_column
    )

    # Reset index
    df = df.reset_index(drop=True)

    return df


def add_time_features(
    df: pd.DataFrame,
    timestamp_column: str
) -> pd.DataFrame:
    """
    Create time-based features useful for energy analysis.
    """

    df = df.copy()

    df["timestamp"] = df[timestamp_column]

    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = (
        df["timestamp"].dt.day_name()
    )

    df["day_of_week_number"] = (
        df["timestamp"].dt.dayofweek
    )

    df["month"] = df["timestamp"].dt.month

    df["year"] = df["timestamp"].dt.year

    # Determine season
    def get_season(month: int) -> str:
        if month in [12, 1, 2]:
            return "Winter"

        if month in [3, 4, 5]:
            return "Spring"

        if month in [6, 7, 8]:
            return "Summer"

        return "Autumn"

    df["season"] = df["month"].apply(
        get_season
    )

    # Peak period
    # We will treat 18:00–22:00 as the initial
    # evening peak period.
    df["is_peak_hour"] = df["hour"].between(
        18,
        22,
        inclusive="both"
    )

    df["period"] = np.where(
        df["is_peak_hour"],
        "Peak",
        "Off-Peak"
    )

    return df


def add_consumption_features(
    df: pd.DataFrame,
    energy_column: str
) -> pd.DataFrame:
    """
    Create consumption-level categories and
    basic statistical features.
    """

    df = df.copy()

    # Rolling average helps compare current consumption
    # with recent historical consumption.
    df["rolling_mean_24"] = (
        df[energy_column]
        .rolling(
            window=24,
            min_periods=1
        )
        .mean()
    )

    # Difference from recent rolling average
    df["consumption_deviation"] = (
        df[energy_column]
        - df["rolling_mean_24"]
    )

    # Dataset-level quantiles
    q50 = df[energy_column].quantile(0.50)
    q75 = df[energy_column].quantile(0.75)
    q90 = df[energy_column].quantile(0.90)

    def classify_consumption(value: float) -> str:

        if value <= q50:
            return "Normal"

        if value <= q75:
            return "Moderate"

        if value <= q90:
            return "High"

        return "Critical"

    df["consumption_level"] = (
        df[energy_column]
        .apply(classify_consumption)
    )

    # Simple anomaly indicator
    mean_value = df[energy_column].mean()
    std_value = df[energy_column].std()

    if std_value == 0 or pd.isna(std_value):
        df["anomaly_score"] = 0.0
    else:
        df["anomaly_score"] = (
            (
                df[energy_column]
                - mean_value
            )
            / std_value
        ).abs()

    df["is_anomaly"] = (
        df["anomaly_score"] >= 2.0
    )

    return df


def create_energy_cases(
    df: pd.DataFrame,
    energy_column: str
) -> pd.DataFrame:
    """
    Create RAG-friendly energy cases.

    Each row represents an energy observation enriched
    with temporal and consumption information.
    """

    df = df.copy()

    df["case_id"] = [
        f"EC_{index + 1:06d}"
        for index in range(len(df))
    ]

    # Create a natural-language case description.
    def build_description(row) -> str:

        energy_value = row[energy_column]

        return (
            f"Energy consumption case {row['case_id']}. "
            f"Timestamp: {row['timestamp']}. "
            f"Season: {row['season']}. "
            f"Period: {row['period']}. "
            f"Hour: {row['hour']}. "
            f"Day: {row['day_of_week']}. "
            f"Energy consumption: {energy_value:.3f}. "
            f"Consumption level: "
            f"{row['consumption_level']}. "
            f"Anomaly detected: "
            f"{row['is_anomaly']}. "
            f"Anomaly score: "
            f"{row['anomaly_score']:.3f}."
        )

    df["case_description"] = df.apply(
        build_description,
        axis=1
    )

    return df


def save_processed_dataset(
    df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Save processed energy cases.
    """

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output,
        index=False
    )

    print(
        f"Processed dataset saved to: {output}"
    )


def preprocess_energy_dataset(
    input_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    """

    # 1. Load
    df = load_raw_dataset(
        input_path
    )

    # 2. Standardize columns
    df = standardize_columns(df)

    # 3. Identify important columns
    timestamp_column = identify_timestamp_column(df)
    energy_column = identify_energy_column(df)

    print(
        f"Timestamp column: {timestamp_column}"
    )

    print(
        f"Energy column: {energy_column}"
    )

    # 4. Clean
    df = clean_dataset(df)

    # 5. Add temporal features
    df = add_time_features(
        df,
        timestamp_column
    )

    # 6. Add consumption features
    df = add_consumption_features(
        df,
        energy_column
    )

    # 7. Create RAG cases
    df = create_energy_cases(
        df,
        energy_column
    )

    # 8. Save
    save_processed_dataset(
        df,
        output_path
    )

    print("\nPreprocessing completed.")
    print(f"Final rows: {len(df)}")
    print(
        f"Final columns: {list(df.columns)}"
    )

    return df


if __name__ == "__main__":

    from src.utils.config import (
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH
    )

    preprocess_energy_dataset(
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH
    )