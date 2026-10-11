import pandas as pd

from src.services.preprocessing import (
    standardize_columns,
    identify_timestamp_column,
    identify_energy_column,
    clean_dataset,
    add_time_features,
    add_consumption_features,
    create_energy_cases
)


def create_sample_data():

    return pd.DataFrame({
        "Date": [
            "2016-01-01 10:00:00",
            "2016-01-01 11:00:00",
            "2016-01-01 12:00:00",
            "2016-01-01 13:00:00"
        ],
        "Appliances": [
            50,
            100,
            200,
            75
        ]
    })


def test_standardize_columns():

    df = create_sample_data()

    result = standardize_columns(df)

    assert "date" in result.columns
    assert "appliances" in result.columns


def test_identify_timestamp_column():

    df = create_sample_data()

    df = standardize_columns(df)

    result = identify_timestamp_column(df)

    assert result == "date"


def test_identify_energy_column():

    df = create_sample_data()

    df = standardize_columns(df)

    result = identify_energy_column(df)

    assert result == "appliances"


def test_clean_dataset():

    df = create_sample_data()

    df = standardize_columns(df)

    result = clean_dataset(df)

    assert len(result) == 4
    assert result["date"].notna().all()
    assert result["appliances"].notna().all()


def test_add_time_features():

    df = create_sample_data()

    df = standardize_columns(df)
    df = clean_dataset(df)

    result = add_time_features(
        df,
        "date"
    )

    assert "timestamp" in result.columns
    assert "hour" in result.columns
    assert "season" in result.columns
    assert "period" in result.columns
    assert "is_peak_hour" in result.columns


def test_add_consumption_features():

    df = create_sample_data()

    df = standardize_columns(df)
    df = clean_dataset(df)

    result = add_time_features(
        df,
        "date"
    )

    result = add_consumption_features(
        result,
        "appliances"
    )

    assert "rolling_mean_24" in result.columns
    assert "consumption_deviation" in result.columns
    assert "consumption_level" in result.columns
    assert "anomaly_score" in result.columns
    assert "is_anomaly" in result.columns


def test_create_energy_cases():

    df = create_sample_data()

    df = standardize_columns(df)
    df = clean_dataset(df)

    df = add_time_features(
        df,
        "date"
    )

    df = add_consumption_features(
        df,
        "appliances"
    )

    result = create_energy_cases(
        df,
        "appliances"
    )

    assert "case_id" in result.columns
    assert "case_description" in result.columns

    assert result["case_id"].iloc[0] == "EC_000001"

    assert (
        "Energy consumption case"
        in result["case_description"].iloc[0]
    )

