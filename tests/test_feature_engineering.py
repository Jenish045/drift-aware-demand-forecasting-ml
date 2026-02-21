import pandas as pd


def create_lag(df, group_col, target_col, lag=1):
    df = df.sort_values("Date")
    df[f"lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    return df


def test_lag_creation():
    data = {
        "Product": ["A", "A", "A"],
        "Date": pd.date_range("2023-01-01", periods=3),
        "Demand": [10, 20, 30]
    }

    df = pd.DataFrame(data)

    df = create_lag(df, "Product", "Demand", lag=1)

    assert pd.isna(df.loc[0, "lag_1"])
    assert df.loc[1, "lag_1"] == 10
    assert df.loc[2, "lag_1"] == 20