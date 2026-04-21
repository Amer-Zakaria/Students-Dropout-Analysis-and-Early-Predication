import pandas as pd


def clean_data(df):
    """
    Although, the data is already cleaned as the dataset source website explicitly states:
    "We performed a rigorous data preprocessing to handle data from anomalies, unexplainable outliers, and missing values." ~Dataset Website
    Am going to do some extra verification...
    """
    df = df.rename(columns={"Nacionality": "Nationality"})

    if df.isnull().sum().sum() != 0:
        raise ValueError("DataFrame contains missing values.")

    if (df == "").sum().sum() != 0:
        raise ValueError("DataFrame contains empty strings.")

    if df.duplicated().sum() != 0:
        raise ValueError("DataFrame contains duplicate rows.")

    if df[(df["International"] == 1) & (df["Nationality"] == 1)].shape[0] != 0:
        raise ValueError(
            "DataFrame contains rows where the student is both portguese and international (the university is in Portugal)."
        )

    return df


def map_labels(df, maps, numerical_features):
    """
    Mapping the encoded categorical featuers into their original values
    """
    categorical_cols = [
        col for col in df.columns if col not in numerical_features + ["Target"]
    ]

    for col in categorical_cols:
        if col in maps:
            df[col] = df[col].map(maps[col])

    return df
