import pandas as pd


def reduce_features(df, numerical_features):
    """
    As I've noticed in the analysis step that the 1st and 2nd semesters are highly correleated
    so I'm going to take only their average for each pairs of featuers
    """
    toBeAveraged = [
        "credited",
        "enrolled",
        "evaluations",
        "approved",
        "grade",
        "without evaluations",
    ]
    for m in toBeAveraged:
        df[f"Curricular units avg {m}"] = df[
            [f"Curricular units 1st sem ({m})", f"Curricular units 2nd sem ({m})"]
        ].mean(axis=1)
        numerical_features.append(f"Curricular units avg {m}")

    old_cols = [f"Curricular units 1st sem ({m})" for m in toBeAveraged] + [
        f"Curricular units 2nd sem ({m})" for m in toBeAveraged
    ]

    # Drop from the original df & In the list of numerical featuers only keep reduced ones
    df = df.drop(columns=old_cols)
    numerical_features = [col for col in numerical_features if col not in old_cols]

    return df, numerical_features


def select_features(df, numerical_features):
    """
    From the analysis step I've noticed that some features are a weakly predictors of the target variable
    """
    numerical_features_to_keep = [
        "Curricular units avg grade",
        "Curricular units avg approved",
        "Curricular units avg enrolled",
        "Age at enrollment",
    ]
    df = df.drop(
        columns=["Nationality", "Educational special needs", "International"]
        + [
            feature
            for feature in numerical_features
            if feature not in numerical_features_to_keep
        ]
    )
    numerical_features = numerical_features_to_keep
    return df, numerical_features


def perform_one_hot_encoding(df, numerical_features):
    """
    prepare the X and y for the model
    X: I performed one hot encoding for non-ordinal categorical features to avoid false ordinal releshionship with the target,
    but for each feature am ignoring categories with less than 15 occurance (where am setting all of the one-hotot-encoded columns to 0)

    y: dropout (0: no or 1: yes)
    """
    categorical_cols = [
        "Marital status",
        "Application mode",
        "Course",
        "Previous qualification",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
        "Gender",
        "Daytime/evening attendance",
    ]

    df_filtered = df.copy()
    for col in categorical_cols:
        counts = df_filtered[col].value_counts()
        significant_categories = counts[counts >= 15].index
        df_filtered[col] = df_filtered[col].where(
            df_filtered[col].isin(significant_categories)
        )  # set categories with less than 15 occurance to NaN

    df_encoded = pd.get_dummies(
        df_filtered, columns=categorical_cols
    )  # By default, NaN category would be encoded as all 0s in the generated binary columns for a given featuer

    y = df["Target"].apply(lambda x: 1 if x == "Dropout" else 0)

    X_final = df_encoded.drop(columns=["Target"])

    return X_final, y
