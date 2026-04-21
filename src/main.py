import data_loader
import preprocessing
import analysis
import feature_engineering
import modeling


def main():
    numerical_features = [
        "Age at enrollment",
        "Application order",
        "Previous qualification (grade)",
        "Admission grade",
        "Inflation rate",
        "Unemployment rate",
        "GDP",
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
    ]

    df = data_loader.load_csv(data_loader.get_data_path("1_splitted.csv"))

    # ================= Data Cleaning & Preprocessing ======================
    df = preprocessing.clean_data(df)

    mappings = data_loader.load_encoded_mappings()
    df = preprocessing.map_labels(df, mappings, numerical_features)
    data_loader.save_csv(df, data_loader.get_data_path("2_cleaned.csv"))

    # ================= Exploratory Data Analysis ======================
    analysis.set_style()

    analysis.plot_target_distribution(df)  # Target labels distribution

    categorical_cols = [
        col for col in df.columns if col not in numerical_features + ["Target"]
    ]
    analysis.plot_categorical_distributions(df, categorical_cols)
    analysis.plot_numerical_distributions(df, numerical_features)

    analysis.save_table(
        analysis.get_numerical_description(df, numerical_features),
        "numerical_description.csv",
    )

    analysis.plot_categorical_relationship_with_target(df, categorical_cols)

    analysis.plot_correlation_heatmap(df, numerical_features)

    analysis.plot_outliers(df, numerical_features)

    sem_corr = analysis.get_semester_correlation(df)
    analysis.save_table(sem_corr, "both_semesters_correlation.csv")

    # ================= Feature Engineering ======================
    df, numerical_features = feature_engineering.reduce_features(df, numerical_features)
    df, numerical_features = feature_engineering.select_features(df, numerical_features)
    X_selected, y = feature_engineering.perform_one_hot_encoding(df, numerical_features)

    df_final = X_selected.copy()
    df_final["Dropped_out"] = y
    data_loader.save_csv(df_final, data_loader.get_data_path("3_preprocessed.csv"))

    # ================= Machine Learning ======================
    # Split
    X_train, X_test, y_train, y_test = modeling.prepare_data(
        df_final, numerical_features, "Dropped_out"
    )

    # Train decision tree with cross-validation
    model, cv_scores = modeling.train_descision_tree(X_train, y_train)
    print(
        f"Decision Tree Cross-Validation Scores (5 folds): {[f"{val:.2f}" for val in cv_scores]}"
    )
    print(f"Mean CV Score: {cv_scores.mean():.2f}")

    # Evaluate the decision tree model using accuracy_score
    y_pred = model.predict(X_test)
    accuracy_score = modeling.evaluate_accuracy_score(y_test, y_pred)
    print(f"Decision Tree accuracy score: {accuracy_score:.2f}")

    # Train logistic regression with cross-validation
    model, cv_scores = modeling.train_logistic_regression(X_train, y_train)
    print(
        f"Logistic Regression Cross-Validation Scores (5 folds): {[f"{val:.2f}" for val in cv_scores]}"
    )
    print(f"Mean CV Score: {cv_scores.mean():.2f}")

    # Evaluate the logistic regression model using accuracy_score
    y_pred = model.predict(X_test)
    accuracy_score = modeling.evaluate_accuracy_score(y_test, y_pred)
    print(f"Logistic Regression accuracy score: {accuracy_score:.2f}")


if __name__ == "__main__":
    main()
