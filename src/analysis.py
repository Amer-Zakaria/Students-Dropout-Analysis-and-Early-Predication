from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os


def set_style():
    sns.set_palette("mako")


def save_table(df, filename):
    """
    Save table to its designated folder
    """
    filename = filename.replace("/", "_")
    path = os.path.join("outputs", "tables", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)


def save_fig(fig, section, filename):
    """
    Save figure to its designated folder
    """
    filename = filename.replace("/", "_")
    path = os.path.join("outputs", "figures", section, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)  # Free memory


def get_numerical_description(df, numerical_features):
    """
    Produce statistics that descripe the numerical featuers
    """
    desc = df[numerical_features].describe().T.drop(columns=["count"])
    return desc


def get_semester_correlation(df):
    """
    I've noticed in the heatmap of all the values with each other, that the correlation between the first and second semester is very high.
    So, this function generate a corrletion between the 1st and 2nd semesters, so that in the featuer enginnering phase I combine them into one to reduce dimensions.
    """
    pairs = [
        ("Curricular units 1st sem (credited)", "Curricular units 2nd sem (credited)"),
        ("Curricular units 1st sem (enrolled)", "Curricular units 2nd sem (enrolled)"),
        (
            "Curricular units 1st sem (evaluations)",
            "Curricular units 2nd sem (evaluations)",
        ),
        ("Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"),
        ("Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"),
        (
            "Curricular units 1st sem (without evaluations)",
            "Curricular units 2nd sem (without evaluations)",
        ),
    ]
    results = []
    for col1, col2 in pairs:
        corr = df[[col1, col2]].corr(method="spearman").iloc[0, 1]
        results.append(
            {
                "Feature": col1.replace("Curricular units ", "").replace(
                    "1st sem ", ""
                ),
                "Spearman Correlation (1st vs 2nd)": corr,
            }
        )
    return pd.DataFrame(results)


def plot_target_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 6))
    df["Target"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    ax.set_title("Target Labels Proportions")
    save_fig(fig, "feature_distribution", "target_distribution.png")


def plot_categorical_distributions(df, categorical_cols):
    """
    Now that we have the true labels in the dataset, we can plot the distributions of the categorical features
    """
    for col in categorical_cols:
        top_values = (
            df[col].value_counts().nlargest(5).index
        )  # Top 5 values(categories) by occurrences
        df_top = df[
            df[col].isin(top_values)
        ]  # Only rows of those top 5 values are kept
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(data=df_top, y=col, order=top_values, ax=ax)
        ax.set_title(f"{col} Distribution {len(top_values) == 5 and '(Top 5)'}")
        plt.tight_layout()
        save_fig(fig, "feature_distribution/categorical", f"{col}_distribution.png")


def plot_numerical_distributions(df, numerical_features):
    """
    Plot both histogram and a KDE to show gaps, counts and the overall trend
    """
    for col in numerical_features:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x=col, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        save_fig(fig, "feature_distribution/numerical", f"{col}_distribution.png")


def plot_correlation_heatmap(df, numerical_features):
    """
    Heatmap the shows the correlation between success(0 = dropout, 1 = enrolled, 2 = graduate) and numerical features
    """
    df_corr = df.copy()
    df_corr["Success"] = OrdinalEncoder(
        categories=[["Dropout", "Enrolled", "Graduate"]]
    ).fit_transform(df_corr[["Target"]])

    corr = df_corr[numerical_features + ["Success"]].corr(method="spearman")
    target_corr = corr[["Success"]].sort_values(by="Success", ascending=False)

    fig, ax = plt.subplots(figsize=(6, 10))
    sns.heatmap(
        target_corr,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.2,
        annot=True,
        ax=ax,
    )
    plt.tight_layout()
    save_fig(fig, "relationships/numerical", "correlation_heatmap.png")


def plot_categorical_relationship_with_target(df, categorical_features):
    """
    Since the heatmap cannot capture the correlation categorical features and the target,
    I've plotted the categorical features separately
    Where I removed categories with less than 15 occurances (to avoid clutter) then compared the categorical featuer by the dropout rate for each category
    """
    # Creat a copy and add the binary dropped_out column to it (dropped_out is going to be the target)
    df_temp = df.copy()
    df_temp["dropped_out"] = df_temp["Target"].apply(
        lambda x: 1 if x == "Dropout" else 0
    )

    for col in categorical_features:
        counts = df_temp[col].value_counts()
        significant_categories = counts[counts > 15].index
        filtered_df = df_temp[
            df_temp[col].isin(significant_categories)
        ]  # only consider categories with more than 15 occurances

        dropout_rate = (
            filtered_df.groupby(col)["dropped_out"].mean().sort_values(ascending=False)
        )  # dropout_rate of each category in the categorical feature
        counts_filtered = filtered_df[col].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=dropout_rate.values,
            y=[
                str(idx) for idx in dropout_rate.index
            ],  # str() to avoid error when the index is Boolean
            ax=ax,
        )

        # Add the number of occurances of each category to the end of its bar
        for i, (label, value) in enumerate(
            zip(dropout_rate.index, dropout_rate.values)
        ):
            ax.text(value, i, f" {counts_filtered[label]}", va="center")

        ax.xaxis.set_major_formatter(
            mtick.PercentFormatter(1.0)
        )  # turn the scale to 100 (0-1 => 0-100)
        ax.set_xlabel("Dropout Percentage")
        ax.set_title(f"Dropout Rate by {col}")
        plt.tight_layout()
        save_fig(
            fig, "relationships/categorical", f'{col.replace("/", "_")}_vs_target.png'
        )


def plot_outliers(df, numerical_features):
    """
    By building a boxplot for each numerical feature, I can detect weather there are suspicious values
    """
    cols_to_plot = [
        c
        for c in numerical_features
        if c
        not in [
            "Admission grade",
            "Previous qualification (grade)",
        ]  # since those has large scale and checked in another step that they fall within their min/max range
    ]
    fig, ax = plt.subplots(figsize=(15, 8))
    df[cols_to_plot].boxplot(rot=0, vert=False, ax=ax)
    ax.set_title("Boxplots for Outlier Detection")
    plt.tight_layout()
    ax.set_xlabel("Values")
    ax.set_ylabel("Numerical features")
    save_fig(fig, "outliers", "numerical_outliers.png")
