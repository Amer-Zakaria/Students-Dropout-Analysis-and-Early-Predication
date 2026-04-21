import pandas as pd
import data_loader
import os


def load_csv(path):
    return pd.read_csv(path)


def save_csv(df, path):
    df.to_csv(path, index=False)


def get_data_path(filename):
    return os.path.join("data", "processed", filename)


def load_encoded_mappings():
    """
    Loading the files that maps the encoded values to their original values
    """
    path_to_mapping_files = data_loader.get_data_path(
        "encoded_columns_description_tables"
    )
    csv_files = {
        "marital_status": os.path.join(path_to_mapping_files, "marital_status.csv"),
        "nationality": os.path.join(path_to_mapping_files, "nationality.csv"),
        "application_mode": os.path.join(path_to_mapping_files, "application_mode.csv"),
        "course_names": os.path.join(path_to_mapping_files, "course_names.csv"),
        "previous_quals": os.path.join(path_to_mapping_files, "previous_quals.csv"),
        "parent_previous_quals": os.path.join(
            path_to_mapping_files, "parent_previous_quals.csv"
        ),
        "parent_occupation": os.path.join(
            path_to_mapping_files, "parent_occupation.csv"
        ),
        "gender": os.path.join(path_to_mapping_files, "gender.csv"),
        "attendance_regime": os.path.join(
            path_to_mapping_files, "attendance_regime.csv"
        ),
        "yes_no": os.path.join(path_to_mapping_files, "yes_no.csv"),
    }

    maps = {}
    for key, file in csv_files.items():
        df = pd.read_csv(file)
        df = df.drop_duplicates(subset="ID")
        df = df.dropna(subset=["ID", "Value"])
        maps[key] = dict(zip(df["ID"], df["Value"]))

    final_maps = {}
    final_maps["Marital status"] = maps["marital_status"]
    final_maps["Nationality"] = maps["nationality"]
    final_maps["Course"] = maps["course_names"]
    final_maps["Gender"] = maps["gender"]
    final_maps["Application mode"] = maps["application_mode"]
    final_maps["Previous qualification"] = maps["previous_quals"]
    final_maps["Father's qualification"] = maps["parent_previous_quals"]
    final_maps["Mother's qualification"] = maps["parent_previous_quals"]
    final_maps["Father's occupation"] = maps["parent_occupation"]
    final_maps["Mother's occupation"] = maps["parent_occupation"]
    final_maps["Daytime/evening attendance"] = maps["attendance_regime"]
    final_maps["Scholarship holder"] = maps["yes_no"]
    final_maps["Debtor"] = maps["yes_no"]
    final_maps["Displaced"] = maps["yes_no"]
    final_maps["Educational special needs"] = maps["yes_no"]
    final_maps["Tuition fees up to date"] = maps["yes_no"]
    final_maps["International"] = maps["yes_no"]

    return final_maps
