import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def group_locations(location):
    if "back" in location:
        if "upper" in location:
            return "upper_back"
        elif "lower" in location:
            return "lower_back"
        else:
            return "back_unspecified"
    elif "chest" in location:
        return "chest"
    elif "arm" in location or "shoulder" in location:
        return "arms_shoulder"
    elif "face" in location or "forehead" in location or "cheek" in location:
        return "face"
    elif "leg" in location or "thigh" in location:
        return "legs_thighs"
    elif "scalp" in location or "head" in location:
        return "scalp_head"
    elif "neck" in location:
        return "neck"
    elif "hand" in location or "finger" in location:
        return "hands_fingers"
    elif "foot" in location or "toe" in location:
        return "feet_toes"
    else:
        return "other"


def tabular_data_preprocessing(dataset):
    columns_to_remove = [
        'midas_record_id', 'midas_iscontrol',
        'midas_pathreport', 'midas_ethnicity', 'midas_race', 'midas_fitzpatrick',
        'clinical_impression_1', 'clinical_impression_2', 'clinical_impression_3'
    ]

    tabular_data = dataset.drop(columns=columns_to_remove, errors='ignore')

    tabular_data = tabular_data.drop(columns=tabular_data.columns[0], errors="ignore")

    tabular_data = tabular_data.dropna()

    tabular_data['midas_grouped_location'] = tabular_data['midas_location'].apply(group_locations)

    tabular_data = tabular_data.drop(columns=['midas_location'], errors="ignore")

    tabular_data['midas_category'] = tabular_data['midas_path'].apply(
        lambda x: 'malignant' if isinstance(x, str) and 'malignant' in x else
                'benign' if isinstance(x, str) and 'benign' in x else
                'other'
    )

    tabular_data.drop(columns=['midas_path'], inplace=True)

    tabular_data = tabular_data[tabular_data['midas_category'].isin(['malignant', 'benign'])]

    tabular_data["midas_melanoma"] = tabular_data["midas_melanoma"].apply(lambda x: 1 if x == "yes" else 0).astype(int)

    tabular_data.loc[:, "midas_category"] = tabular_data["midas_category"].apply(lambda x: 1 if x == "malignant" else 0).astype(int)

    tabular_data["midas_melanoma"] = pd.to_numeric(tabular_data["midas_melanoma"])
    tabular_data["midas_category"] = pd.to_numeric(tabular_data["midas_category"])


    file_names = tabular_data["midas_file_name"]
    tabular_data = tabular_data.drop(columns=["midas_file_name"])

    categorical_columns = tabular_data.select_dtypes(include=["object"]).columns
    numerical_columns = tabular_data.select_dtypes(include=["number"]).drop(columns=["midas_category"]).columns

    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_preprocessor, numerical_columns),
            ("cat", categorical_preprocessor, categorical_columns),
        ]
    )

    X_tabular_data = tabular_data.drop(columns=["midas_category"])
    y_tabular_data = tabular_data["midas_category"]

    X_preprocessed_data = preprocessor.fit_transform(X_tabular_data)

    numerical_feature_names = list(numerical_columns)
    categorical_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_columns)
    processed_feature_columns = numerical_feature_names + list(categorical_feature_names)

    X_preprocessed_data_df = pd.DataFrame(
        X_preprocessed_data.toarray() if hasattr(X_preprocessed_data, "toarray") else X_preprocessed_data,
        columns=processed_feature_columns
    )

    preprocessed_data = X_preprocessed_data_df.copy()
    preprocessed_data["midas_file_name"] = file_names.reset_index(drop=True)
    preprocessed_data["midas_category"] = y_tabular_data.reset_index(drop=True)
    
    return preprocessed_data


def combine_features(feature_vectors, image_ids, preprocessed_data):
    feature_vector_map = {img_id: vec for img_id, vec in zip(image_ids, feature_vectors)}
    preprocessed_data["cnn_features"] = preprocessed_data["midas_file_name"].map(feature_vector_map)

    preprocessed_data = preprocessed_data.dropna(subset=["cnn_features"])

    cnn_feature_columns = [f"cnn_feature_{i}" for i in range(feature_vectors.shape[1])]
    cnn_features_expanded = pd.DataFrame(
        preprocessed_data["cnn_features"].tolist(),
        columns=cnn_feature_columns,
        index=preprocessed_data.index  
    )

    preprocessed_data = pd.concat([preprocessed_data, cnn_features_expanded], axis=1)

    preprocessed_data = preprocessed_data.drop(columns=["cnn_features"])
    
    return preprocessed_data, cnn_feature_columns