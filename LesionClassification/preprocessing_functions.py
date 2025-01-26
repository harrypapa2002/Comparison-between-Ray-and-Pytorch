import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Group locations into clinically relevant clusters (for preprocessing)
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
    
    # Define columns to remove
    columns_to_remove = [
        'midas_record_id', 'midas_iscontrol',
        'midas_pathreport', 'midas_ethnicity', 'midas_race', 'midas_fitzpatrick',
        'clinical_impression_1', 'clinical_impression_2', 'clinical_impression_3'
    ]

    # Remove specified columns
    tabular_data = dataset.drop(columns=columns_to_remove, errors='ignore')

    # Remove the first column as well (Unnamed)
    tabular_data = tabular_data.drop(columns=tabular_data.columns[0], errors="ignore")

    # Drop rows with any missing values (NA)
    tabular_data = tabular_data.dropna()

    # Apply grouping locations to the dataset
    tabular_data['midas_grouped_location'] = tabular_data['midas_location'].apply(group_locations)

    # Remove the midas_location column
    tabular_data = tabular_data.drop(columns=['midas_location'], errors="ignore")
    
    # Create a new column 'midas_category' based on 'midas_path'
    tabular_data['midas_category'] = tabular_data['midas_path'].apply(
        lambda x: 'malignant' if isinstance(x, str) and 'malignant' in x else
                'benign' if isinstance(x, str) and 'benign' in x else
                'other'
    )

    # Drop the 'midas_path' column
    tabular_data.drop(columns=['midas_path'], inplace=True)

    # Filter rows where 'midas_category' is not 'other'
    tabular_data = tabular_data[tabular_data['midas_category'].isin(['malignant', 'benign'])]
    
    # Replace yes/no with 1/0 and ensure numerical dtype
    tabular_data["midas_melanoma"] = tabular_data["midas_melanoma"].apply(lambda x: 1 if x == "yes" else 0).astype(int)

    # Replace malignant/benign with 1/0 and convert to numerical type
    tabular_data.loc[:, "midas_category"] = tabular_data["midas_category"].apply(lambda x: 1 if x == "malignant" else 0).astype(int)

    # Convert these columns explicitly to numeric type
    tabular_data["midas_melanoma"] = pd.to_numeric(tabular_data["midas_melanoma"])
    tabular_data["midas_category"] = pd.to_numeric(tabular_data["midas_category"])
    
    
    # Encoding and Normalization

    # Exclude `midas_file_name` from preprocessing but retain it for mapping
    file_names = tabular_data["midas_file_name"]
    tabular_data = tabular_data.drop(columns=["midas_file_name"])

    # Identify categorical and numerical columns
    categorical_columns = tabular_data.select_dtypes(include=["object"]).columns
    numerical_columns = tabular_data.select_dtypes(include=["number"]).drop(columns=["midas_category"]).columns

    # Preprocessing pipeline
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_preprocessor, numerical_columns),
            ("cat", categorical_preprocessor, categorical_columns),
        ]
    )

    # Separate features and target
    X_tabular_data = tabular_data.drop(columns=["midas_category"])
    y_tabular_data = tabular_data["midas_category"]

    # Fit and transform the dataset
    X_preprocessed_data = preprocessor.fit_transform(X_tabular_data)

    # Extract feature names
    numerical_feature_names = list(numerical_columns)
    categorical_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_columns)
    processed_feature_columns = numerical_feature_names + list(categorical_feature_names)

    # Convert preprocessed features into a DataFrame
    X_preprocessed_data_df = pd.DataFrame(
        X_preprocessed_data.toarray() if hasattr(X_preprocessed_data, "toarray") else X_preprocessed_data,
        columns=processed_feature_columns
    )

    # Add back `midas_file_name` and the target variable to the preprocessed dataset
    preprocessed_data = X_preprocessed_data_df.copy()
    preprocessed_data["midas_file_name"] = file_names.reset_index(drop=True)
    preprocessed_data["midas_category"] = y_tabular_data.reset_index(drop=True)
    
    return preprocessed_data



def combine_features(feature_vectors, image_ids, preprocessed_data):
    # Match feature vectors with preprocessed data using the file name
    feature_vector_map = {img_id: vec for img_id, vec in zip(image_ids, feature_vectors)}
    preprocessed_data["cnn_features"] = preprocessed_data["midas_file_name"].map(feature_vector_map)

    # Drop rows where CNN features are missing (e.g., due to failed processing)
    preprocessed_data = preprocessed_data.dropna(subset=["cnn_features"])

    # Expand CNN features into separate columns
    cnn_feature_columns = [f"cnn_feature_{i}" for i in range(feature_vectors.shape[1])]
    cnn_features_expanded = pd.DataFrame(
        preprocessed_data["cnn_features"].tolist(),
        columns=cnn_feature_columns,
        index=preprocessed_data.index  # Ensure the index aligns with preprocessed_data
    )

    # Concatenate the original data with the expanded CNN features
    preprocessed_data = pd.concat([preprocessed_data, cnn_features_expanded], axis=1)

    # Drop the original "cnn_features" column as it is now expanded
    preprocessed_data = preprocessed_data.drop(columns=["cnn_features"])
    
    return preprocessed_data, cnn_feature_columns