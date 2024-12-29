import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score, roc_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
# from feature_vector_extraction import feature_vector_extraction
# from preprocessing import preprocessing
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Custom PyTorch Dataset class
class MIDASDataset(Dataset):
    def __init__(self, data, cnn_feature_columns):
        self.features = np.hstack([
            data.drop(columns=["midas_file_name", "midas_category"] + cnn_feature_columns).values,
            data[cnn_feature_columns].values
        ])
        self.labels = data["midas_category"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

    
# Classification Layer class
class ClassifierNN(nn.Module):
    def __init__(self, input_dim):
        super(ClassifierNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# Group locations into clinically relevant clusters
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


def preprocessing(dataset):
    
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


def feature_vector_extraction(dataset, image_folder):
    
    # Load the ResNet50 model pre-trained on ImageNet
    base_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    base_model.eval()  # Set the model to evaluation mode

    # Remove the fully connected layer (only want the feature vectors)
    feature_extractor = torch.nn.Sequential(*list(base_model.children())[:-1])

    # Image preprocessing function
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize as per ResNet50
    ])

    def preprocess_image(image_path):
        """Preprocess an image for ResNet50."""
        img = Image.open(image_path).convert("RGB")  # Ensure 3 channels
        img_tensor = preprocess(img)  # Apply the preprocessing transformations
        return img_tensor.unsqueeze(0)  # Add batch dimension

    # Initialize lists to store feature vectors and image IDs
    feature_vectors = []
    image_ids = []

    print("Starting feature extraction...")
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Processing images"):
        image_path = os.path.join(image_folder, row["midas_file_name"])
        try:
            # Preprocess the image
            preprocessed_img = preprocess_image(image_path)
            # Ensure the tensor is on the same device as the model
            with torch.no_grad():  # Disable gradient computation
                features = feature_extractor(preprocessed_img)  # Extract features
            feature_vectors.append(features.squeeze().numpy())  # Flatten the output and convert to NumPy
            image_ids.append(row["midas_file_name"])  # Keep track of the image ID
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    # Convert feature vectors to a NumPy array
    feature_vectors = np.array(feature_vectors)
    
    return feature_vectors, image_ids


def combine_fatures(feature_vectors, image_ids, preprocessed_data):
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


def train(model, train_loader, criterion, optimizer, scheduler):
    epochs = 15
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Step the scheduler at the end of each epoch
        scheduler.step()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")


def test(model, test_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            probabilities = outputs  # For AUC-ROC
            predictions = (outputs >= 0.5).float()  # Default threshold: 0.5

            # Collect labels, predictions, and probabilities
            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.numpy())
            all_probs.extend(probabilities.numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)

    # Step 1: Default Metrics (Threshold = 0.5)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else float('nan')

    print("\nDefault Threshold Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {roc_auc:.4f}")
    
    # Step 2: Optimize Threshold
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"\nOptimal Threshold: {optimal_threshold:.4f}")

    # Step 3: Metrics with Optimal Threshold
    optimal_predictions = (all_probs >= optimal_threshold).astype(int)
    # optimal_predictions = (all_probs >= 0.2).astype(int)

    accuracy_opt = accuracy_score(all_labels, optimal_predictions)
    precision_opt = precision_score(all_labels, optimal_predictions, zero_division=0)
    recall_opt = recall_score(all_labels, optimal_predictions, zero_division=0)
    f1_opt = f1_score(all_labels, optimal_predictions, zero_division=0)
    roc_auc_opt = roc_auc_score(all_labels, all_probs)  # AUC-ROC remains the same

    print("Optimal Threshold Metrics:")
    print(f"  Accuracy:  {accuracy_opt:.4f}")
    print(f"  Precision: {precision_opt:.4f}")
    print(f"  Recall:    {recall_opt:.4f}")
    print(f"  F1 Score:  {f1_opt:.4f}")
    print(f"  AUC-ROC:   {roc_auc_opt:.4f}")
    
    
def classification(tabular_data_path, images_folder):
    
    # Import the tabular_data as dataframe
    tabular_data = pd.read_excel(tabular_data_path)
    
    # Preprocess the tabular data
    preprocessed_data = preprocessing(tabular_data)
    
    # Extract the feature vectors
    #feature_vectors, image_ids = feature_vector_extraction(preprocessed_data, images_folder)
    
    # for testing
    feature_vectors = np.load("feature_vectors_mb.npy")
    image_ids = np.load("image_ids_mb.npy", allow_pickle=True)
    
    # Combine preprocessed tabular data with the corresponding feature vectors
    final_data, cnn_feature_columns = combine_fatures(feature_vectors, image_ids, preprocessed_data)
    
    # Split dataset into training and testing sets
    train_data, test_data = train_test_split(final_data, test_size=0.2, random_state=21, stratify=preprocessed_data["midas_category"])

    # Create PyTorch datasets and loaders
    train_dataset = MIDASDataset(train_data, cnn_feature_columns)
    test_dataset = MIDASDataset(test_data, cnn_feature_columns)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize the model
    input_dim = train_dataset.features.shape[1]
    model = ClassifierNN(input_dim)
    criterion =  nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    train(model, train_loader, criterion, optimizer, scheduler)
    test(model, test_loader)
    
    
def main():
    tabular_data_path = "release_midas.xlsx"
    images_folder = "C:/Users/nikol/Desktop/university/9th_semester/physiological_systems_simulation/project/dataset/midasmultimodalimagedatasetforaibasedskincancer"
    
    classification(tabular_data_path, images_folder)
    

if __name__ == "__main__":
    main()