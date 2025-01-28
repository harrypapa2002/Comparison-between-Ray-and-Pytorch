import os
import numpy as np
import pandas as pd
import time
import json
from PIL import Image
import io
from pyarrow.fs import HadoopFileSystem
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from preprocessing_functions import tabular_data_preprocessing, combine_features


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


class ClassifierNN(nn.Module):
    def __init__(self, input_dim):
        super(ClassifierNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def get_num_nodes():
    try:
        nodes = ray.nodes()
        active_nodes = sum(1 for node in nodes if node["Alive"])
        print(f"Active Ray nodes: {active_nodes}")
        return active_nodes
    except Exception as e:
        print(f"Error getting active nodes: {e}")
        return 0  # Return 0 if an error occurs


def feature_vector_extraction(config, image_id, feature_extractor, hdfs):
    """Extract feature vector from a single image."""

    # Image preprocessing function
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for ResNet50
    ])
    
    try:
        # Construct the full image path in HDFS
        image_path = f"{config['image_data']}/{image_id}"

        # Read the image file as binary
        with hdfs.open_input_file(image_path) as file:
            image_data = file.read()

        # Convert binary data to PIL image
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        img_tensor = preprocess(img)  
        preprocessed_img = img_tensor.unsqueeze(0)

        # Extract feature vector
        with torch.no_grad():
            feature_vector = feature_extractor(preprocessed_img).squeeze().numpy().astype(np.float16)
        
        return feature_vector, image_id 

    except Exception as e:
        print(f"Error processing image {image_id}: {e}")
        return None, None  


@ray.remote
def batch_feature_extraction(config, batch, feature_extractor, hdfs):
    """Extract features for a batch of images at once."""
    batch_results = []
    for image_id in batch:
        result = feature_vector_extraction(config, image_id, feature_extractor, hdfs)
        batch_results.append(result)

    return batch_results  # Return results for the whole batch



def train_and_test(config):
    model = config["model"]
    train_loader = config["train_loader"]
    test_loader = config["test_loader"]

    criterion = config["criterion"]
    optimizer = config["optimizer"]

    epoch_logs, epoch_losses = [], []

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        epoch_logs.append(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {epoch_loss:.4f}")
        print(epoch_logs[-1])  # Log each epoch's loss

    # Evaluation
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            probabilities = torch.sigmoid(outputs)  # Use sigmoid for binary classification
            predictions = (probabilities >= 0.5).float()  # Threshold at 0.5

            # Collect labels, predictions, and probabilities
            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.numpy())
            all_probs.extend(probabilities.numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')

    evaluation_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }

    return epoch_logs, evaluation_metrics, epoch_losses


@ray.remote
def kfold_cross_validation(config, fold_idx, train_data, test_data, cnn_feature_columns):
    """Perform training and testing for a single fold."""
    
    start_time = time.time()
    
    print(f"Starting Fold {fold_idx + 1}...")
    # Create datasets and loaders for this fold
    train_set = MIDASDataset(train_data, cnn_feature_columns)
    test_set = MIDASDataset(test_data, cnn_feature_columns)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)   # config["batch_size"]
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # Initialize model, optimizer, and criterion
    input_dim = train_set.features.shape[1]
    model = ClassifierNN(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001) #config["lr"]
    criterion = nn.BCEWithLogitsLoss()

    # Update config for this fold
    fold_config = config.copy()
    fold_config.update({
        "model": model,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "criterion": criterion,
        "optimizer": optimizer,
    })

    # Perform training and evaluation
    epoch_logs, evaluation_metrics, epoch_losses = train_and_test(fold_config)
    
    duration = time.time() - start_time
    
    result = {
        "fold_idx": fold_idx,
        "epoch_logs": epoch_logs,
        "evaluation_metrics": evaluation_metrics,
        "epoch_losses": epoch_losses,
        "fold_duration": duration,
    }
    return result


def distributed_pipeline(config):
    
    """ 
    ===========================
        Data Preprocessing 
    ===========================
    """
    
    try:
        hdfs = HadoopFileSystem(host=config["hdfs_host"], port=config["hdfs_port"])
        print(f"Connected to HDFS at {config['hdfs_host']}:{config['hdfs_port']}.")
        
    except Exception as e:
        print(f"Failed to connect to HDFS: {e}")
    
    
    print("Loading and preprocessing tabular data...")
    
    try:
        with hdfs.open_input_file(config["tabular_data"]) as file:
            tabular_data = pd.read_excel(file)
            print(f"Successfully loaded {len(tabular_data)} rows from {config['tabular_data']}.")

    except Exception as e:
        print(f"Failed to load data from HDFS: {e}")
        return None 

    
    preprocessed_data = tabular_data_preprocessing(tabular_data)

    print("Feature extraction from images...")
    feature_extraction_start_time = time.time()
    
    # Load the ResNet50 model pre-trained on ImageNet
    base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    base_model.eval()  # Set to evaluation mode

    # Remove the fully connected layer (extract feature vectors only)
    feature_extractor = torch.nn.Sequential(*list(base_model.children())[:-1])
    
    fve_results = []
    futures = []

    for i in range(0, len(preprocessed_data['midas_file_name']), config["batch_size"]):
        batch = preprocessed_data['midas_file_name'][i:i+config["batch_size"]]

        # Submit batch of images as a single task
        batch_futures = batch_feature_extraction.remote(config, batch, feature_extractor, hdfs)
        futures.append(batch_futures)

        # Retrieve and save results after some images to relief memory
        if len(futures) >= config["save_interval"] // config["batch_size"]:
            batch_results = ray.get(futures) 
            for batch in batch_results:
                fve_results.extend(batch) 

            futures = []  # Reset batch futures to avoid memory buildup

    # Retrieve remaining results after all tasks are submitted
    if futures:
        batch_results = ray.get(futures)
        for batch in batch_results:
            fve_results.extend(batch)
        
    feature_vectors, image_ids = [], []
    for fv, ids in fve_results:
        feature_vectors.append(fv)
        image_ids.append(ids)
    feature_vectors = np.vstack(feature_vectors)
    image_ids = np.hstack(image_ids)

    feature_extraction_end_time = time.time()
    feature_extraction_duration = round(feature_extraction_end_time - feature_extraction_start_time, 2)
    config["results"]["Feature Extraction Time"] = feature_extraction_duration

    print("Combining features with preprocessed data...")
    final_data, cnn_feature_columns = combine_features(feature_vectors, image_ids, preprocessed_data)
    
    print("Data preprocessing completed.")
    
    log_text = config['log_text']
    
    log_text.append(f"\n\n=============================================")
    log_text.append(f"          Feature Vector Extraction")        
    log_text.append(f"=============================================")
    log_text.append(f"\nTotal Feature Extraction Duration: {feature_extraction_duration} seconds")
        
    
    """ 
    =============================
        Training and Testing
    =============================
    """
    
    print("Performing 10-Fold Cross Validation...")
    cross_validation_start_time = time.time()
    
    kf = KFold(n_splits=10, shuffle=True, random_state=21)

    # Dispatch each fold as an independent task
    futures = [
        kfold_cross_validation.remote(config, fold_idx, final_data.iloc[train_idx], final_data.iloc[test_idx], cnn_feature_columns)
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(final_data))
    ]

    # Gather results from all workers
    kfold_results = ray.get(futures)
    
    fold_epoch_losses = [result["epoch_losses"] for result in kfold_results]
    num_epochs = len(fold_epoch_losses[0])  # Assumes all folds have the same number of epochs
    mean_epoch_losses = [
        round(np.mean([fold_losses[epoch_idx] for fold_losses in fold_epoch_losses]), 4)
        for epoch_idx in range(num_epochs)
    ]
    
    # Calculate mean metrics across folds
    mean_metrics = {
        "accuracy": round(np.mean([result["evaluation_metrics"]["accuracy"] for result in kfold_results]), 4),
        "precision": round(np.mean([result["evaluation_metrics"]["precision"] for result in kfold_results]), 4),
        "recall": round(np.mean([result["evaluation_metrics"]["recall"] for result in kfold_results]), 4),
        "f1": round(np.mean([result["evaluation_metrics"]["f1"] for result in kfold_results]), 4),
        "roc_auc": round(np.mean([result["evaluation_metrics"]["roc_auc"] for result in kfold_results]), 4),
        "fold_duration": round(np.mean([result["fold_duration"] for result in kfold_results]), 2),
    }
    config["results"]["Mean Accuracy"] = mean_metrics["accuracy"]
    config["results"]["Mean Precision"] = mean_metrics["precision"]
    config["results"]["Mean Recall"] = mean_metrics["recall"]
    config["results"]["Mean F1 Score"] = mean_metrics["f1"]
    config["results"]["Mean AUC-ROC"] = mean_metrics["roc_auc"]
    config["results"]["Mean Time per Fold"] = mean_metrics['fold_duration']
    
    cross_validation_end_time = time.time()
    cross_validation_duration = round(cross_validation_end_time - cross_validation_start_time, 2)
    config["results"]["Cross Validation Time"] = cross_validation_duration
    print("10-Fold Cross Validation Completed.")

    # Log results
    log_text.append(f"\n\n============================================")
    log_text.append(f"          10 Fold Cross Validation")
    log_text.append(f"============================================")
    log_text.append(f"\nTotal 10-Fold Duration: {cross_validation_duration} seconds")
    
    log_text.append(f"\n--- Duration per Fold ---")
    log_text.append(f"Mean Time Per Fold:   {mean_metrics['fold_duration']} seconds\n")
    for result in kfold_results:
        fold_idx = result["fold_idx"]
        log_text.append(f"Fold {fold_idx+1} Duration:   {result['fold_duration']:.2f} seconds")
    
    log_text.append(f"\n--- Mean Loss Per Epoch Across All Folds ---")
    for epoch_idx, mean_loss in enumerate(mean_epoch_losses, start=1):
        log_text.append(f"Epoch [{epoch_idx}/{config['epochs']}], Mean Loss: {mean_loss}")
        
    # Log mean metrics
    log_text.append(f"\n--- Mean Metrics Across All Folds ---")
    log_text.append(f"Mean Accuracy:  {mean_metrics['accuracy']}")
    log_text.append(f"Mean Precision: {mean_metrics['precision']}")
    log_text.append(f"Mean Recall:    {mean_metrics['recall']}")
    log_text.append(f"Mean F1 Score:  {mean_metrics['f1']}")
    log_text.append(f"Mean AUC-ROC:   {mean_metrics['roc_auc']}\n")



def main():
    
    tabular_data_path = "/data/mra_midas/release_midas.xlsx"
    test_data_path = "/data/mra_midas/release_midas_test.xlsx"
    data_1_path = "/data/mra_midas/data_1.xlsx"
    data_2_path = "/data/mra_midas/data_2.xlsx"
    data_3_path = "/data/mra_midas/data_3.xlsx"
    images_folder = "/data/mra_midas/images"
    
    pipeline_start_time = time.time()
    ray.init(address="auto")
    # os.environ["RAY_DEDUP_LOGS"] = "0"
    
    log_text = []
    
    # --- Initialize results dictionary ---
    results = {
        "Framework": "ray",
        "Dataset": None,
        "Nodes": get_num_nodes(),
        "Total Time": None,
        "Feature Extraction Time": None,
        "Cross Validation Time": None,
        "Mean Time per Fold": None,
        "Mean Accuracy": None,
        "Mean Precision": None,
        "Mean Recall": None,
        "Mean F1 Score": None,
        "Mean AUC-ROC": None
    }
    
    config = {
        "hdfs_host": "192.168.0.1",
        "hdfs_port": 9000,
        "num_nodes": get_num_nodes(),
        "epochs": 10,
        "tabular_data": test_data_path,
        "image_data": images_folder,
        "log_text": log_text,
        "results": results,
        "load_precomputed_features": False,
        "batch_size": 10,
        "save_interval": 120
    }
    
    # Define data file paths and sizes (in GB)
    datasets = {
        "data_1.xlsx": 1.05,
        "data_2.xlsx": 2.16,
        "data_3.xlsx": 3.37
    }
    
    # --- Identify dataset number and size ---
    dataset_number = None
    tabular_file = os.path.basename(config["tabular_data"])  # Extract filename
    if tabular_file in datasets:
        dataset_number = int(tabular_file.split("_")[1].split(".")[0])  # Extract dataset number (1, 2, 3)
        dataset_size = datasets[tabular_file]
        dataset_info = f"Dataset Used: {tabular_file} ({dataset_size:.2f} GB)"
        log_filename = f"ray_data{dataset_number}_node{config['num_nodes']}.txt"
        results_filename = f"ray_data{dataset_number}_node{config['num_nodes']}.json"
        results["Dataset"] = dataset_number
    else:
        dataset_info = f"Dataset Used: {tabular_file}"
        log_filename = f"unknown_data_ray_log.txt"
        # results["Dataset"] = "unknown"
            
    
    log_text.append(f"======================================")    
    log_text.append(f"      Pipeline Execution Summary")
    log_text.append(f"======================================")
    log_text.append(f"\nNumber of Nodes: {config['num_nodes']}")
    log_text.append(dataset_info)  # Add dataset name and size (if available)
    
    distributed_pipeline(config)

    pipeline_end_time = time.time()
    pipeline_duration = round(pipeline_end_time - pipeline_start_time, 2)
    config["results"]["Total Time"] = pipeline_duration
    
    log_text.append(f"\nTotal Pipeline Duration: {pipeline_duration} seconds")
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_filename)
    with open(log_file, "w") as file:
        file.write("\n".join(log_text))
    print(f"Log saved to {log_file}")
    
    if dataset_number is not None:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, results_filename)
        with open(results_file, "w") as file:
            json.dump(results, file, indent=4)
        print(f"Results saved to {results_file}")

    ray.shutdown()


if __name__ == "__main__":
    main()
