import os
import numpy as np
import pandas as pd
import tqdm
import time
from PIL import Image
import psutil
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


@ray.remote
def feature_vector_extraction(dataset, image_folder, disable_progress=False):
    """Extract feature vectors with an option to disable progress bar."""
    
    start_time = time.time()  # Start timing the worker's process

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

    feature_vectors, image_ids, cpu_usage = [], [], []
    image_count = 0

    print("Starting feature extraction...")
    for _, row in tqdm.tqdm(dataset.iterrows(), total=len(dataset), desc="Processing images", disable=disable_progress):
        cpu_usage.append(psutil.cpu_percent(interval=0.1))  # Capture CPU usage every 0.1 seconds
        image_path = os.path.join(image_folder, row["midas_file_name"])
        try:
            preprocessed_img = preprocess_image(image_path)
            with torch.no_grad():
                features = feature_extractor(preprocessed_img)
            feature_vectors.append(features.squeeze().numpy())
            image_ids.append(row["midas_file_name"])
            image_count += 1
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    duration = time.time() - start_time
    avg_cpu_usage = np.mean(cpu_usage)
    
    return feature_vectors, image_ids, image_count, duration, avg_cpu_usage



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


def perform_10_fold_cv(config, fold_idx, train_data, test_data, cnn_feature_columns):
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
    
    return {
        "fold_idx": fold_idx,
        "epoch_logs": epoch_logs,
        "evaluation_metrics": evaluation_metrics,
        "epoch_losses": epoch_losses,
        "fold_duration": duration,
    }
    

# Function to process a batch of folds
@ray.remote
def process_fold_batch(config, fold_batch, final_data, cnn_feature_columns, kf):
    
    start_time = time.time()
    
    # Track CPU utilization during processing
    results, cpu_usage = [], []
    fold_count = 0
    for fold_idx in fold_batch:
        train_idx, test_idx = list(kf.split(final_data))[fold_idx]
        train_data = final_data.iloc[train_idx]
        test_data = final_data.iloc[test_idx]
        # Call perform_10_fold_cv for each fold in this batch
        result = perform_10_fold_cv(config, fold_idx, train_data, test_data, cnn_feature_columns)
        results.append(result)
        cpu_usage.append(psutil.cpu_percent(interval=0.1))
        fold_count += 1
        
    duration = time.time() - start_time
    avg_cpu_usage = np.mean(cpu_usage)  # Average CPU usage for this worker
    
    return results, fold_count, duration, avg_cpu_usage


def distributed_pipeline(config):
    
    """ 
    ===========================
        Data Preprocessing 
    ===========================
    """
    
    print("Loading and preprocessing tabular data...")
    tabular_data = pd.read_excel(config['tabular_data'])
    preprocessed_data = tabular_data_preprocessing(tabular_data)

    print("Feature extraction from images...")
    feature_extraction_start_time = time.time()

    stats, worker_durations = [], []
    
    if config.get("load_precomputed_features", True):
        print("Loading precomputed feature vectors and image IDs...")
        feature_vectors = np.load("feature_vectors_mb.npy")
        image_ids = np.load("image_ids_mb.npy", allow_pickle=True)
    else:
        
        batches = np.array_split(preprocessed_data, config["world_size"])
        futures = [
            feature_vector_extraction.remote(batch, config['image_data'], disable_progress=config['disable_progress'])
            for batch in batches
        ]
        results = ray.get(futures)
        
        feature_vectors, image_ids, worker_cpu_usages = [], [], []
        for idx, (fv, ids, count, duration, cpu_usage) in enumerate(results):
            feature_vectors.append(fv)
            worker_durations.append(duration)
            image_ids.append(ids)
            worker_cpu_usages.append(cpu_usage)
            stats.append(f"Worker {idx+1} processed {count} images in {duration:.2f} seconds with avg CPU usage {cpu_usage:.2f}%.")
        feature_vectors = np.vstack(feature_vectors)
        image_ids = np.hstack(image_ids)

    feature_extraction_end_time = time.time()
    feature_extraction_duration = feature_extraction_end_time - feature_extraction_start_time

    print("Combining features with preprocessed data...")
    final_data, cnn_feature_columns = combine_features(feature_vectors, image_ids, preprocessed_data)
    
    print("Data preprocessing completed.")
    
    log_text = config['log_text']
    
    log_text.append(f"\n\n=============================================")
    log_text.append(f"          Feature Vector Extraction")        
    log_text.append(f"=============================================")
    log_text.append(f"Total Feature Extraction Duration: {feature_extraction_duration:.2f} seconds")
    log_text.append(f"\n--- Statistics per Worker ---")
    log_text.extend(stats)
    
    if not config.get("load_precomputed_features", True):
        
        # Calculate average CPU usage across all workers
        overall_avg_cpu_usage = np.mean(worker_cpu_usages)
        log_text.append(f"Overall Average CPU Usage: {overall_avg_cpu_usage:.2f}%")
        
        # Calculate Worker Utilization
        log_text.append("\n--- Worker Utilizations ---")
        worker_utilizations = [
            (duration / feature_extraction_duration) * 100 for duration in worker_durations
        ]
        for idx, utilization in enumerate(worker_utilizations):
            log_text.append(f"Worker {idx+1}: {utilization:.2f}% utilization")
            
        # Calculate global throughput
        log_text.append("\n--- Worker Throughput ---")
        for idx, (fv, ids, count, duration, cpu_usage) in enumerate(results):
            worker_throughput = count / duration
            log_text.append(f"Worker {idx+1}: {worker_throughput:.2f} images/second")
        global_throughput = len(image_ids) / feature_extraction_duration
        log_text.append(f"Global Throughput: {global_throughput:.2f} images/second")
            

    
    
    """ 
    =============================
        Training and Testing
    =============================
    """
    
    print("Performing 10-Fold Cross Validation...")
    train_test_start_time = time.time()
    
    kf = KFold(n_splits=10, shuffle=True, random_state=21)

    # Split folds into batches based on world size
    fold_batches = np.array_split(list(range(10)), config["world_size"])

    # Dispatch fold batches to workers
    fold_futures = []
    for fold_batch in fold_batches:
        future = process_fold_batch.remote(config, fold_batch, final_data, cnn_feature_columns, kf)
        fold_futures.append(future)

    # Gather results from all workers
    worker_results = ray.get(fold_futures)

    # Initialize stats for workers and fold results
    worker_stats, fold_results, worker_durations, worker_cpu_usages = [], [], [], []
    for idx, worker_result in enumerate(worker_results):
        worker_folds, fold_count, duration, cpu_usage = worker_result
        fold_results.extend(worker_folds)  # Flatten fold results
        worker_durations.append(duration)
        worker_cpu_usages.append(cpu_usage)
        worker_stats.append(f"Worker {idx+1} processed {fold_count} folds in {duration:.2f} seconds with avg CPU usage {cpu_usage:.2f}%.")

        print("10-Fold Cross Validation Completed.")
        
    # mean_epoch_losses = None  # To store mean loss across epochs
    # fold_epoch_losses = []    # Collect all folds' epoch losses
    
    # for result in fold_results:
    #     fold_epoch_losses.append(result["epoch_losses"])

    # # Compute mean loss for each epoch
    # num_epochs = len(fold_epoch_losses[0])  # Assumes all folds use the same number of epochs
    # mean_epoch_losses = [
    #     np.mean([fold_losses[epoch_idx] for fold_losses in fold_epoch_losses])
    #     for epoch_idx in range(num_epochs)
    # ]
    
    fold_epoch_losses = [result["epoch_losses"] for result in fold_results]
    num_epochs = len(fold_epoch_losses[0])  # Assumes all folds have the same number of epochs
    mean_epoch_losses = [
        np.mean([fold_losses[epoch_idx] for fold_losses in fold_epoch_losses])
        for epoch_idx in range(num_epochs)
    ]
    
    # Calculate mean metrics across folds
    mean_metrics = {
        "accuracy": np.mean([result["evaluation_metrics"]["accuracy"] for result in fold_results]),
        "precision": np.mean([result["evaluation_metrics"]["precision"] for result in fold_results]),
        "recall": np.mean([result["evaluation_metrics"]["recall"] for result in fold_results]),
        "f1": np.mean([result["evaluation_metrics"]["f1"] for result in fold_results]),
        "roc_auc": np.mean([result["evaluation_metrics"]["roc_auc"] for result in fold_results]),
        "fold_duration": np.mean([result["fold_duration"] for result in fold_results]),
    }
    
    train_test_end_time = time.time()
    train_test_duration = train_test_end_time - train_test_start_time

    # Log results
    log_text.append(f"\n\n============================================")
    log_text.append(f"          10 Fold Cross Validation")
    log_text.append(f"============================================")
    log_text.append(f"Total 10-Fold Duration: {train_test_duration:.2f} seconds")
    log_text.append(f"\n--- Statistics per Worker ---")
    log_text.extend(worker_stats)
    
    # Calculate average CPU usage across all workers
    overall_avg_cpu_usage = np.mean(worker_cpu_usages)
    log_text.append(f"Overall Average CPU Usage: {overall_avg_cpu_usage:.2f}%")
    
    # Calculate Worker Utilization
    log_text.append("\n--- Worker Utilizations ---")
    worker_utilizations = [
        (duration / train_test_duration) * 100 for duration in worker_durations
    ]
    for idx, utilization in enumerate(worker_utilizations):
        log_text.append(f"Worker {idx+1}: {utilization:.2f}% utilization")
    
    log_text.append("\n--- Worker Throughput ---")
    for idx, worker_result in enumerate(worker_results):
        worker_throughput = fold_count / duration
        log_text.append(f"Worker {idx+1}: {worker_throughput:.2f} folds/second")
    global_throughput = 10 / train_test_duration
    log_text.append(f"Global Throughput: {global_throughput:.2f} folds/second")

    # for result in fold_results:
    #     fold_idx = result["fold_idx"]
    #     log_text.append(f"\n--- Fold {fold_idx + 1} ---")
    #     log_text.extend(result["epoch_logs"])
    #     metrics = result["evaluation_metrics"]
    #     log_text.append(f"Accuracy:  {metrics['accuracy']:.4f}")
    #     log_text.append(f"Precision: {metrics['precision']:.4f}")
    #     log_text.append(f"Recall:    {metrics['recall']:.4f}")
    #     log_text.append(f"F1 Score:  {metrics['f1']:.4f}")
    #     log_text.append(f"AUC-ROC:   {metrics['roc_auc']:.4f}\n")
    #     log_text.append(f"Duration:   {result['fold_duration']:.4f}")
    
    log_text.append(f"\n--- Duration per Fold ---")
    log_text.append(f"Mean Time Per Fold:   {mean_metrics['fold_duration']:.4f} seconds\n")
    for result in fold_results:
        fold_idx = result["fold_idx"]
        log_text.append(f"Fold {fold_idx+1} Duration:   {result['fold_duration']:.4f} seconds")
    
    log_text.append(f"\n--- Mean Loss Per Epoch Across All Folds ---")
    for epoch_idx, mean_loss in enumerate(mean_epoch_losses, start=1):
        log_text.append(f"Epoch [{epoch_idx}/{config["epochs"]}], Mean Loss: {mean_loss:.4f}")
        
    # Log mean metrics
    log_text.append(f"\n--- Mean Metrics Across All Folds ---")
    log_text.append(f"Mean Accuracy:  {mean_metrics['accuracy']:.4f}")
    log_text.append(f"Mean Precision: {mean_metrics['precision']:.4f}")
    log_text.append(f"Mean Recall:    {mean_metrics['recall']:.4f}")
    log_text.append(f"Mean F1 Score:  {mean_metrics['f1']:.4f}")
    log_text.append(f"Mean AUC-ROC:   {mean_metrics['roc_auc']:.4f}\n")



def main():
    
    tabular_data_path = "release_midas.xlsx"
    data_1_path = "data_1.xlsx"
    data_2_path = "data_2.xlsx"
    data_3_path = "data_3.xlsx"
    images_folder = "C:/Users/nikol/Desktop/university/9th_semester/physiological_systems_simulation/project/dataset/midasmultimodalimagedatasetforaibasedskincancer"
    
    log_text = []

    config = {
        "world_size": 1,
        "epochs": 10,
        "tabular_data": tabular_data_path,
        "image_data": images_folder,
        "log_text": log_text,
        "disable_progress": False, 
        "load_precomputed_features": False
    }
    
    log_text.append(f"======================================")    
    log_text.append(f"      Pipeline Execution Summary")
    log_text.append(f"======================================")
    log_text.append(f"\nNumber of Workers: {config['world_size']}")
    # log_text.append(f"Learning Rate: {config['lr']}")
    # log_text.append(f"Batch Size: {config['batch_size']}")
    # log_text.append(f"Epochs: {config['epochs']}\n")
    
    pipeline_start_time = time.time()
    ray.init()
    # os.environ["RAY_DEDUP_LOGS"] = "0"
    # ray_init_time = time.time() - pipeline_start_time
    
    # log_text.append(f"=== Ray Initialization ===")
    # log_text.append(f"Time Taken: {ray_init_time:.2f} seconds\n")
    
    distributed_pipeline(config)

    pipeline_end_time = time.time()
    pipeline_duration = pipeline_end_time - pipeline_start_time
    
    log_text.append(f"\nTotal Pipeline Duration: {pipeline_duration:.2f} seconds")
    # log_text.append(f"=== Pipeline Completed ===")
    
    results_dir = "local_logs"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"ray_log_{config['world_size']}.txt")
    with open(results_file, "w") as file:
        file.write("\n".join(log_text))

    print(f"Log saved to {results_file}")

    ray.shutdown()


if __name__ == "__main__":
    main()
