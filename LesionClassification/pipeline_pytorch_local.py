import os
import numpy as np
import pandas as pd
import tqdm
import time
from PIL import Image
import psutil
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
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


class FeatureVectorDataset(Dataset):
    def __init__(self, image_data, image_folder):
        self.image_data = image_data
        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        row = self.image_data.iloc[idx]
        image_path = os.path.join(self.image_folder, row["midas_file_name"])
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, row["midas_file_name"]
    

def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(21)
    

def cleanup():
    dist.destroy_process_group()


def feature_vector_extraction(dataloader, rank, disable_progress=False):
    """Extract feature vectors using PyTorch distributed."""
    
    start_time = time.time()  # Start timing

    # Load ResNet50
    base_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    base_model.eval()  # Set to evaluation mode

    # Remove the fully connected layer
    feature_extractor = torch.nn.Sequential(*list(base_model.children())[:-1])

    feature_vectors, image_ids, cpu_usage = [], [], []
    image_count = 0

    # Process batches assigned to this rank
    for images, file_names in tqdm.tqdm(dataloader, desc=f"Rank {rank} Feature Extraction", disable=disable_progress):
        cpu_usage.append(psutil.cpu_percent(interval=0.1))  # Track CPU usage
        try:
            with torch.no_grad():
                # features = feature_extractor(images).squeeze()
                # Extract features (output shape: [batch_size, 2048, 1, 1])
                features = feature_extractor(images)  # No need for unsqueeze
                # Flatten to [batch_size, 2048]
                features = features.view(features.size(0), -1)
            feature_vectors.extend(features.numpy())
            image_ids.extend(file_names)
            image_count += len(file_names)
        except Exception as e:
            print(f"Error processing images: {e}")

    duration = time.time() - start_time
    avg_cpu_usage = np.mean(cpu_usage)  # Average CPU usage

    print(f"Rank {rank} completed feature extraction in {duration:.2f} seconds with avg CPU usage {avg_cpu_usage:.2f}%")
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
    optimizer = optim.Adam(model.parameters(), lr=0.001) # config["lr"]
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
def process_fold_batch(config, fold_batch, final_data, cnn_feature_columns, kf):
    
    start_time = time.time()
    
    # Track CPU utilization during processing
    results, cpu_usage = [], []
    fold_count = 0  # Initialize fold count
    
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
    
    log_text = config['log_text']
    rank = config["rank"]
    world_size = config["world_size"]
    
    if rank == 0:
        print("Loading and preprocessing tabular data...")
    tabular_data = pd.read_excel(config['tabular_data'])
    preprocessed_data = tabular_data_preprocessing(tabular_data)

    
    # Setup distributed environment
    setup(rank, world_size)
    # log_text.append(f"Process {rank} initialized.")
    
    if rank == 0:
        print("Feature extraction from images...")
    feature_extraction_start_time = time.time()

    stats, worker_durations = [], []
    
    if config.get("load_precomputed_features", True):
        if rank == 0:
            print("Loading precomputed feature vectors and image IDs...")
        feature_vectors = np.load("feature_vectors_mb.npy")
        image_ids = np.load("image_ids_mb.npy", allow_pickle=True)
        
    else:
        
        # Create dataset and dataloader
        dataset = FeatureVectorDataset(preprocessed_data, config['image_data'])
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)    # num_workers = 0 by default to mimic ray
                                                                            # each worker is a process in the distributed enviroment

        # Perform feature extraction
        fv, ids, count, duration, avg_cpu = feature_vector_extraction(
            dataloader, rank, config['disable_progress']
        )   #config['image_data'],

        # Gather results across ranks
        if rank == 0:
            gathered_fv = [None for _ in range(world_size)]
            gathered_ids = [None for _ in range(world_size)]
            gathered_cpu_usages = [None for _ in range(world_size)]
            gathered_durations = [None for _ in range(world_size)]
            gathered_counts = [None for _ in range(world_size)]
        else:
            gathered_fv = None
            gathered_ids = None
            gathered_cpu_usages = None
            gathered_durations = None
            gathered_counts = None   
        

        dist.gather_object(fv, gathered_fv)
        dist.gather_object(ids, gathered_ids)
        dist.gather_object(avg_cpu, gathered_cpu_usages)
        dist.gather_object(duration, gathered_durations)
        dist.gather_object(count, gathered_counts)
        
        
        # Only rank 0 aggregates the results
        if rank == 0:
            feature_vectors = np.vstack(gathered_fv)
            image_ids = np.hstack(gathered_ids)

            # Calculate stats
            for i in range(world_size):
                stats.append(
                    f"Worker {i+1} processed {gathered_counts[i]} images in {gathered_durations[i]:.2f} seconds "
                    f"with avg CPU usage {gathered_cpu_usages[i]:.2f}%."
                )

            worker_durations = gathered_durations


    feature_extraction_end_time = time.time()
    feature_extraction_duration = feature_extraction_end_time - feature_extraction_start_time   

    # Combine features with preprocessed tabular data on rank 0
    if rank == 0:
        final_data, cnn_feature_columns = combine_features(feature_vectors, image_ids, preprocessed_data)
        to_broadcast = [final_data, cnn_feature_columns]
    else:
        to_broadcast = [None, None]

    dist.broadcast_object_list(to_broadcast, src=0)

    final_data = to_broadcast[0]
    cnn_feature_columns = to_broadcast[1]

    # Cleanup distributed environment
    cleanup()
    # log_text.append("\nDistributed process group destroyed\n")

        

    if rank == 0:
        
        print("Data preprocessing completed.")
        
        log_text.append(f"\n\n=============================================")
        log_text.append(f"          Feature Vector Extraction")        
        log_text.append(f"=============================================")
        log_text.append(f"Total Feature Extraction Duration: {feature_extraction_duration:.2f} seconds")
        log_text.append(f"\n--- Statistics per Worker ---")
        log_text.extend(stats)
        
        if not config.get("load_precomputed_features", True):
            
            # Calculate average CPU usage across all workers
            overall_avg_cpu_usage = np.mean(gathered_cpu_usages)
            log_text.append(f"Overall Average CPU Usage: {overall_avg_cpu_usage:.2f}%")
            
            # Calculate Worker Utilization
            log_text.append("\n--- Worker Utilizations ---")
            worker_utilizations = [
                (duration / feature_extraction_duration) * 100 for duration in worker_durations
            ]
            
            for idx, utilization in enumerate(worker_utilizations):
                log_text.append(f"Worker {idx+1}: {utilization:.2f}% utilization")

            log_text.append("\n--- Worker Throughput ---")
            for i in range(world_size):
                worker_throughput = gathered_counts[i] / gathered_durations[i]
                log_text.append(f"Worker {i+1}: {worker_throughput:.2f} images/second")
            global_throughput = len(image_ids) / feature_extraction_duration
            log_text.append(f"Global Throughput: {global_throughput:.2f} images/second")
    
    
    """ 
    =============================
        Training and Testing
    =============================
    """
    
    # Setup distributed environment
    setup(rank, world_size)
    # log_text.append(f"Process {rank} initialized.")
    
    if rank == 0:
        print("Performing 10-Fold Cross Validation...")
    train_test_start_time = time.time()
        
    kf = KFold(n_splits=10, shuffle=True, random_state=21)
    fold_indices = list(range(10))  # Indices of folds

    # Split folds among ranks
    fold_batches = np.array_split(fold_indices, world_size)[rank]
    # fold_count = len(fold_batches)

    # Process folds assigned to this rank
    results, fold_count, duration, avg_cpu_usage = process_fold_batch(
        config, fold_batches, final_data, cnn_feature_columns, kf
    )
    
    # Gather results across ranks
    if rank == 0:
        gathered_results = [None for _ in range(world_size)]
        gathered_durations = [None for _ in range(world_size)]
        gathered_cpu_usages = [None for _ in range(world_size)]
        gathered_fold_counts = [None for _ in range(world_size)]
    else:
        gathered_results = None
        gathered_durations = None
        gathered_cpu_usages = None
        gathered_fold_counts = None

    dist.gather_object(results, gathered_results)
    dist.gather_object(duration, gathered_durations)
    dist.gather_object(avg_cpu_usage, gathered_cpu_usages)
    dist.gather_object(fold_count, gathered_fold_counts)


    # Aggregate results and log metrics (only on rank 0)
    if rank == 0:
        fold_results = [fold for worker_result in gathered_results for fold in worker_result]
        worker_stats = []
        for i in range(world_size):
            worker_stats.append(
                f"Worker {i+1} processed {gathered_fold_counts[i]} folds in {gathered_durations[i]:.2f} seconds "
                f"with avg CPU usage {gathered_cpu_usages[i]:.2f}%."
            )

        fold_epoch_losses = [result["epoch_losses"] for result in fold_results]
        num_epochs = len(fold_epoch_losses[0])  # Assumes all folds have the same number of epochs
        mean_epoch_losses = [
            np.mean([fold_losses[epoch_idx] for fold_losses in fold_epoch_losses])
            for epoch_idx in range(num_epochs)
        ]
        
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
    
    # Cleanup distributed environment
    cleanup()
    # log_text.append("\nDistributed process group destroyed\n")
    
    
    if rank == 0:
        
        print("10-Fold Cross Validation Completed.")

        # Log results
        log_text.append(f"\n\n============================================")
        log_text.append(f"          10 Fold Cross Validation")
        log_text.append(f"============================================")
        log_text.append(f"Total 10-Fold Duration: {train_test_duration:.2f} seconds")
        log_text.append(f"\n--- Statistics per Worker ---")
        log_text.extend(worker_stats)
        
        # Calculate average CPU usage across all workers
        overall_avg_cpu_usage = np.mean(gathered_cpu_usages)
        log_text.append(f"Overall Average CPU Usage: {overall_avg_cpu_usage:.2f}%")
        
        # Worker Utilization
        log_text.append("\n--- Worker Utilizations ---")
        worker_utilizations = [
            (duration / train_test_duration) * 100 for duration in gathered_durations
        ]
        for idx, utilization in enumerate(worker_utilizations):
            log_text.append(f"Worker {idx+1}: {utilization:.2f}% utilization")
        
        
        log_text.append("\n--- Worker Throughput ---")
        for i in range(world_size):
            worker_throughput = gathered_fold_counts[i] / gathered_durations[i]
            log_text.append(f"Worker {i+1}: {worker_throughput:.2f} folds/second")
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
    
    # Rank and world size from environment variables
    rank = int(os.getenv("RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", -1))
    
    # Ensure proper environment variables are set
    if rank == -1 or world_size == -1:
        raise RuntimeError("Environment variables RANK and WORLD_SIZE must be set. Use torchrun to run the program.")


    config = {
        "rank": rank,
        "world_size": world_size,
        "epochs": 10,
        "tabular_data": tabular_data_path,
        "image_data": images_folder,
        "log_text": log_text,
        "disable_progress": False,   # disable tdqm progress bars
        "load_precomputed_features": False
    }
    
    # Log pipeline initialization
    if rank == 0:
        log_text.append(f"======================================")
        log_text.append(f"      Pipeline Execution Summary")
        log_text.append(f"======================================")
        log_text.append(f"\nNumber of Workers: {world_size}")

    pipeline_start_time = time.time()
    
    distributed_pipeline(config)
    
    pipeline_end_time = time.time()
    pipeline_duration = pipeline_end_time - pipeline_start_time

    # Log pipeline duration
    if rank == 0:
        log_text.append(f"\nTotal Pipeline Duration: {pipeline_duration:.2f} seconds")

        # Save logs to file
        results_dir = "local_logs"
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"pytorch_log_{world_size}.txt")
        with open(results_file, "w") as file:
            file.write("\n".join(log_text))
        print(f"Log saved to {results_file}")


if __name__ == "__main__":
    main()
