import os
import numpy as np
import pandas as pd
import time
import json
from PIL import Image
import io
from pyarrow.fs import HadoopFileSystem
import torch
import torch.nn as nn
import torch.distributed as dist
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
    

def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = '192.168.0.1'
    # os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(21)
    

def cleanup():
    dist.destroy_process_group()


def feature_vector_extraction(config, image_id, feature_extractor, hdfs):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    
    try:
        image_path = f"{config['image_data']}/{image_id}"

        with hdfs.open_input_file(image_path) as file:
            image_data = file.read()

        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        img_tensor = preprocess(img)  
        preprocessed_img = img_tensor.unsqueeze(0)

        with torch.no_grad():
            feature_vector = feature_extractor(preprocessed_img).squeeze().numpy().astype(np.float16)
        
        return feature_vector, image_id 

    except Exception as e:
        print(f"Error processing image {image_id}: {e}")
        return None, None  


def batch_feature_extraction(config, batch, feature_extractor, hdfs):
    batch_results = []
    for image_id in batch:
        result = feature_vector_extraction(config, image_id, feature_extractor, hdfs)
        batch_results.append(result)

    return batch_results  


def train_and_test(config):
    model = config["model"]
    train_loader = config["train_loader"]
    test_loader = config["test_loader"]

    criterion = config["criterion"]
    optimizer = config["optimizer"]

    epoch_logs, epoch_losses = [], []

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

    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= 0.5).float() 

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.numpy())
            all_probs.extend(probabilities.numpy())

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


def kfold_cross_validation(config, fold_idx, train_data, test_data, cnn_feature_columns):
    start_time = time.time()
    print(f"Rank {config['rank']}: Starting Fold {fold_idx + 1}...")

    train_set = MIDASDataset(train_data, cnn_feature_columns)
    test_set = MIDASDataset(test_data, cnn_feature_columns)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)   # config["batch_size"]
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    input_dim = train_set.features.shape[1]
    model = ClassifierNN(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    criterion = nn.BCEWithLogitsLoss()

    fold_config = config.copy()
    fold_config.update({
        "model": model,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "criterion": criterion,
        "optimizer": optimizer,
    })

    epoch_logs, evaluation_metrics, epoch_losses = train_and_test(fold_config)
    
    duration = time.time() - start_time
    
    result = {
        "fold_idx": fold_idx,
        "epoch_logs": epoch_logs,
        "evaluation_metrics": evaluation_metrics,
        "epoch_losses": epoch_losses,
        "fold_duration": duration,
    }
    print(f"Rank {config['rank']}: Completed Fold {fold_idx + 1}...")
    return result


def distributed_pipeline(config):
    
    """ 
    ===========================
        Data Preprocessing 
    ===========================
    """
    
    log_text = config['log_text']
    rank = config["rank"]
    world_size = config["world_size"]
    
    try:
        hdfs = HadoopFileSystem(host=config["hdfs_host"], port=config["hdfs_port"])
        print(f"Rank {rank}: Connected to HDFS at {config['hdfs_host']}:{config['hdfs_port']}.")
        
    except Exception as e:
        print(f"Rank {rank}: Failed to connect to HDFS: {e}")
        
    
    if rank == 0:
        print("Loading and preprocessing tabular data...")
    try:
        with hdfs.open_input_file(config["tabular_data"]) as file:
            tabular_data = pd.read_excel(file)
            print(f"Rank {rank}: Successfully loaded {len(tabular_data)} rows from {config['tabular_data']}.")

    except Exception as e:
        print(f"Rank {rank}: Failed to load data from HDFS: {e}")
        return None 

    preprocessed_data = tabular_data_preprocessing(tabular_data)

    setup(rank, world_size)

    if rank == 0:
        print("Feature extraction from images...")
        
    feature_extraction_start_time = time.time()

    base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    base_model.eval() 

    feature_extractor = torch.nn.Sequential(*list(base_model.children())[:-1])
   
    all_images = list(preprocessed_data["midas_file_name"]) 
    batches = [all_images[i:i + config["batch_size"]] for i in range(0, len(all_images), config["batch_size"])]

    assigned_batches = [batches[i] for i in range(rank, len(batches), world_size)]

    feature_vectors, image_ids = [], []
    local_results = [] 

    for batch in assigned_batches:
        batch_results = batch_feature_extraction(config, batch, feature_extractor, hdfs)

        local_results.extend(batch_results)

        if len(local_results) >= config["save_interval"] // config["batch_size"]:
            fv_temp, id_temp = zip(*local_results)
            feature_vectors.extend(fv_temp)
            image_ids.extend(id_temp)
            local_results = [] 

    if local_results:
        fv_temp, id_temp = zip(*local_results)
        feature_vectors.extend(fv_temp)
        image_ids.extend(id_temp)

    if rank == 0:
        gathered_fv = [None for _ in range(world_size)]
        gathered_ids = [None for _ in range(world_size)]
    else:
        gathered_fv = None
        gathered_ids = None

    dist.gather_object(feature_vectors, gathered_fv)
    dist.gather_object(image_ids, gathered_ids)
    dist.barrier()  

    if rank == 0:
        feature_vectors = np.vstack(gathered_fv)
        image_ids = np.hstack(gathered_ids)
        print(f"Feature extraction completed with {len(image_ids)} images.")

    feature_extraction_end_time = time.time()
    feature_extraction_duration = round(feature_extraction_end_time - feature_extraction_start_time, 2)

    if rank == 0:
        config["results"]["Feature Extraction Time"] = feature_extraction_duration
        
        print("Combining features with preprocessed data...")
        final_data, cnn_feature_columns = combine_features(feature_vectors, image_ids, preprocessed_data)
        to_broadcast = [final_data, cnn_feature_columns]
    else:
        to_broadcast = [None, None]

    dist.broadcast_object_list(to_broadcast, src=0)

    final_data = to_broadcast[0]
    cnn_feature_columns = to_broadcast[1]

    if rank == 0:
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
    
    if rank == 0:
        print("Performing 10-Fold Cross Validation...")
        
    cross_validation_start_time = time.time()

    kf = KFold(n_splits=10, shuffle=True, random_state=21)
    all_folds = [(fold_idx, train_idx, test_idx) for fold_idx, (train_idx, test_idx) in enumerate(kf.split(final_data))]

    assigned_folds = all_folds[rank::world_size] if len(all_folds) > rank else []
    
    kfold_results = []
    for fold_idx, train_idx, test_idx in assigned_folds:
        train_data = final_data.iloc[train_idx]
        test_data = final_data.iloc[test_idx]
        result = kfold_cross_validation(config, fold_idx, train_data, test_data, cnn_feature_columns)
        kfold_results.append(result)

    if rank == 0:
        gathered_results = [None for _ in range(world_size)]
    else:
        gathered_results = None

    dist.gather_object(kfold_results, gathered_results)
    dist.barrier()

    if rank == 0:
        gathered_results = [item for sublist in gathered_results for item in sublist] 
        
        fold_epoch_losses = [result["epoch_losses"] for result in gathered_results]
        num_epochs = len(fold_epoch_losses[0]) 
        mean_epoch_losses = [
            round(np.mean([fold_losses[epoch_idx] for fold_losses in fold_epoch_losses]), 4)
            for epoch_idx in range(num_epochs)
        ]

        mean_metrics = {
            "accuracy": round(np.mean([result["evaluation_metrics"]["accuracy"] for result in gathered_results]), 4),
            "precision": round(np.mean([result["evaluation_metrics"]["precision"] for result in gathered_results]), 4),
            "recall": round(np.mean([result["evaluation_metrics"]["recall"] for result in gathered_results]), 4),
            "f1": round(np.mean([result["evaluation_metrics"]["f1"] for result in gathered_results]), 4),
            "roc_auc": round(np.mean([result["evaluation_metrics"]["roc_auc"] for result in gathered_results]), 4),
            "fold_duration": round(np.mean([result["fold_duration"] for result in gathered_results]), 2),
        }
        config["results"]["Mean Accuracy"] = mean_metrics["accuracy"]
        config["results"]["Mean Precision"] = mean_metrics["precision"]
        config["results"]["Mean Recall"] = mean_metrics["recall"]
        config["results"]["Mean F1 Score"] = mean_metrics["f1"]
        config["results"]["Mean AUC-ROC"] = mean_metrics["roc_auc"]
        config["results"]["Mean Time per Fold"] = mean_metrics['fold_duration']
        
    cross_validation_end_time = time.time()
    cross_validation_duration = round(cross_validation_end_time - cross_validation_start_time, 2)

    cleanup()
        
    if rank == 0:
        
        config["results"]["Cross Validation Time"] = cross_validation_duration
        print("10-Fold Cross Validation Completed.")

        log_text.append(f"\n\n============================================")
        log_text.append(f"          10 Fold Cross Validation")
        log_text.append(f"============================================")
        log_text.append(f"\nTotal 10-Fold Duration: {cross_validation_duration} seconds")
        
        log_text.append(f"\n--- Duration per Fold ---")
        log_text.append(f"Mean Time Per Fold:   {mean_metrics['fold_duration']} seconds\n")
        gathered_results.sort(key=lambda x: x["fold_idx"])
        for result in gathered_results:
            fold_idx = result["fold_idx"]
            log_text.append(f"Fold {fold_idx+1} Duration:   {result['fold_duration']:.2f} seconds")
        
        log_text.append(f"\n--- Mean Loss Per Epoch Across All Folds ---")
        for epoch_idx, mean_loss in enumerate(mean_epoch_losses, start=1):
            log_text.append(f"Epoch [{epoch_idx}/{config['epochs']}], Mean Loss: {mean_loss:.4f}")

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
    
    log_text = []

    results = {
        "Framework": "pytorch",
        "Dataset": None,
        "Nodes": None,
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

    rank = int(os.getenv("RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", -1))

    if rank == -1 or world_size == -1:
        raise RuntimeError("Environment variables RANK and WORLD_SIZE must be set. Use torchrun to run the program.")

    config = {
        "hdfs_host": "192.168.0.1",
        "hdfs_port": 9000,
        "tabular_data": data_1_path,
        "image_data": images_folder,
        "log_text": log_text,
        "results": results,
        "rank":  rank,
        "world_size": world_size,
        "num_procs": 4,
        "epochs": 10,
        "batch_size": 10,
        "save_interval": 120
    }

    datasets = {
        "data_1.xlsx": 1.05,
        "data_2.xlsx": 2.16,
        "data_3.xlsx": 3.37
    }
    
    if rank == 0:
        results["Nodes"] = int(config["world_size"]/config["num_procs"])
        print(f"Number of nodes: {int(config['world_size']/config['num_procs'])}")
        print(f"World size: {world_size}")

        dataset_number = None
        tabular_file = os.path.basename(config["tabular_data"])  
        if tabular_file in datasets:
            dataset_number = int(tabular_file.split("_")[1].split(".")[0])  
            dataset_size = datasets[tabular_file]
            dataset_info = f"Dataset Used: {tabular_file} ({dataset_size:.2f} GB)"
            log_filename = f"pytorch_data{dataset_number}_node{int(config['world_size']/config['num_procs'])}.txt"
            results_filename = f"pytorch_data{dataset_number}_node{int(config['world_size']/config['num_procs'])}.json"
            results["Dataset"] = dataset_number
        else:
            dataset_info = f"Dataset Used: {tabular_file}"
            log_filename = f"unknown_data_pytorch_log.txt"

        log_text.append(f"======================================")
        log_text.append(f"      Pipeline Execution Summary")
        log_text.append(f"======================================")
        log_text.append(f"\nNumber of Nodes: {int(config['world_size']/config['num_procs'])}")
        log_text.append(dataset_info)  

    pipeline_start_time = time.time()
    
    distributed_pipeline(config)
    
    pipeline_end_time = time.time()
    pipeline_duration = round(pipeline_end_time - pipeline_start_time, 2)
    
    if rank == 0:
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


if __name__ == "__main__":
    main()
