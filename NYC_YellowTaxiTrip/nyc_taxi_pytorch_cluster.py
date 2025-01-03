import logging
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import folium
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

class KMeansClusterDataset(Dataset):
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate_fn(batch):
    try:
        numeric_batch = np.array(batch, dtype=np.float64)
        return torch.tensor(numeric_batch, dtype=torch.float64)
    except Exception as e:
        logging.error(f"Collate function error: {e}")
        raise ValueError("Batch data contains non-numeric values.")

def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(42)

def cleanup():
    dist.destroy_process_group()

def convert_to_unix(s):
    return time.mktime(pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S").timetuple())

def preprocess_data(data):
    try:
        data["pickup_unix"] = data["tpep_pickup_datetime"].map(convert_to_unix)
        data["dropoff_unix"] = data["tpep_dropoff_datetime"].map(convert_to_unix)
        data["trip_times"] = (data["dropoff_unix"] - data["pickup_unix"]) / 60
        data["Speed"] = 60 * (data["trip_distance"] / data["trip_times"])
        df_cleaned = remove_outliers(data)
        return df_cleaned
    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        return pd.DataFrame()

def remove_outliers(df):
    try:
        df_cleaned = df[
            ((df["dropoff_longitude"] >= -74.15) & (df["dropoff_longitude"] <= -73.7004) &
             (df["dropoff_latitude"] >= 40.5774) & (df["dropoff_latitude"] <= 40.9176)) &
            ((df["pickup_longitude"] >= -74.15) & (df["pickup_latitude"] >= 40.5774) &
             (df["pickup_longitude"] <= -73.7004) & (df["pickup_latitude"] <= 40.9176)) &
            (df["trip_times"] > 0) & (df["trip_times"] < 720) &
            (df["trip_distance"] > 0) & (df["trip_distance"] < 23) &
            (df["Speed"] <= 45.31) & (df["Speed"] >= 0) &
            (df["total_amount"] > 0) & (df["total_amount"] < 1000)
        ]
        return df_cleaned[["pickup_latitude", "pickup_longitude"]]
    except:
        return pd.DataFrame()

def kmeans_cluster(data_chunk, n_clusters):
    if data_chunk is None or len(data_chunk) == 0:
        return {"cluster_data": None, "metrics": None}
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data_chunk)
        silhouette = silhouette_score(data_chunk, labels) if len(set(labels)) > 1 else -1
        cluster_data = []
        for cluster_label in np.unique(labels):
            cluster_points = data_chunk[labels == cluster_label]
            center = cluster_points.mean(axis=0)
            cluster_data.append({
                "label": cluster_label,
                "size": len(cluster_points),
                "center": center.tolist()
            })
        return {"cluster_data": cluster_data, "metrics": {"silhouette": silhouette}}
    except:
        return {"cluster_data": None, "metrics": None}

def aggregate_results(global_cluster_data, global_metrics, n_clusters):
    all_centers = []
    cluster_sizes = [0] * n_clusters
    silhouettes = []
    for batch_data, metrics in zip(global_cluster_data, global_metrics):
        if batch_data:
            for cluster in batch_data:
                all_centers.append(cluster["center"])
                cluster_sizes[cluster["label"]] += cluster["size"]
        if metrics:
            silhouettes.append(metrics["silhouette"])
    if not all_centers:
        return {}, cluster_sizes, -1
    all_centers = np.array(all_centers)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    global_labels = kmeans.fit_predict(all_centers)
    global_clusters = {label: [] for label in range(n_clusters)}
    for idx, center in enumerate(all_centers):
        global_clusters[global_labels[idx]].append(center)
    final_clusters = {}
    for label, points in global_clusters.items():
        points = np.array(points)
        final_clusters[label] = {
            "size": len(points),
            "center": points.mean(axis=0).tolist()
        }
    avg_silhouette = np.mean(silhouettes) if silhouettes else -1
    return final_clusters, cluster_sizes, avg_silhouette

def plot_cluster_centers(cluster_centers, output_path):
    map_osm = folium.Map(location=[40.734695, -73.990372], zoom_start=12)
    for center in cluster_centers:
        folium.Marker(
            location=center,
            popup=f"Lat: {center[0]:.6f}, Lon: {center[1]:.6f}"
        ).add_to(map_osm)
    map_osm.save(output_path)

def process_file(rank, world_size, file_path, chunk_size, n_clusters, output_file):
    setup(rank, world_size)
    start_time = time.time()
    try:
        data_iter = pd.read_csv(file_path, chunksize=chunk_size)
        preprocessed_chunks = [
            preprocess_data(chunk) for idx, chunk in enumerate(data_iter) if idx % world_size == rank
        ]
        preprocessed_data = pd.concat(preprocessed_chunks).reset_index(drop=True)
    except:
        cleanup()
        return
    dataset = KMeansClusterDataset(preprocessed_data.values)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=chunk_size, sampler=sampler, collate_fn=custom_collate_fn)
    local_results = []
    for batch in dataloader:
        result = kmeans_cluster(batch.numpy(), n_clusters)
        local_results.append(result)
    all_results = None
    if rank == 0:
        all_results = [None] * world_size
    dist.gather_object(local_results, all_results, dst=0)
    if rank == 0:
        global_cluster_data = []
        global_metrics = []
        for rank_results in all_results:
            for result in rank_results:
                if result["cluster_data"]:
                    global_cluster_data.append(result["cluster_data"])
                if result["metrics"]:
                    global_metrics.append(result["metrics"])
        aggregated_clusters, cluster_sizes, avg_silhouette = aggregate_results(global_cluster_data, global_metrics, n_clusters)
        cluster_centers = [cluster["center"] for cluster in aggregated_clusters.values()]
        plot_cluster_centers(cluster_centers, f"cluster_centers_{os.path.basename(file_path)}.html")
        end_time = time.time()
        exec_time = end_time - start_time
        with open(output_file, 'a') as f:
            f.write(f"Nodes: {world_size}, File: {os.path.basename(file_path)}, Time: {exec_time}\n")
        print(f"Results stored in {output_file}")
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True, help='List of CSV files to process')
    parser.add_argument('--chunk_size', type=int, default=10000, help='Chunk size for processing')
    parser.add_argument('--n_clusters', type=int, default=40, help='Number of clusters')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--output', type=str, default='torch_results.txt', help='Output file for results')
    args = parser.parse_args()
    os.environ['WORLD_SIZE'] = str(args.nodes)
    rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    for file_path in args.files:
        process_file(rank, world_size, file_path, args.chunk_size, args.n_clusters, args.output)

if __name__ == "__main__":
    main()