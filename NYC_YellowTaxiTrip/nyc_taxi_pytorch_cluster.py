import logging
import os
import time
import argparse
import json
import numpy as np
import pandas as pd
import folium
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from pyarrow import fs
import pyarrow.csv as pv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

class KMeansClusterDataset(Dataset):
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def ensure_directories():
    os.makedirs("results", exist_ok=True)
    os.makedirs("maps", exist_ok=True)

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
    logging.info(f"Process {rank} initialized.")

def cleanup():
    dist.destroy_process_group()
    logging.info("Distributed process group destroyed.")

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
        df = df[
            ((df.dropoff_longitude >= -74.15) & (df.dropoff_longitude <= -73.7004) &
             (df.dropoff_latitude >= 40.5774) & (df.dropoff_latitude <= 40.9176)) &
            ((df.pickup_longitude >= -74.15) & (df.pickup_longitude <= -73.7004) &
             (df.pickup_latitude >= 40.5774) & (df.pickup_latitude <= 40.9176)) &
            (df.trip_times > 0) & (df.trip_times < 720) &
            (df.trip_distance > 0) & (df.trip_distance < 23) &
            (df.Speed <= 45.31) & (df.Speed >= 0) &
            (df.total_amount < 1000) & (df.total_amount > 0)
        ]
        return df[["pickup_latitude", "pickup_longitude"]]
    except Exception as e:
        logging.error(f"Outlier removal error: {e}")
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
    except Exception as e:
        logging.error(f"KMeans clustering error: {e}")
        return {"cluster_data": None, "metrics": None}

def aggregate_clusters(global_cluster_data, global_metrics, n_clusters):
    all_centers = []
    cluster_sizes = [0] * n_clusters
    silhouettes = []
    for rank_data in global_cluster_data:
        for batch_data in rank_data:
            if batch_data:
                for cluster in batch_data:
                    logging.info(f"Cluster: {cluster}")
                    all_centers.append(cluster["center"])
                    cluster_sizes[cluster["label"]] += cluster["size"]

    for rank_metrics in global_metrics:
        for metric in rank_metrics:
            if metric:
                silhouettes.append(metric["silhouette"])

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

def plot_cluster_centers(cluster_centers, output_filename):
    map_path = os.path.join("maps", output_filename)
    map_osm = folium.Map(location=[40.734695, -73.990372], zoom_start=12)
    for center in cluster_centers:
        folium.Marker(
            location=center,
            popup=f"Lat: {center[0]:.6f}, Lon: {center[1]:.6f}"
        ).add_to(map_osm)
    map_osm.save(map_path)
    logging.info(f"Cluster centers plotted to {map_path}")

def get_hdfs_file_system(hdfs_host, hdfs_port):
    try:
        hdfs = fs.HadoopFileSystem(host=hdfs_host, port=hdfs_port)
        logging.info(f"Connected to HDFS at {hdfs_host}:{hdfs_port}.")
        return hdfs
    except Exception as e:
        logging.error(f"Failed to connect to HDFS: {e}")
        return None

def read_and_preprocess_files(hdfs, hdfs_host, hdfs_port, file_paths, batch_size, rank, world_size):
    preprocessed_chunks = []
    for file_path in file_paths:
        try:
            if not file_path.startswith(f"hdfs://{hdfs_host}:{hdfs_port}"):
                file_path = f"hdfs://{hdfs_host}:{hdfs_port}/{file_path.lstrip('/')}"
            logging.info(f"Process {rank}: Processing file: {file_path}")
            with hdfs.open_input_file(file_path) as file:
                read_options = pv.ReadOptions(block_size=batch_size)
                parse_options = pv.ParseOptions(delimiter=',')
                convert_options = pv.ConvertOptions(strings_can_be_null=True)

                csv_reader = pv.open_csv(
                    file,
                    read_options=read_options,
                    parse_options=parse_options,
                    convert_options=convert_options
                )

                for idx, batch in enumerate(csv_reader):
                    if idx % world_size == rank:
                        df_batch = preprocess_data(batch.to_pandas())
                        if not df_batch.empty:
                            preprocessed_chunks.append(df_batch)
        except Exception as e:
            logging.error(f"Process {rank}: Error processing file {file_path}: {e}")
            continue
    if preprocessed_chunks:
        combined_df = pd.concat(preprocessed_chunks, ignore_index=True)
        logging.info(f"Process {rank}: Combined DataFrame has {len(combined_df)} rows.")
        return combined_df
    else:
        logging.error(f"Process {rank}: No data to process after preprocessing.")
        return pd.DataFrame()

def perform_clustering(dataloader, n_clusters):
    local_results = []
    for batch in dataloader:
        data_batch = batch.numpy()
        result = kmeans_cluster(data_batch, n_clusters)
        local_results.append(result)
    return local_results

def handle_results(file_num, all_cluster_data, all_metrics, n_clusters, output_file, start_time):
    aggregated_clusters, cluster_sizes, avg_silhouette = aggregate_clusters(all_cluster_data, all_metrics, n_clusters)
    final_results = {
        "files_processed": file_num,
        "execution_time": time.time() - start_time,
        "clustering_results": [{
            "clusters": aggregated_clusters,
            "metrics": {"average_silhouette": avg_silhouette}
        }]
    }
    output_filepath = os.path.join("results", output_file)
    with open(output_filepath, 'w') as f:
        json.dump(final_results, f, indent=4, default=lambda o: int(o) if isinstance(o, np.integer)
                                                 else float(o) if isinstance(o, np.floating)
                                                 else o.tolist() if isinstance(o, np.ndarray)
                                                 else o)
    logging.info(f"Results saved to {output_filepath}")
    # Plot cluster centers
    cluster_centers = [cluster["center"] for cluster in aggregated_clusters.values()]
    plot_cluster_centers(cluster_centers, f"cluster_centers_{output_file}.html")

def process_files(rank, world_size, file_paths, hdfs_host, hdfs_port, read_block_size, data_loader_batch_size, n_clusters, output_file):
    setup(rank, world_size)
    ensure_directories()
    start_time = time.time()

    hdfs = get_hdfs_file_system(hdfs_host, hdfs_port)
    if not hdfs:
        cleanup()
        return

    combined_df = read_and_preprocess_files(hdfs, hdfs_host, hdfs_port, file_paths, read_block_size, rank, world_size)
    if combined_df.empty:
        logging.error(f"Process {rank}: No data after preprocessing. Exiting.")
        cleanup()
        return

    dataset = KMeansClusterDataset(combined_df.values)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=data_loader_batch_size, sampler=sampler, collate_fn=custom_collate_fn)

    local_results = perform_clustering(dataloader, n_clusters)

    if rank == 0:
        gathered_cluster_data = [None for _ in range(world_size)]
        gathered_metrics = [None for _ in range(world_size)]
    else:
        gathered_cluster_data = None
        gathered_metrics = None
    dist.gather_object([result["cluster_data"] for result in local_results], gathered_cluster_data)
    dist.gather_object([result["metrics"] for result in local_results], gathered_metrics)

    if rank == 0:
        logging.info("Rank 0: Aggregating and saving results.")
        handle_results(len(file_paths), gathered_cluster_data, gathered_metrics, n_clusters, output_file, start_time)

    cleanup()

def main():
    parser = argparse.ArgumentParser(description='PyTorch-based Distributed KMeans clustering on HDFS CSV files.')
    parser.add_argument('--files', nargs='+', required=True, help='List of HDFS CSV files to process (e.g., /data/nyc_taxi/yellow_tripdata_2015-01.csv)')
    parser.add_argument('--hdfs_host', type=str, default="namenode", help='HDFS Namenode host')
    parser.add_argument('--hdfs_port', type=int, default=9000, help='HDFS Namenode port')
    parser.add_argument('--read_block_size', type=int, default=1048576, help='Batch size in bytes for HDFS reading')
    parser.add_argument('--data_loader_batch_size', type=int, default=1024, help='Batch size (number of samples) for DataLoader')
    parser.add_argument('--n_clusters', type=int, default=40, help='Number of clusters for KMeans')
    parser.add_argument('--output', type=str, default='torch_results.json', help='Output file to store execution results')
    args = parser.parse_args()

    rank = int(os.getenv('RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', -1))

    if rank == -1 or world_size == -1:
        logging.error("RANK and WORLD_SIZE environment variables are not set. Ensure you're running the script with torchrun.")
        return

    logging.info(f"Starting PyTorch-based Distributed KMeans clustering with rank {rank} out of {world_size}.")

    process_files(
        rank=rank,
        world_size=world_size,
        file_paths=args.files,
        hdfs_host=args.hdfs_host,
        hdfs_port=args.hdfs_port,
        read_block_size=args.read_block_size,
        data_loader_batch_size=args.data_loader_batch_size,
        n_clusters=args.n_clusters,
        output_file=args.output
    )

    if rank == 0:
        logging.info("Clustering process completed.")

if __name__ == "__main__":
    main()