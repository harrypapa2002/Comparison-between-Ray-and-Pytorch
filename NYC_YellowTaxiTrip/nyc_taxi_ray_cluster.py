import ray
import pandas as pd
import numpy as np
import datetime
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import folium
import argparse
import os

ray.init(ignore_reinit_error=True)

def convert_to_unix(s):
    return time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())

@ray.remote
def preprocess_data(df):
    try:
        df['pickup_unix'] = df['tpep_pickup_datetime'].map(convert_to_unix)
        df['dropoff_unix'] = df['tpep_dropoff_datetime'].map(convert_to_unix)
        df['trip_times'] = (df['dropoff_unix'] - df['pickup_unix']) / 60
        df['Speed'] = 60 * (df['trip_distance'] / df['trip_times'])
        df_cleaned = remove_outliers(df)
        return df_cleaned
    except Exception as e:
        return pd.DataFrame()

def remove_outliers(new_frame):
    new_frame = new_frame[
        ((new_frame.dropoff_longitude >= -74.15) & (new_frame.dropoff_longitude <= -73.7004) &
         (new_frame.dropoff_latitude >= 40.5774) & (new_frame.dropoff_latitude <= 40.9176)) &
        ((new_frame.pickup_longitude >= -74.15) & (new_frame.pickup_latitude >= 40.5774) &
         (new_frame.pickup_longitude <= -73.7004) & (new_frame.pickup_latitude <= 40.9176)) &
        (new_frame.trip_times > 0) & (new_frame.trip_times < 720) &
        (new_frame.trip_distance > 0) & (new_frame.trip_distance < 23) &
        (new_frame.Speed <= 45.31) & (new_frame.Speed >= 0) &
        (new_frame.total_amount < 1000) & (new_frame.total_amount > 0)
    ]
    return new_frame[['pickup_latitude', 'pickup_longitude']]

@ray.remote
def kmeans_cluster(data_chunk, n_clusters):
    if data_chunk.empty:
        return {"cluster_data": None, "metrics": None}
    try:
        data_points = data_chunk.values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data_points)
        silhouette = silhouette_score(data_points, labels) if len(set(labels)) > 1 else -1
        sorted_indices = np.argsort(labels)
        sorted_points = data_points[sorted_indices]
        sorted_labels = labels[sorted_indices]
        unique_labels, counts = np.unique(sorted_labels, return_counts=True)
        cluster_data = []
        start_idx = 0
        for label, count in zip(unique_labels, counts):
            cluster_points = sorted_points[start_idx:start_idx + count]
            center = cluster_points.mean(axis=0)
            cluster_data.append({
                "label": label,
                "size": len(cluster_points),
                "center": center.tolist()
            })
            start_idx += count
        return {"cluster_data": cluster_data, "metrics": {"silhouette": silhouette}}
    except Exception as e:
        return {"cluster_data": None, "metrics": None}

def aggregate_clusters(global_cluster_data, global_metrics, n_clusters):
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
            location=[center[0], center[1]],
            popup=f"Lat: {center[0]:.6f}, Lon: {center[1]:.6f}"
        ).add_to(map_osm)
    map_osm.save(output_path)

def process_files(file_paths, chunk_size, n_clusters, nodes, output_file):
    start_time = time.time()
    results = []
    for file_path in file_paths:
        try:
            with pd.read_csv(file_path, chunksize=chunk_size) as reader:
                raw_chunks = [chunk for chunk in reader]
        except Exception as e:
            continue
        preprocessed_chunks = ray.get([preprocess_data.remote(chunk) for chunk in raw_chunks])
        cluster_results = ray.get([kmeans_cluster.remote(chunk, n_clusters) for chunk in preprocessed_chunks])
        global_cluster_data = [result["cluster_data"] for result in cluster_results if result["cluster_data"]]
        global_metrics = [result["metrics"] for result in cluster_results if result["metrics"]]
        aggregated_clusters, cluster_sizes, avg_silhouette = aggregate_clusters(global_cluster_data, global_metrics, n_clusters)
        results.append({
            "file": os.path.basename(file_path),
            "clusters": aggregated_clusters,
            "silhouette": avg_silhouette
        })
        cluster_centers = [cluster["center"] for cluster in aggregated_clusters.values()]
        plot_cluster_centers(cluster_centers, f"cluster_centers_{os.path.basename(file_path)}.html")
    end_time = time.time()
    execution_time = end_time - start_time
    with open(output_file, 'a') as f:
        f.write(f"Nodes: {nodes}, Files: {len(file_paths)}, Time: {execution_time}\n")
    return results, execution_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True, help='List of CSV files to process')
    parser.add_argument('--chunk_size', type=int, default=10000, help='Chunk size for processing')
    parser.add_argument('--n_clusters', type=int, default=40, help='Number of clusters')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--output', type=str, default='ray_results.txt', help='Output file for results')
    args = parser.parse_args()
    ray.init(num_cpus=args.nodes, ignore_reinit_error=True)
    results, exec_time = process_files(args.files, args.chunk_size, args.n_clusters, args.nodes, args.output)
    with open(args.output, 'a') as f:
        f.write(f"Execution Time: {exec_time}\n")
    print(f"Results stored in {args.output}")

if __name__ == "__main__":
    main()