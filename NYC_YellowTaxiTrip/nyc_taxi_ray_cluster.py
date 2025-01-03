import json
import ray
import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import folium
import argparse
import os
from pyarrow.fs import HadoopFileSystem
import pyarrow.csv as pv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


def ensure_directories():
    """
    Ensures the results and maps directories exist.
    """
    os.makedirs("results", exist_ok=True)
    os.makedirs("maps", exist_ok=True)

def convert_to_unix(s):
    """
    Converts a datetime string to a Unix timestamp.
    """
    return time.mktime(pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S").timetuple())


@ray.remote
def preprocess_data(data):
    """
    Preprocesses the DataFrame by converting datetime fields to Unix timestamps,
    calculating trip times and speed, and removing outliers.
    """
    try:
        # Select only necessary columns and create an explicit copy
        data["pickup_unix"] = data["tpep_pickup_datetime"].map(convert_to_unix)
        data["dropoff_unix"] = data["tpep_dropoff_datetime"].map(convert_to_unix)
        data["trip_times"] = (data["dropoff_unix"] - data["pickup_unix"]) / 60
        data["Speed"] = 60 * (data["trip_distance"] / data["trip_times"])
        df_cleaned = remove_outliers(data)

        logging.info(f"Preprocessed DataFrame with shape {df_cleaned.shape}")
        return df_cleaned

    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        return pd.DataFrame()


def remove_outliers(df):
    """
    Removes outliers based on predefined criteria and selects relevant columns.
    """
    try:
        # Apply outlier filtering and create an explicit copy
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
    except Exception as e:
        logging.error(f"Outlier removal error: {e}")
        return pd.DataFrame()

@ray.remote
def kmeans_cluster(data_chunk, n_clusters):
    """
    Performs KMeans clustering on the provided data chunk.
    Returns cluster centers and silhouette score.
    """
    if data_chunk.empty:
        return {"cluster_data": None, "metrics": None}
    try:
        data_points = data_chunk.values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data_points)
        silhouette = silhouette_score(data_points, labels) if len(set(labels)) > 1 else -1

        # Aggregate cluster data
        cluster_data = []
        for label in np.unique(labels):
            cluster_points = data_points[labels == label]
            center = cluster_points.mean(axis=0)
            cluster_data.append({
                "label": int(label),
                "size": int(len(cluster_points)),
                "center": center.tolist()
            })

        return {"cluster_data": cluster_data, "metrics": {"silhouette": float(silhouette)}}
    except Exception as e:
        logging.error(f"KMeans clustering error: {e}")
        return {"cluster_data": None, "metrics": None}


def aggregate_clusters(global_cluster_data, global_metrics, n_clusters):
    """
    Aggregates clustering results from all data chunks to form global clusters.
    """
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
    for idx, label in enumerate(global_labels):
        global_clusters[label].append(all_centers[idx])

    final_clusters = {}
    for label, points in global_clusters.items():
        points = np.array(points)
        final_clusters[label] = {
            "size": int(len(points)),
            "center": points.mean(axis=0).tolist()
        }

    avg_silhouette = float(np.mean(silhouettes)) if silhouettes else -1
    return final_clusters, cluster_sizes, avg_silhouette


def plot_cluster_centers(cluster_centers, output_filename):
    """
    Plots cluster centers on a Folium map and saves it as an HTML file.
    """
    try:
        output_path = os.path.join("maps", output_filename)
        map_osm = folium.Map(location=[40.734695, -73.990372], zoom_start=12)
        for center in cluster_centers:
            folium.Marker(
                location=[center[0], center[1]],
                popup=f"Lat: {center[0]:.6f}, Lon: {center[1]:.6f}"
            ).add_to(map_osm)
        map_osm.save(output_path)
        logging.info(f"Map saved as '{output_path}'.")
    except Exception as e:
        logging.error(f"Error plotting cluster centers: {e}")


def process_files(file_paths, hdfs_host, hdfs_port, block_size, n_clusters, output_filename):
    """
    Processes multiple CSV files from HDFS: reads, preprocesses, clusters, and aggregates results.
    Concatenates all files before clustering.
    """

    ensure_directories()
    output_file = os.path.join("results", output_filename)

    start_time = time.time()
    final_results = {
        "files_processed": len(file_paths),
        "execution_time": 0,
        "clustering_results": []
    }

    try:
        # Initialize HDFS filesystem connection
        hdfs = HadoopFileSystem(host=hdfs_host, port=hdfs_port)
        logging.info(f"Connected to HDFS at {hdfs_host}:{hdfs_port}.")
    except Exception as e:
        logging.error(f"Failed to connect to HDFS: {e}")
        return final_results, 0

    # List to hold all preprocess tasks
    preprocess_tasks = []

    for file_path in file_paths:
        try:
            if not file_path.startswith(f"hdfs://{hdfs_host}:{hdfs_port}"):
                file_path = f"hdfs://{hdfs_host}:{hdfs_port}/{file_path.lstrip('/')}"

            logging.info(f"Processing file: {file_path}")
            with hdfs.open_input_file(file_path) as file:
                read_options = pv.ReadOptions(block_size=block_size)
                parse_options = pv.ParseOptions(delimiter=',')
                convert_options = pv.ConvertOptions(strings_can_be_null=True)

                # Create a CSV reader
                csv_reader = pv.open_csv(
                    file,
                    read_options=read_options,
                    parse_options=parse_options,
                    convert_options=convert_options
                )

                logging.info(f"Reading file {file_path} in chunks of {block_size} bytes.")

                # Process each batch
                for batch in csv_reader:
                    df_chunk = batch.to_pandas()
                    preprocess_tasks.append(preprocess_data.remote(df_chunk))

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            continue

    if not preprocess_tasks:
        logging.error("No preprocessing tasks were created. Exiting.")
        return final_results, 0

    # Retrieve preprocessed data
    preprocessed_chunks = ray.get(preprocess_tasks)
    logging.info(f"Preprocessed {len(preprocessed_chunks)} chunks.")

    # Filter out empty DataFrames
    preprocessed_chunks = [chunk for chunk in preprocessed_chunks if not chunk.empty]

    if not preprocessed_chunks:
        logging.error("No data to process after preprocessing.")
        return final_results, 0

    # Concatenate all preprocessed data
    combined_df = pd.concat(preprocessed_chunks, ignore_index=True)
    logging.info(f"Combined DataFrame has {len(combined_df)} rows.")

    # Perform KMeans clustering on the combined data
    cluster_tasks = [kmeans_cluster.remote(chunk, n_clusters) for chunk in preprocessed_chunks]
    cluster_results = ray.get(cluster_tasks)

    # Aggregate cluster data and metrics
    global_cluster_data = [result["cluster_data"] for result in cluster_results if result["cluster_data"]]
    global_metrics = [result["metrics"] for result in cluster_results if result["metrics"]]
    aggregated_clusters, cluster_sizes, avg_silhouette = aggregate_clusters(global_cluster_data, global_metrics,
                                                                            n_clusters)

    # Store results
    final_results["clustering_results"].append({
        "files": [os.path.basename(fp) for fp in file_paths],
        "clusters": aggregated_clusters,
        "metrics": {"average_silhouette": avg_silhouette}
    })

    # Plot cluster centers
    cluster_centers = [cluster["center"] for cluster in aggregated_clusters.values()]
    map_filename = f"cluster_centers_{os.path.basename(output_file).replace('.json', '.html')}"
    plot_cluster_centers(cluster_centers, map_filename)

    end_time = time.time()
    final_results["execution_time"] = end_time - start_time

    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=4, default=lambda o: int(o) if isinstance(o, np.integer)
                                               else float(o) if isinstance(o, np.floating)
                                               else o.tolist() if isinstance(o, np.ndarray)
                                               else o)
    logging.info(f"Results saved to {output_file}")


def main():
    """
    Main function to parse arguments and initiate processing.
    """
    parser = argparse.ArgumentParser(description='Ray-based KMeans clustering on HDFS CSV files.')
    parser.add_argument('--files', nargs='+', required=True, help='List of HDFS CSV files to process (e.g., hdfs:///data/nyc_taxi/yellow_tripdata_2015-01.csv)')
    parser.add_argument('--hdfs_host', type=str, default="namenode", help='HDFS Namenode host')
    parser.add_argument('--hdfs_port', type=int, default=9000, help='HDFS Namenode port')
    parser.add_argument('--block_size', type=int, default=20* 1024 * 1024, help='Block size for reading CSV files')
    parser.add_argument('--n_clusters', type=int, default=40, help='Number of clusters for KMeans')
    parser.add_argument('--output', type=str, default='ray_results.json', help='Output file to store execution results')
    args = parser.parse_args()

    logging.info("Ray initialized")

    logging.info("Starting Ray-based KMeans clustering.")

    process_files(
        file_paths=args.files,
        hdfs_host=args.hdfs_host,
        hdfs_port=args.hdfs_port,
        block_size=args.block_size,
        n_clusters=args.n_clusters,
        output_filename=args.output
    )

    logging.info("Clustering process completed.")
    print(f"Results stored in {args.output}")

if __name__ == "__main__":
    main()
