import os
import re
import matplotlib.pyplot as plt

# Create local_plots directory if it doesn't exist
os.makedirs("local_plots", exist_ok=True)

# Parse a log file and extract required metrics
def parse_log(file_path):
    metrics = {}
    with open(file_path, "r") as file:
        content = file.read()
        metrics["total_pipeline_duration"] = float(re.search(r"Total Pipeline Duration: ([\d.]+) seconds", content).group(1))
        metrics["total_feature_extraction_duration"] = float(re.search(r"Total Feature Extraction Duration: ([\d.]+) seconds", content).group(1))
        metrics["total_10_fold_duration"] = float(re.search(r"Total 10-Fold Duration: ([\d.]+) seconds", content).group(1))
    return metrics

# Log file names
ray_logs = ["local_logs/ray_log_1.txt", "local_logs/ray_log_2.txt", "local_logs/ray_log_3.txt"]
pytorch_logs = ["local_logs/pytorch_log_1.txt", "local_logs/pytorch_log_2.txt", "local_logs/pytorch_log_3.txt"]

# Extract data
ray_metrics = [parse_log(log) for log in ray_logs]
pytorch_metrics = [parse_log(log) for log in pytorch_logs]

# Number of workers
workers = [1, 2, 3]

# Plot data
def plot_metric(metric_name, ylabel, filename):
    ray_values = [m[metric_name] for m in ray_metrics]
    pytorch_values = [m[metric_name] for m in pytorch_metrics]

    plt.figure(figsize=(8, 6))
    plt.plot(workers, ray_values, marker='o', label='Ray')
    plt.plot(workers, pytorch_values, marker='o', label='PyTorch')
    plt.title(f"{ylabel} Comparison")
    plt.xlabel("Number of Workers")
    plt.ylabel(ylabel)
    plt.xticks(workers)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"local_plots/{filename}")
    plt.close()

# Create plots
plot_metric("total_pipeline_duration", "Total Pipeline Duration (s)", "total_pipeline_duration.png")
plot_metric("total_feature_extraction_duration", "Total Feature Extraction Time (s)", "total_feature_extraction_duration.png")
plot_metric("total_10_fold_duration", "Total 10-Fold Duration (s)", "total_10_fold_duration.png")

print("Plots saved in the 'local_plots' folder.")
