import os
import pandas as pd
import json

# Define file path and results folder
csv_file = "results.csv"
results_json_folder = "local_results"

# Remove existing CSV if it exists (forcing recreation)
if os.path.exists(csv_file):
    os.remove(csv_file)
    print(f"{csv_file} was deleted and will be recreated.")

# Define column names
columns = [
    "Framework", "Dataset", "Nodes",  # Initial test case columns
    "Total Time", "Feature Extraction Time", "10-Fold Time",  # Timing Metrics
    "Mean Time per Fold",  # Per-fold Timing
    "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean F1 Score", "Mean AUC-ROC"  # Performance Metrics
]

# Create test case combinations
data = []
datasets = [1, 2, 3]  # Dataset numbers
nodes = [1, 2, 3]  # Number of nodes
frameworks = ["Ray", "PyTorch"]  # Frameworks

# Iterate through all possible test cases
for dataset in datasets:
    for node in nodes:
        for framework in frameworks:
            # Initialize result fields as empty (NaN)
            data.append([framework, dataset, node] + [None] * (len(columns) - 3))

# Create DataFrame and save to CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv(csv_file, index=False)
print(f"Created '{csv_file}' with all test cases.")

# If results folder doesn't exist, create it
if os.path.exists(results_json_folder):

    # Load results from JSON files in the results folder
    for json_file in os.listdir(results_json_folder):
        if json_file.endswith(".json"):
            json_path = os.path.join(results_json_folder, json_file)
            
            # Load JSON data
            with open(json_path, "r") as f:
                data = json.load(f)
            
            # Extract relevant identifiers
            framework = data["Framework"].lower()  # Convert to lowercase
            dataset = int(data["Dataset"])  # Convert to int to match CSV format
            nodes = int(data["Nodes"])

            # Find matching row in the DataFrame
            mask = (df["Framework"].str.lower() == framework) & (df["Dataset"] == dataset) & (df["Nodes"] == nodes)

            if mask.sum() == 1:  # Ensure only one match exists
                df.loc[mask, "Total Time"] = float(data["Total Time"])
                df.loc[mask, "Feature Extraction Time"] = float(data["Feature Extraction Time"])
                df.loc[mask, "10-Fold Time"] = float(data["Cross Validation Time"])
                df.loc[mask, "Mean Time per Fold"] = float(data["Mean Time per Fold"])
                df.loc[mask, "Mean Accuracy"] = float(data["Mean Accuracy"])
                df.loc[mask, "Mean Precision"] = float(data["Mean Precision"])
                df.loc[mask, "Mean Recall"] = float(data["Mean Recall"])
                df.loc[mask, "Mean F1 Score"] = float(data["Mean F1 Score"])
                df.loc[mask, "Mean AUC-ROC"] = float(data["Mean AUC-ROC"])

                print(f"Updated results for {framework} - data {dataset} - nodes {nodes}")
            else:
                print(f"No match found for {framework} - data {dataset} - nodes {nodes} in '{csv_file}'")

    # Save updated CSV with results
    df.to_csv(csv_file, index=False)
    print(f"Final results saved to '{csv_file}'")

else:
    print(f"Results directory does not exist.")