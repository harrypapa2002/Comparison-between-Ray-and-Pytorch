import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


RESULTS_DIR = "C://Users//anton//Desktop//Pytorch//Pytorch//PageRank//results//twitter7"  #/results
OUTPUT_DIR = "C://Users//anton//Desktop//Pytorch//Pytorch//PageRank//output" #/output
os.makedirs(OUTPUT_DIR, exist_ok=True)

def infer_dataset_size(filename):
    fname_lower = filename.lower()
    if "1gb" in fname_lower:
        return 1.0
    elif "7_2" in fname_lower:
        return 2.5
    elif "5gb" in fname_lower:
        return 5.0
    else:
        return None

def parse_pytorch_results(lines):
    framework = "Pytorch"
    dataset_size = None
    num_nodes = None
    execution_time = None
    virtual_memory = None 

    for line in lines:
        line = line.strip()

        if line.startswith("File "):
            match = re.search(r"File\s+(.*?)\s+- number of worker machines\s+(\d+)", line)
            if match:
                file_part = match.group(1)       
                world_size_str = match.group(2)  

                dataset_size = infer_dataset_size(file_part)
                if world_size_str.isdigit():
                    world_size = int(world_size_str)
                    num_nodes = world_size // 4   

        if "Execution Time (seconds):" in line:
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    execution_time = float(parts[1].strip())
                except ValueError:
                    pass

        if "Virtual Memory Usage (MB):" in line:
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    virtual_memory = float(parts[1].strip())
                except ValueError:
                    pass

    return {
        "Framework": framework,
        "Dataset Size (GB)": dataset_size,
        "Number of Nodes": num_nodes,
        "Execution Time (s)": execution_time,
        "Virtual Memory Usage (MB)": virtual_memory 
    }


def parse_ray_results(lines):
    framework = "Ray"
    dataset_size = None
    num_nodes = None
    execution_time = None
    virtual_memory = None

    for line in lines:
        line = line.strip()

        if line.startswith("File "):
            match = re.search(r"File\s+(.*?)\s+- using Ray", line)
            if match:
                file_part = match.group(1)  
                dataset_size = infer_dataset_size(file_part)

        if "Alive Ray Nodes:" in line:
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    num_nodes = int(parts[1].strip())
                except ValueError:
                    pass

        if "Execution Time (seconds):" in line:
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    execution_time = float(parts[1].strip())
                except ValueError:
                    pass

        if "Virtual Memory Usage (MB):" in line:
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    virtual_memory = float(parts[1].strip())
                except ValueError:
                    pass

    return {
        "Framework": framework,
        "Dataset Size (GB)": dataset_size,
        "Number of Nodes": num_nodes,
        "Execution Time (s)": execution_time,
        "Virtual Memory Usage (MB)": virtual_memory
    }

all_records = []

for fname in os.listdir(RESULTS_DIR):
    if not fname.endswith(".txt"):
        continue

    filepath = os.path.join(RESULTS_DIR, fname)
    with open(filepath, "r") as f:
        lines = f.readlines()

    if "_ray_" in fname.lower():
        rec = parse_ray_results(lines)
    else:
        rec = parse_pytorch_results(lines)

    if rec["Dataset Size (GB)"] is None or rec["Number of Nodes"] is None or rec["Execution Time (s)"] is None:
        print(f"Warning: Could not fully parse {fname}. Some fields missing.")
        continue

    all_records.append(rec)

df = pd.DataFrame(all_records)

unique_sizes = sorted(df["Dataset Size (GB)"].unique())

sns.set_theme(style="white")

for size in unique_sizes:
    subset = df[df["Dataset Size (GB)"] == size].copy()

    if subset.empty:
        continue

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=subset,
        x="Number of Nodes",
        y="Execution Time (s)",
        hue="Framework",
        marker="o",
        err_style="band",  
        errorbar="sd"            
    )

    plt.xticks([1, 2, 3])
    plt.title(f"Execution Time vs. Number of Nodes (Dataset = {size} GB)")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (s)")
    plt.legend(title="Framework")
    outname = f"execution_time_vs_nodes_{size}GB.png"
    plt.savefig(os.path.join(OUTPUT_DIR, outname))
    plt.close()

print("Done! Plots saved in:", OUTPUT_DIR)

pivot_virtual_memory = df.pivot_table(
    index=["Dataset Size (GB)", "Number of Nodes"],
    columns="Framework",
    values="Virtual Memory Usage (MB)"
).reset_index()

pivot_virtual_memory.to_csv(os.path.join(OUTPUT_DIR, "virtual_memory_usage.csv"), index=False)


print("Done! Plots and Virtual Memory Table saved in:", OUTPUT_DIR)