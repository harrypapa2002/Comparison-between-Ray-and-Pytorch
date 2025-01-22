import os
import pandas as pd

# Paths
base_dir = os.path.dirname("release_midas.xlsx")
images_folder = "C:/Users/nikol/Desktop/university/9th_semester/physiological_systems_simulation/project/dataset/midasmultimodalimagedatasetforaibasedskincancer"

# Dataset files
dataset_files = [
    os.path.join(base_dir, "data_1.xlsx"),
    os.path.join(base_dir, "data_2.xlsx"),
    os.path.join(base_dir, "data_3.xlsx"),
]

# Function to calculate total size of images for a given dataset
def calculate_image_sizes(dataset_path, images_folder):
    try:
        df = pd.read_excel(dataset_path)
    except Exception as e:
        print(f"Error reading {dataset_path}: {e}")
        return None, None

    if "midas_file_name" not in df.columns:
        print(f"The column 'midas_file_name' is not present in {dataset_path}.")
        return None, None

    filenames = df["midas_file_name"].dropna().unique()

    total_size_bytes = 0
    missing_files = []

    for filename in filenames:
        image_path = os.path.join(images_folder, filename)
        if os.path.exists(image_path):
            total_size_bytes += os.path.getsize(image_path)
        else:
            missing_files.append(filename)

    return total_size_bytes, missing_files


# Process each dataset and calculate the image sizes
for dataset_file in dataset_files:
    total_size_bytes, missing_files = calculate_image_sizes(dataset_file, images_folder)

    if total_size_bytes is not None:
        # Convert bytes to MB and GB
        total_size_mb = total_size_bytes / (1024 * 1024)
        total_size_gb = total_size_bytes / (1024 * 1024 * 1024)

        print(f"Dataset: {os.path.basename(dataset_file)}")
        print(f"Total size of images: {total_size_mb:.2f} MB ({total_size_gb:.2f} GB)")

        if missing_files:
            print(f"{len(missing_files)} files were listed in the Excel file but not found in the directory.")
            print("Missing files:", missing_files[:10], "...")  # Show only the first 10 missing files
        print("-" * 50)
