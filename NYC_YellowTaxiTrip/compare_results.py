import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the directory containing the JSON files and output directory
RESULTS_DIR = 'results/'  # Replace with your actual results directory
OUTPUT_DIR = 'output/'  # Directory to store outputs

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize a list to store all records
records = []

# Map data_size to actual size in GB
data_size_map = {
    'data1': 1.99,
    'data2': 3.7,
    'data3': 5.48
}

# Iterate over all files in the directory
for filename in os.listdir(RESULTS_DIR):
    if filename.endswith('.json'):
        filepath = os.path.join(RESULTS_DIR, filename)

        # Determine the framework, data size, and node count from filename
        parts = filename.replace('.json', '').split('_')
        framework = parts[0]  # 'pytorch' or 'ray'
        data_size = parts[1]  # e.g., 'data1'
        node_count = parts[2]  # e.g., 'node3'

        data_size_gb = data_size_map.get(data_size, None)
        if data_size_gb is None:
            print(f"Unknown data size in filename: {filename}")
            continue

        node_count_int = int(node_count.replace('node', ''))

        # Open and parse the JSON file
        with open(filepath, 'r') as f:
            data = json.load(f)
            execution_time = data.get('execution_time', None)
            clustering_results = data.get('clustering_results', [])
            if clustering_results:
                average_silhouette = clustering_results[0].get('metrics', {}).get('average_silhouette', None)
            else:
                average_silhouette = None

        # Append the record
        records.append({
            'Framework': framework.capitalize(),
            'Data Size (GB)': data_size_gb,
            'Number of Nodes': node_count_int,
            'Execution Time (s)': execution_time,
            'Average Silhouette': average_silhouette
        })

# Create a DataFrame
df = pd.DataFrame(records)

# Save the DataFrame as a CSV file
df.to_csv(os.path.join(OUTPUT_DIR, "processed_results.csv"), index=False)

# Display the DataFrame
print(df)

# --- Visualization: Execution Time vs. Data Size ---
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x='Data Size (GB)',
    y='Execution Time (s)',
    hue='Framework',
    marker='o'
)
plt.title('Execution Time vs. Data Size')
plt.xlabel('Data Size (GB)')
plt.ylabel('Execution Time (s)')
plt.legend(title='Framework')
plt.savefig(os.path.join(OUTPUT_DIR, "execution_time_vs_data_size.png"))
plt.close()

# --- Visualization: Execution Time vs. Number of Nodes ---
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x='Number of Nodes',
    y='Execution Time (s)',
    hue='Framework',
    marker='o'
)
plt.title('Execution Time vs. Number of Nodes')
plt.xlabel('Number of Nodes')
plt.ylabel('Execution Time (s)')
plt.legend(title='Framework')
plt.savefig(os.path.join(OUTPUT_DIR, "execution_time_vs_nodes.png"))
plt.close()

# --- Single Table for Silhouette Scores ---
# Create a single table for silhouette scores
pivot_silhouette = df.pivot_table(
    index=['Data Size (GB)', 'Number of Nodes'],
    columns='Framework',
    values='Average Silhouette'
).reset_index()

# Save the table to CSV
pivot_silhouette.to_csv(os.path.join(OUTPUT_DIR, "silhouette_scores_combined.csv"), index=False)

# Display the table
print("Silhouette Scores for All Combinations:")
print(pivot_silhouette)

# Save the table as a styled HTML file for better visualization
styled_table = pivot_silhouette.style.set_table_styles(
    [{'selector': 'th', 'props': [('text-align', 'center')]},
     {'selector': 'td', 'props': [('text-align', 'center')]}]
).set_caption("Silhouette Scores for All Combinations")

styled_table.to_html(os.path.join(OUTPUT_DIR, "silhouette_scores_combined.html"))