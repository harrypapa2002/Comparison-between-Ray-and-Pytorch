import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_DIR = 'results/' 
OUTPUT_DIR = 'output/'  

os.makedirs(OUTPUT_DIR, exist_ok=True)

records = []

data_size_map = {
    'data1': 1.05,
    'data2': 2.16,
    'data3': 3.37
}

for filename in os.listdir(RESULTS_DIR):
    if filename.endswith('.json'):
        filepath = os.path.join(RESULTS_DIR, filename)

        parts = filename.replace('.json', '').split('_')
        framework = parts[0] 
        data_size = parts[1]  
        node_count = parts[2]  

        data_size_gb = data_size_map.get(data_size, None)
        if data_size_gb is None:
            print(f"Unknown data size in filename: {filename}")
            continue

        node_count_int = int(node_count.replace('node', ''))

        with open(filepath, 'r') as f:
            data = json.load(f)
            execution_time = data.get('Total Time', None)
            mean_accuracy = data.get('Mean Accuracy', None)
            mean_precision = data.get('Mean Precision', None)
            mean_recall = data.get('Mean Recall', None)
            mean_f1_score = data.get('Mean F1 Score', None)
            mean_auc_roc = data.get('Mean AUC-ROC', None)

        records.append({
            'Framework': framework.capitalize(),
            'Data Size (GB)': data_size_gb,
            'Number of Nodes': node_count_int,
            'Execution Time (s)': execution_time,
            "Mean Accuracy": mean_accuracy,
            "Mean Precision": mean_precision,
            "Mean Recall": mean_recall,
            "Mean F1 Score": mean_f1_score,
            "Mean AUC-ROC": mean_auc_roc
        })

df = pd.DataFrame(records)

df.to_csv(os.path.join(OUTPUT_DIR, "processed_results.csv"), index=False)

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

# --- Single Table for Mean Metrics ---
# Create a single table for mean metrics (Accuracy, Precision, Recall, F1 Score, AUC-ROC)
pivot_metrics = df.pivot_table(
    index=['Data Size (GB)', 'Number of Nodes'],
    columns='Framework',
    values=["Mean Accuracy", "Mean Precision", "Mean Recall", "Mean F1 Score", "Mean AUC-ROC"]
).reset_index()


pivot_metrics.to_csv(os.path.join(OUTPUT_DIR, "mean_metrics_combined.csv"), index=False)

print("Mean Metrics for All Combinations:")
print(pivot_metrics)

