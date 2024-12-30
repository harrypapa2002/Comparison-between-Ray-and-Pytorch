import os
import json
import time
import gc
import torch
from torch_ppr import page_rank
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pyarrow.fs as fs
import pyarrow.csv as pv
import psutil

# Custom Dataset to handle graph edges in chunks
class GraphDataset(Dataset):
    def __init__(self, chunk):
        self.edges = self.format_edges(chunk)

    def format_edges(self, chunk):
        nodes1 = chunk.column('node1').to_pylist()
        nodes2 = chunk.column('node2').to_pylist()
        return torch.tensor([nodes1, nodes2], dtype=torch.long)

    def __len__(self):
        return self.edges.size(1)

    def __getitem__(self, idx):
        return self.edges[:, idx]

# Distributed setup for VMs
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.0.1'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# Clean up process group and intermediate files
def cleanup():
    dist.destroy_process_group()
    temp_dir = '~/PyTorch/pagerank/intermediate_results'
    temp_dir = os.path.expanduser(temp_dir)
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))

# Map nodes to consecutive indices using torch.unique
def format_input(chunk):
    nodes1, nodes2 = chunk
    all_nodes = torch.cat([nodes1, nodes2])
    unique_nodes, inverse_indices = torch.unique(all_nodes, return_inverse=True)
    mapped_nodes1 = inverse_indices[:len(nodes1)]
    mapped_nodes2 = inverse_indices[len(nodes1):]
    return torch.stack([mapped_nodes1, mapped_nodes2]), unique_nodes

# Save intermediate PageRank results using tensor serialization for faster I/O
def save_partial_results(results, chunk_id):
    path = '~/Comparison-between-Ray-and-Pytorch/PageRank/intermediate_results'
    os.makedirs(os.path.expanduser(path), exist_ok=True)
    torch.save(results, os.path.expanduser(f'{path}/result_chunk_{chunk_id}.pt'))

# Aggregate results from all chunks
def aggregate_results():
    directory = os.path.expanduser('~/Comparison-between-Ray-and-Pytorch/PageRank/results')
    aggregated = {}
    for file in os.listdir(directory):
        if file.endswith('.pt'):
            chunk_results = torch.load(os.path.join(directory, file))
            for node, score in chunk_results.items():
                aggregated[node] = aggregated.get(node, 0) + score
    return aggregated

# Normalize PageRank scores to sum to 1
def normalize_results(results):
    total_score = sum(results.values())
    for node in results:
        results[node] /= total_score
    return results

# Display and save results to file on master node
def display_results(start_time, aggregated_results, config):
    end_time = time.time()
    execution_time = end_time - start_time

    normalized_results = normalize_results(aggregated_results)
    top_nodes = dict(sorted(normalized_results.items(), key=lambda item: item[1], reverse=True)[:10])
    memory_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # Memory usage in MB

    results_text = (
        f"Execution Time (seconds): {execution_time:.2f}\n"
        f"Memory Usage (MB): {memory_usage:.2f}\n"
        f"Top 10 Nodes by PageRank (Normalized):\n{json.dumps(top_nodes, indent=4)}\n"
    )
    print(results_text)

    # Save results to file
    directory = os.path.expanduser('~/Comparison-between-Ray-and-Pytorch/PageRank/results')
    os.makedirs(directory, exist_ok=True)
    file_name = f'{config["datafile"].split(".")[0]}_pagerank_results.txt'
    with open(f'{directory}/{file_name}', 'w') as f:
        f.write(results_text)

# PageRank calculation in distributed mode
def distributed_pagerank(rank, world_size):
    config = {
    "datafile": "twitter7/twitter7_100mb.csv",  
    "batch_size": 1024 * 1024 * 50,  
    "hdfs_host": '192.168.0.1',
    "hdfs_port": 50000
}
    setup(rank, world_size)
    hdfs = fs.HadoopFileSystem(host=config['hdfs_host'], port=config['hdfs_port'])
    file_to_read = f'/data/{config["datafile"]}'

    start_time = time.time()
    global_results = {}

    with hdfs.open_input_file(file_to_read) as file:
        block_size = config['batch_size']
        reader = pv.open_csv(file, read_options=pv.ReadOptions(block_size=block_size))
        for i, chunk in enumerate(reader):
            dataset = GraphDataset(chunk)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
            dataloader = DataLoader(dataset, batch_size=1024 * 1024, sampler=sampler)

            for batch in dataloader:
                pr_input, nodes = format_input(batch)
                pr_scores = page_rank(edge_index=pr_input).tolist()
                global_results = {nodes[idx]: pr_scores[idx] for idx in range(len(nodes))}

            if i % 10 == 0:
                save_partial_results(global_results, i)
                global_results.clear()
                gc.collect()

    if global_results:
        save_partial_results(global_results, 'final')

    dist.barrier()  # Synchronize across nodes

    if rank == 0:
        aggregated_results = aggregate_results()
        display_results(start_time, aggregated_results, config)

    cleanup()

if __name__ == "__main__":
    rank = int(os.getenv('RANK'))
    world_size = int(os.getenv('WORLD_SIZE'))
    distributed_pagerank(rank, world_size)
