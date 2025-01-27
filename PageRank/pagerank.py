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
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


# Clean up process group and intermediate files
def cleanup():
    rank = dist.get_rank()
    dist.destroy_process_group()
    if rank == 0:
        temp_dir = '~/Comparison-between-Ray-and-Pytorch/PageRank/intermediate_results'
        temp_dir = os.path.expanduser(temp_dir)
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))


# Format input edges and map nodes to indices
def format_input(chunk):
    nodes1 = chunk[:, 0].tolist()
    nodes2 = chunk[:, 1].tolist()

    all_nodes_set = set(nodes1 + nodes2)
    all_nodes = list(all_nodes_set)
    all_nodes.sort()
    node_map = {node: idx for idx, node in enumerate(all_nodes)}

    mapped_nodes1 = [node_map[node] for node in nodes1]
    mapped_nodes2 = [node_map[node] for node in nodes2]

    nodes1_tensor = torch.tensor(mapped_nodes1, dtype=torch.int32)
    nodes2_tensor = torch.tensor(mapped_nodes2, dtype=torch.int32)

    return torch.stack([nodes1_tensor, nodes2_tensor], dim=0), all_nodes


# Normalize final PageRank scores
def normalize_page_rank(scores):
    total_score = scores.sum()
    return scores / total_score


# Save intermediate results
def save_intermediate_results(results, chunk_id,rank):
    path = '~/Comparison-between-Ray-and-Pytorch/PageRank/intermediate_results'
    full_path = os.path.expanduser(path)
    os.makedirs(full_path, exist_ok=True)

    temp_file = os.path.join(full_path, f'temp_result_chunk_{chunk_id}_rank_{rank}.json')
    final_file = os.path.join(full_path, f'result_chunk_{chunk_id}_rank_{rank}.json')

    try:
        # Write to a temporary file first
        with open(temp_file, 'w') as f:
            json.dump(results, f)

        # Rename the temp file to final to ensure atomic write
        os.replace(temp_file, final_file)

        print(f"Intermediate results for chunk {chunk_id} saved successfully.")
    except Exception as e:
        print(f"Error saving intermediate results for chunk {chunk_id}: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)  # Clean up temp file if error occurs

def add_self_loops(edge_index, num_nodes):
    loop_index = torch.arange(0, num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index

def prune_disconnected_nodes(edge_index):
    unique_nodes = torch.unique(edge_index)
    node_map = {old.item(): new for new, old in enumerate(unique_nodes)}
    
    # Filter out edges with unmapped nodes
    mapped_edges = []
    for i in range(edge_index.size(1)):
        src, dst = edge_index[:, i]
        if src.item() in node_map and dst.item() in node_map:
            mapped_edges.append([node_map[src.item()], node_map[dst.item()]])
    
    if not mapped_edges:
        raise ValueError("No valid edges remain after pruning disconnected nodes.")
    
    mapped_edges = torch.tensor(mapped_edges, dtype=torch.long).t()
    return mapped_edges, len(unique_nodes)

# Aggregate intermediate results
def aggregate_results():
    directory = os.path.expanduser('~/Comparison-between-Ray-and-Pytorch/PageRank/intermediate_results')
    aggregated = {}
    for file in os.listdir(directory):
        if file.startswith('result_chunk_') and file.endswith('.json'):
            full_path = os.path.join(directory,file)
            with open(full_path, 'r') as f:
                chunk_results = json.load(f)
                for node, score in chunk_results.items():
                    aggregated[node] = aggregated.get(node, 0) + score
    return aggregated


# Display and save results to file
def display_results(start_time, aggregated_results, config):
    end_time = time.time()
    execution_time = end_time - start_time

    normalized_results = normalize_page_rank(torch.tensor(list(aggregated_results.values())))
    top_nodes = dict(sorted(zip(aggregated_results.keys(), normalized_results.tolist()),
                             key=lambda item: item[1], reverse=True)[:10])
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    peak_memory = memory_info.rss / (1024 * 1024)  
    virtual_memory = memory_info.vms / (1024 * 1024)

    results_text = (
        f"File {config['datafile'].split('.')[0]} - number of worker machines {config['world_size']} - batch size {config['batch_size']}:\n"
        f"Execution Time (seconds): {execution_time:.2f}\n"
        f"Peak Memory Usage (MB): {peak_memory:.2f}\n"
        f"Virtual Memory Usage (MB): {virtual_memory:.2f}\n"
        f"Top 10 Nodes by PageRank (Normalized):\n{json.dumps(top_nodes, indent=4)}\n"
    )
    print(results_text)

    directory = os.path.expanduser('~/Comparison-between-Ray-and-Pytorch/PageRank/results')
    os.makedirs(directory, exist_ok=True)
    file_name = f"{config['datafile'].split('.')[0]}_{config['world_size']}_results.txt"
    with open(f'{directory}/{file_name}', 'w') as f:
        f.write(results_text)


# Distributed PageRank computation
def distributed_pagerank(rank, world_size):
    config = {
        "datafile": "twitter7/twitter7_5gb.csv",
        "batch_size": 1024 * 1024 * 30,
        "hdfs_host": '192.168.0.1',
        "hdfs_port": 9000,
        "world_size": world_size
    }
    setup(rank, world_size)
    hdfs = fs.HadoopFileSystem(host=config['hdfs_host'], port=config['hdfs_port'])
    file_to_read = f'/data/{config["datafile"]}'

    start_time = time.time()
    global_results = {}

    with hdfs.open_input_file(file_to_read) as file:
        reader = pv.open_csv(file, read_options=pv.ReadOptions(block_size=config['batch_size']))
        for i, chunk in enumerate(reader):
            dataset = GraphDataset(chunk)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
            dataloader = DataLoader(dataset, batch_size=1024 * 1024, sampler=sampler)

            for batch in dataloader:
                pr_input, nodes = format_input(batch)
                num_nodes = len(nodes)
                pr_input = add_self_loops(pr_input, num_nodes)
                pr_input, num_nodes = prune_disconnected_nodes(pr_input)
                pr_scores = page_rank(edge_index=pr_input).tolist()
                #global_results.update({int(nodes[idx]): pr_scores[idx] for idx in range(len(pr_scores))})
                for idx, score in enumerate(pr_scores):
                    node_id = int(nodes[idx])
                    global_results[node_id] = global_results.get(node_id,0.0) + score

            if i % 10 == 0:
                save_intermediate_results(global_results, i,rank)
                global_results.clear()
                gc.collect()

    if global_results:
        save_intermediate_results(global_results, 'final',rank)
    
    dist.barrier()

    if rank == 0:
        aggregated_results = aggregate_results()
        display_results(start_time, aggregated_results, config)

    cleanup()


if __name__ == "__main__":
    rank = int(os.getenv('RANK'))
    world_size = int(os.getenv('WORLD_SIZE'))
    distributed_pagerank(rank, world_size)
