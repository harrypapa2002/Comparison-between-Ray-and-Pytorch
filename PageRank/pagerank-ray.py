import os
import json
import time
import gc
import torch
import psutil
from torch_ppr import page_rank
import pyarrow.fs as fs
import pyarrow.csv as pv
import ray


def get_alive_nodes():
    return len([n for n in ray.nodes() if n["Alive"]])

def add_to_dict(global_pr, scores, nodes):
    for idx, node in enumerate(nodes):
        global_pr[node] = global_pr.get(node, 0.0) + scores[idx]
    return global_pr

def aggregate_dicts(main_dict, partial_dict):
    for node, score in partial_dict.items():
        main_dict[node] = main_dict.get(node, 0.0) + score
    return main_dict

def top_k(scores_dict, k):
    items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    return dict(items[:k])

def normalize_scores(scores_dict):
    total_score = sum(scores_dict.values())
    return {n: sc / total_score for n, sc in scores_dict.items()} if total_score > 0 else scores_dict

def save_intermediate_results(results_dict, chunk_id):
    path = os.path.expanduser('~/Comparison-between-Ray-and-Pytorch/PageRank/intermediate_results')
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"result_chunk_{chunk_id}.json")
    with open(file_path, 'w') as f:
        json.dump(results_dict, f)

def add_self_loops(edge_index, num_nodes):
    loop_index = torch.arange(0, num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index

def prune_disconnected_nodes(edge_index):
    unique_nodes = torch.unique(edge_index)
    node_map = {old.item(): new for new, old in enumerate(unique_nodes)}

    mapped_edges = []
    for i in range(edge_index.size(1)):
        src, dst = edge_index[:, i]
        mapped_edges.append([node_map[src.item()], node_map[dst.item()]])
    if not mapped_edges:
        raise ValueError("No valid edges remain after pruning disconnected nodes.")

    mapped_edges = torch.tensor(mapped_edges, dtype=torch.long).t()
    return mapped_edges, len(unique_nodes)

def load_and_merge_intermediate_results(top_k_size=10000):
    path = os.path.expanduser('~/Comparison-between-Ray-and-Pytorch/PageRank/intermediate_results')
    merged_dict = {}
    for fname in os.listdir(path):
        if fname.startswith("result_chunk_") and fname.endswith(".json"):
            file_path = os.path.join(path, fname)
            with open(file_path, 'r') as f:
                partial_dict = json.load(f)
                merged_dict = aggregate_dicts(merged_dict, partial_dict)
                # Keep only top 10k to prevent unbounded growth
                merged_dict = top_k(merged_dict, top_k_size)
    return merged_dict

def cleanup_intermediate():
    path = os.path.expanduser('~/Comparison-between-Ray-and-Pytorch/PageRank/intermediate_results')
    if os.path.exists(path):
        for fname in os.listdir(path):
            if fname.startswith("result_chunk_") and fname.endswith(".json"):
                os.remove(os.path.join(path, fname))


def format_input(chunk):
    nodes1 = chunk[:, 0].tolist()
    nodes2 = chunk[:, 1].tolist()
    all_nodes = list(set(nodes1 + nodes2))
    all_nodes.sort()
    node_map = {node: idx for idx, node in enumerate(all_nodes)}
    mapped_nodes1 = [node_map[x] for x in nodes1]
    mapped_nodes2 = [node_map[x] for x in nodes2]
    edge_index = torch.tensor([mapped_nodes1, mapped_nodes2], dtype=torch.long)
    return edge_index, all_nodes

def page_rank_per_chunk(chunk_data):
    edge_index, node_list = format_input(chunk_data)
    pr_scores = page_rank(edge_index=edge_index).tolist()
    return node_list, pr_scores


@ray.remote
def process_chunk_remote(chunk_data):
    dataset = chunk_data  
    nodes1 = dataset.column('node1').to_pylist()
    nodes2 = dataset.column('node2').to_pylist()
    all_nodes = list(set(nodes1 + nodes2))
    all_nodes.sort()
    node_map = {node: idx for idx, node in enumerate(all_nodes)}
    mapped_nodes1 = [node_map[n] for n in nodes1]
    mapped_nodes2 = [node_map[n] for n in nodes2]
    edge_index = torch.tensor([mapped_nodes1, mapped_nodes2], dtype=torch.long)
    scores = page_rank(edge_index=edge_index).tolist()
    partial_dict = {}
    for idx, node_id in enumerate(all_nodes):
        partial_dict[node_id] = scores[idx]
    return partial_dict



def display_results(start_time, final_scores, config):
    end_time = time.time()
    elapsed = end_time - start_time
    alive_nodes = get_alive_nodes()
    top_nodes = top_k(final_scores, 10)
    sum_scores = sum(final_scores.values())
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    peak_memory = memory_info.rss / (1024 * 1024)

    virtual_memory = memory_info.vms / (1024 * 1024)
    if sum_scores != 0:
        normalized_top10 = {n: (sc / sum_scores) for n, sc in top_nodes.items()}
    else:
        normalized_top10 = top_nodes

    msg = (
        f"File {config['datafile']} - using Ray\n"
        f"Alive Ray Nodes: {alive_nodes}\n"
        f"Execution Time (seconds): {elapsed:.2f}\n"
        f"Peak Memory Usage (MB): {peak_memory:.2f}\n"
        f"Virtual Memory Usage (MB): {virtual_memory:.2f}\n"
        f"Top 10 Nodes by PageRank (Normalized):\n{json.dumps(normalized_top10, indent=4)}\n"
    )
    print(msg)
    results_dir = os.path.expanduser('~/Comparison-between-Ray-and-Pytorch/PageRank/results')
    os.makedirs(results_dir, exist_ok=True)
    fname = f"{config['datafile'].split('.')[0]}_ray_{alive_nodes}.txt"
    with open(os.path.join(results_dir, fname), 'w') as f:
        f.write(msg)


def distributed_pagerank_ray():
    config = {
        "datafile": "twitter7/twitter7_5gb.csv",
        "batch_size": 1024 * 1024 * 10,
        "hdfs_host": "192.168.0.1",
        "hdfs_port": 9000
    }
    ray.init(address="auto")
    cleanup_intermediate()
    start_time = time.time()

    hdfs = fs.HadoopFileSystem(host=config["hdfs_host"], port=config["hdfs_port"])
    file_path = f"/data/{config['datafile']}"

    futures = []
    with hdfs.open_input_file(file_path) as f:
        reader = pv.open_csv(f, read_options=pv.ReadOptions(block_size=config["batch_size"]))
        for i, chunk in enumerate(reader):
            fut = process_chunk_remote.remote(chunk)
            futures.append(fut)

            
            if (i+1) % 20 == 0:
                partial_dict_list = ray.get(futures)
                futures = []
                merged_dict = {}
                for pd in partial_dict_list:
                    merged_dict = aggregate_dicts(merged_dict, pd)
                merged_dict = top_k(merged_dict, 10000)
                save_intermediate_results(merged_dict, f"chunk_{i+1}")
                merged_dict.clear()
                gc.collect()

    if futures:
        partial_dict_list = ray.get(futures)
        futures = []
        merged_dict = {}
        for pd in partial_dict_list:
            merged_dict = aggregate_dicts(merged_dict, pd)
        merged_dict = top_k(merged_dict, 10000)
        save_intermediate_results(merged_dict, "chunk_final")
        merged_dict.clear()
        gc.collect()

    
    final_scores = load_and_merge_intermediate_results(top_k_size=10000)
    final_scores = normalize_scores(final_scores)

    display_results(start_time, final_scores, config)
    ray.shutdown()

if __name__ == "__main__":
    distributed_pagerank_ray()
