import os
import json
import time
import gc
import torch
from torch_ppr import page_rank
from torch.utils.data import Dataset, DataLoader
import pyarrow.csv as pv
import pandas as pd


class GraphDataset(Dataset):
    def __init__(self, chunk):
        self.edges = self.format_edges(chunk)

    def format_edges(self, chunk):
        nodes1 = chunk['node1'].tolist()
        nodes2 = chunk['node2'].tolist()
        return torch.tensor([nodes1, nodes2], dtype=torch.long)

    def __len__(self):
        return self.edges.size(1)

    def __getitem__(self, idx):
        return self.edges[:, idx]


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
def normalize_pr(scores):
    total_score = scores.sum()
    return scores / total_score


# Add self-loops to handle dangling nodes
def add_self_loops(edge_index, num_nodes):
    loop_index = torch.arange(0, num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index


# Prune disconnected nodes
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



# Check for isolated nodes
def check_isolated_nodes(edge_index, num_nodes):
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes))
    col_sum = torch.sparse.sum(adj, dim=0).to_dense()
    isolated = (col_sum == 0).sum().item()
    print(f"Isolated nodes: {isolated} / {num_nodes}")


# PageRank computation locally
def run_pagerank_local(file_path):
    start_time = time.time()
    global_results = {}

    data = pd.read_csv(file_path)
    dataset = GraphDataset(data)
    dataloader = DataLoader(dataset, batch_size=1024 * 1024)

    for batch in dataloader:
        pr_input, all_nodes = format_input(batch)
        num_nodes = len(all_nodes)
        
        # Add self-loops and prune disconnected nodes
        pr_input = add_self_loops(pr_input, num_nodes)
        pr_input, num_nodes = prune_disconnected_nodes(pr_input)
        
        # Check for isolated nodes (optional)
        check_isolated_nodes(pr_input, num_nodes)

        # Perform PageRank
        pr_scores = page_rank(edge_index=pr_input).tolist()
        global_results.update({idx: pr_scores[idx] for idx in range(len(pr_scores))})


    # Normalize and display results
    normalized_results = normalize_pr(torch.tensor(list(global_results.values())))
    top_nodes = dict(sorted(zip(global_results.keys(), normalized_results.tolist()),
                             key=lambda item: item[1], reverse=True)[:10])

    execution_time = time.time() - start_time
    print(f"Execution Time (seconds): {execution_time:.2f}")
    print(f"Top 10 Nodes by PageRank (Normalized):\n{json.dumps(top_nodes, indent=4)}")


if __name__ == "__main__":
    file_path = 'C:/Users/anton/Desktop/Pytorch data/twitter7_100mb.csv'  # Replace with actual path
    run_pagerank_local(file_path)
