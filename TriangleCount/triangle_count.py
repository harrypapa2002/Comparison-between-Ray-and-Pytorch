import os
import json
import gc
import time
import torch
import torch.distributed as dist
import pyarrow.fs as fs
import pyarrow.csv as pv
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class GraphEdgeDataset(Dataset):
    def __init__(self, chunk):
        self.edges = self.load_chunk(chunk)

    def load_chunk(self, chunk):
        nodes1 = chunk.column('node1').to_pylist()
        nodes2 = chunk.column('node2').to_pylist()
        edge_list = torch.tensor([nodes1, nodes2], dtype=torch.int32)
        return edge_list

    def __len__(self):
        return self.edges.size(1)

    def __getitem__(self, idx):
        return self.edges[:, idx]


# ------- Distributed Setup -------
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.0.1'  
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


# ------- Cleanup -------
def cleanup():
    dist.destroy_process_group()


# ------- Triangle Counting via Matrix Multiplication -------
def triangle_count(adjacency_matrix):
    A3 = torch.matmul(adjacency_matrix, torch.matmul(adjacency_matrix, adjacency_matrix))
    triangle_counts = torch.diag(A3)
    total_triangles = torch.sum(triangle_counts) // 6  # Each triangle is counted 6 times
    return total_triangles.item()


# ------- Process Data Chunk -------
def process_chunk(batch):
    node_pairs = batch.t()
    max_node = torch.max(node_pairs).item() + 1
    adjacency_matrix = torch.zeros((max_node, max_node), dtype=torch.float32)

    # Fill adjacency matrix
    for i in range(node_pairs.size(0)):
        adjacency_matrix[node_pairs[i, 0], node_pairs[i, 1]] = 1
        adjacency_matrix[node_pairs[i, 1], node_pairs[i, 0]] = 1  # Undirected graph

    return triangle_count(adjacency_matrix)


# ------- Distributed Triangle Count -------
def distributed_triangle_count(rank, world_size):
    config = {
        "datafile": "twitter7_10gb.csv",
        "batch_size": 1024 * 1024 * 50,  # 50MB chunks
        "hdfs_host": "namenode",
        "hdfs_port": 50000
    }

    setup(rank, world_size)

    hdfs = fs.HadoopFileSystem(host=config["hdfs_host"], port=config["hdfs_port"])
    file_to_read = f'/data/{config["datafile"]}'

    total_triangles = 0
    with hdfs.open_input_file(file_to_read) as file:
        read_options = pv.ReadOptions(block_size=config["batch_size"])
        csv_reader = pv.open_csv(file, read_options=read_options)

        for i, chunk in enumerate(csv_reader):
            dataset = GraphEdgeDataset(chunk)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
            dataloader = DataLoader(dataset, batch_size=1024 * 1024, sampler=sampler, shuffle=False)

            for batch in dataloader:
                triangles = process_chunk(batch)
                total_triangles += triangles

                # Save intermediate results every 10 chunks
                if i % 10 == 0:
                    save_intermediate_results(rank, triangles, f"result_chunk_{i}.json")

    # Aggregate results across all nodes
    total_triangles_tensor = torch.tensor([total_triangles], dtype=torch.float32)
    gathered_results = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(gathered_results, total_triangles_tensor)

    if rank == 0:
        final_count = sum([t.item() for t in gathered_results])
        print(f"Total Triangle Count: {final_count}")

    cleanup()


# ------- Save Intermediate Results to HDFS -------
def save_intermediate_results(rank, result, filename):
    directory = '/data/triangle_count/results'
    hdfs_path = f"{directory}/{filename}"

    hdfs = fs.HadoopFileSystem(host="namenode", port=50000)
    with hdfs.open_output_stream(hdfs_path) as f:
        json.dump({"rank": rank, "triangles": result}, f)


# ------- Main -------
def main():
    rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    distributed_triangle_count(rank, world_size)


if __name__ == "__main__":
    main()
