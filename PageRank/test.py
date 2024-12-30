import torch
from torch_ppr import page_rank

# Simulate a small graph (triangle graph)
def simulate_graph():
    # Simple graph: 3 nodes, 3 edges (triangle)
    edges = [
        (0, 1),
        (1, 2),
        (2, 0)
    ]

    nodes1, nodes2 = zip(*edges)  # Separate source and destination
    nodes1_tensor = torch.tensor(nodes1, dtype=torch.int32)
    nodes2_tensor = torch.tensor(nodes2, dtype=torch.int32)

    # Stack to form edge_index of shape [2, N]
    edge_index = torch.stack([nodes1_tensor, nodes2_tensor], dim=0)
    
    return edge_index

# PageRank on small graph
def test_pagerank():
    edge_index = simulate_graph()
    
    # Print the edge_index to verify the format
    print("Edge Index (Before PageRank):")
    print(edge_index)
    print("Shape:", edge_index.shape)

    # Ensure correct shape [2, N]
    if edge_index.shape[0] != 2:
        raise ValueError(f"Invalid edge index shape: {edge_index.shape}. Expected shape [2, N].")
    
    # Run PageRank (without distributed setup)
    try:
        pr_scores = page_rank(edge_index=edge_index).tolist()
        print("\nPageRank Scores (Success):", pr_scores)
    except Exception as e:
        print("\nError during PageRank execution:")
        print(e)

if __name__ == "__main__":
    test_pagerank()
