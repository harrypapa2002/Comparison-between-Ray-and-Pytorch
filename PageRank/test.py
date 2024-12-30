import torch
def normalize_adj_manual(edge_index, num_nodes):
    values = torch.ones(edge_index.size(1), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=values,
        size=(num_nodes, num_nodes)
    )
    col_sum = torch.sparse.sum(adj, dim=0).to_dense()  # Dense sum of columns
    col_sum[col_sum == 0] = 1  # Avoid division by zero
    normalized_values = values / col_sum[edge_index[1]]  # Normalize by column sum

    return torch.sparse_coo_tensor(
        indices=edge_index,
        values=normalized_values,
        size=(num_nodes, num_nodes)
    )


def test_normalization():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # Simple cycle graph
    num_nodes = torch.max(edge_index) + 1

    adj = normalize_adj_manual(edge_index, num_nodes)
    col_sum = torch.sparse.sum(adj, dim=0).to_dense()

    print("Adjacency Matrix Normalized Column Sum:", col_sum)

test_normalization()


