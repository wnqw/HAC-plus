import torch
import torch.nn as nn
import torch.nn.functional as F


def build_knn_graph(x, k=8):
    """Build k-NN graph.
    Args:
        x (Tensor): [N, 3] anchor positions.
    Returns:
        edge_index (LongTensor): [2, E] edge index.
    """
    N = x.shape[0]
    dist = torch.cdist(x, x)
    knn = dist.topk(k + 1, largest=False).indices[:, 1:]
    row = torch.arange(N, device=x.device).unsqueeze(1).repeat(1, k).flatten()
    col = knn.flatten()
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


class AnchorGNN(nn.Module):
    """Simple two-layer graph convolution network for anchors."""

    def __init__(self, in_dim, hidden_dim=32, out_dim=16):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        row, col = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])
        h = F.relu(self.lin1(agg))
        agg2 = torch.zeros_like(h)
        agg2.index_add_(0, row, h[col])
        out = self.lin2(agg2)
        return out


def kmeans(x, num_clusters, num_iters=10):
    """Basic k-means clustering."""
    N = x.shape[0]
    indices = torch.randperm(N, device=x.device)[:num_clusters]
    centers = x[indices]
    for _ in range(num_iters):
        dist = torch.cdist(x, centers)
        labels = dist.argmin(dim=1)
        for i in range(num_clusters):
            mask = labels == i
            if mask.any():
                centers[i] = x[mask].mean(dim=0)
    dist = torch.cdist(x, centers)
    labels = dist.argmin(dim=1)
    return labels


def cluster_anchors(anchors, features, num_clusters, k=8):
    """Cluster anchors with a GNN."""
    edge_index = build_knn_graph(anchors, k)
    gnn = AnchorGNN(features.shape[1]).to(features.device)
    with torch.no_grad():
        emb = gnn(features, edge_index)
    cluster_ids = kmeans(emb, num_clusters)
    return cluster_ids
