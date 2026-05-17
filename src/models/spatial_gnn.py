"""
Spatial GNN: GAT-based message passing over patch adjacency graph.

Builds an 8-neighbor graph from patch grid positions, runs multi-hop
GATv2Conv message passing, and outputs per-patch spatial embeddings
that are injected as additional FiLM modulation into SPADE decoder blocks.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


def build_grid_graph(positions, k=8):
    """Build k-nearest-neighbor edges from a regular grid of patch positions.

    Args:
        positions: [N, 2] float tensor of (row, col) grid positions
        k: number of neighbors (8 = all adjacent including diagonals)

    Returns:
        edge_index: [2, E] long tensor of (src, dst) edges
    """
    N = positions.shape[0]
    dists = torch.cdist(positions, positions, p=2)  # [N, N]
    dists.fill_diagonal_(float('inf'))

    _, indices = dists.topk(min(k, N - 1), dim=1, largest=False)

    src = torch.arange(N, device=positions.device).repeat_interleave(min(k, N - 1))
    dst = indices.flatten()
    edge_index = torch.stack([src, dst], dim=0)

    return edge_index


class SpatialGNN(nn.Module):
    """GAT-based spatial message passing over a patch grid.

    Takes per-patch encoder features + grid positions, builds an adjacency
    graph, runs GATv2Conv message passing, and outputs per-patch spatial
    embeddings for SPADE conditioning.

    Args:
        feature_dim: input feature dimension (encoder bottleneck channels, e.g. 512)
        hidden_dim: hidden dimension in GAT layers
        output_dim: output spatial embedding dimension (e.g. 64)
        num_layers: number of GAT hops (3=ER/PR, 5=HER2, 7=Ki67)
        heads: number of attention heads per GAT layer
        dropout: attention dropout
    """

    def __init__(self, feature_dim=512, hidden_dim=256, output_dim=64,
                 num_layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                          dropout=dropout, add_self_loops=False)
            )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

        # Xavier-init output projection (zero-init would block gradient flow
        # when combined with zero-init SPADE spatial_gamma/beta)
        nn.init.xavier_uniform_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, patch_features, positions, edge_index=None):
        """
        Args:
            patch_features: [B*N, feature_dim] encoder features per patch
            positions: [B*N, 2] (row, col) grid positions
            edge_index: [2, E] pre-built edge index (auto-built if None)

        Returns:
            spatial_emb: [B*N, output_dim] per-patch spatial embeddings
        """
        if edge_index is None:
            edge_index = build_grid_graph(positions, k=8)

        device = patch_features.device
        edge_index = edge_index.to(device)

        x = self.input_proj(patch_features)

        for gat in self.gat_layers:
            x_new = gat(x, edge_index)
            x = x + x_new
            x = torch.relu(x)

        spatial_emb = self.output_proj(x)
        return spatial_emb
