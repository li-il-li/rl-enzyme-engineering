import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.utils import unbatch
from torch_geometric.data import Data, Batch
import numpy as np

class CrossAttentionGraphBlock(nn.Module):
    def __init__(self, num_heads, node_feature_size, latent_size, dropout=0.1):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU()
        self.q_dense = nn.Linear(node_feature_size, node_feature_size)
        self.k_dense = nn.Linear(latent_size, node_feature_size)
        self.v_dense = nn.Linear(latent_size, node_feature_size)
        self.attention = nn.MultiheadAttention(node_feature_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(node_feature_size)
        self.dense1 = nn.Linear(node_feature_size, node_feature_size)
        self.ln2 = nn.LayerNorm(node_feature_size)

    def forward(self, graph_nodes, graph_batch, conditioning_vector, conditioning_attention_mask):
        q = self.q_dense(graph_nodes)
        k = self.k_dense(conditioning_vector)
        v = self.v_dense(conditioning_vector)
        attn_output, _ = self.attention(q, k, v, key_padding_mask=conditioning_attention_mask)
        x = self.ln1(attn_output + graph_nodes)
        x = self.leaky_relu(self.dense1(x)) + x
        x = self.ln2(x)
        return x

class CustomGraphNet(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_sizes, device):
        super().__init__()
        self.device = device
        self.action_shape = action_shape

        # Graph Convolutions
        self.conv1 = gnn.GATv2Conv(64, 128, edge_dim=2, negative_slope=0.01)
        self.ln1 = gnn.LayerNorm(128)

        # Cross Attention
        self.crossattention = CrossAttentionGraphBlock(num_heads=8, node_feature_size=128, latent_size=1280)

        # Final Convolutions
        self.conv2 = gnn.GATv2Conv(128, 256, edge_dim=2, negative_slope=0.01)
        self.ln2 = gnn.LayerNorm(256)

        # Global Pooling
        self.pool = gnn.global_mean_pool

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, int(np.prod(action_shape)))
        )

        self.output_dim = int(np.prod(action_shape))

    def forward(self, obs, state=None, info=None):
        # Extract relevant information from the observation
        x = obs['bind_conv5_x'].to(self.device)
        edge_index = obs['bind_conv5_a'].to(self.device)
        edge_attr = obs['bind_conv5_e'].to(self.device)
        batch = obs['bind_crossattention4_graph_batch'].to(self.device)
        hidden_states = obs['bind_crossattention4_hidden_states_30'].to(self.device).squeeze(0)  # Remove batch dimension
        attention_mask = obs['bind_crossattention4_padding_mask'].to(self.device).squeeze(0)  # Remove batch dimension

        # Graph convolutions
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.ln1(x)

        # Cross attention
        x = self.crossattention(x, batch, hidden_states, attention_mask)

        # Final convolutions
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.ln2(x)

        # Global pooling
        x = self.pool(x, batch)

        # Final MLP
        logits = self.mlp(x)

        return logits, state

    def get_output_dim(self):
        return self.output_dim