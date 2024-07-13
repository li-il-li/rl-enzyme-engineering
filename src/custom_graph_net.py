import torch
from torch import nn
import torch_geometric.nn as gnn
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
import numpy as np

class CustomGraphNet(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_sizes, device):
        super().__init__()
        self.device = device
        self.action_shape = action_shape

        # Assuming state_shape is a dictionary with the following keys:
        self.node_feature_size = state_shape['node_features']
        self.edge_feature_size = state_shape['edge_features']
        self.latent_size = state_shape['text_embedding']

        self.leaky_relu = nn.LeakyReLU()

        # Node and edge embeddings
        self.x_embed = nn.Linear(self.node_feature_size, 16)
        self.x_embed_ln = gnn.LayerNorm(16)
        self.e_embed = nn.Linear(self.edge_feature_size, 2)
        self.e_embed_ln = nn.LayerNorm(2)

        # Graph convolution layers
        self.conv1 = gnn.GATv2Conv(16, 64, edge_dim=2, negative_slope=0.01)
        self.ln1 = gnn.LayerNorm(64)
        self.conv2 = gnn.GATv2Conv(64, 64, edge_dim=2, negative_slope=0.01)
        self.ln2 = gnn.LayerNorm(64)

        # Cross-attention layers
        self.crossattention1 = CrossAttentionGraphBlock(num_heads=16, node_feature_size=64, latent_size=self.latent_size)
        self.crossattention2 = CrossAttentionGraphBlock(num_heads=16, node_feature_size=64, latent_size=self.latent_size)

        # Global pooling
        self.pool = gnn.global_mean_pool

        # Final layers
        self.dense1 = nn.Linear(64, hidden_sizes[-1])
        self.ln_final = nn.LayerNorm(hidden_sizes[-1])
        self.output_layer = nn.Linear(hidden_sizes[-1], int(np.prod(action_shape)))

        self.output_dim = int(np.prod(action_shape))

    def forward(self, obs, state=None, info={}):
        # Assuming obs is a dictionary with 'graph_data' and 'text_embeddings' keys
        graph_data = obs['graph_data'].to(self.device)  # This should be a PyG Data or Batch object
        text_embeddings = obs['text_embeddings'].to(self.device)  # This should be a tensor of shape (batch_size, latent_size)

        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        batch = graph_data.batch

        # Node and edge embeddings
        x = self.leaky_relu(self.x_embed_ln(self.x_embed(x)))
        edge_attr = self.leaky_relu(self.e_embed_ln(self.e_embed(edge_attr)))

        # Graph convolutions
        x = self.leaky_relu(self.ln1(self.conv1(x, edge_index, edge_attr)))
        x = self.leaky_relu(self.ln2(self.conv2(x, edge_index, edge_attr)))

        # Cross-attention
        x, _ = self.crossattention1(x, batch, text_embeddings, None)
        x, _ = self.crossattention2(x, batch, text_embeddings, None)

        # Global pooling
        x = self.pool(x, batch)

        # Final layers
        x = self.leaky_relu(self.ln_final(self.dense1(x)))
        logits = self.output_layer(x)

        return logits, state

    def get_output_dim(self):
        return self.output_dim

    def get_input_dim(self):
        # This should return the total dimension of your input
        # Including both graph features and text embeddings
        return self.node_feature_size + self.edge_feature_size + self.latent_size

class CrossAttentionGraphBlock(nn.Module):
    def __init__(self, num_heads, node_feature_size, latent_size, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(node_feature_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(node_feature_size)
        self.dense1 = nn.Linear(node_feature_size, node_feature_size)
        self.ln2 = nn.LayerNorm(node_feature_size)

    def forward(self, graph_nodes, graph_batch, conditioning_vector, conditioning_attention_mask):
        attn_output, _ = self.attention(graph_nodes, conditioning_vector, conditioning_vector)
        x = self.ln1(attn_output + graph_nodes)
        x = self.dense1(x) + x
        x = self.ln2(x)
        return x, None