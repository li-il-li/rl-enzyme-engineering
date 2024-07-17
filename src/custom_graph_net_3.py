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
        self.ln1 = gnn.LayerNorm(node_feature_size)
        self.dense1 = nn.Linear(node_feature_size, node_feature_size)
        self.ln2 = gnn.LayerNorm(node_feature_size)

    def forward(self, graph_nodes, graph_batch, conditioning_vector, conditioning_attention_mask):
        q = self.q_dense(graph_nodes)
        k = self.k_dense(conditioning_vector)
        v = self.v_dense(conditioning_vector)
        attn_output, _ = self.attention(q, k, v, key_padding_mask=conditioning_attention_mask)
        x = self.ln1(attn_output + graph_nodes, graph_batch)
        x = self.leaky_relu(self.dense1(x)) + x
        x = self.ln2(x, graph_batch)
        return x

class CustomGraphNet(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self._device = device
        self.action_shape = action_shape

        # Extract relevant dimensions from state_shape
        # Extract relevant dimensions from state_shape
        self.node_feature_dim = state_shape['bind_conv5_x'].shape[1]  # 64
        self.edge_feature_dim = state_shape['bind_conv5_e'].shape[1]  # 2
        self.hidden_state_dim = state_shape['bind_crossattention4_hidden_states_30'].shape[2]  # 1280

        # Calculate input dimension
        self.input_dim = (
            np.prod(state_shape['bind_conv5_x'].shape) +
            np.prod(state_shape['bind_conv5_a'].shape) +
            np.prod(state_shape['bind_conv5_e'].shape) +
            np.prod(state_shape['bind_crossattention4_graph_batch'].shape) +
            np.prod(state_shape['bind_crossattention4_hidden_states_30'].shape) +
            np.prod(state_shape['bind_crossattention4_padding_mask'].shape)
        )

        self.leaky_relu = nn.LeakyReLU()

        # Graph Convolutions
        self.conv1 = gnn.GATv2Conv(self.node_feature_dim, 64, edge_dim=self.edge_feature_dim, negative_slope=0.01)
        self.ln1 = gnn.LayerNorm(64)

        # Cross Attention
        self.crossattention = CrossAttentionGraphBlock(num_heads=16, node_feature_size=64, latent_size=self.hidden_state_dim)
        
        # Final Convolutions
        self.conv2 = gnn.GATv2Conv(64, 64, edge_dim=self.edge_feature_dim, negative_slope=0.01)
        self.ln2 = gnn.LayerNorm(64)

        # Global Pooling
        self.pool = gnn.LCMAggregation(64, 1024)

        # Final MLP
        self.dense1 = nn.Linear(1024, 512)
        self.lnf2 = nn.LayerNorm(512)

        # Output layer
        self.output_layer = nn.Linear(512, int(np.prod(action_shape)))

        self.output_dim = int(np.prod(action_shape))

    def forward(self, obs, state=None, info=None):
        # Convert numpy arrays to PyTorch tensors and move to the correct device
        x = torch.as_tensor(obs['bind_conv5_x'], dtype=torch.float32, device=self._device)
        a = torch.as_tensor(obs['bind_conv5_a'], dtype=torch.long, device=self._device)
        e = torch.as_tensor(obs['bind_conv5_e'], dtype=torch.float32, device=self._device)
        batch = torch.as_tensor(obs['bind_crossattention4_graph_batch'], dtype=torch.long, device=self._device)
        hidden_states = torch.as_tensor(obs['bind_crossattention4_hidden_states_30'], dtype=torch.float32, device=self._device).squeeze(0)
        attention_mask = torch.as_tensor(obs['bind_crossattention4_padding_mask'], dtype=torch.bool, device=self._device).squeeze(0)

        print(f"Initial tensor shapes:")
        print(f"  x: {x.shape}")
        print(f"  a: {a.shape}")
        print(f"  e: {e.shape}")
        print(f"  batch: {batch.shape}")
        print(f"  hidden_states: {hidden_states.shape}")
        print(f"  attention_mask: {attention_mask.shape}")

        
        # Graph convolutions
        x = self.conv1(x, a, e)
        x = self.leaky_relu(x)
        x = self.ln1(x, batch)

        # Cross attention
        x = self.crossattention(x, batch, hidden_states, attention_mask)

        # Final convolutions
        x = self.conv2(x, a, e)
        x = self.leaky_relu(x)
        x = self.ln2(x, batch)

        # Global pooling
        x = self.pool(x, batch)

        # Final MLP
        x = self.dense1(x)
        x = self.leaky_relu(x)
        x = self.lnf2(x)

        # Output layer
        logits = self.output_layer(x)

        return logits, state

    def get_output_dim(self):
        return self.output_dim

    def get_input_dim(self):
        return self.input_dim