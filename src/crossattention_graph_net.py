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
        unbatched_sequences = unbatch(graph_nodes, graph_batch)
        largest_batch_nodes = max([x.shape[0] for x in unbatched_sequences])
        feature_size = unbatched_sequences[0].shape[1]

        all_padded_batches = []
        attention_masks = []

        for current_batch in unbatched_sequences:
            number_pad_nodes = largest_batch_nodes - current_batch.shape[0]
            pad_zeros = torch.zeros([number_pad_nodes, feature_size], device=current_batch.device)
            padded_batch = torch.cat([current_batch, pad_zeros], dim=0)
            all_padded_batches.append(padded_batch)
            attention_mask = [1] * current_batch.shape[0] + [0] * number_pad_nodes
            attention_masks.append(attention_mask)

        batch_nodes = torch.stack(all_padded_batches, dim=0)
        node_mask = torch.tensor(attention_masks, dtype=torch.bool, device=batch_nodes.device)

        q = self.q_dense(batch_nodes)
        k = self.k_dense(conditioning_vector)
        v = self.v_dense(conditioning_vector)

        attn_output, _ = self.attention(q, k, v, key_padding_mask=conditioning_attention_mask)

        x = self.ln1(attn_output + batch_nodes)
        x = self.leaky_relu(self.dense1(x)) + x
        x = self.ln2(x)

        # Node reconstruction
        graphs = []
        for batch_i in range(x.shape[0]):
            current_nodes = x[batch_i]
            current_mask = node_mask[batch_i]
            current_nodes = current_nodes[current_mask]
            current_graph = Data(x=current_nodes)
            graphs.append(current_graph)

        new_batch = Batch.from_data_list(graphs)
        return new_batch.x

class CustomGraphNet(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self._device = device
        self.action_shape = action_shape

        self.node_feature_dim = state_shape['bind_conv5_x'].shape[-1]  # 64
        self.edge_feature_dim = state_shape['bind_conv5_e'].shape[-1]  # 2
        self.hidden_state_dim = state_shape['bind_crossattention4_hidden_states_30'].shape[-1]  # 1280

        self.leaky_relu = nn.LeakyReLU()

        self.conv1 = gnn.GATv2Conv(self.node_feature_dim, 64, edge_dim=self.edge_feature_dim, negative_slope=0.01)
        self.ln1 = gnn.LayerNorm(64)

        self.crossattention = CrossAttentionGraphBlock(num_heads=16, node_feature_size=64, latent_size=self.hidden_state_dim)
        
        self.conv2 = gnn.GATv2Conv(64, 64, edge_dim=self.edge_feature_dim, negative_slope=0.01)
        self.ln2 = gnn.LayerNorm(64)

        self.pool = gnn.LCMAggregation(64, 1024)

        self.dense1 = nn.Linear(1024, 512)
        self.lnf2 = nn.LayerNorm(512)

        self.output_layer = nn.Linear(512, int(np.prod(action_shape)))

        self.output_dim = int(np.prod(action_shape))

    def forward(self, obs, state=None, info=None):
        x = torch.as_tensor(obs['bind_conv5_x'], dtype=torch.float32, device=self._device)
        a = torch.as_tensor(obs['bind_conv5_a'], dtype=torch.long, device=self._device)
        e = torch.as_tensor(obs['bind_conv5_e'], dtype=torch.float32, device=self._device)
        batch = torch.as_tensor(obs['bind_crossattention4_graph_batch'], dtype=torch.long, device=self._device)
        hidden_states = torch.as_tensor(obs['bind_crossattention4_hidden_states_30'], dtype=torch.float32, device=self._device).squeeze(1)
        attention_mask = torch.as_tensor(obs['bind_crossattention4_padding_mask'], dtype=torch.bool, device=self._device).squeeze(1)

        if x.dim() == 3:  # If batched
            batch_size, num_nodes, _ = x.shape
            x = x.view(-1, self.node_feature_dim)
            a = a.permute(0, 2, 1).reshape(2, -1)  # Reshape to (2, batch_size * num_edges)
            e = e.view(-1, self.edge_feature_dim)
            batch = torch.arange(batch_size, device=self._device).repeat_interleave(num_nodes)

        x = self.conv1(x, a, e)
        x = self.leaky_relu(x)
        x = self.ln1(x, batch)

        x = self.crossattention(x, batch, hidden_states, attention_mask)

        x = self.conv2(x, a, e)
        x = self.leaky_relu(x)
        x = self.ln2(x, batch)

        x = self.pool(x, batch)

        x = self.dense1(x)
        x = self.leaky_relu(x)
        x = self.lnf2(x)

        logits = self.output_layer(x)

        return logits, state

    def get_output_dim(self):
        return self.output_dim