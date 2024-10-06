# Binding INteraction Determination
import os
import numpy as np
import torch, torch_geometric, transformers, networkx
from transformers import logging, AutoModel, AutoTokenizer
from torch_geometric.data import Batch
import sys
import logging
from .models.BIND import loading
from .models.BIND.data import BondType
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.data import Data
from tqdm import tqdm
from Bio import SeqIO

log = logging.getLogger(__name__)

def predict_binder(bind_model, esm_model, esm_tokeniser, device, sequences, ligand_smile):

    all_scores = []

    for sequence in sequences:
        
        encoded_input = esm_tokeniser([sequence], padding="longest", truncation=False, return_tensors="pt")
        esm_output = esm_model.forward(**encoded_input.to(device), output_hidden_states=True)
        hidden_states = esm_output.hidden_states

        hidden_states = [x.to(device).detach() for x in hidden_states]
        attention_mask = encoded_input["attention_mask"].to(device)

        ligand = get_graph(ligand_smile)

        current_graphs = Batch.from_data_list([ligand]).to(device).detach()
        output = bind_model.forward(current_graphs, hidden_states, attention_mask)

        output = [x.detach().cpu().numpy() for x in output]
        probability = sigmoid(output[-1])

        output = output + [probability]

        scores = (['id', sequence ,ligand_smile] + [np.array2string(np.squeeze(x), precision=5) for x in output])
        score = {
            'id': scores[0],
            'sequence': scores[1],
            'smile': scores[2],
            'pKi': scores[3],
            'pIC50': scores[4],
            'pKd': scores[5],
            'pEC50': scores[6],
            'logit': scores[7],
            'non_binder_prob': scores[8]
        }
        all_scores.append(score)

    return all_scores
  
def init_BIND(device):
    esm_tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm_model.eval()
    esm_model.to(device)

    model = torch.load("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models/BIND/saves/BIND_checkpoint_12042024.pth", map_location=device)
    model.eval()
    model.to(device)
    
    #log.info(f"Model BIND: {model}")
    
    crossattention4_graph_batch = None
    crossattention4_hidden_states_30 = None
    crossattention4_padding_mask = None
    def hook_fn_crossattention4(module, input, output):
        nonlocal crossattention4_graph_batch, crossattention4_hidden_states_30, crossattention4_padding_mask
        crossattention4_graph_batch, crossattention4_hidden_states_30, crossattention4_padding_mask = (
            input[1].detach().cpu().numpy(),
            input[2].detach().cpu().numpy(),
            input[3].detach().cpu().numpy()
        )
    hook_crossattention4 = model.crossattention4.register_forward_hook(hook_fn_crossattention4)
    
    def get_crossattention4_inputs():
        return crossattention4_graph_batch, crossattention4_hidden_states_30, crossattention4_padding_mask
    
    conv5_x = None
    conv5_a = None
    conv5_e = None
    def hook_fn_conv5(module, input, output):
        nonlocal conv5_x, conv5_a, conv5_e
        conv5_x, conv5_a, conv5_e = (
            input[0].detach().cpu().numpy(),
            input[1].detach().cpu().numpy(),
            input[2].detach().cpu().numpy()
        )
    hook_conv5 = model.conv5.register_forward_hook(hook_fn_conv5)

    def get_conv5_inputs():
        return conv5_x, conv5_a, conv5_e

    return model, esm_model, esm_tokeniser, get_conv5_inputs, get_crossattention4_inputs


def get_graph(smiles):
    graph = loading.get_data(smiles, apply_paths=False, parse_cis_trans=False, unknown_atom_is_dummy=True)

    x, a, e = loading.convert(*graph, bonds=[BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC, BondType.NOT_CONNECTED])
    x = torch.Tensor(x)
    a = dense_to_sparse(torch.Tensor(a))[0]
    e = torch.Tensor(e)

    # Given an xae
    graph = Data(x=x, edge_index=a, edge_features=e)

    return graph


def sigmoid(x):
  return 1 / (1 + np.exp(-x))