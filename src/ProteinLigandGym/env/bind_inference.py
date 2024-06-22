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
        output, intermediate_activations = bind_model.forward(current_graphs, hidden_states, attention_mask, return_intermediate=True)

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

    return model, esm_model, esm_tokeniser


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