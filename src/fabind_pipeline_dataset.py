from torch_geometric.data import Dataset
import pandas as pd
from tqdm import tqdm
import os
from EnzymeGym.models.FABind.FABind_plus.fabind.utils.inference_pdb_utils import extract_protein_structure, extract_esm_feature
from EnzymeGym.models.FABind.FABind_plus.fabind.utils.inference_mol_utils import read_smiles, extract_torchdrug_feature_from_mol, generate_conformation
from torch_geometric.data import HeteroData
import torch


class PipelineDataset(Dataset):
    def __init__(self, protein_ligand_input_dicts):
        super().__init__(None, None, None, None)
        
        self.data = []
        for input_dict in protein_ligand_input_dicts:
            self.data.append(input_dict)

    def len(self):
        return len(self.data)

    def get(self, idx):
        input_dict = self.data[idx]
        protein_node_xyz = torch.tensor(input_dict['protein_structure']['coords'])[:, 1]
        protein_seq = input_dict['protein_structure']['seq']
        protein_esm_feature = input_dict['protein_esm_feature']
        smiles = input_dict['molecule_smiles']
        rdkit_coords, compound_node_features, input_atom_edge_list, LAS_edge_index = input_dict['molecule_info']
        
        n_protein_whole = protein_node_xyz.shape[0]
        n_compound = compound_node_features.shape[0]

        data = HeteroData()

        data.coord_offset = protein_node_xyz.mean(dim=0).unsqueeze(0)
        protein_node_xyz = protein_node_xyz - protein_node_xyz.mean(dim=0)
        coords_init = rdkit_coords - rdkit_coords.mean(axis=0)
        
        # compound graph
        data['compound'].node_feats = compound_node_features.float()
        data['compound', 'LAS', 'compound'].edge_index = LAS_edge_index
        data['compound'].node_coords = coords_init - coords_init.mean(dim=0)
        data['compound'].rdkit_coords = coords_init
        data['compound'].smiles = smiles
        data['compound_atom_edge_list'].x = (input_atom_edge_list[:,:2].long().contiguous() + 1).clone()
        data['LAS_edge_list'].x = (LAS_edge_index + 1).clone().t()

        data.node_xyz_whole = protein_node_xyz
        data.seq_whole = protein_seq
        data.idx = idx
        data.uid = input_dict['protein_structure']['name']
        data.mol = input_dict['molecule']
        data.ligand_id = input_dict['ligand_id']

        # complex whole graph
        data['complex_whole_protein'].node_coords = torch.cat( # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                coords_init - coords_init.mean(dim=0), # for pocket prediction module, the ligand is centered at the protein center/origin
                torch.zeros(1, 3), 
                protein_node_xyz
            ), dim=0
        ).float()
        data['complex_whole_protein'].node_coords_LAS = torch.cat( # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                rdkit_coords,
                torch.zeros(1, 3), 
                torch.zeros_like(protein_node_xyz)
            ), dim=0
        ).float()

        segment = torch.zeros(n_protein_whole + n_compound + 2)
        segment[n_compound+1:] = 1 # compound: 0, protein: 1
        data['complex_whole_protein'].segment = segment # protein or ligand
        mask = torch.zeros(n_protein_whole + n_compound + 2)
        mask[:n_compound+2] = 1 # glb_p can be updated
        data['complex_whole_protein'].mask = mask.bool()
        is_global = torch.zeros(n_protein_whole + n_compound + 2)
        is_global[0] = 1
        is_global[n_compound+1] = 1
        data['complex_whole_protein'].is_global = is_global.bool()

        data['complex_whole_protein', 'c2c', 'complex_whole_protein'].edge_index = input_atom_edge_list[:,:2].long().t().contiguous() + 1
        data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index = LAS_edge_index + 1

        data['protein_whole'].node_feats = protein_esm_feature
        
        return data

