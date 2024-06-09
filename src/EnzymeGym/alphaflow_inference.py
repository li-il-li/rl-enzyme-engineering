import sys
sys.path.append("/root/projects/rl-enzyme-engineering/src/AlphaFlow")

import torch, tqdm, os, wandb, json, time
import io
import gzip
from Bio.PDB import PDBParser
import pandas as pd
import pytorch_lightning as pl
import numpy as np
from collections import defaultdict
from AlphaFlow.alphaflow.model.wrapper import ESMFoldWrapper
from AlphaFlow.alphaflow.data.data_modules import collate_fn
from AlphaFlow.alphaflow.utils.tensor_utils import tensor_tree_map
from AlphaFlow.alphaflow.config import model_config
from AlphaFlow.alphaflow.utils.protein import protein
from AlphaFlow.alphaflow.data.inference import seq_to_tensor
from openfold.data.data_transforms import make_atom14_masks

def init_esmflow(ckpt='', device='cuda'):
    ckpt = torch.load(ckpt, map_location=device)
    model = ESMFoldWrapper(**ckpt['hyper_parameters'], training=False)
    model.model.load_state_dict(ckpt['params'], strict=False)
    model = model.to(device)
    # TODO return also esm

    return model

def generate_conformation_ensemble(model: ESMFoldWrapper, cfg, AA_sequences=[]):
    samples = cfg.alphaflow.samples
    noisy_first = cfg.alphaflow.noisy_first
    no_diffusion = cfg.alphaflow.no_diffusion
    self_cond = cfg.alphaflow.self_cond
    tmax = cfg.alphaflow.tmax
    steps = cfg.alphaflow.steps

    schedule = np.linspace(tmax, 0, steps+1)
    
    batch = create_batch(AA_sequences)
    conformation_structures = []
    # Use DataLoader
    # Create an ID for every sequence generated
    model.eval()
    with torch.no_grad():
        for i, item in enumerate(batch):
            result = []
            for j in range(samples):
                batch = collate_fn([item])
                batch = tensor_tree_map(lambda x: x.cuda(), batch)  
                prots = model.inference(batch, as_protein=True, noisy_first=noisy_first,
                            no_diffusion=no_diffusion, schedule=schedule, self_cond=self_cond)
                result.append(prots[-1])

            pdb_str = protein.prots_to_pdb(result)
            protein_structure = pdb_string_to_structure(pdb_str)
            conformation_structures.append(protein_structure)
    
    return conformation_structures

def create_batch(AA_sequences=[]):
    batch = []
    for seq in AA_sequences:
        entry= {
            'name': 'id', # TODO Use ID here
            'seqres': seq,
            'aatype': seq_to_tensor(seq),
            'residue_index': torch.arange(len(seq)),
            'pseudo_beta_mask': torch.ones(len(seq)),
            'seq_mask': torch.ones(len(seq))
        }
        make_atom14_masks(entry)
        batch.append(entry)
        
    return batch

def pdb_string_to_structure(pdb_string):
    pdb_string = pdb_string.encode() # Convert to bytes
    pdb_io = io.BytesIO(pdb_string) 
    with gzip.GzipFile(fileobj=pdb_io, mode='r') as fake_file:
        parser = PDBParser()
        structure = parser.get_structure("TEMP", fake_file)

    return structure





