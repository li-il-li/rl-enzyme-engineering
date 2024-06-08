import sys
sys.path.append("/root/projects/rl-enzyme-engineering/src/AlphaFlow")

import torch, tqdm, os, wandb, json, time
import pandas as pd
import pytorch_lightning as pl
import numpy as np
from collections import defaultdict
from AlphaFlow.alphaflow.model.wrapper import ESMFoldWrapper
from AlphaFlow.alphaflow.data.data_modules import collate_fn
from AlphaFlow.alphaflow.utils.tensor_utils import tensor_tree_map
from AlphaFlow.alphaflow.config import model_config
from AlphaFlow.alphaflow.data.inference import seq_to_tensor
from openfold.data.data_transforms import make_atom14_masks

def init_esmflow(ckpt='', device='cuda'):
    ckpt = torch.load(ckpt, map_location=device)
    model = ESMFoldWrapper(**ckpt['hyper_parameters'], training=False)
    model.model.load_state_dict(ckpt['params'], strict=False)
    model = model.to(device)
    model.eval()

    return model

def generate_conformation_ensemble(AA_sequences=[], model: ESMFoldWrapper, cfg):
    samples = cfg.alphaflow.samples
    noisy_first = cfg.alphaflow.noisy_first
    no_diffusion = cfg.alphaflow.no_diffusion
    self_cond = cfg.alphaflow.self_cond
    
    batch = create_batch(AA_sequences)
    # Use DataLoader
    # Create an ID for every sequence generated

    @torch.no_grad()
    
    return



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





