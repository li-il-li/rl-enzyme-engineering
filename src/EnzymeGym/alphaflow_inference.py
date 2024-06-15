import torch
from io import StringIO
from Bio.PDB import PDBParser
import numpy as np
from models.AlphaFlow.alphaflow.model.wrapper import ESMFoldWrapper
from models.AlphaFlow.alphaflow.data.data_modules import collate_fn
from models.AlphaFlow.alphaflow.utils.tensor_utils import tensor_tree_map
from models.AlphaFlow.alphaflow.config import model_config
from models.AlphaFlow.alphaflow.utils.protein import prots_to_pdb 
from models.AlphaFlow.alphaflow.data.inference import seq_to_tensor
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
        pdb_files = []
        for i, item in enumerate(batch):
            print("Working on Batch")
            result = []
            for j in range(samples):
                print("Creating sample")
                batch = collate_fn([item])
                batch = tensor_tree_map(lambda x: x.cuda(), batch)  
                prots = model.inference(batch, as_protein=True, noisy_first=noisy_first,
                            no_diffusion=no_diffusion, schedule=schedule, self_cond=self_cond)
                result.append(prots[-1])

            pdb_str = prots_to_pdb(result)
            protein_structure = pdb_string_to_structure(pdb_str)
            conformation_structures.append(protein_structure)
            
            # While testing, write out every pdb -> maybe async in future
            file_name = f'{item["name"]}.pdb'
            with open(file_name, 'w') as f:
                f.write(prots_to_pdb(result))
            pdb_files.append(file_name)
    
    return conformation_structures, pdb_files

def create_batch(AA_sequences=[]):
    batch = []
    for i,seq in enumerate(AA_sequences):
        entry= {
            'name': i,
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
    pdb_io = StringIO(pdb_string)  # use StringIO instead of BytesIO
    parser = PDBParser()
    structure = parser.get_structure("TEMP", pdb_io)

    return structure





