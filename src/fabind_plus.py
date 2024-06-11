# %%
import sys
sys.path.append("/root/projects/rl-enzyme-engineering/src/EnzymeGym/models/FABind/FABind_plus/fabind")
# %%
import os
import argparse
import torch
#from torch_geometric.loader import DataLoader
from EnzymeGym.models.FABind.FABind_plus.fabind.utils import *
from EnzymeGym.models.FABind.FABind_plus.fabind.utils.parsing import parse_train_args
#from FABind.FABind_plus.fabind.data import get_data
from EnzymeGym.models.FABind.FABind_plus.fabind.models.model import FABindPlus
import sys
import argparse
#from accelerate import Accelerator
#from accelerate import DistributedDataParallelKwargs
#from accelerate.utils import set_seed
import shlex
import time

from tqdm import tqdm

#from FABind.FABind_plus.fabind.utils.fabind_inference_dataset import InferenceDataset
#from FABind.FABind_plus.fabind.utils.inference_mol_utils import write_mol
#from FABind.FABind_plus.fabind.utils.post_optim_utils import post_optimize_compound_coords
import pandas as pd

"""
data_path=/root/projects/rl-enzyme-engineering/data/FABind/pdbbind2020
ckpt_path=/root/projects/rl-enzyme-engineering/ckpts/FABind/FABind_plus/confidence_model.bin
sample_size=10

python fabind/tools/generate_esm2_t33.py ${data_path}

python fabind/test_sampling_fabind.py \
    --batch_size 8 \
    --data-path ${data_path} \
    --resultFolder ./results \
    --exp-name test_exp \
    --ckpt ${ckpt_path} --use-clustering --infer-dropout \
    --sample-size ${sample_size} \
    --symmetric-rmsd ${data_path}/renumber_atom_index_same_as_smiles \
    --save-rmsd-dir ./rmsd_result
"""
# %%
sys.argv = [''] # Fix issues with jupyter notebook -f flag
parser = argparse.ArgumentParser(description='FABind model testing.')

parser.add_argument("--ckpt", type=str, default='../checkpoints/pytorch_model.bin')
parser.add_argument("--data-path", type=str, default="/PDBbind_data/pdbbind2020",
                    help="Data path.")
parser.add_argument("--resultFolder", type=str, default="./result",
                    help="information you want to keep a record.")
parser.add_argument("--exp-name", type=str, default="",
                    help="data path.")
parser.add_argument('--seed', type=int, default=600,
                    help="seed to use.")
parser.add_argument("--batch_size", type=int, default=8,
                    help="batch size.")
parser.add_argument("--write-mol-to-file", type=str, default=None)
parser.add_argument("--infer-logging", action='store_true', default=False)
parser.add_argument("--use-clustering", action='store_true', default=False)
parser.add_argument("--dbscan-eps", type=float, default=9.0)
parser.add_argument("--dbscan-min-samples", type=int, default=2)
parser.add_argument("--choose-cluster-prob", type=float, default=0.5)
parser.add_argument("--save-rmsd-dir", type=str, default=None)
parser.add_argument("--infer-dropout", action='store_true', default=False)
parser.add_argument("--symmetric-rmsd", default=None, type=str, help="path to the raw molecule file")
parser.add_argument("--command", type=str, default=None)
parser.add_argument("--sample-size", type=int, default=1)

parser.add_argument("--post-optim", action='store_true', default=False)
parser.add_argument('--post-optim-mode', type=int, default=0)
parser.add_argument('--post-optim-epoch', type=int, default=1000)
parser.add_argument('--sdf-output-path-post-optim', type=str, default="")

parser.add_argument('--index-csv', type=str, default=None)
parser.add_argument('--pdb-file-dir', type=str, default="")
parser.add_argument('--preprocess-dir', type=str, default="")

test_args = parser.parse_args()
_, train_parser = parse_train_args(test=True)
train_parser.add_argument("--stack-mlp", action='store_true', default=False)
train_parser.add_argument("--confidence-dropout", type=float, default=0.1)
train_parser.add_argument("--confidence-use-ln-mlp", action='store_true', default=False)
train_parser.add_argument("--confidence-mlp-hidden-scale", type=int, default=2)
train_parser.add_argument("--ranking-loss", type=str, default='logsigmoid', choices=['logsigmoid', 'dynamic_hinge'])
train_parser.add_argument("--num-copies", type=int, default=1)
train_parser.add_argument("--keep-cls-2A", action='store_true', default=False)


if test_args.command is not None:
    command = test_args.command
else:
    command = 'fabind/main_fabind.py --stack-mlp --confidence-dropout 0.2 --confidence-mlp-hidden-scale 1 --confidence-use-ln-mlp --batch_size 2 --label baseline --addNoise 5 --resultFolder fabind_reg --seed 224 --total-epochs 1500 --exp-name fabind_plus_regression --coord-loss-weight 1.5 --pair-distance-loss-weight 1 --pair-distance-distill-loss-weight 1 --pocket-cls-loss-weight 1 --pocket-distance-loss-weight 0.05 --pocket-radius-loss-weight 0.05 --lr 5e-5 --lr-scheduler poly_decay --n-iter 8 --mean-layers 5 --hidden-size 512 --pocket-pred-hidden-size 128 --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer --clip-grad --expand-clength-set --cut-train-set --random-n-iter --use-ln-mlp --mlp-hidden-scale 1 --permutation-invariant --use-for-radius-pred ligand --dropout 0.1 --use-esm2-feat --dis-map-thres 15 --pocket-radius-buffer 5 --min-pocket-radius 20'
command = shlex.split(command)

args = train_parser.parse_args(command[1:])
# print(vars(test_args))
for attr in vars(test_args):
    # Set the corresponding attribute in args
    setattr(args, attr, getattr(test_args, attr))
# Overwrite or set specific attributes as needed
args.tqdm_interval = 0.1
args.disable_tqdm = False
args.confidence_inference = True
args.confidence_training = True

# My settings
args.ckpt = '/root/projects/rl-enzyme-engineering/ckpts/FABind/FABind_plus/fabind_plus_best_ckpt.bin'
args.data_path = '/root/projects/rl-enzyme-engineering/data/FABind/pdbbind2020'
args.resultFolder = './result'
args.exp_name = 'experiment1'
args.seed = 224
args.batch_size = 8
args.sample_size = 10
args.post_optim = True

print(args.ckpt)

# %%
# Load model
model = FABindPlus(args, args.hidden_size, args.pocket_pred_hidden_size)
model.load_state_dict(torch.load(args.ckpt), strict=False)
# %%
model.to('cuda')
# %%
model.eval()

# %%
# Load ESM2
import esm
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model_esm.to(device)

def extract_esm_feature(protein, model, alphabet):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                    'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                    'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                    'N': 2, 'Y': 18, 'M': 12}

    num_to_letter = {v:k for k, v in letter_to_num.items()}

    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    data = [
        ("protein1", protein['seq']),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
    token_representations = results["representations"][33][0][1: -1]
    assert token_representations.shape[0] == len(protein['seq'])
    return token_representations

# %%
# Preprocess

# Protein
from pprint import pprint
from Bio.PDB import *
from EnzymeGym.models.FABind.FABind_plus.fabind.utils.inference_pdb_utils import get_clean_res_list, get_protein_structure

pdb_id = "6g3c"

pdb_file = "/root/projects/rl-enzyme-engineering/src/EnzymeGym/models/FABind/FABind_plus/inference_examples/pdb_files/6g3c.pdb"

esm2_dict = {}
protein_dict = {}

parser = PDBParser(QUIET=True)
structure = parser.get_structure(pdb_id, pdb_file)
res_list = get_clean_res_list(structure.get_residues(), verbose=False, ensure_ca_exist=True)
protein_structure = get_protein_structure(res_list)
protein_structure['name'] = pdb_id

seq_embedding = extract_esm_feature(protein_structure, model_esm, alphabet)

protein_dict[pdb_id] = protein_structure
esm2_dict[pdb_id] = seq_embedding

pprint(esm2_dict)


# %%
from EnzymeGym.models.FABind.FABind_plus.fabind.utils.inference_mol_utils import read_smiles, extract_torchdrug_feature_from_mol, generate_conformation
# Ligand
smile = "CC(C)CCN1c2nc(Nc3cc(F)c(O)c(F)c3)ncc2N(C)C(=O)C1(C)C"
mol = read_smiles(smile)
mol = generate_conformation(mol)
molecule_info = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)

pprint(molecule_info)


# %%
# Load data
input_dict = {}
input_dict['protein_esm_feature'] = seq_embedding
input_dict['protein_structure'] = protein_structure
input_dict['molecule'] = mol
input_dict['molecule_smiles'] = smile
input_dict['molecule_info'] = molecule_info
input_dict['ligand_id'] = pdb_id

from fabind_pipeline_dataset import PipelineDataset

dataset =  PipelineDataset([input_dict])

# %%
com_coord_per_sample_list = []
uid_list = []
smiles_list = []
sdf_name_list = []
mol_list = []
com_coord_pred_per_sample_list = []
com_coord_offset_per_sample_list = []

# %%
def post_optim_mol(
    args,
    device,
    data,
    com_coord_pred,
    com_coord_pred_per_sample_list,
    com_coord_offset_per_sample_list,
    com_coord_per_sample_list,
    compound_batch,
    LAS_tmp,
    rigid=False
):
    print("Running post optim")
    post_optim_device='cpu'
    for i in range(compound_batch.max().item()+1):
        print("Post Optim Loop")
        i_mask = (compound_batch == i)
        com_coord_pred_i = com_coord_pred[i_mask]
        com_coord_i = data['compound'].rdkit_coords
        com_coord_pred_center_i = com_coord_pred_i.mean(dim=0).reshape(1, 3)
        print(f"com_coord_i: {com_coord_i}")
        print(f"com_coord_pred_i: {com_coord_pred_i}")
        print(f"LAS_tmp: {LAS_tmp}")
        print(com_coord_pred_i.to(post_optim_device).requires_grad)
        
        if args.post_optim:
            predict_coord, loss, rmsd = post_optimize_compound_coords(
                reference_compound_coords=com_coord_i.to(post_optim_device),
                predict_compound_coords=com_coord_pred_i.to(post_optim_device),
                # LAS_edge_index=(data[i]['complex', 'LAS', 'complex'].edge_index - data[i]['complex', 'LAS', 'complex'].edge_index.min()).to(post_optim_device),
                LAS_edge_index=LAS_tmp[i].to(post_optim_device),
                mode=args.post_optim_mode,
                total_epoch=args.post_optim_epoch,
            )
            predict_coord = predict_coord.to(device)
            predict_coord = predict_coord - predict_coord.mean(dim=0).reshape(1, 3) + com_coord_pred_center_i
            com_coord_pred[i_mask] = predict_coord
        
        com_coord_pred_per_sample_list.append(com_coord_pred[i_mask])
        com_coord_per_sample_list.append(com_coord_i)
        com_coord_offset_per_sample_list.append(data.coord_offset[i])
        
        mol_list.append(data.mol[i])
        uid_list.append(data.uid[i])
        smiles_list.append(data['compound'].smiles[i])
        sdf_name_list.append(data.ligand_id[i] + '.sdf')

    return

# %%
from torch_geometric.loader import DataLoader
from EnzymeGym.models.FABind.FABind_plus.fabind.utils.parsing import parse_train_args
from EnzymeGym.models.FABind.FABind_plus.fabind.utils.post_optim_utils import post_optimize_compound_coords

data_loader = DataLoader(dataset, batch_size=args.batch_size, follow_batch=['x'], shuffle=False, pin_memory=False, num_workers=0)
for i_batch, batch in enumerate(data_loader):
    pprint(batch)
    batch = batch.to(device)
    LAS_tmp = []
    for i in range(len(batch)):
        LAS_tmp.append(batch[i]['compound', 'LAS', 'compound'].edge_index.detach().clone())
    with torch.no_grad():
        stage=1
        com_coord_pred, compound_batch = model.inference(batch)        

    post_optim_mol(
        args,
        device,
        batch,
        com_coord_pred,
        com_coord_pred_per_sample_list,
        com_coord_offset_per_sample_list,
        com_coord_per_sample_list,
        compound_batch,
        LAS_tmp=LAS_tmp
    )

# %%
from rdkit.Geometry import Point3D 

def create_mol(reference_mol, coords):
    mol = reference_mol
    if mol is None:
        raise Exception("Reference mol should not be None.")
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x, y, z = coords[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return mol
# %%
mols = []
info = pd.DataFrame({'uid': uid_list, 'smiles': smiles_list, 'sdf_name': sdf_name_list})
for i in tqdm(range(len(info))):
    save_coords = com_coord_pred_per_sample_list[i] + com_coord_offset_per_sample_list[i]
    mol = create_mol(reference_mol=mol_list[i], coords=save_coords)
    mols.append(mol)

pprint(mols)



# %%
# Visualize
pdb_file = "/root/projects/rl-enzyme-engineering/src/EnzymeGym/models/FABind/FABind_plus/inference_examples/pdb_files/6g3c.pdb"
sdf_file = "/root/projects/rl-enzyme-engineering/src/EnzymeGym/models/FABind/FABind_plus/inference_examples/inference_output/6g3c.sdf"
import nglview as nv

# Create the structures as separate components
structure1 = nv.FileStructure(pdb_file)
structure2 = nv.FileStructure(sdf_file)

# Create a new NGLView widget
view = nv.NGLWidget()

# Add the structures to the view
view.add_component(structure1)
view.add_component(structure2)

view

# %%
import io
from Bio.PDB import PDBIO

def pdb_to_string(structure):
    stream = io.StringIO()
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save(stream)
    return stream.getvalue()

pdb_string = pdb_to_string(structure)

# %%
from rdkit.Chem import AllChem as Chem

def mol_to_sdf_string(mol):
    stream = io.StringIO()
    writer = Chem.SDWriter(stream)
    writer.write(mol)
    writer.flush()
    return stream.getvalue()

sdf_string = mol_to_sdf_string(mols[0])

# %%
import nglview as nv

# Create the structures as components from strings
structure1 = nv.TextStructure(pdb_string, ext='pdb')
structure2 = nv.TextStructure(sdf_string, ext='sdf')

# Create a new NGLView widget
view = nv.NGLWidget()
view.add_component(structure1)
view.add_component(structure2)

# Display the widget
view
