# %%
import requests
import time

import sys
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models/BIND/")
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models")
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env")
sys.path.append("/root/projects/rl-enzyme-engineering/src/")
from bind_inference import init_BIND, predict_binder
# import run function from algorithm
from main import run
from hydra import compose, initialize
from omegaconf import OmegaConf

# %%
class ProteinLigandComplex:

    def __init__(self, pdb_id, bind_model):
        self.pdb_id = pdb_id
        self.bind_model = bind_model

        self.protein_AA_seq = self.__get_prot_AA_seq(self.pdb_id)
        self.ligand_smiles = self.__get_ligand_smiles(self.pdb_id)

        self.bind_score = self.__score_BIND(self.protein_AA_seq, self.ligand_smiles)

    def __str__(self):
        return f"""
        PDB ID:
        {self.pdb_id}

        Ligand SMILES:
        {self.ligand_smiles}
        
        Protein Amino Acid Sequence:
        {self.protein_AA_seq}

        ---------------------------------------
        """

    def __get_prot_AA_seq(self, pdb_id):
        response = requests.get(f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1")
        data = response.json()

        return data['entity_poly']['pdbx_seq_one_letter_code_can']
    
    def __get_ligand_smiles(self, pdb_id):
        response = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}")
        data = response.json()
        ligand_comp_id = data['rcsb_binding_affinity'][0]['comp_id']

        response = requests.get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_comp_id}")
        data = response.json()
        ligand_smiles = data['rcsb_chem_comp_descriptor']['smiles']

        return ligand_smiles
    
    def __score_BIND(self, prot_AA_seq, ligand_SMILES ):
        scores = predict_binder(*self.bind_model, [prot_AA_seq], ligand_SMILES)[0]
        return {
            "pKi": scores["pKi"],
            "pIC50": scores["pIC50"],
            "pKd": scores["pKd"],
            "pEC50": scores["pEC50"],
            "non_binder_prob": scores["non_binder_prob"]
        }


# %%
# Get protein AA-sequence and ligand SMILE based on RCSB ID

validation_complexes_pdb_ids = [
    '3dxg', # Riconuclease uridine 5' phosphate complex dpK 3.3
    '1bcu', # Thrombin Light Chain complexed with Proflavin dpK 4.25
    '1p1q', # Glutamate receptor in complex with AMPA dpK 2.8
    '2BRB', # pyrrolopyrimidine inhibitors of the Chk1 kinase, an oncology target dpK 1.7
    '1o0H', # Ribonuclease A in complex with 5'-ADP dpK 2.4
    '3ebp', # glycogen phosphorylase (GP) in complex with flavopiridol inhibitor dpK 2.3
    '3PRS', # Endothiapepsin in complex with protease inhibitor ritonavir (large ligand) dpK 3.0
    '2YMD', # Engineered Serotonin-Binding Protein engineered to recognize the agonist serotonin dpK 5.5
]

# %%
# Initialize BIND model
device = 'cuda'
ba_model, esm_model, esm_tokeniser , _, _ = init_BIND(device)
bind_model = (ba_model, esm_model, esm_tokeniser, device)

# %%
# Create list with protein ligand complexes
validation_protein_ligand_complexes = [ProteinLigandComplex(pdb_id, bind_model) for pdb_id in validation_complexes_pdb_ids] 

# %%
for protligcomp in validation_protein_ligand_complexes:
    print(protligcomp.pdb_id, protligcomp.bind_score['non_binder_prob'])
    
# %%
# Load config
with initialize(version_base=None, config_path="../"):
    cfg = compose(config_name="conf")#, overrides=["db=mysql", "db.user=me"])
    print(cfg['experiment']['wildtype_AA_seq'])
# %%


# %%
# Define: function with termination condition

# Start execution
# should return the directory of runtime

# Load JSON

# Do comparisson
