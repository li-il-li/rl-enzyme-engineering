# %%
import requests
import time

import sys
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models/BIND/")
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models")
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env")

from bind_inference import init_BIND, predict_binder

# %%
class ProteinLigandComplex:

    def __init__(self, pdb_id, bind_model):
        self.pdb_id = pdb_id
        self.bind_model = bind_model

        self.protein_AA_seq = self.__get_prot_AA_seq(self.pdb_id)
        self.ligand_smiles = self.__get_ligand_smiles(self.pdb_id)

        self.non_binder_probability = self.__score_BIND(self.protein_AA_seq, self.ligand_smiles)

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
    
    def __score_BIND(self, ):
        # Add score function here        
        print('Scoring')


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

device = 'cuda'
ba_model, esm_model, esm_tokeniser , _, _ = init_BIND(device)


# Create list with protein ligand complexes
validation_protein_ligand_complexes = [ProteinLigandComplex(pdb_id, ba_model) for pdb_id in validation_complexes_pdb_ids] 

# %%
for  protligcomp in validation_protein_ligand_complexes:
    print(protligcomp)
    
# %%
from ...src.ProteinLigandGym.env.bind_inference import predict_binder
