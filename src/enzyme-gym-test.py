# %%
import sys
sys.path.append("/root/projects/rl-enzyme-engineering/src/EnzymeGym/models/AlphaFlow")
sys.path.append("/root/projects/rl-enzyme-engineering/src/EnzymeGym/models/FABind/FABind_plus/fabind")
sys.path.append("/root/projects/rl-enzyme-engineering/src/EnzymeGym/models/DSMBind")
import EnzymeGym
import gymnasium
import numpy as np
from pprint import pprint
from hydra import initialize, compose
import os
import time

# Configuration
with initialize(version_base=None, config_path="../conf/"):
    cfg = compose(config_name='conf_dev',
                  overrides=[
                      "hydra.verbose=true"
                  ],
                  return_hydra_config=True)
    
dir_path = cfg.experiment.directory
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

os.chdir(dir_path)

print(os.getcwd())

# Create Environment
env = gymnasium.make('EnzymeGym/ProteinLigandInteraction-v0',
                     wildtype_aa_seq=cfg.experiment.wildtype_AA_seq,
                     ligand_smile=cfg.experiment.ligand_smile,
                     device=cfg.experiment.device,
                     config=cfg)

# %%
# Reset Environment
observation, info = env.reset()
pprint(observation)
pprint(info)

# Action
start_time = time.time()

action = np.array([3, 5])
observation, reward, terminated, truncated , info = env.step(action)

end_time = time.time()
execution_time = end_time - start_time
print(f"The execution time is {execution_time} seconds.")

pprint(observation)
pprint(info)

# %%
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.Seq import Seq

# define two amino acid sequences
seq1 = Seq("MAGWGSNGS")
seq2 = Seq("MAGCSNGS")

# global alignment
alignments = pairwise2.align.globalxx(seq1, seq2)

# print all the alignments
for a in alignments:
    print(format_alignment(*a))