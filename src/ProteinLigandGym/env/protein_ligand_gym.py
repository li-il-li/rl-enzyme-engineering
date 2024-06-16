import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from copy import copy
from pettingzoo import AECEnv
import functools
# Hydra
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
# EvoDiff
from evodiff.pretrained import OA_DM_38M, OA_DM_640M
from evodiff.conditional_generation import inpaint_simple
# AlphaFlow
from ProteinLigandGym.env.alphaflow_inference import init_esmflow, generate_conformation_ensemble
# FABind+
from fabind_plus_inference import init_fabind, prepare_ligand, create_FABindPipelineDataset, dock_proteins_ligand
# DSMBind
from dsmbind_inference import init_DSMBind, DrugAllAtomEnergyModel

log = logging.getLogger(__name__)


class ProteinLigandInteractionEnv(AECEnv):

    metadata = {
        "name": "protein_ligand_gym_v0",
    }

    def __init__(self, render_mode=None,
                 wildtype_aa_seq: str = 'AA',
                 ligand_smile: str = 'SMILE',
                 device = 'cuda',
                 config={}):
        log.info("Initializing environment...")
        
        # Hydra Config
        self.config = config

        log.info(f"Preparing Ligand: {ligand_smile}")
        # Ligand
        self.ligand_dict = {}
        self.ligand_dict['smile'] = ligand_smile
        self.ligand_dict['mol_structure'], self.ligand_dict['mol_features'] = prepare_ligand(ligand_smile)

        # Protein
        self.wildtype_aa_seq = wildtype_aa_seq

        # Models
        self.device = device
        log.info("Loading sequence model ...")
        self.sequence_model, self.sequenze_tokenizer = self._init_evodiff()
        log.info("Loading folding model ...")
        self.folding_model = init_esmflow(ckpt = config.alphaflow.ckpt, device=device)
        log.info("Loading docking model ...")
        self.docking_model, self.structure_tokenizer, self.structure_alphabet = init_fabind(device=device)
        log.info("Loading binding affinity prediction model ...")
        self.ba_model = init_DSMBind(device=device)
        
        # RL
        self.timestep = None
        self.possible_agents = ["mutation_site_picker", "mutation_site_filler"]
        
        self.action_space = {
            "mutation_site_picker": spaces.Box(low = 0,high = len(self.wildtype_aa_seq),shape = (2,),dtype=np.int32),
            "mutation_site_filler": spaces.Text(min_length = len(self.wildtype_aa_seq), max_length = len(self.wildtype_aa_seq)) # Use AA charset
        } 
        
        self.observation_space = {
            "mutation_site_picker": spaces.Dict(
                {
                    "mutation_aa_seq": spaces.Text(min_length=len(self.wildtype_aa_seq), max_length=len(self.wildtype_aa_seq)),
                    "protein_ligand_conformation_latent": spaces.Box(low=-100.0, high=100.0, shape=(2,2), dtype=np.float32)
                }
            ),
            "mutation_site_filler": spaces.Dict(
                {
                    "mutation_aa_seq": spaces.Text(min_length=len(self.wildtype_aa_seq), max_length=len(self.wildtype_aa_seq)),
                    "mutation_site": spaces.Box(low = 0,high = len(self.wildtype_aa_seq),shape = (2,),dtype=np.int32)
                }
            )
        }

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.mutant_aa_seq = self.wildtype_aa_seq
        self.mutation_site = np.zeros(2,)
        self.protein_ligand_conformation_latent = np.zeros((2,2), dtype=np.float32)
        self.binding_affinity = 0

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observations = self._get_obs()
        infos = self._get_infos()

        return observations, infos

    def step(self, action):
        
        if self.agent_selection == "mutation_site_picker":
            #aa_seq_hole_start_idx, aa_seq_hole_end_idx = np.sort(action)
            self.mutation_site = np.sort(action)

        elif self.agent_selection == "mutation_site_filler": 
            self.mutant_aa_seq = action
        
            #log.info("Sample sequences ...")
            #sample, entire_sequence, generated_idr = inpaint_simple(
            #    self.sequence_model,
            #    self.mutant_aa_seq,
            #    aa_seq_hole_start_idx,
            #    aa_seq_hole_end_idx,
            #    tokenizer=self.sequenze_tokenizer,
            #    device=self.device
            #)
        
            log.info("Generate conformations ...")
            conformation_structures, pdb_files = generate_conformation_ensemble(self.folding_model,
                                                                     self.config,
                                                                     [self.mutant_aa_seq])
            self.conformation_structures = conformation_structures
            
            log.info("Dock Proteins to ligand ...")
            fabind_dataset = create_FABindPipelineDataset(conformation_structures,
                                                          self.ligand_dict,
                                                          self.structure_tokenizer,
                                                          self.structure_alphabet)
            protein_ligand_conformations_mols = dock_proteins_ligand(fabind_dataset, self.docking_model, self.device)
            
            self.binding_affinity = self.ba_model.virtual_screen(pdb_files[0], protein_ligand_conformations_mols)

        rewards = {
            "mutation_site_picker": self.binding_affinity,
            "mutation_site_filler": self.binding_affinity
        }

        # Check termination conditions
        # Check model properties (if folding prop is too low)
        terminations = { "mutation_site_picker": False, "mutation_site_filler": False}
        
        # Check truncation conditions (overwrites termination conditions)
        truncations = { "mutation_site_picker": False, "mutation_site_filler": False}
        if self.timestep > 15: # Make configurable
            truncations = { "mutation_site_picker": True, "mutation_site_filler": True }
        self.timestep += 1

        observations = self._get_obs()
        infos = self._get_infos()
        
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_obs(self):
        return {
            "mutation_site_picker": {
                "mutation_aa_seq": self.mutant_aa_seq,
                "protein_ligand_conformation_latent": self.protein_ligand_conformation_latent
            },
            "mutation_site_filler": {
                "mutation_aa_seq": self.mutant_aa_seq,
                "mutation_site": self.mutation_site
            }
        }

    def _get_infos(self):
        return { 
            "placeholder": 10,
        }
    
    def _init_evodiff(self):
        model, collater, tokenizer, scheme = OA_DM_640M()
        model.to(device=self.device)

        return model, tokenizer