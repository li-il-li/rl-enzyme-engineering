import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from copy import copy
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import functools
# Hydra
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
# AlphaFlow
from ProteinLigandGym.env.alphaflow_inference import init_esmflow, generate_conformation_ensemble
# FABind+
from fabind_plus_inference import init_fabind, prepare_ligand, create_FABindPipelineDataset, dock_proteins_ligand
# DSMBind
from dsmbind_inference import init_DSMBind, DrugAllAtomEnergyModel
# BIND
from ProteinLigandGym.env.bind_inference import init_BIND, predict_binder


amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

log = logging.getLogger(__name__)


class ProteinLigandInteractionEnv(AECEnv):

    metadata = {
        "name": "protein_ligand_gym_v0",
        "render_modes": ["human"]
    }

    def __init__(self, render_mode=None,
                 wildtype_aa_seq: str = 'AA',
                 ligand_smile: str = 'SMILE',
                 device = 'cuda',
                 config={}):
        log.debug("Initializing environment...")
        
        # Hydra Config
        self.config = config

        log.debug(f"Preparing Ligand: {ligand_smile}")
        # Ligand
        self.ligand_dict = {}
        self.ligand_dict['smile'] = ligand_smile
        self.ligand_dict['mol_structure'], self.ligand_dict['mol_features'] = prepare_ligand(ligand_smile)

        # Protein
        self.wildtype_aa_seq = wildtype_aa_seq

        # Models
        self.device = device
        log.debug("Loading sequence based binding affinity model ...")
        self.ba_model, self.esm_model, self.esm_tokeniser = init_BIND(device) # still small model

        #log.info("Loading folding model ...")
        #self.folding_model = init_esmflow(ckpt = config.alphaflow.ckpt, device=device)
        #log.info("Loading docking model ...")
        #self.docking_model, self.structure_tokenizer, self.structure_alphabet = init_fabind(device=device)
        #log.info("Loading binding affinity prediction model ...")
        #self.ba_model = init_DSMBind(device=device)
        
        # PettingZoo Env
        self.timestep = None
        self.possible_agents = ["mutation_site_picker", "mutation_site_filler"]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
        # Action space is the same for both agents (Tianshou limitation)
        action_space = spaces.Box(
            low=np.array([0] * (len(self.wildtype_aa_seq) + 2)),
            high=np.array([len(amino_acids)-1] * len(self.wildtype_aa_seq) + [len(self.wildtype_aa_seq) - 1] * 2),
            dtype=np.int32
        )
        self._action_spaces = {
            "mutation_site_picker": action_space,
            "mutation_site_filler": action_space
        }
        log.debug(f"Shape action space: {self._action_spaces['mutation_site_picker'].shape}")

        self._observation_spaces = {
            "mutation_site_picker": spaces.Dict(
                {
                    "mutation_aa_seq": spaces.Text(min_length=len(self.wildtype_aa_seq), max_length=len(self.wildtype_aa_seq)),
                    "mutation_site": spaces.Box(low = 0,high = len(self.wildtype_aa_seq)-1,shape = (2,),dtype=np.uint32),
                    "protein_ligand_conformation_latent": spaces.Box(low=-100.0, high=100.0, shape=(2,2), dtype=np.float32)
                }
            ),
            "mutation_site_filler": spaces.Dict(
                {
                    "mutation_aa_seq": spaces.Text(min_length=len(self.wildtype_aa_seq), max_length=len(self.wildtype_aa_seq)),
                    "mutation_site": spaces.Box(low = 0,high = len(self.wildtype_aa_seq)-1,shape = (2,),dtype=np.uint32),
                    "protein_ligand_conformation_latent": spaces.Box(low=-100.0, high=100.0, shape=(2,2), dtype=np.float32)
                }
            )
        }
        
        self.render_mode = render_mode


    def reset(self, seed=None, options=None):
        log.info("Executing 'reset' ...")

        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self.mutant_aa_seq = self.wildtype_aa_seq
        self.mutation_site = np.zeros(2,)
        self.protein_ligand_conformation_latent = np.zeros((2,2), dtype=np.float32)
        self.binding_affinity = 0

        self.observations = self._get_obs()
        self.infos = self._get_infos()
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        return self.observations, self.infos

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return self.observations[agent]

    def step(self, action):
        log.debug(f"Action space: {self._action_spaces['mutation_site_picker'].shape}")
        log.debug(f"Executing action: {action}")
        
        
        if self.agent_selection == "mutation_site_picker":
            log.debug(f"Agent in execution: {self.agent_selection}")
            action = action[-2:]
            self.mutation_site = np.sort(action)
            log.debug(f"Filling : {self.mutation_site}")
            #aa_seq_hole_start_idx, aa_seq_hole_end_idx = np.sort(action)

        elif self.agent_selection == "mutation_site_filler": 
            log.info(f"Agent in execution: {self.agent_selection}")
            self.mutant_aa_seq = self.action_to_aa_sequence(action)
            
            # TODO add if else for structure or sequence based training

            #log.info("Generate conformations ...")
            #conformation_structures, pdb_files = generate_conformation_ensemble(self.folding_model,
            #                                                         self.config,
            #                                                         [self.mutant_aa_seq])
            #self.conformation_structures = conformation_structures
            #
            #log.info("Dock Proteins to ligand ...")
            #fabind_dataset = create_FABindPipelineDataset(conformation_structures,
            #                                              self.ligand_dict,
            #                                              self.structure_tokenizer,
            #                                              self.structure_alphabet)
            #protein_ligand_conformations_mols = dock_proteins_ligand(fabind_dataset, self.docking_model, self.device)
            #
            #self.binding_affinity = self.ba_model.virtual_screen(pdb_files[0], protein_ligand_conformations_mols)
            
            score = predict_binder(self.ba_model, self.esm_model, self.esm_tokeniser, self.device,
                                   [self.mutant_aa_seq], self.ligand_dict['smile'])
            
            self.binding_affinity = score[0]['non_binder_prob']

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
        
         # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            log.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == 2:

            def insert_char(string, char, indexes):
                pos_cor = 0
                for i, index in enumerate(indexes):
                    c = pos_cor + i
                    string = string[:c] + char + string[c:]
                    ++pos_cor
                return string

            sequence_edit = insert_char(self.mutant_aa_seq, "|", self.mutation_site)
            string = f"{sequence_edit}"

        else:
            string = "!!!!!!!!!!!!   Episode finished   !!!!!!!!!!!!"

        log.info(string)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        log.info(f"Getting action space for agent {agent}.")
        log.info(f"Action space {self._action_spaces[agent]}.")
        return self._action_spaces[agent]

    def _get_obs(self):
        
        mask = np.eye(len(self.wildtype_aa_seq)+2, dtype=bool)
        mask[-2:, -2:] = False
        #mask = np.ones(len(self.wildtype_aa_seq)+2, dtype=bool)
        #mask[-2:] = False

        mutation_site_picker_mask = ~mask
        mutation_site_filler_mask = mask
        
        log.info(f"Mutation Mask Shape: {mutation_site_filler_mask.shape}")
        log.info(f"Mutation Mask: {mutation_site_filler_mask}")
        return {
            "mutation_site_picker": {
                "agent_id": self.agents[0],
                "obs": {
                    "mutation_aa_seq": self.mutant_aa_seq,
                    "protein_ligand_conformation_latent": self.protein_ligand_conformation_latent,
                },
                "mask": mutation_site_picker_mask
            },
            "mutation_site_filler": {
                "agent_id": self.agents[1],
                "obs": {
                    "mutation_aa_seq": self.mutant_aa_seq,
                    "mutation_site": self.mutation_site,
                },
                "mask": mutation_site_filler_mask
            }
        }

    def _get_infos(self):
        return {
            "mutation_site_picker": {},
            "mutation_site_filler": {}
        }
    
    def action_to_aa_sequence(self,action):
        lookup_table_int_to_aa = {idx: amino_acid for idx, amino_acid in enumerate(amino_acids)}
        return ''.join(lookup_table_int_to_aa[act] for act in action)

    def aa_sequence_to_action(self,aa_sequence):
        lookup_table_aa_to_int = {amino_acid: np.uint32(idx) for idx, amino_acid in enumerate(amino_acids)}
        return np.array([lookup_table_aa_to_int[aa] for aa in aa_sequence], dtype=np.uint32)