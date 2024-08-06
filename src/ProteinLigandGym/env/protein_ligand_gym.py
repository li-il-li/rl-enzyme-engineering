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

NUM_ITERS = 10

log = logging.getLogger(__name__)

class ProteinLigandInteractionEnv(AECEnv):

    metadata = {
        "name": "protein_ligand_gym_v0",
        "render_modes": ["human"]
    }

    def __init__(
        self, render_mode=None,
        wildtype_aa_seq: str = 'AA',
        ligand_smile: str = 'SMILE',
        max_steps = 100,
        device = 'cuda',
        config = {}
    ):
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
        (
            self.ba_model,
            self.get_ba_activations,
            self.latent_vector_size,
            self.esm_model,
            self.esm_tokeniser,
            self.get_conv5_inputs,
            self.get_crossattention4_inputs
        ) = init_BIND(device) # still small model

        log.info("Loading folding model ...")
        self.folding_model = init_esmflow(ckpt = config.alphaflow.ckpt, device=device)
        log.info("Loading docking model ...")
        self.docking_model, self.structure_tokenizer, self.structure_alphabet = init_fabind(device=device)
        log.info("Loading binding affinity prediction model ...")
        self.ba_model_struct = init_DSMBind(device=device)
        
        # PettingZoo Env
        self.timestep = None
        self.max_steps = max_steps
        self.possible_agents = ["mutation_site_picker", "mutation_site_filler"]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.mask = np.ones(len(self.wildtype_aa_seq)-1, dtype=bool)
        
        # Action space is the same for both agents (Tianshou limitation)
        self.amino_acids_sequence_actions = ['0', '1', 'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        action_space = spaces.MultiDiscrete(np.array([len(self.amino_acids_sequence_actions)-1] * len(self.wildtype_aa_seq)))

        self._action_spaces = {
            "mutation_site_picker": action_space,
            "mutation_site_filler": action_space
        }
        log.debug(f"Shape action space: {self._action_spaces['mutation_site_picker'].shape}")

        self._observation_spaces = {
            "mutation_site_picker": spaces.Dict(
                {
                    "mutation_aa_seq": spaces.Text(min_length=len(self.wildtype_aa_seq), max_length=len(self.wildtype_aa_seq)),
                    "mutation_site": spaces.MultiDiscrete(np.array([len(self.amino_acids_sequence_actions)-1] * len(self.wildtype_aa_seq))),
                    "protein_ligand_conformation_latent": spaces.Box(low=-100.0, high=100.0, shape=(self.latent_vector_size,), dtype=np.float32),
                    "protein_ligand_protein_sequence": spaces.Box(low=-100.0, high=100.0, shape=(self.latent_vector_size + len(self.wildtype_aa_seq),), dtype=np.float32),
                    "bind_crossattention4_graph_batch": spaces.Box(low=-100.0, high=100.0, shape=[48], dtype=np.int64), # Torch dtypes sadly unsupported
                    "bind_crossattention4_hidden_states_30": spaces.Box(low=-100.0, high=100.0, shape=[1, 307, 1280], dtype=np.float32),
                    "bind_crossattention4_padding_mask": spaces.Box(low=0, high=1, shape=[1, 307], dtype=np.bool),
                    "bind_conv5_x": spaces.Box(low=-100, high=100, shape=[48, 64], dtype=np.float32),
                    "bind_conv5_a": spaces.Box(low=-100, high=100, shape=[2, 102], dtype=np.int64),
                    "bind_conv5_e": spaces.Box(low=-100, high=100, shape=[102, 2], dtype=np.float32),
                }
            ),
            "mutation_site_filler": spaces.Dict(
                {
                    "mutation_aa_seq": spaces.Text(min_length=len(self.wildtype_aa_seq), max_length=len(self.wildtype_aa_seq)),
                    "mutation_site": spaces.MultiDiscrete(np.array([len(self.amino_acids_sequence_actions)-1] * len(self.wildtype_aa_seq))),
                    "protein_ligand_conformation_latent": spaces.Box(low=-100.0, high=100.0, shape=(self.latent_vector_size,), dtype=np.float32),
                    "protein_ligand_protein_sequence": spaces.Box(low=-100.0, high=100.0, shape=(self.latent_vector_size + len(self.wildtype_aa_seq),), dtype=np.float32),
                    "bind_crossattention4_graph_batch": spaces.Box(low=-100.0, high=100.0, shape=[48], dtype=np.int64), # Torch dtypes sadly unsupported
                    "bind_crossattention4_hidden_states_30": spaces.Box(low=-100.0, high=100.0, shape=[1, 307, 1280], dtype=np.float32),
                    "bind_crossattention4_padding_mask": spaces.Box(low=0, high=1, shape=[1, 307], dtype=np.bool),
                    "bind_conv5_x": spaces.Box(low=-100, high=100, shape=[48, 64], dtype=np.float32),
                    "bind_conv5_a": spaces.Box(low=-100, high=100, shape=[2, 102], dtype=np.int64),
                    "bind_conv5_e": spaces.Box(low=-100, high=100, shape=[102, 2], dtype=np.float32),
                }
            )
        }
        
        self.render_mode = render_mode


    def reset(self, seed=None, options=None):
        log.info("Executing 'reset' ...")

        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.rewards = {agent: 0 for agent in self.agents}
        self.mask_penalty = 0 # remove
        self.binding_reward = 0
        self.clustering_score = 0
        self.large_cluster_penalty = 0
        self.edit_penalty_score = 0
        self.num_edits = 0

        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self.mutant_aa_seq = self.wildtype_aa_seq
        self.mutation_site = np.zeros(len(self.mutant_aa_seq))
        self.protein_ligand_conformation_latent = np.zeros((self.latent_vector_size), dtype=np.float32)
        self.protein_ligand_protein_sequence = np.zeros((self.latent_vector_size + len(self.wildtype_aa_seq)), dtype=np.float32)

        crossattention4_graph_batch_space= self._observation_spaces["mutation_site_picker"]["bind_crossattention4_graph_batch"]
        self.crossattention4_graph_batch = np.zeros(crossattention4_graph_batch_space.shape, dtype=crossattention4_graph_batch_space.dtype)
        crossattention4_hidden_states_space = self._observation_spaces["mutation_site_picker"]["bind_crossattention4_hidden_states_30"]
        self.crossattention4_hidden_states = np.zeros(crossattention4_hidden_states_space.shape, dtype=crossattention4_hidden_states_space.dtype)
        crossattention4_padding_mask_space= self._observation_spaces["mutation_site_picker"]["bind_crossattention4_padding_mask"]
        self.crossattention4_padding_mask = np.zeros(crossattention4_padding_mask_space.shape, dtype=crossattention4_padding_mask_space.dtype)
        bind_conv5_x_space= self._observation_spaces["mutation_site_picker"]["bind_conv5_x"]
        self.bind_conv5_x = np.zeros(bind_conv5_x_space.shape, dtype=bind_conv5_x_space.dtype)
        bind_conv5_a_space = self._observation_spaces["mutation_site_picker"]["bind_conv5_a"]
        self.bind_conv5_a = np.zeros(bind_conv5_a_space.shape, dtype=bind_conv5_a_space.dtype)
        bind_conv5_e_space = self._observation_spaces["mutation_site_picker"]["bind_conv5_e"]
        self.bind_conv5_e = np.zeros(bind_conv5_e_space.shape, dtype=bind_conv5_e_space.dtype)

        self.binding_affinity = 0
        self.number_holes = 0

        self.observations = self._get_obs()
        self.infos = self._get_infos()
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        return self.observations, self.infos

    def observe(self, agent):
        # observation of one agent is the previous state of the other
        observation = self.observations[agent]
        return observation

    def step(self, action):
        log.debug(f"Action space: {self._action_spaces['mutation_site_picker'].shape}")
        log.debug(f"Executing action: {action}")
        log.debug("Step")
        
        if self.agent_selection == "mutation_site_picker":
            log.debug(f"Agent in execution: {self.agent_selection}")
            self.mutation_site = self._expand_mutation_site(action)
            self.number_holes = np.sum(action)
            log.debug(f"Mutation site: {self.mutation_site}")

        elif self.agent_selection == "mutation_site_filler": 
            log.debug(f"Agent in execution: {self.agent_selection}")
            self.mutant_aa_seq = self.decode_aa_sequence(action)
            log.debug(f"Action sequence: {self.mutant_aa_seq}")
            

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
            
            self.binding_affinity = self.ba_model_struct.virtual_screen(pdb_files[0], protein_ligand_conformations_mols)
            
            score = predict_binder(self.ba_model, self.esm_model, self.esm_tokeniser, self.device,
                                   [self.mutant_aa_seq], self.ligand_dict['smile'])
            
            # Here Sequence
            self.binding_affinity = score[0]['non_binder_prob']
            (
                self.protein_ligand_conformation_latent,
                self.crossattention4_graph_batch,
                self.crossattention4_hidden_states,
                self.crossattention4_padding_mask,
                self.bind_conv5_x,
                self.bind_conv5_a,
                self.bind_conv5_e,
            ) = self._get_ba_model_activation()

            aa_seq_encoded = self.encode_aa_sequence(self.mutant_aa_seq).astype(np.float32).reshape(1,-1)
            self.protein_ligand_protein_sequence = np.concatenate((self.protein_ligand_conformation_latent, aa_seq_encoded),axis=1)

            (
                reward,
                self.binding_reward,
                self.clustering_score,
                self.large_cluster_penalty,
                self.edit_penalty_score,
                self.num_edits
            ) = self._calculate_comprehensive_reward(self.mutation_site)
            self.rewards = {
                "mutation_site_picker": self.binding_reward, #reward,
                "mutation_site_filler": self.binding_reward #reward
            }

            # Adds .rewards to ._cumulative_rewards
            self._accumulate_rewards()

            # Check truncation conditions (overwrites termination conditions)
            self.truncations = { "mutation_site_picker": False, "mutation_site_filler": False}

            self.timestep += 1

            if self.timestep == self.max_steps:
                self.truncations = { "mutation_site_picker": True, "mutation_site_filler": True }

            self.render()



        # Check termination conditions
        # Check model properties (if folding prop is too low)
        
        # If uncommented skips actuall mutation
        # if self.mask_penalty >= (1 * self.config.agents.binding_affinity_k):
        #     self.rewards = {
        #         "mutation_site_picker": np.random.randint(0, self.config.agents.binding_affinity_k) - self.mask_penalty,
        #         "mutation_site_filler": np.random.randint(0, self.config.agents.binding_affinity_k) - self.mask_penalty 
        #     }
        #     self.terminations = { "mutation_site_picker": True, "mutation_site_filler": True}
        #     self._accumulate_rewards()
        #     self.render()

        self.observations = self._get_obs()
        self.infos = self._get_infos()

        if any(self.terminations.values()) or all(self.truncations.values()):
            self.agents = []

        self.agent_selection = self._agent_selector.next()
        
        return self.observations, self.rewards, self.truncations, self.terminations, self.infos


    def render(self):
        #log.info(f"Rendering...")
        if self.render_mode is None:
            log.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == 2:
            
            mask = self._mask_string(self.mutant_aa_seq,self.mutation_site)
            sequence = self.mutant_aa_seq
            string = f"""

Step:                   {self.timestep}
Reward:                 {self.rewards[self.agent_selection]}  

Binding Reward:         {self.binding_reward}
Clustering Score:       {self.clustering_score}
Large Cluster Penalty:  {self.large_cluster_penalty}
Edit Penalty Score:     {self.edit_penalty_score}

Num of Edites:          {self.num_edits}

{mask}
{sequence}
            """
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
        return self._action_spaces[agent]
    
    
    def _get_obs(self):
        
        return {
            "mutation_site_picker": {
                "agent_id": self.agents[0],
                "mutation_aa_seq": self.mutant_aa_seq,
                "mutation_site": self.mutation_site,
                "protein_ligand_conformation_latent": self.protein_ligand_conformation_latent,
                "protein_ligand_protein_sequence": self.protein_ligand_protein_sequence,
                "bind_crossattention4_graph_batch": self.crossattention4_graph_batch,
                "bind_crossattention4_hidden_states_30": self.crossattention4_hidden_states,
                "bind_crossattention4_padding_mask": self.crossattention4_padding_mask,
                "bind_conv5_x": self.bind_conv5_x,
                "bind_conv5_a": self.bind_conv5_a,
                "bind_conv5_e": self.bind_conv5_e,
                "mask": self.mask
            },
            "mutation_site_filler": {
                "agent_id": self.agents[1],
                "mutation_aa_seq": self.mutant_aa_seq,
                "mutation_site": self.mutation_site,
                "protein_ligand_conformation_latent": self.protein_ligand_conformation_latent,
                "protein_ligand_protein_sequence": self.protein_ligand_protein_sequence,
                "bind_crossattention4_graph_batch": self.crossattention4_graph_batch,
                "bind_crossattention4_hidden_states_30": self.crossattention4_hidden_states,
                "bind_crossattention4_padding_mask": self.crossattention4_padding_mask,
                "bind_conv5_x": self.bind_conv5_x,
                "bind_conv5_a": self.bind_conv5_a,
                "bind_conv5_e": self.bind_conv5_e,
                "mask": self.mask
            }
        }

    def _get_infos(self):
        return {
            "mutation_site_picker": {
                "number_holes": self.number_holes,
            },
            "mutation_site_filler": {
                "number_holes": self.number_holes,
            }
        }
    
    def _get_ba_model_activation(self):
        dense = self.get_ba_activations()
        conv5_x, conv5_a, conv5_e = self.get_conv5_inputs()
        crossattention4_graph_batch, crossattention4_hidden_states_30, crossattention4_padding_mask = self.get_crossattention4_inputs()
        return  dense, crossattention4_graph_batch, crossattention4_hidden_states_30, crossattention4_padding_mask, conv5_x, conv5_a, conv5_e
    
    def decode_aa_sequence(self,action):
        vocabulary = self.amino_acids_sequence_actions
        lookup_table_int_to_aa = {idx: amino_acid for idx, amino_acid in enumerate(vocabulary)}
        return ''.join(lookup_table_int_to_aa[act] for act in action)

    def encode_aa_sequence(self,aa_sequence):
        vocabulary = self.amino_acids_sequence_actions
        lookup_table_aa_to_int = {amino_acid: np.uint32(idx) for idx, amino_acid in enumerate(vocabulary)}
        return np.array([lookup_table_aa_to_int[aa] for aa in aa_sequence], dtype=np.uint32)
    
    def _mask_string(self, string, mask):
        char_array = np.array(list(string))
        char_array[mask == 1] = '_'
        masked_string = ''.join(char_array)
        return masked_string
    
    def _expand_mutation_site(self,action):
        hole_size=self.config.agents.picker.self_determination 
        # Find the index of the 1 in the action vector
        one_index = np.where(action == 1)[0][0]
        
        # Calculate the start and end indices for the hole
        start = max(0, one_index - hole_size // 2)
        end = min(len(action), one_index + hole_size // 2 + 1)
        
        # Create a new vector with the expanded hole
        expanded_action = np.zeros_like(action)
        expanded_action[start:end] = 1
        
        return expanded_action
    
    def _calculate_reward(self,mask):
        affinity_reward = (1.0 - float(self.binding_affinity))
        #mask_penalty = self._calculate_mask_penalty(mask)
        clustering_reward = self._calculate_clustering_reward(mask)
        
        total_reward = (
            affinity_reward
            + self.config.agents.clustering_weight * clustering_reward
            #- mask_penalty
        )
        
        return total_reward
    
    def _calculate_clustering_reward(self,mask):
        runs = np.diff(np.where(np.concatenate(([mask[0]], mask[:-1] != mask[1:], [True])))[0])
        masked_runs = runs[mask[:-1] == 1]
        if len(masked_runs) == 0:
            return 0
        
        avg_cluster_size = np.mean(masked_runs)
        num_clusters = len(masked_runs)
        
        # Reward larger clusters and fewer clusters
        clustering_score = avg_cluster_size / (num_clusters + 1)  # Adding 1 to avoid division by zero
        return clustering_score
    
    def _calculate_mask_penalty(self, mask):
        threshold = self.config.agents.sequence_edit_target_ratio
        k = self.config.agents.sequence_edit_target_ratio_penalty_k
        ratio = np.mean(mask)
        if ratio == 0:
            return 100
        else:
            return k * (ratio - threshold)**3
    
    def _calculate_comprehensive_reward(self,mask):
        # Config
        max_cluster_size_ratio = self.config.agents.max_cluster_size_ratio
        large_cluster_penalty=self.config.agents.large_cluster_penalty
        edit_penalty=self.config.agents.edit_penalty
        no_edit_penalty=self.config.agents.no_edit_penalty
        binding_affinity_weight=self.config.agents.binding_affinity_weight
        clustering_score_k = self.config.agents.clustering_score_k
        
        sequence_length = len(mask)
        num_edits = np.sum(mask)
        max_cluster_size = int(max_cluster_size_ratio * sequence_length)

        # Early return if no edits
        if num_edits == 0:
            return -no_edit_penalty

        # Binding affinity reward (assuming lower is better)
        binding_reward = binding_affinity_weight * (1.0 - float(self.binding_affinity))

        # Cluster calculation
        changes = np.diff(np.concatenate(([0], mask, [0])))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        masked_runs = ends - starts

        if len(masked_runs) > 0:
            avg_cluster_size = np.mean(masked_runs)
            num_clusters = len(masked_runs)
            
            # Basic clustering score
            clustering_score = clustering_score_k *  (avg_cluster_size / (num_clusters + 1))
            
            # Penalty for clusters larger than max_cluster_size
            large_cluster_penalty = sum(max(0, run - max_cluster_size) for run in masked_runs) * large_cluster_penalty
        else:
            clustering_score = 0
            large_cluster_penalty = 0
        
        # Penalty for total number of edits
        edit_penalty_score = edit_penalty * num_edits
        
        # Final reward
        final_reward = binding_reward + clustering_score - large_cluster_penalty - edit_penalty_score
        
        return final_reward, binding_reward, clustering_score, large_cluster_penalty, edit_penalty_score, num_edits