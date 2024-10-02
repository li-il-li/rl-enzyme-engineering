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
# BIND
from ProteinLigandGym.env.bind_inference import init_BIND, predict_binder

import heapq
import json
import threading
import queue

class TopSequencesTracker:
    def __init__(self, max_size=1000, filename='top_sequences.json'):
        self.max_size = max_size
        self.filename = filename
        self.sequences = []
        self.load_from_file()
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

    def add_sequence(self, amino_acid, value1, value2):
        # Use value1 directly for max-heap behavior
        if len(self.sequences) < self.max_size:
            heapq.heappush(self.sequences, (value1, amino_acid, value2))
        elif value1 > self.sequences[0][0]:
            heapq.heapreplace(self.sequences, (value1, amino_acid, value2))
        self.save_queue.put(self.get_top_sequences())

    def get_top_sequences(self):
        return sorted([(seq[1], seq[0], seq[2]) for seq in self.sequences],
                      key=lambda x: x[1], reverse=True)

    def _save_worker(self):
        while True:
            data = self.save_queue.get()
            with open(self.filename, 'w') as f:
                json.dump(data, f)
            self.save_queue.task_done()

    def load_from_file(self):
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
            self.sequences = [(value1, amino_acid, value2)
                              for amino_acid, value1, value2 in data]
            heapq.heapify(self.sequences)
        except FileNotFoundError:
            print(f"File {self.filename} not found. Starting with an empty list.")

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

        # Tracker
        self.tracker = TopSequencesTracker()
        
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
                    "bind_crossattention4_graph_batch": spaces.Box(low=-100.0, high=100.0, shape=[26], dtype=np.int64), # Torch dtypes sadly unsupported
                    "bind_crossattention4_hidden_states_30": spaces.Box(low=-100.0, high=100.0, shape=[1, 307, 1280], dtype=np.float32),
                    "bind_crossattention4_padding_mask": spaces.Box(low=0, high=1, shape=[1, 307], dtype=bool),
                    "bind_conv5_x": spaces.Box(low=-100, high=100, shape=[26, 64], dtype=np.float32),
                    "bind_conv5_a": spaces.Box(low=-100, high=100, shape=[2, 52], dtype=np.int64),
                    "bind_conv5_e": spaces.Box(low=-100, high=100, shape=[52, 2], dtype=np.float32),
                }
            ),
            "mutation_site_filler": spaces.Dict(
                {
                    "mutation_aa_seq": spaces.Text(min_length=len(self.wildtype_aa_seq), max_length=len(self.wildtype_aa_seq)),
                    "mutation_site": spaces.MultiDiscrete(np.array([len(self.amino_acids_sequence_actions)-1] * len(self.wildtype_aa_seq))),
                    "protein_ligand_conformation_latent": spaces.Box(low=-100.0, high=100.0, shape=(self.latent_vector_size,), dtype=np.float32),
                    "protein_ligand_protein_sequence": spaces.Box(low=-100.0, high=100.0, shape=(self.latent_vector_size + len(self.wildtype_aa_seq),), dtype=np.float32),
                    "bind_crossattention4_graph_batch": spaces.Box(low=-100.0, high=100.0, shape=[26], dtype=np.int64), # Torch dtypes sadly unsupported
                    "bind_crossattention4_hidden_states_30": spaces.Box(low=-100.0, high=100.0, shape=[1, 307, 1280], dtype=np.float32),
                    "bind_crossattention4_padding_mask": spaces.Box(low=0, high=1, shape=[1, 307], dtype=bool),
                    "bind_conv5_x": spaces.Box(low=-100, high=100, shape=[26, 64], dtype=np.float32),
                    "bind_conv5_a": spaces.Box(low=-100, high=100, shape=[2, 52], dtype=np.int64),
                    "bind_conv5_e": spaces.Box(low=-100, high=100, shape=[52, 2], dtype=np.float32),
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
        self.binding_affinity_struct = 0
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

            self.binding_reward = self._calculate_binding_reward()
            self.rewards = {
                "mutation_site_picker": self.binding_reward,
                "mutation_site_filler": self.binding_reward
            }

            self.tracker.add_sequence(self.mutant_aa_seq, self.binding_reward, 0)

            # Adds .rewards to ._cumulative_rewards
            self._accumulate_rewards()

            # Check truncation conditions (overwrites termination conditions)
            self.truncations = { "mutation_site_picker": False, "mutation_site_filler": False}

            self.timestep += 1

            if self.timestep == self.max_steps:
                self.truncations = { "mutation_site_picker": True, "mutation_site_filler": True }

            self.render()

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
    
    # Create bigger holes in sequence at location -> more control for pLM
    def _expand_mutation_site(self,action):
        hole_size=self.config.agents.filler_plm.self_determination 
        # Find the index of the 1 in the action vector
        one_index = np.where(action == 1)[0][0]
        
        # Calculate the start and end indices for the hole
        start = max(0, one_index - hole_size // 2)
        end = min(len(action), one_index + hole_size // 2 + 1)
        
        # Create a new vector with the expanded hole
        expanded_action = np.zeros_like(action)
        expanded_action[start:end] = 1
        
        return expanded_action

    def _calculate_binding_reward(self):
        binding_reward = (1.0 - float(self.binding_affinity))
        return binding_reward