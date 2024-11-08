import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from copy import copy
#from pettingzoo import AECEnv
#from pettingzoo.utils import agent_selector
import functools

# Hydra
from omegaconf import DictConfig, OmegaConf
import hydra
import logging

# BIND
from ProteinLigandGym.env.bind_inference import init_BIND, predict_binder, get_graph

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

log = logging.getLogger(__name__)

amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def encode_aa_sequence(aa_sequence):
    vocabulary = amino_acids 
    lookup_table_aa_to_int = {amino_acid: np.uint32(idx) for idx, amino_acid in enumerate(vocabulary)}
    return np.array([lookup_table_aa_to_int[aa] for aa in aa_sequence], dtype=np.uint32)

class ProteinLigandInteractionEnv(gym.Env):

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

        self.config = config # Hydra config

        # Gym Env Parameters
        self.timestep = None
        self.total_timesteps = 0
        self.max_steps = max_steps
        self.reward = None
        self.terminated = None
        self.truncated = None

        # Environment / World Model (BIND)
        self.ligand_smiles = ligand_smile # ligand
        self.smile_graph = get_graph(self.config.experiment.ligand_smile)
        self.tracker = TopSequencesTracker() # best sequences tracker
        self.wildtype_aa_seq = wildtype_aa_seq # wildtype protein (initial)

        self.device = device
        log.debug("Loading sequence based binding affinity model ...")
        (
            self.ba_model,
            self.esm_model,
            self.esm_tokeniser,
            self.get_conv5_inputs,
            self.get_crossattention4_inputs
        ) = init_BIND(device)
        
        
        # Observation and Action Space
        self.amino_acids_sequence_actions = amino_acids
        self.action_space = spaces.MultiDiscrete(np.array([len(self.amino_acids_sequence_actions)-1] * len(self.wildtype_aa_seq)))
        self.observation_space = spaces.Dict(
            {
                "mutation_aa_seq": spaces.Text(min_length=len(self.wildtype_aa_seq), max_length=len(self.wildtype_aa_seq)),
                "bind_crossattention4_graph_batch": spaces.Box(low=-100.0, high=100.0, shape=[self.smile_graph.x.size()[0]], dtype=np.int64), # Torch dtypes sadly unsupported
                "bind_crossattention4_hidden_states_30": spaces.Box(low=-100.0, high=100.0, shape=[1, len(self.wildtype_aa_seq) + 2, 1280], dtype=np.float32),
                "bind_crossattention4_padding_mask": spaces.Box(low=0, high=1, shape=[1, len(self.wildtype_aa_seq) + 2], dtype=bool),
                "bind_conv5_x": spaces.Box(low=-100, high=100, shape=[self.smile_graph.x.size()[0], 64], dtype=np.float32),
                "bind_conv5_a": spaces.Box(low=-100, high=100, shape=[self.smile_graph.edge_index.size()[0],self.smile_graph.edge_index.size()[1]], dtype=np.int64),
                "bind_conv5_e": spaces.Box(low=-100, high=100, shape=[self.smile_graph.edge_index.size()[1],self.smile_graph.edge_index.size()[0]], dtype=np.float32),
            }
        )
        
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        log.info("Executing 'reset' ...")

        self.timestep = 0
        # TODO get initial value of protein ligand pair:
        self.reward = 0
        self.terminated = False
        self.truncated = False

        # Observation Space Init
        # TODO: This is wrong -> run a single inference step here
        self.mutant_aa_seq = self.wildtype_aa_seq
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

        # TODO: This is wrong -> run a single inference step here
        self.binding_affinity = 0

        self.observations = self._get_obs()
        self.infos = self._get_infos()
        
        return self.observations, self.infos

    def step(self, action):
        log.debug(f"Executing action: {action}")
        
        # Score mutated sequence in BIND
        self.mutant_aa_seq = self.decode_aa_sequence(action)
        score = predict_binder(self.ba_model, self.esm_model, self.esm_tokeniser, self.device,
                               [self.mutant_aa_seq], self.ligand_dict['smile'])
        
        # Get Latent Represetations BIND
        self.binding_affinity = score[0]['non_binder_prob']
        (
            self.crossattention4_graph_batch,
            self.crossattention4_hidden_states,
            self.crossattention4_padding_mask,
            self.bind_conv5_x,
            self.bind_conv5_a,
            self.bind_conv5_e,
        ) = self._get_ba_model_activation()
        
        # Reward
        self.binding_reward = self._calculate_binding_reward()
        self.reward = self.binding_reward

        # Track sequence
        self.tracker.add_sequence(self.mutant_aa_seq, self.binding_reward, 0)

        # Termination
        self.timestep += 1
        self.total_timesteps += 1
        if self.timestep == self.max_steps:
            self.terminated= True

        # Render
        if (self.total_timesteps % self.config.experiment.render_timesteps == 0):
            self.render()

        self.observations = self._get_obs()
        self.infos = self._get_infos()

        return self.observations, self.reward, self.terminated, self.truncated, self.infos


    def render(self):
        #log.info(f"Rendering...")
        if self.render_mode is None:
            log.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        else:    
            #mask = self._mask_string(self.mutant_aa_seq,self.mutation_site)
            sequence = self.mutant_aa_seq
            string = f"""

Step:                   {self.timestep}
Reward:                 {self.reward}  

{sequence}
            """
            
        log.info(string)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """

        # Stop TopSequenceTracker thread
        if hasattr(self, 'tracker'):
            self.tracker.save_queue.put(None)
            self.tracker.save_thread.join()
            
        # Clear any large data structures
        self.ba_model = None
        self.esm_model = None
        self.esm_tokeniser = None
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            
        log.info("Environment closed and resources released.")

    
    def _get_obs(self):
        return {
            {
                "mutation_aa_seq": self.mutant_aa_seq,
                "bind_crossattention4_graph_batch": self.crossattention4_graph_batch,
                "bind_crossattention4_hidden_states_30": self.crossattention4_hidden_states,
                "bind_crossattention4_padding_mask": self.crossattention4_padding_mask,
                "bind_conv5_x": self.bind_conv5_x,
                "bind_conv5_a": self.bind_conv5_a,
                "bind_conv5_e": self.bind_conv5_e,
            }
        }
        
    def _get_infos(self):
        return {}

    def _get_ba_model_activation(self):
        conv5_x, conv5_a, conv5_e = self.get_conv5_inputs()
        crossattention4_graph_batch, crossattention4_hidden_states_30, crossattention4_padding_mask = self.get_crossattention4_inputs()
        return  crossattention4_graph_batch, crossattention4_hidden_states_30, crossattention4_padding_mask, conv5_x, conv5_a, conv5_e
    
    def _decode_aa_sequence(self,action):
        vocabulary = self.amino_acids_sequence_actions
        lookup_table_int_to_aa = {idx: amino_acid for idx, amino_acid in enumerate(vocabulary)}
        return ''.join(lookup_table_int_to_aa[act] for act in action)

    # TODO: reintroduce but now through difference between input sequence and old sequence
    #def _mask_string(self, string, mask):
    #    char_array = np.array(list(string))
    #    char_array[mask == 1] = '_'
    #    masked_string = ''.join(char_array)
    #    return masked_string

    def _calculate_binding_reward(self):
        binding_reward = (1.0 - float(self.binding_affinity))
        return binding_reward