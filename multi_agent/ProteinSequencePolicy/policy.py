from tianshou.policy.base import BasePolicy, TrainingStats
from tianshou.data import Batch
from evodiff.pretrained import OA_DM_38M, OA_DM_640M
#from evodiff.conditional_generation import inpaint_simple
from .evodiff_inference import inpaint
import numpy as np
import logging
import ProteinSequencePolicy.fsa as fsa
import gymnasium as gym
from dataclasses import dataclass

log = logging.getLogger(__name__)

class ProteinSequencePolicy(BasePolicy):
    def __init__(
            self,
            model_size_parameters,
            sequence_encoder,
            action_space: gym.Space,
            device,
        ):

        super().__init__(
            action_space=action_space
        )

        log.debug("Loading sequence model...")
        self.device = device
        self.sequence_model, self.sequenze_tokenizer = self._init_evodiff(device, model_size_parameters)
        self.sequence_enocder = sequence_encoder

    def forward(self, batch, state=None, model=None):
        """Compute action over the given batch data."""
        n = len(batch.obs.obs)  # number of states in the batch

        #log.info(f"ProtSeqPolicy Batch Observation: {batch.obs}")
        #log.info(f"ProtSeqPolicy Batch: {batch}")

        log.debug(f"Sample sequence...")
        mutant_sequence = batch.obs.obs.mutation_aa_seq[0]
        mutation_mask = batch.obs.obs.mutation_site[0]
        indices = np.where(mutation_mask == 1)[0]
        #log.info(f"Indices: {indices}")
        
        _, entire_sequence  = inpaint(
            self.sequence_model,
            mutant_sequence,
            indices,
            tokenizer=self.sequenze_tokenizer,
            device=self.device
        )

        log.debug(f"Mutation: {entire_sequence}")
        
        encoded_sequence = self.sequence_enocder(entire_sequence)
        
        action_space = self.action_space
        mask = batch.obs.mask[0]
        log.debug(f"Actionspace: {action_space.shape}")
        log.debug(f"Mask: {mask.shape}")
        
        action = np.zeros(action_space.shape)
        action[:len(encoded_sequence)] = encoded_sequence
        log.debug(f"Action: {action}")
        
        return Batch(act=[action]) 

    def learn(self, batch, **kwargs):
        """This is a random policy, so no learning is involved."""
        #log.info(f"Training ProteinSequencePolicy: {batch}")
        
        return TrainingStats(
            train_time=1.5,
            smoothed_loss={'loss1': 0.5, 'loss2': 0.3}
        )
    
    def _init_evodiff(self, device, model_size):
        model, collater, tokenizer, scheme = OA_DM_640M() if model_size == 640 else OA_DM_38M()
        model.to(device)

        return model, tokenizer


