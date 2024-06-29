from tianshou.policy import BasePolicy
from tianshou.data import Batch
from evodiff.pretrained import OA_DM_38M, OA_DM_640M
#from evodiff.conditional_generation import inpaint_simple
from .evodiff_inference import inpaint
import numpy as np
import logging
import ProteinSequencePolicy.fsa as fsa
import gymnasium as gym

log = logging.getLogger(__name__)

class ProteinSequencePolicy(BasePolicy):
    def __init__(
            self,
            sequence_encoder,
            action_space: gym.Space,
            device,
        ):

        super().__init__(
            action_space=action_space
        )

        log.debug("Loading sequence model...")
        self.device = device
        self.sequence_model, self.sequenze_tokenizer = self._init_evodiff(device)
        self.sequence_enocder = sequence_encoder

    def forward(self, batch, state=None, model=None):
        """Compute action over the given batch data."""

        n = len(batch.obs)  # number of states in the batch

        #log.info(f"ProtSeqPolicy Batch Observation: {batch.obs}")
        #log.info(f"ProtSeqPolicy Batch: {batch}")

        log.debug(f"Sample sequence...")
        mutant_sequence = batch.obs.mutation_aa_seq[0]
        #log.info(f"Mutant sequence: {mutant_sequence}")
        #log.info(f"Mutation Mask: {batch.obs.mutation_site[0]}")
        mutation_mask = batch.obs.mutation_site[0]
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
        pass
    
    def _init_evodiff(self, device):
        # TODO change back to big model
        model, collater, tokenizer, scheme = OA_DM_38M() #OA_DM_640M()
        model.to(device)

        return model, tokenizer


