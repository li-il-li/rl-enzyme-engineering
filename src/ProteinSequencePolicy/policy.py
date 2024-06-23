from tianshou.policy import BasePolicy
from tianshou.data import Batch
from evodiff.pretrained import OA_DM_38M, OA_DM_640M
from evodiff.conditional_generation import inpaint_simple
import numpy as np
import logging
import ProteinSequencePolicy.fsa as fsa
import gymnasium as gym

log = logging.getLogger(__name__)

class ProteinSequencePolicy(BasePolicy):
    def __init__(
            self,
            action_space: gym.Space,
            device,
        ):

        super().__init__(
            action_space=action_space
        )

        log.debug("Loading sequence model...")
        self.device = device
        self.sequence_model, self.sequenze_tokenizer = self._init_evodiff(device)

    def forward(self, batch, state=None, model=None):
        """Compute action over the given batch data."""

        n = len(batch.obs)  # number of states in the batch

        #log.info(f"ProtSeqPolicy Batch Observation: {batch.obs}")
        #log.info(f"ProtSeqPolicy Batch: {batch}")

        log.debug(f"Sample sequence...")
        mutant_sequence = batch.obs.mutation_aa_seq[0]
        mutation_site_start_idx, mutation_site_end_idx = batch.obs.mutation_site[0].astype(int)
        
        log.debug(f"Mutant sequence: {mutant_sequence}")
        log.debug(f"Mutation Sites Batch Ops: {mutation_site_start_idx, mutation_site_end_idx}")
        
        sample, entire_sequence, generated_idr = inpaint_simple(
            self.sequence_model,
            mutant_sequence,
            mutation_site_start_idx, mutation_site_end_idx,
            tokenizer=self.sequenze_tokenizer,
            device=self.device
        )

        log.debug(f"Mutation: {entire_sequence}")
        
        encoded_sequence = fsa.aa_sequence_to_action(entire_sequence)
        
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


