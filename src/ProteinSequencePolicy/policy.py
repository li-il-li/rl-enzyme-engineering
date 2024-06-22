from tianshou.policy import BasePolicy
from tianshou.data import Batch
from evodiff.pretrained import OA_DM_38M, OA_DM_640M
from evodiff.conditional_generation import inpaint_simple
import numpy as np
import logging
import ProteinSequencePolicy.fsa as fsa

log = logging.getLogger(__name__)

class ProteinSequencePolicy(BasePolicy):
    def __init__(self, device):
        super().__init__()
        log.info("Loading sequence model...")
        self.device = device
        self.sequence_model, self.sequenze_tokenizer = self._init_evodiff(device)

    def forward(self, batch, state=None, model=None):
        """Compute action over the given batch data."""

        n = len(batch.obs)  # number of states in the batch

        log.info(f"ProtSeqPolicy Batch: {batch}")
        log.info(f"ProtSeqPolicy Batch Observation: {batch.obs}")

        log.info(f"Sample sequence...")
        mutant_sequence = batch.obs.obs.mutation_aa_seq[0]
        mutation_site_start_idx, mutation_site_end_idx = batch.obs.obs.mutation_site[0].astype(int)
        
        log.info(f"Mutant sequence: {mutant_sequence}")
        log.info(f"Mutation sites: {mutation_site_start_idx, mutation_site_end_idx}")
        
        sample, entire_sequence, generated_idr = inpaint_simple(
            self.sequence_model,
            mutant_sequence,
            mutation_site_start_idx, mutation_site_end_idx,
            tokenizer=self.sequenze_tokenizer,
            device=self.device
        )

        log.info(f"Mutation: {entire_sequence}")
        
        encoded_sequence = fsa.aa_sequence_to_action(entire_sequence)
        
        action = self.action_space
        log.info(f"Action: {action}")
        print()
        
        return Batch(act=action) 

    def learn(self, batch, **kwargs):
        """This is a random policy, so no learning is involved."""
        pass
    
    def _init_evodiff(self, device):
        # TODO change back to big model
        model, collater, tokenizer, scheme = OA_DM_38M() #OA_DM_640M()
        model.to(device)

        return model, tokenizer


