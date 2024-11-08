from tianshou.policy import PPOPolicy
from tianshou.data.types import ObsBatchProtocol, DistBatchProtocol
from tianshou.data import Batch
from typing import Any, Optional
import numpy as np
import torch
from typing import cast

from evodiff.pretrained import OA_DM_38M, OA_DM_640M
from evodiff.utils import Tokenizer

class pLMPPOPolicy(PPOPolicy):
    def __init__(
            self,
            device,
            model_size_parameters,
            sequence_encoder,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.print_prefix = "pLMPPO"
        self.device = device
        self.sequence_model, self.sequenze_tokenizer = self._init_evodiff(device, model_size_parameters)
        self.sequence_encoder = sequence_encoder
    
    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | Batch | np.ndarray | None = None,
        **kwargs: Any,
    ) -> DistBatchProtocol:
        #Forward output: Batch(logits=action_dist_input_BD, act=act_B, state=hidden_BH, dist=dist)
        mutation_site_action_batch = super().forward(batch, state, **kwargs)
        
        sequences = batch.obs.mutation_aa_seq
        mutation_site_indices = mutation_site_action_batch.act.nonzero()[:, 1].reshape(-1, 1).tolist()

        mutated_sequences = []
        for sequence, indices in zip(sequences, mutation_site_indices):
            _, mutated_sequence  = self.inpaint(
                self.sequence_model,
                sequence,
                indices,
                tokenizer=self.sequenze_tokenizer,
                device=self.device
            )
            mutated_sequences.append(self.sequence_encoder(mutated_sequence))

        mutation_site_action_batch.act = torch.tensor(np.array(mutated_sequences, dtype=np.uint8),dtype=torch.uint8, device=self.device)

        return mutation_site_action_batch
    
    def inpaint(self, model, sequence, mask_idx, tokenizer=Tokenizer(), device='cuda'):

        def _replace_at_indices(string, indices):
            char_list = list(string)
            
            for index in indices:
                char_list[index] = '#'
            
            result = ''.join(char_list)

            return result

        all_aas = tokenizer.all_aas

        masked_sequence = _replace_at_indices(sequence, mask_idx)

        tokenized_sequence = torch.tensor(tokenizer.tokenizeMSA(masked_sequence))
        sample = tokenized_sequence.to(torch.long)
        sample = sample.to(device)

        np.random.shuffle(mask_idx)
        with torch.no_grad():
            for i in mask_idx:
                timestep = torch.tensor([0]) # placeholder but not called in model <- TODO: check this
                timestep = timestep.to(device)
                prediction = model(sample.unsqueeze(0), timestep)
                p = prediction[:, i, :len(all_aas)-6]
                p = torch.nn.functional.softmax(p, dim=1)
                p_sample = torch.multinomial(p, num_samples=1)
                sample[i] = p_sample.squeeze()
        untokenized_seq = tokenizer.untokenize(sample)

        return sample, untokenized_seq

    def _init_evodiff(self, device, model_size):
        model, collater, tokenizer, scheme = OA_DM_640M() if model_size == 640 else OA_DM_38M()
        model.to(device)

        return model, tokenizer