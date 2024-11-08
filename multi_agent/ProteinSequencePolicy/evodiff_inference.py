import torch
import numpy as np
from evodiff.utils import Tokenizer

def replace_at_indices(string, indices):
    char_list = list(string)
    
    for index in indices:
        char_list[index] = '#'
    
    result = ''.join(char_list)

    return result

def inpaint(model, sequence, mask_idx, tokenizer=Tokenizer(), device='cuda'):
    all_aas = tokenizer.all_aas

    masked_sequence = replace_at_indices(sequence, mask_idx)

    tokenized_sequence = torch.tensor(tokenizer.tokenizeMSA(masked_sequence))

    sample = tokenized_sequence.to(torch.long)
    sample = sample.to(device)
    np.random.shuffle(mask_idx)
    with torch.no_grad():
        for i in mask_idx:
            timestep = torch.tensor([0]) # placeholder but not called in model
            timestep = timestep.to(device)
            prediction = model(sample.unsqueeze(0), timestep)
            p = prediction[:, i, :len(all_aas)-6]
            p = torch.nn.functional.softmax(p, dim=1)
            p_sample = torch.multinomial(p, num_samples=1)
            sample[i] = p_sample.squeeze()
    untokenized_seq = tokenizer.untokenize(sample)

    return sample, untokenized_seq