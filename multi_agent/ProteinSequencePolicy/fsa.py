import numpy as np

amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def action_to_aa_sequence(action):
    lookup_table_int_to_aa = {idx: amino_acid for idx, amino_acid in enumerate(amino_acids)}
    return ''.join(lookup_table_int_to_aa[act] for act in action)

def aa_sequence_to_action(aa_sequence):
    lookup_table_aa_to_int = {amino_acid: np.uint32(idx) for idx, amino_acid in enumerate(amino_acids)}
    return np.array([lookup_table_aa_to_int[aa] for aa in aa_sequence], dtype=np.uint32)