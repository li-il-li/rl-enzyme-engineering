# %%
# Load fasta file
from Bio import SeqIO

def read_fasta_file(file_path):
    sequences = []
    for seq_record in SeqIO.parse(file_path, "fasta"):
        sequences.append(seq_record)
    return sequences

def read_partial_fasta_file(file_path, num_sequences):
    sequences = []
    for i, seq_record in enumerate(SeqIO.parse(file_path, "fasta")):
        sequences.append(seq_record)
        if i+1 >= num_sequences:  # stop after reading num_sequences
            break
    return sequences

sequences = read_partial_fasta_file('data/uniprot-all-enzymes.fasta', 10000)

seqs_shorter_400 = []
for seq in sequences:
    if len(seq) <= 400:
        seqs_shorter_400.append(seq)

for seq in seqs_shorter_400:
    print(seq.)
    print(seq.format('fasta'))
