# %%
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import Counter
import torch
from esm.pretrained import esmfold_v1
import biotite.structure as struc
import biotite.structure.io as bsio
import biotite.structure.io as strucio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# %%

sequence_file = '/root/projects/rl-enzyme-engineering/results/data/sequence_rl_low_ent/top_sequences.json'

# Load the JSON data
with open(sequence_file, 'r') as f:
    data = json.load(f)

# Sort the data based on the second element (index 1) in descending order
sorted_data = sorted(data, key=lambda x: x[1], reverse=True)

# Get the top 100 elements
top_100 = sorted_data[:100]


# %%
# Assuming the data is stored in a variable called 'data'
# Extract sequences and rewards
sequences = [item[0] for item in top_100]
rewards = [item[1] for item in top_100]

# Create a DataFrame
df = pd.DataFrame({'Sequence': sequences, 'Reward': rewards})

# Truncate sequences to first 50 characters for better visibility
df['Truncated Sequence'] = df['Sequence'].str[:120] + '...'

# Create a figure and axis
fig, ax = plt.subplots(figsize=(15, len(df) * 0.5))  # Increased figure width

# Hide axes
ax.axis('tight')
ax.axis('off')

# Define column widths (as fractions of table width)
col_widths = [0.8, 0.2]  # 80% width for sequence, 20% for reward

# Create table with specified column widths
table = ax.table(cellText=df[['Truncated Sequence', 'Reward']].values,
                 colLabels=['Sequence', 'Reward'],
                 cellLoc='left',
                 loc='center',
                 colWidths=col_widths)

# Adjust table style
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Save the figure
plt.savefig('sequence_reward_table.png', bbox_inches='tight', dpi=300)
plt.close()

print("Table saved as 'sequence_reward_table.png'")

# %%

def analyze_mutations(data, wildtype):
    sequences = [item[0] for item in data]
    rewards = [item[1] for item in data]
    
    # Function to compare sequence with wildtype
    def compare_to_wildtype(seq):
        return ''.join([s if s != w else '.' for s, w in zip(seq, wildtype)])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Sequence': sequences,
        'Reward': rewards,
        'Mutations': [compare_to_wildtype(seq) for seq in sequences]
    })
    
    # Count mutations per position
    mutation_counts = Counter()
    for mutations in df['Mutations']:
        mutation_counts.update([i for i, m in enumerate(mutations) if m != '.'])
    
    # Calculate statistics
    df['Mutation_Count'] = df['Mutations'].apply(lambda x: sum(1 for m in x if m != '.'))
    avg_mutations = df['Mutation_Count'].mean()
    correlation = df['Mutation_Count'].corr(df['Reward'])
    
    # Most common mutations
    common_mutations = Counter(m for mutations in df['Mutations'] for m in mutations if m != '.')
    
    # Visualizations
    plt.figure(figsize=(12, 8))
    
    # 1. Mutation frequency by position
    plt.subplot(3, 2, (1,2))
    sns.barplot(x=list(mutation_counts.keys()), y=list(mutation_counts.values()))
    plt.title('Mutation Frequency by Position')
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    
    # 2. Mutation count vs Reward
    plt.subplot(3, 2, 3)
    sns.scatterplot(data=df, x='Mutation_Count', y='Reward')
    plt.title('Mutation Count vs Reward')
    
    # 3. Top 10 most common mutations
    plt.subplot(3, 2, 4)
    common_mutations_df = pd.DataFrame.from_dict(common_mutations, orient='index').sort_values(0, ascending=False).head(10)
    sns.barplot(x=common_mutations_df.index, y=common_mutations_df[0])
    plt.title('Top 10 Most Common Mutations')
    plt.xlabel('Mutation')
    plt.ylabel('Frequency')
    
    # 4. Mutation heatmap
    plt.subplot(3, 2, (5,6))
    mutation_matrix = np.array([[1 if m != '.' else 0 for m in mutations] for mutations in df['Mutations']])
    sns.heatmap(mutation_matrix, cmap='YlOrRd', cbar_kws={'label': 'Mutation'})
    plt.title('Mutation Heatmap')
    plt.xlabel('Position')
    plt.ylabel('Sequence')
    
    plt.tight_layout()
    plt.savefig('mutation_analysis.png')
    plt.close()
    
    return {
        'avg_mutations': avg_mutations,
        'correlation': correlation,
        'common_mutations': common_mutations,
        'mutation_counts': mutation_counts
    }

# Run the analysis
wildtype = 'MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA'
results = analyze_mutations(top_100, wildtype)

print(f"Average mutations per sequence: {results['avg_mutations']:.2f}")
print(f"Correlation between mutation count and reward: {results['correlation']:.2f}")
print("Top 5 most common mutations:", dict(results['common_mutations'].most_common(5)))
print("Mutation analysis visualizations saved as 'mutation_analysis.png'")

# %%
def export_sequences_to_fasta(data, filename='sequences.fasta'):
    """
    Export sequences to a FASTA file.
    
    :param data: List of [sequence, reward] pairs
    :param filename: Name of the output FASTA file
    """
    records = []
    for i, (sequence, reward) in enumerate(data):
        # Create a SeqRecord for each sequence
        record = SeqRecord(
            Seq(sequence),
            id=f"seq_{i+1}",
            description=f"reward={reward:.4f}"
        )
        records.append(record)
    
    # Write the records to a FASTA file
    with open(filename, "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")
    
    print(f"Sequences exported to {filename}")


# %%
top_100 = [sublist[:-1] for sublist in data]
# %%
top_100

# %%
export_sequences_to_fasta(top_100, 'my_sequences.fasta')

# %%
model = esmfold_v1()
model = model.eval().cuda()

# %%
export_sequences_to_fasta([[wildtype,0]], 'wildtype.fasta')

# %%
len(top_100)

# %%
df = pd.DataFrame(top_100, columns=['sequence', 'reward'])

# %%
carp = pd.read_csv('./data/sequence_rl_low_ent/COMPSS/sequence/carp_640M_logp.tsv', sep='\t')
esm = pd.read_csv('./data/sequence_rl_low_ent/COMPSS/sequence/esm_results.tsv', sep='\t')
esm6 = pd.read_csv('./data/sequence_rl_low_ent/COMPSS/sequence/esm_results6.tsv', sep='\t')

# %%
df['carp640'] = carp['logp']
df['esm'] = esm['score']
df['esm6'] = esm['score']
# %%
df

# %%
df['sequence'] = df.iloc[:, 0].astype(str)
numerical_data = df.iloc[:, 1:].astype(float)

# Standardize the numerical data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Calculate the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])

# Plot the explained variance ratio
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Component')
plt.show()

# Plot the first two principal components
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title('PCA: First Two Principal Components')
plt.show()

# Print the explained variance ratio
print("Explained Variance Ratio:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {ratio:.4f}")

# Print the feature loadings
print("\nFeature Loadings:")
feature_loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
    index=numerical_data.columns
)
print(feature_loadings)



# %%
pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
pca_df['sequence'] = df['sequence']

top_20_pc1 = pca_df.nlargest(20, 'PC1') # models
top_20_pc2 = pca_df.nlargest(20, 'PC2') # reward

# %%
top_20_pc1

# %%
top_20_pc2

# %%
# Function to fold a single sequence
def fold_sequence(sequence):
    with torch.no_grad():
        output = model.infer_pdb(sequence)
        return output
        
for _, row in top_20_pc1.iterrows():
    seq = row['sequence']
    pdb_string = fold_sequence(seq)
    
    # Save the PDB file
    with open(f"./data/sequence_rl_low_ent/pdb/pc1/{row.name}.pdb", "w") as f:
        f.write(pdb_string)
    print(f"Folded structure {row.name} saved as {row.name}.pdb")

# %%
for _, row in top_20_pc2.iterrows():
    seq = row['sequence']
    pdb_string = fold_sequence(seq)
    
    # Save the PDB file
    with open(f"./data/sequence_rl_low_ent/pdb/pc2/{row.name}.pdb", "w") as f:
        f.write(pdb_string)
    print(f"Folded structure {row.name} saved as {row.name}.pdb")


# %%
pdb_string = fold_sequence(wildtype)

# Save the PDB file
with open(f"./data/sequence_rl_low_ent/pdb/wildtype.pdb", "w") as f:
    f.write(pdb_string)


# %%
struct = bsio.load_structure("./data/sequence_rl_low_ent/pdb/wildtype.pdb", extra_fields=["b_factor"])
print(struct.b_factor.mean())  # this will be the pLDDT

# %%
import os

directory = './data/sequence_rl_low_ent/pdb/pc1/'

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        struct = bsio.load_structure(file_path, extra_fields=["b_factor"])
        print(struct.b_factor.mean())  # this will be the pLDDT

# %%
directory = './data/sequence_rl_low_ent/pdb/pc2/'

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        struct = bsio.load_structure(file_path, extra_fields=["b_factor"])
        print(struct.b_factor.mean())  # this will be the pLDDT


# %%
top_100[141]

# %%
df

# %%
def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Remove dominated points
            is_efficient[i] = True
    return is_efficient

# Get the Pareto optimal solutions
costs = df[['reward', 'esm']].values
pareto_optimal = df[is_pareto_efficient(-costs)]  # Use negative because we want to maximize

# Sort by reward and esm (both descending) to get the "top" solutions
top_10_pareto = pareto_optimal.sort_values(['reward', 'esm'], ascending=[False, False]).head(10)

print(top_10_pareto)

# %%
def fold_sequence(sequence):
    with torch.no_grad():
        output = model.infer_pdb(sequence)
        return output
        
for _, row in top_10_pareto.iterrows():
    seq = row['sequence']
    pdb_string = fold_sequence(seq)
    
    # Save the PDB file
    with open(f"./data/sequence_rl_low_ent/pdb/paretto/{row.name}.pdb", "w") as f:
        f.write(pdb_string)
    print(f"Folded structure {row.name} saved as {row.name}.pdb")
    
# %%
top_10_pareto['pLDDT'] = None
# %% 
directory = './data/sequence_rl_low_ent/pdb/paretto/'

for index, row in top_10_pareto.iterrows():
    file_path = os.path.join(directory, f"{row.name}.pdb")
    struct = bsio.load_structure(file_path, extra_fields=["b_factor"])
    plddt = struct.b_factor.mean()
    top_10_pareto.at[index, 'pLDDT'] = plddt 

# %%
top_10_pareto

# %%
for index, row in top_10_pareto.iterrows():
    print(row.name)
    print(row['sequence'])

# %%
diffdock_prob = [
    '',
    
]