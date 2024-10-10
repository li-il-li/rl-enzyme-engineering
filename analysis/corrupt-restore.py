# %%
# 1. Select Protein Ligand pair with high affinity based on BindingDB or plinder
# 2. Check that it is newer than Juli this year
# 3. Check affinity score or binding probability of BIND for pair => expected to be high
# 4. Corrupt for a set number of locations eg n = 10
# 5. Run algorithm with stopping condition if either sequence was restored or binding prob. about the hight pair

# %%
import pandas as pd

# %%
binddb_df = pd.read_csv('data/BindingDB_All.tsv', sep='\t', on_bad_lines='skip') 

# %%
for column in binddb_df.columns:
    print(column)