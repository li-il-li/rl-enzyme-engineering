
# %%
import polars as pl

# %%
# Load TSV BidningDB
binddb_df = pl.read_csv('data/BindingDB_All.tsv', separator='\t', truncate_ragged_lines=True) 

# %%
binddb_df = binddb_df.select(['Target Name','Ligand SMILES','Kd (nM)'])
#binddb_df = binddb_df.with_columns(pl.col('Ki (nM)').str.strip_chars().cast(pl.Float64))

# %%
binddb_df_sorted_Ki = binddb_df.sort('Ki (nM)', descending=True, nulls_last=True)

# %%
binddb_df_sorted_Ki.shape

# %%
i = int((binddb_df_sorted_Ki.shape[0]-1)*0.8)
range = 200
complex = binddb_df_sorted_Ki.slice(i-int(range/2), range)
complex.head()

# %%
print(complex['Target Name'][0])

