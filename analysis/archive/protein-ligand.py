# %%
import sys
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models/BIND/")
# %%
from src.ProteinLigandGym.env.bind_inference import init_BIND, predict_binder

# %%
device="cuda"
mutant_aa_seq = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"

ligands = [
    'CC(=O)CCc1ccccc1',        # Benzylacetone
    'C1=CC=C(C=C1)CCC(=O)O',   # Benzenepropionic acid (3-Phenylpropionic acid)
    'COC1=CC=CC(CCC(O)=O)=C1', # 3-(3-methoxyphenyl)propionic acid
    'COc1cccc(CCC(O)C)c1O',    # Dihydroconiferyl alcohol
    'COc1cc(CCC(=O)O)ccc1O',   # Dihydroferulic acid
    'O=C(O)CCc1ccc2OCOc2c1',   # 3-(3,4-Methylenedioxyphenyl)propionic acid (MDPPA)
    'OC(=O)CCc1ccccc1',        # Hydrocinnamic acid (3-Phenylpropionic acid, duplicate removed)
    'COc1cc(CCCO)ccc1O',       # Dihydroconiferyl alcohol (different representation)
    'COc1cc(CCCO)cc(OC)c1O',   # Syringyl propanol
    'OC(=O)Cc1ccccc1',         # Phenylacetic acid
    'CC(=O)CCc1ccc2OCOc2c1'    # 3-(3,4-Methylenedioxyphenyl)-2-butanone
]

ligands_extended = [
    # Original compounds
    'CC(=O)CCc1ccccc1',        # Benzylacetone
    'C1=CC=C(C=C1)CCC(=O)O',   # Benzenepropionic acid (3-Phenylpropionic acid)
    'COC1=CC=CC(CCC(O)=O)=C1', # 3-(3-methoxyphenyl)propionic acid
    'COc1cccc(CCC(O)C)c1O',    # Dihydroconiferyl alcohol
    'COc1cc(CCC(=O)O)ccc1O',   # Dihydroferulic acid
    'O=C(O)CCc1ccc2OCOc2c1',   # 3-(3,4-Methylenedioxyphenyl)propionic acid (MDPPA)
    'OC(=O)CCc1ccccc1',        # Hydrocinnamic acid (3-Phenylpropionic acid)
    'COc1cc(CCCO)ccc1O',       # Dihydroconiferyl alcohol (different representation)
    'COc1cc(CCCO)cc(OC)c1O',   # Syringyl propanol
    'OC(=O)Cc1ccccc1',         # Phenylacetic acid
    'CC(=O)CCc1ccc2OCOc2c1',   # 3-(3,4-Methylenedioxyphenyl)-2-butanone

    # Additional lignin monomers and related compounds
    'OC/C=C/c1ccc(O)cc1',      # p-Coumaryl alcohol
    'COc1cc(/C=C/CO)ccc1O',    # Coniferyl alcohol
    'COc1cc(/C=C/CO)cc(OC)c1O',# Sinapyl alcohol
    'O=C(O)/C=C/c1ccc(O)cc1',  # p-Coumaric acid
    'COc1cc(/C=C/C(=O)O)ccc1O',# Ferulic acid
    'COc1cc(/C=C/C(=O)O)cc(OC)c1O', # Sinapic acid
    'COc1cc(C=CC=O)ccc1O',     # Coniferyl aldehyde
    'COc1cc(C=CC=O)cc(OC)c1O', # Sinapaldehyde

    # Lignin degradation products and model compounds
    'COc1cccc(CC=O)c1',        # 3-Methoxyphenylacetaldehyde
    'COc1ccc(C=CC=O)cc1O',     # Vanillin
    'COc1cc(C=O)ccc1O',        # Vanillin (different representation)
    'COc1cc(C(=O)O)ccc1O',     # Vanillic acid
    'O=Cc1ccc(O)c(O)c1',       # Protocatechualdehyde
    'O=C(O)c1ccc(O)c(O)c1',    # Protocatechuic acid
    'COc1cc(C(C)=O)ccc1O',     # Acetovanillone
    'COc1cc(CC=O)ccc1O',       # Homovanillin
    'COc1cc(CCC=O)ccc1O',      # Homovanillin alcohol
    'COc1cc(CCO)ccc1O',        # Homovanillyl alcohol
    'Oc1ccccc1O',              # Catechol
    'COc1ccccc1O',             # Guaiacol
    'COc1cc(OC)c(O)c(OC)c1',   # Syringol
    'CC(=O)c1ccc(O)cc1',       # p-Hydroxyacetophenone
    'O=Cc1ccc(O)cc1',          # p-Hydroxybenzaldehyde
    'O=C(O)c1ccc(O)cc1',       # p-Hydroxybenzoic acid
    'COc1cc(C(C)=O)cc(OC)c1O', # Acetosyringone
    'COc1cc(C=O)cc(OC)c1O',    # Syringaldehyde
    'COc1cc(C(=O)O)cc(OC)c1O', # Syringic acid

    # Additional related compounds
    'OC(=O)c1ccccc1',          # Benzoic acid
    'O=Cc1ccccc1',             # Benzaldehyde
    'OCC1=CC=C(CO)C=C1',       # 1,4-Benzenedimethanol
    'OC(=O)C1=CC=C(C(=O)O)C=C1', # Terephthalic acid
    'OC1=CC=C(C=CC(=O)O)C=C1', # p-Coumaric acid (different representation)
    'COc1ccc(/C=C/C(=O)O)cc1', # Ferulic acid (different representation)
    'OC(=O)CCc1ccc(O)cc1',     # 3-(4-Hydroxyphenyl)propionic acid
    'OC(=O)Cc1ccc(O)cc1',      # 4-Hydroxyphenylacetic acid
    'CC(O)c1ccc(O)cc1',        # 1-(4-Hydroxyphenyl)ethanol
    'OC(=O)C1=CC(O)=C(O)C=C1', # 3,4-Dihydroxybenzoic acid
    'OC1=CC=C(CCO)C=C1',       # 4-(2-Hydroxyethyl)phenol
    'OC1=CC=C(CCCO)C=C1',      # 4-(3-Hydroxypropyl)phenol
    'COC1=C(O)C=CC(=CC=O)C1=O',# Coniferyl aldehyde quinone methide
    'COc1cc(C2OCC3C(c4cc(OC)c(O)cc4)OCC23)ccc1O', # Pinoresinol (a lignin dimer)
    'COc1cc(CC(CO)C(CO)Cc2ccc(O)c(OC)c2)ccc1O',   # Guaiacylglycerol-β-guaiacyl ether (β-O-4 lignin model compound)
]


ba_model, get_ba_activations, latent_vector_size, esm_model, esm_tokeniser, get_conv5_inputs, get_crossattention4_inputs = init_BIND(device)

# %%
for ligand in ligands:
    score = predict_binder(ba_model, esm_model, esm_tokeniser, device, [mutant_aa_seq], ligand)
    print(score)
    
# %% markdown
"""

Lignin is a complex organic polymer that's a key structural component in the cell walls of many plants, especially in wood and bark.
It's one of the most abundant natural polymers on Earth, second only to cellulose. Lignin is composed of various phenylpropanoid units, which gives it a complex, three-dimensional structure.

The compounds in your list relate to lignin in several ways:

1. Lignin Monomers and Derivatives:
   Many of these compounds are either direct monomers of lignin or closely related derivatives. The three main monolignols (building blocks) of lignin are:
   - p-coumaryl alcohol
   - coniferyl alcohol
   - sinapyl alcohol

2. Lignin Degradation Products:
   Several compounds in your list are products of lignin degradation or depolymerization. When lignin breaks down (either naturally or through industrial processes), it produces a variety of smaller aromatic compounds.

3. Model Compounds:
   Some of these molecules are used as model compounds in lignin research to study lignin's properties, reactivity, and potential applications.

Let's go through some specific examples from your list:

1. Dihydroconiferyl alcohol: This is a reduced form of coniferyl alcohol, one of the primary lignin monomers. It's often found in lignin degradation products.
2. Dihydroferulic acid: This is closely related to ferulic acid, which is a precursor in the biosynthesis of lignin monomers. It's also a common product of lignin degradation.
3. 3-(3,4-Methylenedioxyphenyl)propionic acid (MDPPA): While not a direct lignin monomer, this compound has structural similarities to lignin building blocks, particularly in its aromatic ring structure.
4. Hydrocinnamic acid (3-Phenylpropionic acid): This simple aromatic acid is structurally similar to the phenylpropanoid units found in lignin.
5. Syringyl propanol: This is related to sinapyl alcohol, one of the three main lignin monomers. The syringyl unit is particularly abundant in hardwood lignins.
6. Phenylacetic acid: While not directly derived from lignin, this compound shares structural similarities with lignin degradation products.
7. 3-(3,4-Methylenedioxyphenyl)-2-butanone: This compound has structural similarities to lignin monomers and could be a product of lignin modification or degradation.

These compounds are important in various fields:

- Biofuel Research: Many of these compounds are studied in the context of converting lignin into valuable chemicals and fuels.
- Paper Industry: Understanding lignin structure and degradation is crucial for paper production and wood processing.
- Agriculture: Lignin-related compounds play roles in plant defense and are studied for their effects on soil and plant health.
- Pharmacology: Some lignin-derived compounds have potential medicinal properties and are studied for various health applications.

In summary, these compounds represent a spectrum of molecules related to lignin's structure, biosynthesis, and degradation. They are valuable in understanding lignin's complex chemistry and in developing new applications for this abundant natural resource.
"""

# %%
for ligand in ligands_extended:
    score = predict_binder(ba_model, esm_model, esm_tokeniser, device, [mutant_aa_seq], ligand)
    print(score)
    
# %%
lignin_focus_compounds = [
    # Lignin Monomers
    'OC/C=C/c1ccc(O)cc1',      # p-Coumaryl alcohol
    'COc1cc(/C=C/CO)ccc1O',    # Coniferyl alcohol
    'COc1cc(/C=C/CO)cc(OC)c1O',# Sinapyl alcohol

    # Key Intermediates
    'COc1ccc(/C=C/C(=O)O)cc1', # Ferulic acid
    'COc1ccccc1O',             # Guaiacol
    'O=C(O)c1ccc(O)c(O)c1',    # Protocatechuic acid

    # High-Value Products
    'COc1cc(C=O)ccc1O',        # Vanillin
    'COc1cc(C(=O)O)ccc1O',     # Vanillic acid
    'COc1cc(C=O)cc(OC)c1O',    # Syringaldehyde

    # Model Compounds for Lignin Linkages
    'COc1cc(C2OCC3C(c4cc(OC)c(O)cc4)OCC23)ccc1O', # Pinoresinol (a lignin dimer)
    'COc1cc(CC(CO)C(CO)Cc2ccc(O)c(OC)c2)ccc1O',   # Guaiacylglycerol-β-guaiacyl ether (β-O-4 lignin model compound)

    # Platform Chemicals
    'OC1=CC=C(CCO)C=C1',       # 4-(2-Hydroxyethyl)phenol
    'OC1=CC=C(CCCO)C=C1',      # 4-(3-Hydroxypropyl)phenol

    # Additional Important Compounds
    'COc1cc(/C=C/C(=O)O)cc(OC)c1O', # Sinapic acid
    'COc1cc(C=CC=O)ccc1O',     # Coniferyl aldehyde
    'COc1cc(C=CC=O)cc(OC)c1O', # Sinapaldehyde
    'Oc1ccccc1O',              # Catechol
    'COc1cc(OC)c(O)c(OC)c1',   # Syringol
    'O=Cc1ccc(O)c(O)c1',       # Protocatechualdehyde
    'OC(=O)CCc1ccc(O)cc1',     # 3-(4-Hydroxyphenyl)propionic acid
]


for ligand in lignin_focus_compounds:
    score = predict_binder(ba_model, esm_model, esm_tokeniser, device, [mutant_aa_seq], ligand)
    print(score)
    
# %%
lignin_focus_compounds = [
    # Lignin Monomers and Derivatives
    'OCC(O)Cc1ccc(O)c(OC)c1',  # Dihydroconiferyl alcohol
    'OCC(O)Cc1cc(OC)c(O)c(OC)c1',  # Dihydrosinapyl alcohol
    'CCCc1ccc(O)c(OC)c1',  # 4-Propylguaiacol
    'CCCc1cc(OC)c(O)c(OC)c1',  # 4-Propylsyringol
    'COc1cc(/C=C/C)ccc1O',  # trans-Isoeugenol
    'COc1cc(/C=C/C)cc(OC)c1O',  # trans-4-Propenylsyringol
    'OCC=Cc1ccc(O)c(OC)c1',  # Coniferyl alcohol
    'OCC=Cc1cc(OC)c(O)c(OC)c1',  # Sinapyl alcohol

    # Hydroxycinnamic Acids
    'COc1cc(/C=C/C(=O)O)ccc1O',  # Ferulic acid
    'COc1cc(/C=C/C(=O)O)cc(OC)c1O',  # Sinapic acid
    'OC(=O)/C=C/c1ccc(O)c(O)c1',  # Caffeic acid
    'OC(=O)/C=C/c1ccc(O)cc1',  # p-Coumaric acid

    # Other Relevant Compounds
    'COc1cccc(CCC(=O)O)c1',  # 3-3-Methoxyphenylpropionic acid

    # Compounds from your original list that are also relevant
    'COc1cc(C=O)ccc1O',  # Vanillin
    'COc1cc(C(=O)O)ccc1O',  # Vanillic acid
    'COc1cc(C=O)cc(OC)c1O',  # Syringaldehyde
    'COc1ccccc1O',  # Guaiacol
    'COc1cc(OC)c(O)c(OC)c1',  # Syringol
    'Oc1ccccc1O',  # Catechol
    'O=Cc1ccc(O)c(O)c1',  # Protocatechualdehyde
    'OC(=O)CCc1ccc(O)cc1',  # 3-(4-Hydroxyphenyl)propionic acid
]

for ligand in lignin_focus_compounds:
    score = predict_binder(ba_model, esm_model, esm_tokeniser, device, [mutant_aa_seq], ligand)
    print(score)