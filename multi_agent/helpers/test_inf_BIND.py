# %%
import sys
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models/BIND/")

from ProteinLigandGym.env.bind_inference import init_BIND, predict_binder

device = 'cuda'

bind_model, esm_model, esm_tokeniser = init_BIND(device)


sequence = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"
#smile = "CC(=O)CCc1ccc2OCOc2c1"
smile = "c1(ccc(cc1)c1ccccc1c1[n-]nnn1)Cn1c(c(nc1CCCC)Cl)CO"

scores = predict_binder(bind_model, esm_model, esm_tokeniser, device, [sequence], smile)

print(scores)