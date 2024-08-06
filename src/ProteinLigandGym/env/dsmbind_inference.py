import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random, math
import esm
from sidechainnet.utils.measure import get_seq_coords_and_angles
from prody import *
from rdkit import Chem
from rdkit.Chem import PandasTools
from models.DSMBind.bindenergy.models import FARigidModel
from models.DSMBind.bindenergy.models.drug import MPNEncoder
from models.DSMBind.bindenergy.models.frame import FAEncoder, AllAtomEncoder
from models.DSMBind.bindenergy.utils.utils import _expansion, _density, _score
from models.DSMBind.bindenergy.data.drug import DrugDataset
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND


def init_DSMBind(device):
    
    fn = '/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models/DSMBind/ckpts/model.recA'
    model_dsm_ckpt, opt_ckpt, model_args = torch.load(fn)
    model_dsm = DrugAllAtomEnergyModel(model_args, device=device)
    model_dsm.load_state_dict(model_dsm_ckpt, strict=False)
    model_dsm.to(device)
    model_dsm.eval()

    return model_dsm
    

class DrugAllAtomEnergyModel(FARigidModel):

    def __init__(self, args, device):
        super(DrugAllAtomEnergyModel, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.threshold = args.threshold
        self.args = args
        self.mpn = MPNEncoder(args)
        self.encoder = AllAtomEncoder(args)
        self.W_o = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.SiLU(),
                nn.Linear(args.hidden_size, args.hidden_size),
        )
        self.U_o = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.SiLU(),
        )
        self.theta_range = np.linspace(0.1, np.pi, 100)
        self.sigma_range = np.linspace(0, 10.0, 100) + 0.1
        self.expansion = [_expansion(self.theta_range, sigma) for sigma in self.sigma_range]
        self.density = [_density(exp, self.theta_range) for exp in self.expansion]
        self.score = [_score(exp, self.theta_range, sigma) for exp, sigma in zip(self.expansion, self.sigma_range)]
        self.eye = torch.eye(3).unsqueeze(0).cuda()

        self.tokkenizer, self.alphabet = self._init_tokkenizer(device)

    def _init_tokkenizer(self, device):
        model_tokkenizer, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        model_tokkenizer.to(device)
        model_tokkenizer.eval()
        return model_tokkenizer, alphabet

    def sc_rotate(self, X, w):
        w = w.unsqueeze(1).expand_as(X)  # [B,N,3]
        c = w.norm(dim=-1, keepdim=True)  # [B,N,1]
        c1 = torch.sin(c) / c.clamp(min=1e-6)
        c2 = (1 - torch.cos(c)) / (c ** 2).clamp(min=1e-6)
        cross = lambda a,b : torch.cross(a, b, dim=-1)
        return X + c1 * cross(w, X) + c2 * cross(w, cross(w, X))

    def forward(self, binder, target, use_sidechain=True):
        print(f"Type of binder: {type(binder)}")
        print(f"Length of binder: {len(binder) if isinstance(binder, (list, tuple)) else 'N/A'}")
        print(f"Contents of binder: {binder}")
        true_X, mol_batch, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target
        bind_S = self.mpn(mol2graph(mol_batch))

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_mask = bind_A[:,:,1].clamp(max=1).float()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()
        bind_A = bind_A * (true_X.norm(dim=-1) > 1e-4).long() 
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long() 
        sc_mask = (tgt_A[:,:,4:] > 0).float().view(B*M, 10)
        has_sc = sc_mask.sum(dim=-1).clamp(max=1)

        # Random backbone rotation + translation
        sidx = [random.randint(0, 99) for _ in range(B)]
        sigma = torch.tensor([self.sigma_range[i] for i in sidx]).float().cuda()
        tidx = [np.random.choice(list(range(100)), p=self.density[i]) for i in sidx]
        theta = torch.tensor([self.theta_range[i] for i in tidx]).float().cuda()
        w = torch.randn(B, 3).cuda()
        hat_w = F.normalize(w, dim=-1)
        w = hat_w * theta.unsqueeze(-1)
        eps = np.random.uniform(0.1, 1.0, size=B)
        eps = torch.tensor(eps).float().cuda().unsqueeze(-1)
        hat_t = torch.randn(B, 3).cuda() * eps
        # Apply
        center = self.mean(true_X[:,:,1], bind_mask)
        bind_X = true_X - center[:,None,None,:]
        bind_X = self.rotate(bind_X, w) + hat_t[:,None,None,:]
        bind_X = bind_X + center[:,None,None,:]
        bind_X = bind_X.requires_grad_()

        # Random side chain rotation
        aidx = [random.randint(0, 99) for _ in range(B * M)]
        sigma = torch.tensor([self.sigma_range[i] for i in aidx]).float().cuda()
        bidx = [np.random.choice(list(range(100)), p=self.density[i]) for i in aidx]
        theta = torch.tensor([self.theta_range[i] for i in bidx]).float().cuda()
        u = torch.randn(B*M, 3).cuda() 
        hat_u = F.normalize(u, dim=-1)
        u = hat_u * theta.unsqueeze(-1)
        # Apply
        backbone = tgt_X[:,:,:4].clone()
        center = tgt_X[:,:,1:2,:].clone()  # CA is the rotation center
        tgt_X = tgt_X - center
        tgt_X = tgt_X.view(B*M, 14, 3)
        tgt_X = self.sc_rotate(tgt_X, u)
        tgt_X = tgt_X.view(B, M, 14, 3) + center
        tgt_X = torch.cat((backbone, tgt_X[:,:,4:]), dim=-2)
        tgt_X = tgt_X * (tgt_A > 0).float().unsqueeze(-1)
        tgt_X = tgt_X.requires_grad_()

        # Contact map
        mask_2D = (bind_A > 0).float().view(B,N*14,1) * (tgt_A > 0).float().view(B,1,M*14)
        D = (bind_X.view(B,N*14,1,3) - tgt_X.view(B,1,M*14,3)).norm(dim=-1)  # [B,N*14,M*14]
        mask_2D = mask_2D * (D < self.threshold).float()
        # Energy
        h = self.encoder(
                (bind_X, bind_S, bind_A, None), 
                (tgt_X, tgt_S, tgt_A, None), 
        )  # [B,N+M,14,H]
        bind_h = self.W_o(h[:, :N]).view(B, N*14, -1)
        tgt_h = self.U_o(h[:, N:]).view(B, M*14, -1)
        energy = torch.matmul(bind_h, tgt_h.transpose(1,2))  # [B,N*14,M*14]
        energy = (energy * mask_2D).sum(dim=(1,2))  # [B]

        # force
        f_bind, f_tgt = torch.autograd.grad(energy.sum(), [bind_X, tgt_X], create_graph=True, retain_graph=True)
        t = self.mean(f_bind[:,:,1], bind_mask)

        # Backbone torque
        center = self.mean(bind_X[:,:,1], bind_mask)
        bind_X = bind_X - center[:,None,None,:]    # set rotation center to zero
        G = torch.cross(bind_X[:,:,1], f_bind[:,:,1], dim=-1)  # [B,N,3]
        G = (G * bind_mask[...,None]).sum(dim=1)   # [B,3] angular momentum
        I = self.inertia(bind_X[:,:,1], bind_mask) # [B,3,3] inertia matrix
        w = torch.linalg.solve(I.detach(), G)  # angular velocity

        # Side chain torque
        center = tgt_X[:,:,1:2,:]  # CA is the rotation center
        tgt_X = tgt_X[:,:,4:] - center   # set rotation center to zero
        tgt_X = tgt_X.view(B*M, 10, 3)
        f_tgt = f_tgt[:,:,4:].reshape(B*M, 10, 3)
        G = torch.cross(tgt_X, f_tgt, dim=-1)
        G = (G * sc_mask[...,None]).sum(dim=1)   # [B*N,3] angular momentum
        I = self.inertia(tgt_X, sc_mask)    # [B*N,3,3] inertia matrix
        I = I + self.eye * (1 - has_sc)[:,None,None]  # avoid zero inertia
        u = torch.linalg.solve(I.detach(), G)  # [B*N, 3] angular velocity

        # Backbone score matching loss
        score = torch.tensor([self.score[i][j] for i,j in zip(sidx, tidx)]).float().cuda()
        wloss = self.mse_loss(w, hat_w * score.unsqueeze(-1))
        tloss = self.mse_loss(t * eps, -hat_t / eps)
        # Side chain score matching loss
        score = torch.tensor([self.score[i][j] for i,j in zip(aidx, bidx)]).float().cuda()
        uloss = (u - hat_u * score.unsqueeze(-1)) ** 2
        uloss = (uloss.sum(-1) * has_sc).sum() / has_sc.sum().clamp(min=1e-6)
        return wloss + tloss + uloss * int(use_sidechain)

    def predict(self, binder, target, visual=False):
        bind_X, mol_batch, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target
        bind_S = self.mpn(mol2graph(mol_batch))

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_mask = bind_A[:,:,1].clamp(max=1).float()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()
        bind_A = bind_A * (bind_X.norm(dim=-1) > 1e-4).long() 
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long() 

        mask_2D = (bind_A > 0).float().view(B,N*14,1) * (tgt_A > 0).float().view(B,1,M*14)
        D = (bind_X.view(B,N*14,1,3) - tgt_X.view(B,1,M*14,3)).norm(dim=-1)  # [B,N*14,M*14]
        mask_2D = mask_2D * (D < self.threshold).float()

        h = self.encoder(
                (bind_X, bind_S, bind_A, None), 
                (tgt_X, tgt_S, tgt_A, None), 
        )  # [B,N+M,14,H]

        bind_h = self.W_o(h[:, :N]).view(B, N*14, -1)
        tgt_h = self.U_o(h[:, N:]).view(B, M*14, -1)
        energy = torch.matmul(bind_h, tgt_h.transpose(1,2))  # [B,N*14,M*14]
        if visual:
            energy = energy.view(B, N, 14, M, 14).transpose(2, 3)
            mask_2D = mask_2D.view(B, N, 14, M, 14).transpose(2, 3)
            return (energy * mask_2D).sum(dim=(-1, -2))
        else:
            return (energy * mask_2D).sum(dim=(1,2))  # [B]
    
    def virtual_screen(self, protein_pdb, mols, batch_size=1):
        hchain = parsePDB(protein_pdb, model=1)
        print(chain)
        _, hcoords, hseq, _, _ = get_seq_coords_and_angles(hchain)
        hcoords = hcoords.reshape((len(hseq), 14, 3))

        all_data = []
        for mol in mols:
            entry = {
                "binder_mol": mol, "target_seq": hseq, "target_coords": hcoords,
            }
            all_data.append(entry)
        
        embedding = load_esm_embedding(self.tokkenizer, self.alphabet ,all_data)
        all_data = DrugDataset.process(all_data, self.args.patch_size)
        all_score = []

        for i in range(0, len(all_data), batch_size):
            batch = all_data[i : i + batch_size]
            binder, target = DrugDataset.make_bind_batch(batch, embedding, self.args)
            score = self.predict(binder, target)
            for entry,aff in zip(batch, score):
                smiles = Chem.MolToSmiles(entry['binder_mol'])
                all_score.append((smiles, aff))
        return all_score

    def gaussian_forward(self, binder, target):
        true_X, mol_batch, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target
        bind_S = self.mpn(mol2graph(mol_batch))

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_mask = bind_A[:,:,1].clamp(max=1).float()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()
        bind_A = bind_A * (true_X.norm(dim=-1) > 1e-4).long()
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long()
        atom_mask = (bind_A > 0).float().unsqueeze(-1)

        eps = np.random.uniform(0.1, 1.0, size=B)
        eps = torch.tensor(eps).float().cuda()[:,None,None,None]
        hat_t = torch.randn_like(true_X).cuda() * eps
        bind_X = true_X + hat_t * atom_mask
        bind_X = bind_X.requires_grad_()

        # Contact map
        mask_2D = (bind_A > 0).float().view(B,N*14,1) * (tgt_A > 0).float().view(B,1,M*14)
        D = (bind_X.view(B,N*14,1,3) - tgt_X.view(B,1,M*14,3)).norm(dim=-1)  # [B,N*14,M*14]
        mask_2D = mask_2D * (D < self.threshold).float()

        # Energy
        h = self.encoder(
                (bind_X, bind_S, bind_A, None),
                (tgt_X, tgt_S, tgt_A, None),
        )  # [B,N+M,14,H]
        bind_h = self.W_o(h[:, :N]).view(B, N*14, -1)
        tgt_h = self.U_o(h[:, N:]).view(B, M*14, -1)
        energy = torch.matmul(bind_h, tgt_h.transpose(1,2))  # [B,N*14,M*14]
        energy = (energy * mask_2D).sum(dim=(1,2))  # [B]

        # force
        f_bind = torch.autograd.grad(energy.sum(), bind_X, create_graph=True, retain_graph=True)[0]
        loss = (f_bind * eps + hat_t / eps) * atom_mask
        return loss.sum() / atom_mask.sum()
    
def load_esm_embedding(model, alphabet, data):
    batch_converter = alphabet.get_batch_converter()
    model = model.cuda()
    model.eval()
    embedding = {}
    with torch.no_grad():
        seqs = [d['target_seq'] for d in data if d['target_seq'] not in embedding]
        for s in sorted(set(seqs)):
            batch_labels, batch_strs, batch_tokens = batch_converter([(s, s)])
            batch_tokens = batch_tokens.cuda()
            results = model(batch_tokens, repr_layers=[36], return_contacts=False)
            embedding[s] = results["representations"][36][0, 1:len(s)+1].cpu()
    return embedding