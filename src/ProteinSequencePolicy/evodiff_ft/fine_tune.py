import argparse
import json
import os
from datetime import datetime, timedelta
import pathlib

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.cuda.amp import GradScaler

from evodiff.model import ByteNetLMTime
from evodiff.utils import Tokenizer
from torch.utils.data import Subset
from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from sequence_models.datasets import UniRefDataset
from sequence_models.constants import MSA_ALPHABET
from evodiff.collaters import OAMaskCollater, D3PMCollater
from evodiff.losses import OAMaskedCrossEntropyLoss, D3PMCELoss, D3PMLVBLoss
from sequence_models.metrics import MaskedAccuracy
from sequence_models.utils import warmup 
import sys

from lora_pytorch import LoRA

from dataset_loaders.textfile_loader import TextfileDataset
from evodiff.pretrained import OA_DM_38M, OA_DM_640M


sys.setrecursionlimit(1000) # must be as large as diffusion timesteps for Q_bar calculation

### SET RANDOM SEEDS ###
torch.cuda.empty_cache() # empty caches

home = str(pathlib.Path.home())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fpath', type=str)
    parser.add_argument('--out_fpath', type=str)
    parser.add_argument('--train_fpath', type=str)
    parser.add_argument('--valid_fpath', type=str)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--large_model', action='store_true')
    parser.add_argument('--LoRA', type=int, default=0)
    parser.add_argument('--task', default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mask', type=str, default='oadm')
    parser.add_argument('--final_norm', action='store_true')
    parser.add_argument('--norm_first', action='store_true') # turns norm_first on in transformer model
    parser.add_argument('--checkpoint_freq', type=float, default=1)  # in minutes
    parser.add_argument('--log_freq', type=float, default=10)  # in steps
    parser.add_argument('--reweighting_term', type=float, default=0)  # lambda reweighting term from Austin D3PM
    parser.add_argument('--random_seed', type=int, default=0)  # lambda reweighting term from Austin D3PM

    args = parser.parse_args()
    train_loop(args)

def train_loop(args):
    rs = torch.random.manual_seed(args.random_seed)
    rs = np.random.seed(int(args.random_seed))
    device = torch.device('cuda')
    with open(args.config_fpath, 'r') as f:
        config = json.load(f)
    n_tokens = len(MSA_ALPHABET)
    d_embed = config['d_embed']
    d_model = config['d_model']
    n_layers = config['n_layers']
    kernel_size = config['kernel_size']
    r = config['r']
    if 'slim' in config:
        slim = config['slim']
    else:
        slim = True
    if 'activation' in config:
        activation = config['activation']
    else:
        activation = 'relu'
    bucket_size = config['bucket_size']
    max_tokens = config['max_tokens']
    max_batch_size = config['max_batch_size']
    epochs = config['epochs']
    lr = config['lr']
    warmup_steps = config['warmup_steps']
    if 'rank' in config:
        weight_rank = config['rank']
    else:
        weight_rank = None
    if args.task is not None:
        config['task'] = args.task
    if args.dataset is not None:
        config['dataset'] = args.dataset
    try:
        data_top_dir = os.getenv('PT_DATA_DIR') + '/'
    except:
        data_top_dir = home + '/Desktop/DMs/data/'
    data_dir = data_top_dir + config['dataset'] + '/'
    # ----------------------------------------------------------
    ### COLLATORS ###
    # ----------------------------------------------------------
    if args.mask == 'oadm':
        tokenizer = Tokenizer()
        collater = OAMaskCollater(tokenizer=tokenizer)
        diffusion_timesteps = None # Not input to model
    # elif args.mask == 'so':
    #     tokenizer = Tokenizer()
    #     raise Exception("Autoreg in other script")
    #     collater = BertMaskCollater(tokenizer=tokenizer)
    #     diffusion_timesteps = None  # Not input to model
    elif args.mask in ('random', 'blosum'):
        diffusion_timesteps = config['diffusion_timesteps']
        tokenizer = Tokenizer(path_to_blosum=data_top_dir+"blosum62-special-MSA.mat", sequences=True)
        if args.mask == 'random':
            Q_prod, Q_t = tokenizer.q_random_schedule(timesteps=diffusion_timesteps)
        else:
            Q_prod, Q_t = tokenizer.q_blosum_schedule(timesteps=diffusion_timesteps)
        collater = D3PMCollater(tokenizer=tokenizer, num_timesteps=diffusion_timesteps, Q=Q_t, Q_bar=Q_prod)
    else:
        raise RuntimeError("mask must be: 'oadm', 'blosum', or 'random'")
    causal = False
    if args.mask == 'so':
        causal = True
    # ----------------------------------------------------------
    ### DATALOADER ###
    # ----------------------------------------------------------
    if args.large_model:
        init_model, collater, tokenizer, _ = OA_DM_640M()
    else:
        init_model, collater, tokenizer, _ = OA_DM_38M()
    ds_train = TextfileDataset(args.train_fpath)
    len_train = len(ds_train.data)
    dl_train = DataLoader(dataset=ds_train,
                          batch_size=config["batch_size"],
                          shuffle=True,
                          collate_fn=collater)
    ds_valid = TextfileDataset(args.valid_fpath)
    len_valid = len(ds_valid.data)
    dl_valid = DataLoader(dataset=ds_valid,
                          batch_size=config["batch_size"],
                          shuffle=True,
                          collate_fn=collater)
    # ----------------------------------------------------------
    # Initiate model
    # ----------------------------------------------------------
    padding_idx = tokenizer.pad_id  # PROTEIN_ALPHABET.index(PAD)
    optimizer = Adam(init_model.parameters(), lr=lr, weight_decay=args.weight_decay)
    outputs = os.listdir(args.out_fpath)
    if len(outputs) > 0:
        last_epoch = 0
        for output in outputs:
            if 'checkpoint' in output:
                starting_epoch = int(output.split('checkpoint')[-1][:-4])
                if starting_epoch > last_epoch:
                    args.state_dict = args.out_fpath + output
                    last_epoch = starting_epoch
    init_model = init_model.to(device)
    scaler = GradScaler()

    # Add LoRA wrapper if LoRA was specified.
    if args.LoRA > 0:
        model = LoRA.from_module(init_model, rank=16)
    else:
        model = init_model

    # ----------------------------------------------------------
    # Loss Function
    # ----------------------------------------------------------
    # Turn off scheduler for now since we are fine-tuning only.
    #scheduler = LambdaLR(optimizer, warmup(warmup_steps), verbose=False)
    if args.mask == 'oadm' or args.mask == 'so':
        loss_func = OAMaskedCrossEntropyLoss(reweight=True)
    elif args.mask == 'blosum' or args.mask == 'random':
        # Austin = LVB + lambda * CE
        loss_func1 = D3PMLVBLoss(tmax=diffusion_timesteps, tokenizer=tokenizer)
        loss_func2 = D3PMCELoss(tokenizer=tokenizer)
        _lambda = args.reweighting_term
    accu_func = MaskedAccuracy()
    

    initial_epoch = 0
    total_steps = 0
    total_tokens = 0
    # ----------------------------------------------------------
    # Run
    # ----------------------------------------------------------
    def epoch(model, train, current_step=0, current_tokens=0):
        start_time = datetime.now()
        if train:
            model = model.train()
            loader = dl_train
            t = 'Training:'
        else:
            model = model.eval()
            loader = dl_valid
            t = 'Validating:'
        losses = []
        nll_losses = []
        accus = []
        ns = []
        num_seqs = []
        chunk_time = datetime.now()
        n_seen = 0
        tokens_trained = current_tokens
        if train:
            n_total = len(ds_train)
        else:
            n_total = len(ds_valid)
        for i, batch in enumerate(loader):
            # restarting from a checkpoint
            new_loss, new_nll_loss, new_accu, new_n, new_seqs, new_processed = step(model, batch, train)
            if train:
                new_loss = new_loss.sum()
                new_nll_loss = new_nll_loss.sum()
                new_accu = new_accu.sum()
                new_n = new_n.sum()
                new_seqs = new_seqs.sum()
                #dist.reduce(new_loss, 0, op=dist.ReduceOp.SUM)
                #dist.reduce(new_nll_loss, 0, op=dist.ReduceOp.SUM)
                #dist.reduce(new_accu, 0, op=dist.ReduceOp.SUM)
                #dist.reduce(new_n, 0, op=dist.ReduceOp.SUM)
                #dist.reduce(new_seqs, 0, op=dist.ReduceOp.SUM)
            losses.append(new_loss.item())
            nll_losses.append(new_nll_loss.item())
            accus.append(new_accu.item())
            ns.append(new_n.item())
            num_seqs.append(new_seqs.item())
            n_seen += new_seqs.item()
            total_n = sum(ns)
            r_loss = sum(losses) / total_n
            r_nll_loss = sum(nll_losses) / total_n
            raccu = sum(accus) / total_n
            if train:
                nsteps = current_step + i + 1
                tokens_trained += new_processed.item()
            else:
                nsteps = i
            print('%s Epoch %d of %d Step %d ntokens %d Example %d of %d loss = %.4f nll loss = %.4f accu = %.4f\n'
                      % (t, e + 1, epochs, nsteps, tokens_trained, n_seen, n_total, r_loss, r_nll_loss, raccu),
                  flush=True)
            if train:
                losses = losses[-999:]
                accus = accus[-999:]
                ns = ns[-999:]
                num_seqs = num_seqs[-999:]
                nll_losses = nll_losses[-999:]
                if nsteps % args.log_freq == 0:  # write to checkpoint frequency
                    with open(args.out_fpath + 'train-metrics.csv', 'a') as f:
                        f.write(','.join([str(r_loss), str(r_nll_loss), str(raccu), str(int(current_tokens)), str(nsteps), str(e)]))
                        f.write('\n')
                if ((datetime.now() - chunk_time) > timedelta(minutes=args.checkpoint_freq)) or (n_seen == n_total):
                    print('Writing to checkpoint at', chunk_time, flush=True)
                    with torch.no_grad():
                        ckpt_fpath = args.out_fpath + 'checkpoint%d.tar' % nsteps
                        torch.save({
                                'step': nsteps,
                                'tokens': tokens_trained,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': e
                                }, ckpt_fpath)
                        _ = epoch(model, False, current_step=nsteps, current_tokens=tokens_trained)
                    chunk_time = datetime.now()
        if not train:
            with open(args.out_fpath + 'valid-metrics.csv', 'a') as f:
                f.write(','.join([str(r_loss), str(r_nll_loss), str(raccu), str(int(current_tokens)), str(current_step), str(e)]))
                f.write('\n')
            print('Validation complete in ' + str(datetime.now() - start_time), flush=True)
        else:
            print('Epoch complete in ' + str(datetime.now() - start_time), flush=True)
        return i, tokens_trained

    def step(model, batch, train):
        if args.mask == 'blosum' or args.mask == 'random':
            src, src_onehot, timestep, tgt, tgt_onehot, Q, Q_bar, q = batch
            q = q.to(device)
            Q = Q.to(device)
            Q_bar = Q_bar.to(device)
            src_onehot = src_onehot.to(device)
            tgt_onehot = tgt_onehot.to(device)
        else:
            src, timestep, tgt, mask = batch
            mask = mask.to(device)
        timestep = timestep.to(device)
        src = src.to(device)
        tgt = tgt.to(device)
        input_mask = (src != padding_idx).float()

        if args.mask == 'blosum' or args.mask == 'random':
            n_tokens = input_mask.sum()
        else:
            n_tokens = mask.sum()

        n_processed = input_mask.sum()
        n_seqs = torch.tensor(len(src), device=device)
        # step through model
        if train:
            optimizer.zero_grad() # reset gradients of model parameters

        # Enables autocasting for the forward pass (model + loss)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            outputs = model(src, timestep, input_mask=input_mask.unsqueeze(-1))
            # The next few lines are bad syntax -- this is from the original file,
            # I have not updated (jlp). However, since we are only using OADM for
            # now, this is not currently an issue...
            if args.mask in ('random', 'blosum'):
                lvb_loss = loss_func1(src_onehot, q, outputs, tgt, tgt_onehot, input_mask, timestep, Q, Q_bar)
                ce_loss = loss_func2(outputs, tgt, input_mask)
                lvb_loss = lvb_loss.to(torch.float32)
                ce_loss = ce_loss.to(torch.float32)
                loss = (lvb_loss + (_lambda * ce_loss)) * n_tokens
                nll_loss = ce_loss * n_tokens
                accu = accu_func(outputs, tgt, input_mask) * n_tokens
            elif args.mask == 'oadm' or args.mask=='so':
                ce_loss, nll_loss = loss_func(outputs, tgt, mask, timestep, input_mask)  # sum(loss per token)
                loss = ce_loss
                accu = accu_func(outputs, tgt, mask) * n_tokens
            else:
                raise RuntimeError("Unexpected mask.")
        if train:
            # Exit the context manager before backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            # Skip scheduler for now...
            #skip_scheduler = (scale > scaler.get_scale())
            #if not skip_scheduler:
            #    scheduler.step()

        if loss <= 0 or loss >= 1000000:
            print(loss, lvb_loss, ce_loss, nll_loss, n_tokens, _lambda)
            print(timestep)
            print([tokenizer.untokenize(t) for t in tgt])
            print([tokenizer.untokenize(s) for s in src])
        return loss, nll_loss, accu, n_tokens, n_seqs, n_processed

    n_parameters = sum(p.numel() for p in model.parameters())
    print('%d model parameters' %n_parameters, flush=True)
    print('%d training sequences' %len_train, flush=True)
    print('%d validation sequences' %len_valid, flush=True)
    for e in range(initial_epoch, epochs):
        s, t = epoch(model, True, current_step=total_steps, current_tokens=total_tokens)
        total_steps += s
        total_tokens += t

    print("All done.", flush=True)
    ckpt_fpath = os.path.join(args.out_fpath, 'FINAL_MODEL.pt')
    if args.LoRA > 0:
        new_model = model.merge_lora(inplace=True)
        torch.save(new_model.state_dict(), ckpt_fpath)
    else:
        torch.save(model.state_dict(), ckpt_fpath)


if __name__ == '__main__':
    main()
