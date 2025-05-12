#  ML_Model/train.py 
import os
import sys
import math
import random
import argparse
import warnings
import distutils.util
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.io as sio
from tqdm import tqdm
import wandb                                 # ← ADD THIS
from wandb import Settings

# ensure repo root on PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from ML_Model.dataset        import HUTUBS
from ML_Model.model          import EquivariantSHPredictor
from ML_Model.baseline_model import ConvNNHrtfSht
from ML_Model.metrics import rotationally_averaged_lsd, _make_grid_pts

def str2bool(v):
    return bool(distutils.util.strtobool(v))

def init_params():
    p = argparse.ArgumentParser()
    p.add_argument('--seed',            type=int,   default=688)
    p.add_argument('-o', '--out_fold',  type=str,   required=True)
    p.add_argument('-a', '--anthro_mat_path',   type=str, required=True)
    p.add_argument('-t', '--hrtf_SHT_mat_path', type=str, required=True)
    p.add_argument('-v', '--shvec_path',         type=str, required=True)
    p.add_argument('-i', '--val_idx',    type=int,   default=0)
    p.add_argument('--norm_anthro',      type=str2bool, nargs='?', const=True, default=True)
    p.add_argument('--anthro_norm_method', type=str, default='chun2017', choices=['standard','chun2017'])
    p.add_argument('--ear_anthro_dim', type=int, default=12)
    p.add_argument('--head_anthro_dim',type=int, default=13)
    p.add_argument('--freq_bin',       type=int, default=41)
    p.add_argument('--ear_emb_dim',  type=int, default=32)
    p.add_argument('--head_emb_dim', type=int, default=32)
    p.add_argument('--lr_emb_dim',   type=int, default=16)
    p.add_argument('--freq_emb_dim', type=int, default=16)
    p.add_argument('--condition_dim',type=int, default=256)
    p.add_argument('--num_epochs',  type=int,   default=1000)
    p.add_argument('--batch_size',  type=int,   default=1024)
    p.add_argument('--lr',          type=float, default=5e-4)
    p.add_argument('--lr_decay',    type=float, default=0.8)
    p.add_argument('--interval',    type=int,   default=100)
    p.add_argument('--beta_1',      type=float, default=0.9)
    p.add_argument('--beta_2',      type=float, default=0.999)
    p.add_argument('--eps',         type=float, default=1e-8)
    p.add_argument('--gpu',         type=str,   default="0")
    p.add_argument('--num_workers', type=int,   default=0)
    p.add_argument('--norm',   type=str,      default='layer', choices=['batch','layer','instance'])
    p.add_argument('--target', type=str,      default='sht',   choices=['sht','hrtf'])
    p.add_argument('--test_only', action='store_true')
    p.add_argument('--equivariant', type=str2bool, nargs='?', const=True, default=False,
                   help="True → use SO(3)-equivariant model; False → baseline ConvNN")
    args = p.parse_args()

    # reproducibility & directories
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.test_only:
        os.makedirs(args.out_fold, exist_ok=True)
        os.makedirs(os.path.join(args.out_fold, 'checkpoint'), exist_ok=True)

    # sanity-check input files
    for path in (args.anthro_mat_path, args.hrtf_SHT_mat_path, args.shvec_path):
        assert os.path.exists(path), f"Missing file: {path}"

    # save config + initialize empty logs
    with open(os.path.join(args.out_fold, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    for lg in ('train_loss.log', 'dev_loss.log', 'test_loss.log'):
        open(os.path.join(args.out_fold, lg), 'w').write(f"Starting {lg}\n")

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args

def train(args):
    # —— load SH‐vector basis robustly ——
    mat = sio.loadmat(args.shvec_path)
    for key in ('Ynm_measured','Ynm','Ynm_val','shvec','Y'):
        if key in mat and isinstance(mat[key], np.ndarray) and mat[key].size:
            Ynm_np = mat[key].astype(np.float32)
            break
    else:
        # fallback to first non-private ndarray
        Ynm_np = next(
            v.astype(np.float32)
            for k,v in mat.items()
            if not k.startswith('__') and isinstance(v, np.ndarray)
        )
    Ynm_full = torch.from_numpy(Ynm_np).to(args.device)

    L_max = int(math.isqrt(Ynm_full.shape[1]) - 1)
    pts   = _make_grid_pts(L_max, args.device)

    # initialize W&B with a longer timeout
    wandb.init(
        project='hrtf-ablation',
        config=vars(args),
        settings=Settings(init_timeout=300)
    )

    for fold in range(93):
        print(f"\n─── Fold {fold}/92 ───")
        args.val_idx = fold

        tr_ds = HUTUBS(args, val=False)
        va_ds = HUTUBS(args, val=True)
        tr_ld = DataLoader(tr_ds, args.batch_size, shuffle=True,  num_workers=args.num_workers)
        va_ld = DataLoader(va_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = (EquivariantSHPredictor(num_freq_bins=args.freq_bin).to(args.device)
                 if args.equivariant else
                 ConvNNHrtfSht(args).to(args.device))

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta_1, args.beta_2),
            eps=args.eps,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        mse_fn = nn.MSELoss()

        for epoch in range(1, args.num_epochs + 1):
            # — training —
            model.train()
            running_train = 0.0
            train_batches = 0
            for head, ear, sht, aux, subj, freq, side in tr_ld:
                head, ear, sht, aux, freq, side = [
                    x.to(args.device) for x in (head, ear, sht, aux, freq, side)
                ]
                optimizer.zero_grad()

                if args.equivariant:
                    # Equivariant: (head, ear, freq, side)
                    pred = model(head, ear, freq, side)
                else:
                    # Baseline:   (ear, head, freq, side)
                    pred = model(ear, head, freq, side)

                loss = mse_fn(pred, sht)
                loss.backward()
                optimizer.step()
                running_train += loss.item()
                train_batches += 1
            avg_train_loss = running_train / train_batches
            scheduler.step()

            # — validation —
            model.eval()
            running_val = 0.0
            running_rot = 0.0
            val_batches = 0
            with torch.no_grad():
                for head, ear, sht, aux, subj, freq, side in va_ld:
                    head, ear, sht, aux, freq, side = [
                        x.to(args.device) for x in (head, ear, sht, aux, freq, side)
                    ]
                    if args.equivariant:
                        pred = model(head, ear, freq, side)
                    else:
                        pred = model(ear, head, freq, side)

                    running_val += mse_fn(pred, sht).item()
                    running_rot += rotationally_averaged_lsd( sht, pred, L_max,).item()
                    H_lin = (10.0**(pred/20.0)) @ Ynm_full[..., :pred.size(1)].T
                    val_batches += 1
            avg_val_loss = running_val / val_batches
            avg_rot_lsd  = running_rot / val_batches
            n = len(va_ld)
            wandb.log({
            'fold':       fold,
            'epoch':      epoch,
            'train_loss': avg_train_loss,
            'val_loss':   avg_val_loss,
            'rotLSD':     avg_rot_lsd,
            'lr':         scheduler.get_last_lr()[0]
            })

            with open(os.path.join(args.out_fold, 'train_loss.log'), 'a') as f:
                f.write(f"{epoch}\t{avg_train_loss:.6f}\n")
            with open(os.path.join(args.out_fold, 'dev_loss.log'), 'a') as f:
                f.write(f"{epoch}\t{avg_val_loss:.6f}\n")

            if epoch % 10 == 0:
                print(f"[fold {fold}] epoch {epoch:4d} | "
                f"train {avg_train_loss:7.2f} | val {avg_val_loss:7.2f} | "
                f"rotLSD {avg_rot_lsd:6.2f}")

        # save this fold’s model
        torch.save(
            model.state_dict(),
            os.path.join(args.out_fold, f"fold{fold}.pt")
        )

    wandb.finish()

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning)
    args = init_params()
    if not args.test_only:
        train(args)
