#!/usr/bin/env python3
"""
ML_Model/new_train.py (ablation: first 5 folds, equivariant mul=6)
"""
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
import wandb  # ← We log to W&B
from wandb import Settings

# ensure repo root on PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from ML_Model.dataset import HUTUBS
from ML_Model.model import EquivariantSHPredictor
from ML_Model.baseline_model import ConvNNHrtfSht
from ML_Model.metrics import rotationally_averaged_lsd, _make_grid_pts


def str2bool(v):
    return bool(distutils.util.strtobool(v))


def init_params():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=688)
    p.add_argument('-o', '--out_fold', type=str, required=True,
                   help='output folder, e.g. equivariant_8_run')
    p.add_argument('-a', '--anthro_mat_path', type=str, required=True,
                   help='CSV of anthropometric data')
    p.add_argument('-t', '--hrtf_SHT_mat_path', type=str, required=True,
                   help='.mat with HRTF SHT data')
    p.add_argument('-v', '--shvec_path', type=str, required=True,
                   help='.mat with SH_Vec_matrix')
    p.add_argument('--freq_bin', type=int, default=41)
    p.add_argument('--ear_emb_dim', type=int, default=32)
    p.add_argument('--head_emb_dim', type=int, default=32)
    p.add_argument('--lr_emb_dim', type=int, default=16)
    p.add_argument('--freq_emb_dim', type=int, default=16)
    p.add_argument('--condition_dim', type=int, default=256)
    p.add_argument('--ear_anthro_dim', type=int, default=12)
    p.add_argument('--head_anthro_dim', type=int, default=13)
    p.add_argument('--norm_anthro', type=str2bool, nargs='?', const=True, default=True)
    p.add_argument('--anthro_norm_method', type=str, default='chun2017',
                   choices=['standard','chun2017'])
    p.add_argument('-i', '--val_idx', type=int, default=0)
    p.add_argument('--norm', type=str, default='layer',
                   choices=['batch','layer','instance'])
    p.add_argument('--target', type=str, default='sht', choices=['sht','hrtf'])
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--num_epochs', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--equivariant', type=str2bool, nargs='?', const=True, default=True,
                   help='Enable SO(3)-equivariant model')
    p.add_argument('--mul', type=int, default=6,
                   help='multiplicity for Clebsch–Gordan layers')
    p.add_argument('--test_only', action='store_true')
    args = p.parse_args()

    # reproducibility & directories
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.test_only:
        os.makedirs(args.out_fold, exist_ok=True)
        os.makedirs(os.path.join(args.out_fold, 'checkpoint'), exist_ok=True)

    # save config
    with open(os.path.join(args.out_fold, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args


def train(args):
    # load SH basis
    mat = sio.loadmat(args.shvec_path)
    for key in ('Ynm_measured','Ynm','Ynm_val','shvec','Y'):
        if key in mat and isinstance(mat[key], np.ndarray) and mat[key].size:
            Ynm_np = mat[key].astype(np.float32)
            break
    else:
        Ynm_np = next(v.astype(np.float32) for k,v in mat.items()
                      if not k.startswith('__') and isinstance(v, np.ndarray))
    Ynm_full = torch.from_numpy(Ynm_np).to(args.device)

    L_max = int(math.isqrt(Ynm_full.shape[1]) - 1)
    pts   = _make_grid_pts(L_max, args.device)

    # init W&B
    wandb.init(project='equivariant_8_ablation', config=vars(args),
               settings=Settings(init_timeout=300))

    # only first 5 folds
    for fold in range(5):
        print(f"\n─── Fold {fold}/4 ───")
        args.val_idx = fold

        tr_ds = HUTUBS(args, val=False)
        va_ds = HUTUBS(args, val=True)
        tr_ld = DataLoader(tr_ds, args.batch_size, shuffle=True,  num_workers=args.num_workers)
        va_ld = DataLoader(va_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

        # choose model
        if args.equivariant:
            model = EquivariantSHPredictor(num_freq_bins=args.freq_bin,
                                           L=L_max, mul=args.mul).to(args.device)
        else:
            model = ConvNNHrtfSht(args).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        mse_fn = nn.MSELoss()

        for epoch in range(1, args.num_epochs+1):
            # training
            model.train()
            running_train = 0.0
            for head, ear, sht, aux, subj, freq, side in tr_ld:
                head, ear, sht, aux, freq, side = [x.to(args.device) for x in (head, ear, sht, aux, freq, side)]
                optimizer.zero_grad()
                if args.equivariant:
                    pred = model(head, ear, freq, side)
                else:
                    pred = model(ear, head, freq, side)
                loss = mse_fn(pred, sht)
                loss.backward()
                optimizer.step()
                running_train += loss.item()
            avg_train = running_train / len(tr_ld)
            scheduler.step()

            # validation
            model.eval()
            running_val = 0.0
            running_rot = 0.0
            with torch.no_grad():
                for head, ear, sht, aux, subj, freq, side in va_ld:
                    head, ear, sht, aux, freq, side = [x.to(args.device) for x in (head, ear, sht, aux, freq, side)]
                    if args.equivariant:
                        pr = model(head, ear, freq, side)
                    else:
                        pr = model(ear, head, freq, side)
                    running_val += mse_fn(pr, sht).item()
                    running_rot += rotationally_averaged_lsd(sht, pr, L_max, K=5).item()
            avg_val = running_val / len(va_ld)
            avg_rot = running_rot / len(va_ld)

            wandb.log({'fold':fold,'epoch':epoch,
                       'train_loss':avg_train,'val_loss':avg_val,
                       'rotLSD':avg_rot,'lr':scheduler.get_last_lr()[0]})
            if epoch % 50==0:
                print(f"[fold {fold}] epoch {epoch} | train {avg_train:.2f} | val {avg_val:.2f} | rotLSD {avg_rot:.2f}")

        # save checkpoint
        torch.save(model.state_dict(), os.path.join(args.out_fold, f"fold{fold}.pt"))

    wandb.finish()


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    args = init_params()
    if not args.test_only:
        train(args)
