#!/usr/bin/env python3
"""
Compute aggregate LSD numbers for the **baseline CNN** (non-equivariant)
trained on the HUTUBS database.

Outputs (averaged over all folds that have a .pt file):
• √MSE grid-LSD   vs inverse-SHT reconstruction
• √MSE grid-LSD   vs measured HRTF
"""

import os
import math
import json
import argparse
import numpy as np
import torch
import scipy.io as sio
from tqdm import tqdm
from torch.utils.data import DataLoader

from ML_Model.dataset        import HUTUBS
from ML_Model.baseline_model import ConvNNHrtfSht
# from ML_Model.metrics        import rotationally_averaged_lsd  # ← no longer used

EPS = 1e-8  # avoid log/0

def rms(x):        # root-mean-square in dB domain
    return math.sqrt(float(np.mean(x)))

def mse_db(x, y):  # element-wise (x-y)², x/y are already in dB
    return (x - y).pow(2)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", required=True,
                   help="folder that contains fold0.pt … fold92.pt + args.json")
    p.add_argument("--hrtfsht",    required=True,
                   help="Processed_ML/…/HUTUBS_matrix_measured.mat")
    p.add_argument("--shvec",      required=True,
                   help="Processed_ML/…/Measured_Ynm.mat (64×440)")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()

def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # static data 
    mat      = sio.loadmat(args.hrtfsht)
    freq_idx = mat["freq_logind"].squeeze(0).astype(int)            # 41 indices
    # measured HRTF: [93 subjects, 41 F, 440 S, 2 ears]  (float32)
    H_meas_all = torch.from_numpy(mat["hrtf_freq_allDB"][:93, freq_idx]).float().to(device)

    # SH basis on the 440-pt grid  – shape [F=41, C=64, S=440]
    shvec_np  = sio.loadmat(args.shvec)["SH_Vec_matrix"].astype(np.float32)  # [64,440]
    shvec     = torch.from_numpy(shvec_np).to(device)
    shvec     = shvec.unsqueeze(0).repeat(len(freq_idx), 1, 1)               # [41,64,440]

    C  = shvec.shape[1]                       # 64
    L  = int(math.isqrt(C) - 1)               # → 7

    # reload training-time args once 
    with open(os.path.join(args.models_dir, "args.json")) as f:
        cfg = json.load(f)

    # containers for fold-level metrics
    lsd_vs_sht, lsd_vs_meas = [], []

    print("\n▶ Evaluating baseline CNN (non-equivariant)")
    for fold in tqdm(range(93), desc="folds"):
        wpath = os.path.join(args.models_dir, f"fold{fold}.pt")
        if not os.path.isfile(wpath):
            continue                          # skip missing weights

        #rebuild model for this fold 
        nn_args            = argparse.Namespace(**cfg)
        nn_args.device     = device
        nn_args.val_idx    = fold
        model              = ConvNNHrtfSht(nn_args).to(device)
        model.load_state_dict(torch.load(wpath, map_location=device))
        model.eval()

        # validation loader 
        val_ds   = HUTUBS(nn_args, val=True)
        val_ld   = DataLoader(val_ds, batch_size=nn_args.batch_size,
                              shuffle=False, num_workers=nn_args.num_workers)

        gt_list, pr_list = [], []
        with torch.no_grad():
            for head, ear, sht, *rest in val_ld:
                # rest = [aux, subj, freq, side]
                freq, side = rest[-2].to(device), rest[-1].to(device)

                head, ear, sht = head.to(device), ear.to(device), sht.to(device)
                pr             = model(ear, head, freq, side)   # [B, 64]
                gt_list.append(sht)
                pr_list.append(pr)

        gt_sh = torch.cat(gt_list, 0)      # [Btot,64]
        pr_sh = torch.cat(pr_list, 0)

        #1. grid-LSD vs inverse-SHT ("smoothed")
        Btot = gt_sh.shape[0]
        F    = len(freq_idx)
        E    = Btot // F
        H_gt, H_pr = [], []
        for f in range(F):
            idx = slice(f*E, (f+1)*E)
            Yf  = shvec[f].T             # [64,440]
            H_gt.append((gt_sh[idx] @ Yf))
            H_pr.append((pr_sh[idx] @ Yf))
        H_gt = torch.cat(H_gt, 0)       # [Btot,440]
        H_pr = torch.cat(H_pr, 0)
        lsd_vs_sht.append(mse_db(H_gt, H_pr).mean().item())

        #  2. grid-LSD vs measured HRTF 
        H_meas = H_meas_all[fold]         # [41,440,2]
        H_meas = H_meas.permute(0,2,1).reshape(Btot, 440)
        lsd_vs_meas.append(mse_db(H_meas, H_pr).mean().item())

        #skip rotational-LSD at test time! 
        # rot_lsd.append(rotationally_averaged_lsd(gt_sh, pr_sh, L).item())

    # final report 
    print("\n──────── averaged over {:d} folds ────────".format(len(lsd_vs_sht)))
    print(f"√MSE grid-LSD vs  SHT-recon  : {math.sqrt(np.mean(lsd_vs_sht)):6.2f} dB")
    print(f"√MSE grid-LSD vs  measured   : {math.sqrt(np.mean(lsd_vs_meas)):6.2f} dB")

if __name__ == "__main__":
    main()
