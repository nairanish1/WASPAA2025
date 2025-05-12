#!/usr/bin/env python3
"""
Evaluate an SO(3)–equivariant SH predictor on HUTUBS,
auto-inferring the correct L and mul from your checkpoints,
and slicing your full SH grid down to the trained subspace.
"""

import os
import math
import argparse
import json

import numpy as np
import torch
import scipy.io as sio
from tqdm import tqdm
from torch.utils.data import DataLoader

from ML_Model.model   import EquivariantSHPredictor
from ML_Model.dataset import HUTUBS

N_SUBJ = 93


def mse_db(x, y):
    return (x - y).pow(2)


def infer_L_and_C_from_ckpt(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    ls = [
        int(k.split(".")[1])
        for k in state
        if k.startswith("out_proj.") and k.endswith(".weight")
    ]
    if not ls:
        raise RuntimeError(f"Can't find any out_proj.*.weight in {ckpt_path}")
    L       = max(ls)
    C_model = (L + 1) ** 2
    return L, C_model


def infer_mul_from_ckpt(ckpt_path, L):
    state = torch.load(ckpt_path, map_location="cpu")
    key   = "blocks.0.layers.0.weight"
    if key not in state:
        return None
    total = state[key].numel()
    mul, rem = divmod(total, L + 1)
    return mul if rem == 0 and mul >= 1 else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models_dir", required=True,
        help="folder with fold0.pt … fold92.pt + args.json"
    )
    p.add_argument(
        "--hrtfsht", required=True,
        help=".mat with measured HRTF (hrtf_freq_allDB)"
    )
    p.add_argument(
        "--shvec", required=True,
        help=".mat with SH_Vec_matrix"
    )
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ─── infer L and C_model from first checkpoint ─────────────────
    ck0 = os.path.join(args.models_dir, "fold0.pt")
    if not os.path.isfile(ck0):
        raise FileNotFoundError(f"No fold0.pt in {args.models_dir}")
    L, C_model = infer_L_and_C_from_ckpt(ck0)
    mul        = infer_mul_from_ckpt(ck0, L) or 4
    print(f"\n▶ Found L={L}, C_model=(L+1)²={C_model}, mul={mul}")

    # ─── load the *raw* SH grid and correct its orientation ────────
    mat = sio.loadmat(args.shvec)
    raw = mat["SH_Vec_matrix"].astype(np.float32)
    # if it's 440×64, transpose to 64×440; otherwise assume 64×440 already
    if raw.shape[0] == C_model and raw.shape[1] != C_model:
        sh_full_np = raw
    elif raw.shape[1] == C_model and raw.shape[0] != C_model:
        sh_full_np = raw.T
    else:
        raise ValueError(f"SH_Vec_matrix has shape {raw.shape}, can't match C_model={C_model}")

    sh_full = torch.from_numpy(sh_full_np).to(device)   # [C_model, G]
    _, G    = sh_full.shape                             # G should be 440

    #  measured HRTF (for "vs measured") 
    m        = sio.loadmat(args.hrtfsht)
    freq_idx = m["freq_logind"].squeeze(0).astype(int)        # [41]
    H_meas   = torch.from_numpy(
                   m["hrtf_freq_allDB"][:N_SUBJ, freq_idx]
               ).float().to(device)                          # [93,41,440,2]

    # slice & repeat SH into [F, C_model, G] 
    F  = len(freq_idx)
    sh = sh_full.unsqueeze(0).repeat(F, 1, 1)  # [41, 64, 440]

    # reload loader‐cfg so we can rebuild HUTUBS dataloaders 
    cfg_path = os.path.join(args.models_dir, "args.json")
    cfg      = json.load(open(cfg_path)) if os.path.isfile(cfg_path) else {}

    lsd_vs_sht  = []
    lsd_vs_meas = []

    for fold in tqdm(range(N_SUBJ), desc="folds"):
        wpath = os.path.join(args.models_dir, f"fold{fold}.pt")
        if not os.path.isfile(wpath):
            continue

        # rebuild & load the equivariant model
        model = EquivariantSHPredictor(num_freq_bins=F, L=L, mul=mul)
        model.load_state_dict(torch.load(wpath, map_location=device))
        model.to(device).eval()

        # val‐only DataLoader for this fold
        ds_args        = argparse.Namespace(**cfg)
        ds_args.device = device
        ds_args.val_idx= fold
        val_ds = HUTUBS(ds_args, val=True)
        val_ld = DataLoader(
            val_ds,
            batch_size   = ds_args.batch_size,
            shuffle      = False,
            num_workers  = ds_args.num_workers
        )

        gt_list, pr_list = [], []
        with torch.no_grad():
            for head, ear, sht, *rest in val_ld:
                freq, side = rest[-2].to(device), rest[-1].to(device)
                head, ear, sht = head.to(device), ear.to(device), sht.to(device)
                pr = model(head, ear, freq, side)
                gt_list.append(sht)
                pr_list.append(pr)

        gt_sh = torch.cat(gt_list, 0)     # [Btot,  C_model]
        pr_sh = torch.cat(pr_list, 0)

        # 1) grid-LSD vs inverse-SHT
        Btot = gt_sh.size(0)
        E    = Btot // F
        H_gt, H_pr = [], []
        for f in range(F):
            sl = slice(f*E, (f+1)*E)
            Yf = sh[f]                   # <— no .T here, shape [C_model, G]
            H_gt.append(gt_sh[sl] @ Yf)  # [E×C] @ [C×G] → [E×G]
            H_pr.append(pr_sh[sl] @ Yf)
        H_gt = torch.cat(H_gt, 0)        # [Btot, G]
        H_pr = torch.cat(H_pr, 0)
        lsd_vs_sht.append(mse_db(H_gt, H_pr).mean().item())

        # 2) grid-LSD vs measured HRTF
        Hm = H_meas[fold]               # [41, 440, 2]
        Hm = Hm.permute(0,2,1).reshape(Btot, G) 
        lsd_vs_meas.append(mse_db(Hm, H_pr).mean().item())

    # final report
    print(f"\n──────── averaged over {len(lsd_vs_sht)} folds ────────")
    print(f"√MSE grid-LSD vs SHT-recon : {math.sqrt(np.mean(lsd_vs_sht)):6.2f} dB")
    print(f"√MSE grid-LSD vs measured  : {math.sqrt(np.mean(lsd_vs_meas)):6.2f} dB")


if __name__ == "__main__":
    main()
