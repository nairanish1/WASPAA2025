#!/usr/bin/env python3
"""
rotate_sh_surface.py

Per-(φ,θ) “heat‐map” of single‐angle rot‐LSD for:
  • Baseline CNN
  • SO(3)-Equivariant SH model

This version builds its own equirectangular SH→grid basis of size G=(L+1)^2,
so there are no shape mismatches.
"""
import os, sys, math, argparse, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401

# repo root on PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from ML_Model.dataset        import HUTUBS
from ML_Model.baseline_model import ConvNNHrtfSht
from ML_Model.model          import EquivariantSHPredictor
from ML_Model.metrics       import _make_grid_pts

import e3nn.o3 as o3


def infer_L_from_ckpt(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    ls = [int(k.split(".")[1])
          for k in sd
          if k.startswith("out_proj.") and k.endswith(".weight")]
    if not ls:
        raise RuntimeError(f"No out_proj.*.weight found in {ckpt_path}")
    return max(ls)


def infer_mul_from_ckpt(ckpt_path, L):
    state = torch.load(ckpt_path, map_location="cpu")
    key   = "blocks.0.layers.0.weight"
    if key not in state:
        return None
    total = state[key].numel()
    mul, rem = divmod(total, L+1)
    return mul if rem == 0 and mul >= 1 else None


def rotate_SH_coeffs(sh, phi, theta):
    """
    Rotate SH coeffs sh [1,C] by Euler(φ,θ,0) via spatial reprojection
    on our equirectangular grid.
    """
    device = sh.device
    global SH_FULL, GRID_PTS, L

    # 1) SH→grid: [1,C] @ [C,G] → [1,G]
    H = sh @ SH_FULL

    # 2) build Z(φ)Y(θ): [3,3]
    φ = math.radians(phi)
    θ = math.radians(theta)
    Z = torch.tensor([
        [ math.cos(φ), -math.sin(φ), 0],
        [ math.sin(φ),  math.cos(φ), 0],
        [           0,            0, 1],
    ], device=device)
    Y = torch.tensor([
        [ math.cos(θ), 0, math.sin(θ)],
        [           0, 1,           0],
        [-math.sin(θ), 0, math.cos(θ)],
    ], device=device)
    R = Z @ Y  # [3,3]

    # 3) rotate grid points: [G,3] @ [3,3].T → [G,3]
    pts_rot = GRID_PTS @ R.T

    # 4) re-evaluate SH basis at rotated points: yields [G, C]
    Ys = [
        o3.spherical_harmonics(l, pts_rot, normalize="component").real
        for l in range(L+1)
    ]
    Y_rot = torch.cat(Ys, dim=1)  # [G, C]

    # 5) pseudo-inverse: [C, G]
    pinv = torch.pinverse(Y_rot)

    # 6) back-project: [1, G] @ [G, C] → [1, C]
    return H @ pinv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",       type=int, required=True)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--base_dir",   required=True)
    parser.add_argument("--eq_dir",     required=True)
    parser.add_argument("--gpu",        type=int, default=0)
    args = parser.parse_args()

    # Device
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 1) infer SH order and capacity
    ckpt_eq = os.path.join(args.eq_dir, f"fold{args.fold}.pt")
    L  = infer_L_from_ckpt(ckpt_eq)
    C  = (L+1)**2
    mul = infer_mul_from_ckpt(ckpt_eq, L) or 4

    # 2) build equirectangular sampling grid of G = (L+1)^2 points
    GRID_PTS = _make_grid_pts(L, device=device)  # [G,3]
    G = GRID_PTS.shape[0]

    # 3) build SH_FULL: [C, G]
    Ys     = [o3.spherical_harmonics(l, GRID_PTS, normalize="component").real
              for l in range(L+1)]
    Y_full = torch.cat(Ys, dim=1)  # [G, C]
    SH_FULL = Y_full.T             # [C, G]

    print(f"▶ L={L}, C={C}, mul={mul}, using equirec grid G={G}")

    # 4) load one validation sample
    cfgb = json.load(open(os.path.join(args.base_dir, "args.json")))
    nsb  = argparse.Namespace(**cfgb)
    nsb.device  = device
    nsb.val_idx = args.fold
    ds  = HUTUBS(nsb, val=True)
    head, ear, true_sh, *rest = ds[args.sample_idx]
    head, ear, true_sh = [x.unsqueeze(0).to(device)
                          for x in (head, ear, true_sh)]
    freq, side = rest[-2], rest[-1]
    freq = torch.tensor([freq], dtype=torch.long, device=device)
    side = torch.tensor([side], dtype=torch.long, device=device)

    # 5) load both models
    base = ConvNNHrtfSht(nsb).to(device)
    base.load_state_dict(torch.load(os.path.join(
        args.base_dir, f"fold{args.fold}.pt"
    ), map_location=device))
    base.eval()

    cfge = json.load(open(os.path.join(args.eq_dir, "args.json")))
    eq = EquivariantSHPredictor(
        num_freq_bins=cfge["freq_bin"], L=L, mul=mul
    ).to(device)
    eq.load_state_dict(torch.load(ckpt_eq, map_location=device))
    eq.eval()

    # 6) forward pass once for SH predictions
    with torch.no_grad():
        pb = base(ear, head, freq, side)  # [1, C]
        pe = eq  (head, ear, freq, side)  # [1, C]

    # 7) sweep φ,θ and compute LSD surface
    phis   = np.arange(0, 360, 5)
    thetas = np.arange(0, 181, 5)
    LSD_b  = np.zeros((len(thetas), len(phis)))
    LSD_e  = np.zeros_like(LSD_b)

    for i, θ in enumerate(thetas):
        for j, φ in enumerate(phis):
            t_r = rotate_SH_coeffs(true_sh, φ, θ)
            b_r = rotate_SH_coeffs(pb,       φ, θ)
            e_r = rotate_SH_coeffs(pe,       φ, θ)

            Ht = t_r @ SH_FULL
            Hb = b_r @ SH_FULL
            He = e_r @ SH_FULL

            eps = 1e-6
            LSD_b[i,j] = (20 * torch.log10((Ht.abs()+eps)/(Hb.abs()+eps))
                          ).mean().item()
            LSD_e[i,j] = (20 * torch.log10((Ht.abs()+eps)/(He.abs()+eps))
                          ).mean().item()

    # 8) plot
    Φ, Θ = np.meshgrid(phis, thetas)
    fig = plt.figure(figsize=(12,5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(Φ, Θ, LSD_b, cmap="magma", edgecolor='none')
    ax1.set_title("Baseline CNN rot‑LSD")
    ax1.set_xlabel("φ (deg)")
    ax1.set_ylabel("θ (deg)")
    ax1.set_zlabel("LSD (dB)")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(Φ, Θ, LSD_e, cmap="viridis", edgecolor='none')
    ax2.set_title("SO(3)‑Equivariant rot‑LSD")
    ax2.set_xlabel("φ (deg)")
    ax2.set_ylabel("θ (deg)")
    ax2.set_zlabel("LSD (dB)")

    plt.tight_layout()
    # save a high‑res figure for your paper
    plt.savefig("rot_lsd_surface.png", dpi=150, bbox_inches="tight")
    plt.show()
