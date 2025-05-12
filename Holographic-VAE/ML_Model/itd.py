#!/usr/bin/env python3
"""
ML_Model/itd.py

Reconstruct and plot per‑frequency ITD (via minimum‑phase),
ILD, and HRTF magnitude for fold 0
(comparing Baseline CNN vs SO(3)-Equivariant SH model).
"""
import os, sys
import argparse, json

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# add repo root to PYTHONPATH
top = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, top)

from ML_Model.dataset        import HUTUBS
from ML_Model.baseline_model import ConvNNHrtfSht
from ML_Model.model          import EquivariantSHPredictor
from ML_Model.metrics       import _make_grid_pts
import e3nn.o3 as o3


def infer_L_and_mul(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    ls = [int(k.split('.')[1])
          for k in sd if k.startswith('out_proj.') and k.endswith('.weight')]
    L = max(ls) if ls else 0
    key = 'blocks.0.layers.0.weight'
    if key in sd:
        total = sd[key].numel()
        mul, rem = divmod(total, L+1)
        if rem == 0 and mul >= 1:
            return L, mul
    return L, 4


def minimum_phase_itd(mag_db):
    """
    approximate ITD from mag only via Hilbert minimum‑phase
    """
    mag_lin  = 10.0**(mag_db / 20.0)
    log_mag  = np.log(np.maximum(mag_lin, 1e-8))
    analytic = hilbert(log_mag)
    phase_mp = -np.imag(analytic)
    ω        = np.linspace(0, np.pi, len(phase_mp))
    return -np.gradient(np.unwrap(phase_mp), ω)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True, help="baseline_run directory")
    p.add_argument("--eq_dir",   required=True, help="equivariant_run directory")
    p.add_argument("--dir_idx",  type=int, default=0,
                   help="which of the G sampled directions (0…G−1) to plot")
    p.add_argument("--gpu",      type=int, default=0)
    p.add_argument("--out",      default="fold0_itd_ild_mag.png")
    args = p.parse_args()

    # ─── device ─────────────────────────────────────────────────────────────
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    fold   = 0
    # ─── 1) infer SH order & multiplicity ──────────────────────────────────
    ckpt_eq = os.path.join(args.eq_dir, f"fold{fold}.pt")
    L, mul  = infer_L_and_mul(ckpt_eq)
    C       = (L+1)**2

    # ─── 2) build equirectangular grid & SH basis ─────────────────────────
    GRID    = _make_grid_pts(L, device=device)          # [G×3]
    G       = GRID.shape[0]
    assert 0 <= args.dir_idx < G, f"--dir_idx must be in [0..{G-1}]"
    Ys      = [o3.spherical_harmonics(l, GRID, normalize='component').real
               for l in range(L+1)]
    Y_full  = torch.cat(Ys, dim=1)                     # [G×C]
    SH_FULL = Y_full.T.to(device)                      # [C×G]

    # ─── 3) load fold‑0 validation set ────────────────────────────────────
    cfg        = json.load(open(os.path.join(args.base_dir, "args.json")))
    nsb        = argparse.Namespace(**cfg)
    nsb.device = device
    nsb.val_idx= fold
    ds         = HUTUBS(nsb, val=True)
    from torch.utils.data import DataLoader
    ld         = DataLoader(ds,
                            batch_size=nsb.batch_size,
                            shuffle=False,
                            num_workers=nsb.num_workers)

    # ─── 4) load both models ───────────────────────────────────────────────
    base = ConvNNHrtfSht(nsb).to(device)
    base.load_state_dict(torch.load(os.path.join(
        args.base_dir, f"fold{fold}.pt"), map_location=device))
    base.eval()

    cfge = json.load(open(os.path.join(args.eq_dir, "args.json")))
    eq   = EquivariantSHPredictor(
              num_freq_bins=cfge["freq_bin"],
              L=L, mul=mul
          ).to(device)
    eq.load_state_dict(torch.load(ckpt_eq, map_location=device))
    eq.eval()

    # ─── 5) predict SH coeffs for all freq×ear ────────────────────────────
    all_pb, all_pe = [], []
    with torch.no_grad():
        for head, ear, sht, *rest in ld:
            head, ear = head.to(device), ear.to(device)
            freq, side= rest[-2].to(device), rest[-1].to(device)
            all_pb.append(base(ear, head, freq, side))
            all_pe.append(eq  (head, ear, freq, side))

    # assemble into [F×2×C]
    pb = torch.cat(all_pb).view(-1, 2, C).cpu().numpy()
    pe = torch.cat(all_pe).view(-1, 2, C).cpu().numpy()
    F  = pb.shape[0]   # number of frequency bins

    # ─── 6) reconstruct HRTF magnitude [F×2×G] ───────────────────────────
    SH_np = SH_FULL.cpu().numpy()                      # [C×G]
    Hb    = np.einsum("fec,cg->feg", pb, SH_np)
    He    = np.einsum("fec,cg->feg", pe, SH_np)

    d      = args.dir_idx
    f_axis = np.arange(F)

    # ─── 7) ILD & ITD ─────────────────────────────────────────────────────
    ild_base = Hb[:,0,d] - Hb[:,1,d]
    ild_equi = He[:,0,d] - He[:,1,d]

    itd_base = minimum_phase_itd(Hb[:,0,d]) - minimum_phase_itd(Hb[:,1,d])
    itd_equi = minimum_phase_itd(He[:,0,d]) - minimum_phase_itd(He[:,1,d])

    # ─── 8) plot ───────────────────────────────────────────────────────────
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,4), sharex=True)

    ax1.plot(f_axis, itd_base, 'r-',  label='Baseline')
    ax1.plot(f_axis, itd_equi, 'b--', label='Equivariant')
    ax1.set(title=f'ITD @ dir {d}', xlabel='Freq bin', ylabel='ITD (samples)')
    ax1.legend()

    ax2.plot(f_axis, ild_base, 'r-',  label='Baseline')
    ax2.plot(f_axis, ild_equi, 'b--', label='Equivariant')
    ax2.set(title=f'ILD @ dir {d}', xlabel='Freq bin', ylabel='ILD (dB)')
    ax2.legend()

    ax3.plot(f_axis, Hb[:,0,d], 'r-',  label='Base L')
    ax3.plot(f_axis, He[:,0,d], 'b--', label='Equi L')
    ax3.plot(f_axis, Hb[:,1,d], 'r:',  label='Base R')
    ax3.plot(f_axis, He[:,1,d], 'b-.', label='Equi R')
    ax3.set(title=f'HRTF mag @ dir {d}', xlabel='Freq bin', ylabel='Mag (dB)')
    ax3.legend(fontsize='small')

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.show()
