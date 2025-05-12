# ── ML_Model/model.py ────────────────────────────────────────────────
from __future__ import annotations
import os, sys, math, torch, torch.nn as nn
from typing import Dict, List
import e3nn.o3 as o3
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from holographic_vae.cg_coefficients.get_w3j_coefficients import get_w3j_coefficients
from holographic_vae.nn.blocks import CGBlock



def load_w3j(lmax: int, device: str, dtype: torch.dtype):
    tbl = get_w3j_coefficients(lmax=lmax)
    return {k: torch.tensor(v, device=device, dtype=dtype) for k, v in tbl.items()}


def _pad_to_multiple(t: torch.Tensor, mult: int = 16) -> torch.Tensor:
    """Right-pad *t* (B, L) with zeros so L is a multiple of *mult*."""
    B, L = t.shape
    if L % mult == 0:
        return t
    pad = mult - (L % mult)
    return torch.nn.functional.pad(t, (0, pad), value=0.0)


class EquivariantSHPredictor(nn.Module):
    """
    head(13) + ear(≤64) + freq + L/R  →  64 SH coeffs (L = 7)
    ~56 k parameters,  SO(3)‑equivariant.
    """

    def __init__(self, num_freq_bins: int = 41, L: int = 7, mul: int = 4):
        super().__init__()
        self.L, self.mul = L, mul

        
        hidden = 76
        self.mlp_ae = nn.Sequential(
            nn.Linear(29, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.emb_f  = nn.Embedding(num_freq_bins, 16)
        self.emb_lr = nn.Embedding(2, 8)
        cond_dim = hidden + 16 + 8                               # =100

        # ─ Wigner‑3j tables (CPU—even when model on GPU) 
        dtype = torch.get_default_dtype()
        w3j = load_w3j(L, "cpu", dtype)
        for k, v in w3j.items():
            self.register_buffer(f"w3j_{k}", v, persistent=False)
        w3j_fn = lambda: {k: getattr(self, f"w3j_{k}") for k in w3j}

        # ─ scalar‑to‑tensor “lift”
        self.lift = nn.ModuleDict({
            str(l): nn.Linear(cond_dim, mul * (2*l + 1)) for l in range(L + 1)
        })

        # ─ equivariant trunk (2 CG blocks)
        ir_hidden = o3.Irreps("+".join(f"{mul}x{l}e" for l in range(L + 1)))
        self.blocks = nn.ModuleList([
            CGBlock(
                irreps_in      = ir_hidden,
                irreps_hidden  = ir_hidden,
                w3j_matrices   = w3j_fn(),
                norm_type      = "component",
                normalization  = "norm",
                norm_affine    = True,
                norm_nonlinearity = "relu",
                ch_nonlin_rule = "full",
                ls_nonlin_rule = "full",
            )
            for _ in range(2)
        ])

        # ─ project each ℓ back to single copy 
        self.out_proj = nn.ModuleDict({
            str(l): nn.Linear(2*l + 1, 2*l + 1, bias=False)
            for l in range(L + 1)
        })

    
    def forward(
        self,
        head: torch.Tensor,          # [B, 13]
        ear:  torch.Tensor,          # [B, ≤64]  (will pad)
        freq_idx: torch.Tensor,      # [B,]
        ear_idx:  torch.Tensor       # [B,]
    ) -> torch.Tensor:               # [B, 64]
        B = head.size(0)

        # guard 0 : pad ear‑vector to 16‑multiple (GPU conv needs it)
        ear = _pad_to_multiple(ear, 16)

        # guard 1 : embedding‑index sanity 
        assert freq_idx.max() < self.emb_f.num_embeddings, (
            f"freq index {freq_idx.max().item()} outside [0,{self.emb_f.num_embeddings-1}]"
        )
        assert ear_idx.max()  < self.emb_lr.num_embeddings, "ear‑side index must be 0/1"

        # guard 2 : no NaN / Inf in inputs
        for name, t in (("head", head), ("ear", ear)):
            if torch.isinf(t).any() or torch.isnan(t).any():
                bad = torch.isnan(t) | torch.isinf(t)
                first_bad = torch.nonzero(bad, as_tuple=False)[0]
                raise ValueError(f"{name} has invalid value at {first_bad.tolist()}")

        # ─ concatenated scalar codeword (ℓ = 0)
        scalars = torch.cat([
            self.mlp_ae(torch.cat([head, ear], dim=-1)),   # [B, hidden]
            self.emb_f(freq_idx),                          # [B, 16]
            self.emb_lr(ear_idx)                           # [B,  8]
        ], dim=-1)                                         # [B, 100]

        # lift to irrep tensors  h[ℓ] : [B, mul, 2ℓ+1] 
        h: Dict[int, torch.Tensor] = {
            l: self.lift[str(l)](scalars).view(B, self.mul, 2*l + 1)
            for l in range(self.L + 1)
        }

        # equivariant mixing 
        for blk in self.blocks:
            h = blk(h)

        # average multiplicities, project each ℓ, concat -------------
        parts: List[torch.Tensor] = [
            self.out_proj[str(l)](h[l].mean(1)) for l in range(self.L + 1)
        ]
        return torch.cat(parts, dim=-1)                    # [B, 64]


# smoke‑test (CPU) 
#if __name__ == "__main__":
    #B = 4
    #head = torch.randn(B, 13)
    #ear  = torch.randn(B, 12)           # un‑padded on purpose
    #fidx = torch.randint(0, 41, (B,))
    #eidx = torch.randint(0, 2,  (B,))

    #net = EquivariantSHPredictor(41).cpu()
    #out = net(head, ear, fidx, eidx)
    #print("output OK:", out.shape)      # → (B, 64)

