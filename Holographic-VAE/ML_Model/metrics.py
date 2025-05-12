import math
import torch
import scipy.io as sio
from e3nn import o3

# cache for your 440-point set if you use measured points
_MEASURED_PTS = None

def _load_measured_pts(path: str, device: torch.device) -> torch.Tensor:
    """
    Load and cache spherical point coordinates [S,3] from a MAT file.
    """
    global _MEASURED_PTS
    if _MEASURED_PTS is None:
        data = sio.loadmat(path)
        pts_np = data['pts']         # expected shape [S,3]
        _MEASURED_PTS = torch.from_numpy(pts_np).to(device).float()
    return _MEASURED_PTS


def _make_grid_pts(L: int, device: torch.device) -> torch.Tensor:
    """
    Generate an equirectangular grid of (L+1)^2 points on the sphere.
    """
    S = (L + 1)**2
    N = int(math.isqrt(S))
    az = torch.linspace(0, 2*math.pi*(1 - 1/N), N, device=device)
    el = torch.linspace(0, math.pi, N, device=device)
    azg, elg = torch.meshgrid(az, el, indexing='ij')
    x = torch.sin(elg).flatten() * torch.cos(azg).flatten()
    y = torch.sin(elg).flatten() * torch.sin(azg).flatten()
    z = torch.cos(elg).flatten()
    return torch.stack([x, y, z], dim=1)  # [S,3]


def rotationally_averaged_lsd(
    sh_t: torch.Tensor,
    sh_p: torch.Tensor,
    L: int,
    K: int = 5,
    eps: float = 1e-3,
    measured_pts_path: str = None,
) -> torch.Tensor:
    """
    Spectral log‐spectral distance averaged over K random rotations.
    The R‐invariant metric on spherical Harmonics coefficients.
    """
    B, _ = sh_t.shape
    device = sh_t.device

    # choose evaluation points
    pts = _load_measured_pts(measured_pts_path, device) if measured_pts_path else _make_grid_pts(L, device)

    acc = torch.zeros((), device=device)

    for _ in range(K):
        # random Euler angles
        alpha = torch.rand(1).item() * 2*math.pi
        beta  = torch.rand(1).item() * math.pi
        gamma = torch.rand(1).item() * 2*math.pi
        ca, sa = math.cos(alpha), math.sin(alpha)
        cb, sb = math.cos(beta),  math.sin(beta)
        cg, sg = math.cos(gamma), math.sin(gamma)

        Rz1 = torch.tensor([
            [ ca, -sa, 0],
            [ sa,  ca, 0],
            [  0,   0, 1]
        ], device=device, dtype=pts.dtype)
        Ry = torch.tensor([
            [ cb, 0, sb],
            [  0, 1,  0],
            [-sb, 0, cb]
        ], device=device, dtype=pts.dtype)
        Rz2 = torch.tensor([
            [ cg, -sg, 0],
            [ sg,  cg, 0],
            [  0,   0, 1]
        ], device=device, dtype=pts.dtype)

        R = Rz1 @ Ry @ Rz2                  # [3,3]
        pts_r = pts @ R.T                   # [S,3]

        # compute spherical harmonics at rotated points
        Y_r = torch.cat([
            o3.spherical_harmonics(l, pts_r, normalize='component').real
            for l in range(L+1)
        ], dim=1)                          # [S, (L+1)^2]

        Ht = (sh_t @ Y_r.T).abs()          # [B, S]
        Hp = (sh_p @ Y_r.T).abs()          # [B, S]

        ratio = ((Ht + eps) / (Hp + eps)).clamp(min=1e-10)
        acc += (20.0 * torch.log10(ratio).abs()).mean()

    return acc / K


def directional_grad_energy(
    H: torch.Tensor,
    L: int = None,
    measured_pts_path: str = None,
    K: int = 6
) -> torch.Tensor:
    """
    Approximate RMS angular‐gradient on an arbitrary spherical point set.

    H : tensor of shape [B, S]
    L : grid parameter if using an (L+1)^2 equirectangular grid
    measured_pts_path : path to .mat containing 'pts' [S,3] for measured sets
    K : number of neighbors in k‑NN graph

    Returns scalar: sqrt(mean((H_i - H_j)^2)) over all edges i→j.
    """
    B, S = H.shape
    device = H.device

    # load or infer sampling points
    if measured_pts_path:
        pts = _load_measured_pts(measured_pts_path, device)
        assert pts.shape[0] == S, (
            f"H has {S} points but measured_pts has {pts.shape[0]}; they must match"
        )
    else:
        # grid must be (L+1)^2 for square sampling
        N = int(math.isqrt(S))
        assert N * N == S, (
            f"H has {S} points which is not a square grid; pass measured_pts_path instead"
        )
        if L is None:
            L = N - 1
        pts = _make_grid_pts(L, device)

    # compute pairwise squared distances
    diff = pts.unsqueeze(1) - pts.unsqueeze(0)
    dist2 = diff.pow(2).sum(-1)
    dist2.fill_diagonal_(float('inf'))

    # find nearest K neighbors
    knn_idx = dist2.topk(K, largest=False)[1]  # [S, K]

    # gather neighbor values and compute energy
    H_neighbors = H[:, knn_idx]                # [B, S, K]
    H_center    = H.unsqueeze(-1)              # [B, S, 1]
    diffs       = H_center - H_neighbors       # [B, S, K]

    mse = diffs.pow(2).mean()
    return torch.sqrt(mse)
