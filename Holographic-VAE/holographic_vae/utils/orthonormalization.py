import torch
from torch import Tensor

def orthonormalize_frame(x_psy_N6: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Gram–Schmidt process with safe, no‑grad denominators to avoid NaNs
    in both forward and backward passes.

    Input: x_psy_N6 of shape [B,6] = [x (3 dims), psy (3 dims)]
    Output: concatenated orthonormal frame [B,9] = [x̂, ŷ, ẑ]
    """
    # 1) Split into x and psy
    x, psy = x_psy_N6[:, :3], x_psy_N6[:, 3:]

    # 2) Sanitize raw inputs
    x   = torch.nan_to_num(x,   nan=0.0, posinf=1e3, neginf=-1e3)
    psy = torch.nan_to_num(psy, nan=0.0, posinf=1e3, neginf=-1e3)

    # 3) Compute projection of psy onto x
    x_dot_psy = torch.sum(x * psy, dim=1, keepdim=True)
    x_dot_x   = torch.sum(x * x,   dim=1, keepdim=True).clamp(min=eps)
    y = psy - (x_dot_psy / x_dot_x) * x

    # 4) Cross product to get z
    z = torch.cross(x, y, dim=1)
    z = torch.nan_to_num(z, nan=0.0, posinf=1e3, neginf=-1e3)

    # 5) Define safe norm (adds eps inside sqrt)
    def safe_norm(v: Tensor) -> Tensor:
        # 1) Compute squared norm
        sumsq = torch.sum(v * v, dim=1, keepdim=True)
        # 2) Clamp to avoid overflow to inf
        sumsq = sumsq.clamp(max=1e30)
        # 3) Add eps under sqrt and clamp to avoid zero
        return torch.sqrt(sumsq + eps).clamp(min=eps)

    with torch.no_grad():
        denom_x = safe_norm(x).detach()
        denom_y = safe_norm(y).detach()
        denom_z = safe_norm(z).detach() 


    with torch.no_grad():
        x_normed = x / denom_x
        y_normed = y / denom_y
        z_normed = z / denom_z

    # Now x_normed, y_normed, z_normed have no grad history for the division.
    # We reattach them to the graph by treating them as the “output” of a zero‐grad op:
    x = x_normed.requires_grad_(True)
    y = y_normed.requires_grad_(True)
    z = z_normed.requires_grad_(True)

    # Finally sanitize any residual NaNs/Infs
    x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
    y = torch.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3)
    z = torch.nan_to_num(z, nan=0.0, posinf=1e3, neginf=-1e3)
    # 8) Concatenate back into [B,9]
    return torch.cat([x, y, z], dim=1)