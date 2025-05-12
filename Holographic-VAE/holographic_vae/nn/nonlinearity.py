import torch
from torch import nn, Tensor
import e3nn
from e3nn import o3
from typing import Dict, List, Optional, Tuple, Union

def get_edges_for_l3_and_L(l3, L, optimize_speed=True):
    import numpy as np
    import networkx as nx

    edges = []
    for l1 in range(L + 1):
        for l2 in range(l1, L + 1):
            if l3 >= np.abs(l1 - l2) and l3 <= l1 + l2:
                if optimize_speed:
                    edges.append((l1, l2, (2 * l1 + 1) * (2 * l2 + 1)))
                else:
                    edges.append((l1, l2, 1))

    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    MST = nx.minimum_spanning_tree(G, weight="weight")

    # Add self-connections
    for l in range(L + 1):
        if l3 <= l + l:
            MST.add_edge(l, l)

    # Ensure (l1, l2) with l1>=l2, then sort
    edges = sorted((max(a,b), min(a,b)) for a,b in MST.edges)
    return edges

def get_efficient_connections(
    L_in: int, L_out: Union[int, Tuple[int, int]]
) -> Dict[int, Dict[int, List[int]]]:
    if isinstance(L_out, int):
        L_out = (0, L_out)
    connections: Dict[int, Dict[int, List[int]]] = {}
    for l3 in range(L_out[0], L_out[1] + 1):
        for l1, l2 in get_edges_for_l3_and_L(l3, L_in):
            connections.setdefault(l1, {}).setdefault(l2, []).append(l3)
    return connections

class TP_nonlinearity(nn.Module):
    """
    Implements an SO(3) tensor-product nonlinearity, registering each Wigner-3j
    coefficient tensor as a buffer so that .to(device) moves them onto CUDA.
    """
    def __init__(
        self,
        irreps_in: o3.Irreps,
        w3j_matrices: Dict[Tuple[int,int,int], Tensor],
        filter_ir_out: Optional[List[Union[str, o3.Irrep]]] = None,
        ls_rule: str = "full",
        channel_rule: str = "full",
        filter_symmetric: bool = True,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.filter_symmetric = filter_symmetric

        # register each 3j-matrix as a buffer:
        for (l1, l2, l3), mat in w3j_matrices.items():
            self.register_buffer(f"w3j_{l1}_{l2}_{l3}", mat)

        # parity check
        for item in irreps_in:
            ir = item[1] if isinstance(item, tuple) else item
            assert ir.p == 1, f"Expected parity 1 but got {ir.p}"

        self.all_ls = sorted(set(irreps_in.ls))
        assert ls_rule in ["full", "elementwise", "efficient"]
        assert channel_rule in ["full", "elementwise"]
        self.ls_rule = ls_rule
        self.channel_rule = channel_rule

        if filter_ir_out is not None:
            filter_ir_out = [o3.Irrep(ir) if isinstance(ir, str) else ir
                             for ir in filter_ir_out]

        # build irreps_out & ls_out & set_ls_out
        if ls_rule in ["full", "elementwise"]:
            out = []
            for mul1, ir1 in irreps_in:
                for mul2, ir2 in irreps_in:
                    if filter_symmetric and ir2.l < ir1.l:
                        continue
                    if ls_rule=="elementwise" and ir1!=ir2:
                        continue
                    for ir_out in ir1 * ir2:
                        if filter_ir_out and ir_out not in filter_ir_out:
                            continue
                        if channel_rule=="full":
                            out.append((mul1*mul2, ir_out))
                        else:  # elementwise
                            assert mul1==mul2
                            out.append((mul1, ir_out))
            self.irreps_out = o3.Irreps(out).sort().irreps.simplify()
        else:  # efficient
            ls_in = [ir.l for _,ir in irreps_in]
            L_in = ls_in[-1]
            ls_out = [o3.Irrep(ir).l for ir in filter_ir_out]  # type: ignore
            self.connections = get_efficient_connections(L_in, (ls_out[0], ls_out[-1]))
            l3_mul = {}
            for mul1, ir1 in irreps_in:
                for mul2, ir2 in irreps_in:
                    for l3 in self.connections.get(ir1.l, {}).get(ir2.l, []):
                        l3_mul[l3] = l3_mul.get(l3, 0) + (mul1*mul2 if channel_rule=="full" else mul1)
            self.irreps_out = o3.Irreps([(c,f"{l3}e") for l3,c in l3_mul.items()]).sort().irreps.simplify()

        self.ls_out = [ir.ir.l for ir in self.irreps_out]
        self.set_ls_out = set(self.ls_out)

    def forward(self, x: Dict[int, Tensor]) -> Dict[int, Tensor]:
        output = {l3: [] for l3 in self.ls_out}
        available = sorted(x.keys())

        def get_w3j(l1,l2,l3):
            return getattr(self, f"w3j_{l1}_{l2}_{l3}")

        if self.ls_rule in ["full", "elementwise"]:
            for l1 in available:
                for l2 in available:
                    if self.ls_rule=="elementwise" and l1!=l2: continue
                    if self.filter_symmetric and l2<l1: continue
                    possible = range(abs(l1-l2), l1+l2+1)
                    outs = [l for l in possible if l in self.set_ls_out]
                    if not outs: continue

                    if self.channel_rule=="full":
                        op = torch.einsum("bim,bjn->bijmn", x[l1], x[l2])
                        b,i,j,m,n = op.shape
                        op = op.reshape(b, i*j, m, n)
                    else:
                        op = torch.einsum("bim,bin->bimn", x[l1], x[l2])

                    for l3 in outs:
                        w = get_w3j(l1,l2,l3)
                        output[l3].append(torch.einsum("mnM,bimn->biM", w, op))

        else:  # efficient
            for l1, sub in self.connections.items():
                for l2, outs in sub.items():
                    if self.channel_rule=="full":
                        op = torch.einsum("bim,bjn->bijmn", x[l1], x[l2])
                        b,i,j,m,n = op.shape
                        op = op.reshape(b, i*j, m, n)
                    else:
                        op = torch.einsum("bim,bin->bimn", x[l1], x[l2])

                    for l3 in outs:
                        w = get_w3j(l1,l2,l3)
                        output[l3].append(torch.einsum("mnM,bimn->biM", w, op))

        # concatenate or zero-size
        for l3 in self.ls_out:
            if output[l3]:
                output[l3] = torch.cat(output[l3], dim=1)
            else:
                batch = next(iter(x.values())).shape[0]
                output[l3] = torch.zeros(batch, 0, device=next(iter(x.values())).device)
        return output
