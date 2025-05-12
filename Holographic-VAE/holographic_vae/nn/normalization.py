
import numpy as np
import torch
from torch import nn
from torch import Tensor
import e3nn
from e3nn import o3

from typing import *


NONLIN_TO_ACTIVATION_MODULES = {
    'swish': 'torch.nn.SiLU()',
    'sigmoid': 'torch.nn.Sigmoid()',
    'relu': 'torch.nn.ReLU()',
    'identity': 'torch.nn.Identity()'
}


class magnitudes_norm(torch.nn.Module):
    '''
    Basically just removes the norms!
    '''
    def __init__(self,
                 irreps_in: o3.Irreps,
                 return_magnitudes: bool = False,
                 eps: float = 1e-8):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_in
        self.return_magnitudes = return_magnitudes
        self.eps = eps

    def forward(self, x: Dict[int, Tensor]) -> Tuple[Tensor, Dict[int, Tensor]]:
        '''
        'magnitude' is equivalent to 'component' in the e3nn naming convention.
        '''
        if self.return_magnitudes:
            magnitudes = []

        directions = {}
        for irr in self.irreps_in:
            feat = x[irr.ir.l]
            norm = feat.pow(2).sum(-1).mean(-1) # [batch]
            norm = torch.sqrt(norm + self.eps)

            directions[irr.ir.l] = feat / norm.unsqueeze(-1)
            
            if self.return_magnitudes:
                magnitudes.append(norm.squeeze())
        
        if self.return_magnitudes:
            return torch.cat(magnitudes, dim=-1), directions
        else:
            return directions


class signal_norm(torch.nn.Module):
    def __init__(self,
                 irreps_in: o3.Irreps,
                 normalization: str = 'component', # norm, component
                 affine: Optional[str] = 'per_feature', # None, unique, per_l, per_feature
                 balanced: Union[bool, float] = False,
                 eps: float = 1e-8):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_in
        self.normalization = normalization
        self.affine = affine
        self.balanced = balanced
        self.eps = eps

        valid_affine_types = [None, 'unique', 'per_l', 'per_feature']
        if self.affine not in valid_affine_types:
            raise NotImplementedError('Affine of type "{}" not implemented. Implemented types include {}'.format(self.affine, valid_affine_types))
        
        valid_normalization_types = ['norm', 'component']
        if self.normalization not in valid_normalization_types:
            raise NotImplementedError('Normalization of type "{}" not implemented. Implemented types include {}'.format(self.normalization, valid_normalization_types))

        if self.balanced: # True or float
            multiplicative_value = 1.0 if isinstance(self.balanced, bool) else self.balanced
            if self.normalization == 'norm':
                self.balancing_constant = self.irreps_in.dim / multiplicative_value
            elif self.normalization == 'component':
                muls, ls = [], []
                for mul, ir in self.irreps_in:
                    muls.append(mul)
                    ls.append(ir.l)
                avg_mul = np.mean(muls)
                num_ls = len(ls)
                self.balancing_constant = float(avg_mul*num_ls) / multiplicative_value
        else:
            self.balancing_constant = 1.0
        
        weights = {}
        if self.affine == 'unique':
            self.weight = torch.nn.parameter.Parameter(torch.ones(1))
        
        if self.affine == 'per_l':
            for irr in self.irreps_in:
                weights[str(irr.ir.l)] = torch.nn.parameter.Parameter(torch.ones(1))
            self.weights = torch.nn.ParameterDict(weights)

            self.bias = torch.nn.parameter.Parameter(torch.zeros(1))
            
        elif self.affine == 'per_feature':
            for irr in self.irreps_in:
                weights[str(irr.ir.l)] = torch.nn.parameter.Parameter(torch.ones(irr.mul))
            self.weights = torch.nn.ParameterDict(weights)

            num_scalar = sum(mul for mul, ir in irreps_in if ir.is_scalar())
            self.bias = torch.nn.parameter.Parameter(torch.zeros((num_scalar, 1)))

    def forward(self, x: Dict[int, Tensor]) -> Dict[int, Tensor]:
        # compute normalization factors in a batch
        norm_factors = 0.0
        for l, feat in x.items():
            if self.normalization == 'norm':
                norm_factors += feat.pow(2).sum(-1).sum(-1)
            elif self.normalization == 'component':
                norm_factors += feat.pow(2).mean(-1).sum(-1)
            
        norm_factors = torch.sqrt(norm_factors + self.eps) / np.sqrt(self.balancing_constant)
        
        # normalize!
        output = {}
        for l in x:
            if self.affine == 'unique':
                output[l] = torch.einsum('bim,b->bim', x[l], torch.reciprocal(norm_factors)) * self.weight
            if self.affine == 'per_l':
                output[l] = torch.einsum('bim,b->bim', x[l], torch.reciprocal(norm_factors)) * self.weights[str(l)]
            elif self.affine == 'per_feature':
                output[l] = torch.einsum('bim,b,i->bim', x[l], torch.reciprocal(norm_factors), self.weights[str(l)])
            else: # no affine
                output[l] = torch.einsum('bim,b->bim', x[l], torch.reciprocal(norm_factors))
            
            if l == 0:
                if self.affine in ['per_l', 'per_feature']:
                    output[l] += self.bias
        
        return output


# adapted from `https://github.com/NVIDIA/DeepLearningExamples/blob/master/DGLPyTorch/DrugDiscovery/SE3Transformer/se3_transformer/model/layers/norm.py`
class layer_norm_nonlinearity(nn.Module):
    """
    Norm-based SE(3)-equivariant nonlinearity.
                 ┌──> feature_norm ──> LayerNorm() ──> Nonlinearity() ──┐
    feature_in ──┤                                              * ──> feature_out
                 └──> feature_phase ────────────────────────────────────┘
    """

    def __init__(self,
                 irreps_in: o3.Irreps,
                 nonlinearity: Optional[Union[nn.Module, str]] = nn.Identity(),
                 affine: bool = True,
                 normalization: str = 'component', # norm, component --> component further normalizes by 2l+1
                 eps: float = 1e-6):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_in
        if isinstance(nonlinearity, str):
            self.nonlinearity = eval(NONLIN_TO_ACTIVATION_MODULES[nonlinearity])
        else:
            self.nonlinearity = nonlinearity
        self.normalization = normalization
        self.eps = eps

        valid_normalization_types = ['norm', 'component']
        if self.normalization not in valid_normalization_types:
            raise NotImplementedError('Normalization of type "{}" not implemented. Implemented types include {}'.format(self.normalization, valid_normalization_types))

        self.layer_norms = nn.ModuleDict({
            str(irr.ir.l): nn.LayerNorm(irr.mul, elementwise_affine=affine)
            for irr in irreps_in
        })

    def forward(self, x: Dict[int, Tensor], *args, **kwargs) -> Dict[int, Tensor]:
        output = {}
        for l, feat in x.items():
            if self.normalization == 'norm':
                norm = feat.pow(2).sum(-1, keepdim=True)
            elif self.normalization == 'component':
                norm = feat.pow(2).mean(-1, keepdim=True)
            
            norm = torch.sqrt(norm + self.eps) # comment this out for experiment on pretrained model for CSE543 report

            new_norm = self.nonlinearity(self.layer_norms[str(l)](norm.squeeze(-1)).unsqueeze(-1))
            output[l] = new_norm * feat / norm

        return output


class batch_norm(nn.Module):
    """
    Batch normalization for dictionaries of steerable tensors.
    This version iterates only over the keys present in the input dictionary.
    If a given degree tensor is empty, it substitutes a zero tensor with the expected shape.
    It prints debugging information and updates running statistics per degree.
    """
    def __init__(self,
                 irreps_in: o3.Irreps,
                 instance: bool = False,
                 layer: bool = False,
                 affine: bool = True,
                 normalization: str = 'component',  # 'norm' or 'component'
                 reduce: str = 'mean',              # 'mean' or 'max'
                 momentum: float = 0.1,
                 eps: float = 1e-5):
        super().__init__()
        assert normalization in ['norm', 'component']
        assert reduce in ['mean', 'max']
        self.irreps_in = irreps_in
        self.irreps_out = irreps_in
        self.layer = layer
        self.instance = instance
        assert not (self.instance and self.layer)
        self.affine = affine
        self.normalization = normalization
        self.reduce = reduce
        self.momentum = momentum
        self.eps = eps

        # Build a lookup from degree l to expected (mul, dim)
        self.expected = {}
        for mul, ir in irreps_in:
            self.expected[ir.l] = (mul, 2 * ir.l + 1)

        # Create running statistics buffers by concatenating expected sizes over degrees.
        running_mean_list = []
        running_var_list = []
        for l in sorted(self.expected.keys()):
            mul, dim = self.expected[l]
            size = mul * dim
            running_mean_list.append(torch.zeros(size))
            running_var_list.append(torch.ones(size))
        self.register_buffer('running_mean', torch.cat(running_mean_list))
        self.register_buffer('running_var', torch.cat(running_var_list))

        if affine:
            total_features = sum(mul for mul, _ in self.expected.values())
            self.weight = nn.Parameter(torch.ones(total_features))
            self.bias = nn.Parameter(torch.zeros(total_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in}, eps={self.eps}, momentum={self.momentum})"

    def _roll_avg(self, curr: Tensor, update: Tensor) -> Tensor:
        return (1 - self.momentum) * curr + self.momentum * update.detach()

    def forward(self, x: Dict[int, Tensor]):
        # Debug: print keys and shapes
        #for key, value in x.items():
            #print(f"[batch_norm] Degree {key} input shape: {value.shape}")

        output = {}
        # Dictionaries to accumulate new running stats per degree.
        new_means = {}
        new_vars = {}

        # First, compute a global normalization factor over the available keys.
        norm_factors = 0.0
        keys = sorted(x.keys())
        for l in keys:
            feat = x[l]
            if self.normalization == 'norm':
                norm_factors += feat.pow(2).sum(-1).sum(-1)
            elif self.normalization == 'component':
                norm_factors += feat.pow(2).mean(-1).sum(-1)
        norm_factors = torch.sqrt(norm_factors + self.eps)
        total_expected = sum(mul * dim for mul, dim in self.expected.values())
        norm_factors = norm_factors / np.sqrt(total_expected)
        
        # Process each degree present in x.
        for l in keys:
            field = x[l]
            orig_shape = field.shape
            if len(orig_shape) == 3:
                batch, mul, d = orig_shape
                field = field.view(batch, 1, mul, d)
            elif len(orig_shape) == 4:
                batch, sample, mul, d = orig_shape
            else:
                raise RuntimeError(f"Unexpected shape {orig_shape} for degree {l}")
            
            # If the current field is empty, substitute with zeros.
            if mul == 0:
                expected_mul, expected_d = self.expected.get(l, (0, 0))
                field = torch.zeros(batch, sample, expected_mul, expected_d,
                                    device=field.device, dtype=field.dtype)
                output[l] = field
                print(f"[batch_norm] Degree {l} is empty; substituting zeros with shape {field.shape}")
                continue

            # For degree 0, subtract the mean.
            if l == 0:
                field_mean = field.mean([0, 1])
                field = field - field_mean.unsqueeze(0).unsqueeze(0)
                new_means[l] = field_mean.reshape(-1)
            # Compute per-degree normalization.
            if self.normalization == 'norm':
                field_norm = torch.sqrt(field.pow(2) + self.eps)
            else:
                field_norm = torch.sqrt(field.pow(2) + self.eps)
            field_norm = field_norm.mean(1) if self.reduce == 'mean' else field_norm.max(1).values
            if not self.instance:
                field_norm = field_norm.mean(0)
                new_vars[l] = field_norm.reshape(-1)
            field_norm = (field_norm + self.eps).pow(-0.5)
            if self.affine:
                # Compute offset from lower degrees.
                offset = sum(self.expected[k][0] for k in sorted(self.expected.keys()) if k < l)
                expected_mul, _ = self.expected[l]
                weight_slice = self.weight[offset: offset + expected_mul]
                weight_slice = weight_slice.view(expected_mul, 1)
                field_norm = field_norm * weight_slice.unsqueeze(0)
            field = field * field_norm.unsqueeze(1)
            if self.affine and l == 0:
                offset = 0
                expected_mul, _ = self.expected.get(l, (0, 0))
                bias_slice = self.bias[offset: offset + expected_mul]
                bias_slice = bias_slice.view(expected_mul, 1)
                field = field + bias_slice.view(1, 1, expected_mul, 1)
            field = field.view(*orig_shape)
            output[l] = field
            #print(f"[batch_norm] Degree {l}: output shape = {output[l].shape}")

        # Update running statistics per degree.
        if self.training and not self.instance and new_means and new_vars:
            new_running_mean = self.running_mean.clone()
            new_running_var = self.running_var.clone()
            offset = 0
            for l in sorted(self.expected.keys()):
                mul, dim = self.expected[l]
                size = mul * dim
                if l in new_means:
                    if new_means[l].numel() != size:
                        print(f"Warning: For degree {l}, new_mean has {new_means[l].numel()} elements; expected {size}. Skipping update for this degree.")
                    else:
                        new_running_mean[offset: offset + size] = self._roll_avg(
                            self.running_mean[offset: offset + size],
                            new_means[l]
                        )
                if l in new_vars:
                    if new_vars[l].numel() != size:
                        print(f"Warning: For degree {l}, new_var has {new_vars[l].numel()} elements; expected {size}. Skipping update for this degree.")
                    else:
                        new_running_var[offset: offset + size] = self._roll_avg(
                            self.running_var[offset: offset + size],
                            new_vars[l]
                        )
                offset += size
            self.running_mean = new_running_mean
            self.running_var = new_running_var

        return output
